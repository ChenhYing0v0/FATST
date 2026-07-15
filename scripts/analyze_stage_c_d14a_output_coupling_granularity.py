#!/usr/bin/env python3
"""Aggregate StageC D14-A0 artifacts and apply the preregistered gates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


CANONICAL_SCALES = (1, 48, 144, 360, 720)
INTERMEDIATE_SCALES = (48, 144, 360)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_smoke:
        missing = [
            name
            for name in ("input_root", "design", "output_dir")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(f"missing required arguments: {', '.join(missing)}")
    return args


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def relative_gain(reference: float, candidate: float) -> float:
    return (reference - candidate) / max(reference, np.finfo(np.float64).tiny)


def arm_index(arms: np.ndarray, name: str) -> int:
    matches = np.flatnonzero(arms == name)
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one arm named {name}, found {len(matches)}")
    return int(matches[0])


def crossing_pairs(
    canonical_bin_loss: np.ndarray, margin: float
) -> tuple[list[str], list[dict[str, Any]]]:
    pairs: list[str] = []
    rows: list[dict[str, Any]] = []
    for left_index, left_scale in enumerate(CANONICAL_SCALES):
        for right_index in range(left_index + 1, len(CANONICAL_SCALES)):
            right_scale = CANONICAL_SCALES[right_index]
            gains = (
                canonical_bin_loss[right_index] - canonical_bin_loss[left_index]
            ) / np.maximum(
                canonical_bin_loss[right_index], np.finfo(np.float64).tiny
            )
            pair = f"s{left_scale}_vs_s{right_scale}"
            crossing = bool(gains.max() >= margin and gains.min() <= -margin)
            if crossing:
                pairs.append(pair)
            rows.append(
                {
                    "pair": pair,
                    "left_scale": left_scale,
                    "right_scale": right_scale,
                    "short_relative_gain_left_vs_right": float(gains[0]),
                    "mid_relative_gain_left_vs_right": float(gains[1]),
                    "long_relative_gain_left_vs_right": float(gains[2]),
                    "max_relative_gain": float(gains.max()),
                    "min_relative_gain": float(gains.min()),
                    "crossing": crossing,
                }
            )
    return pairs, rows


def analyze_dataset(
    input_root: Path, dataset: str, design: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    dataset_dir = input_root / dataset
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    fold_rows: list[dict[str, Any]] = []
    crossing_rows: list[dict[str, Any]] = []
    pair_counts: Counter[str] = Counter()
    all_finite = True
    max_orthogonality_gap = 0.0
    max_condition_number = 0.0
    for fold_meta in metadata["folds"]:
        fold = int(fold_meta["fold"])
        archive = np.load(dataset_dir / f"validation_bin_losses_fold{fold}.npz")
        arms = archive["arms"]
        mse = archive["mse"].astype(np.float64)
        if mse.ndim != 3 or mse.shape[2] != 3:
            raise RuntimeError(f"unexpected bin loss shape for {dataset} fold {fold}: {mse.shape}")
        canonical_indices = [arm_index(arms, f"canonical_s{scale}") for scale in CANONICAL_SCALES]
        random_indices = [
            arm_index(arms, "canonical_s1"),
            *[arm_index(arms, f"random_s{scale}") for scale in INTERMEDIATE_SCALES],
            arm_index(arms, "canonical_s720"),
        ]
        shifted_indices = [
            arm_index(arms, "canonical_s1"),
            *[arm_index(arms, f"shifted_s{scale}") for scale in INTERMEDIATE_SCALES],
            arm_index(arms, "canonical_s720"),
        ]
        selected_index = arm_index(arms, "train_selected_best")
        mean_index = arm_index(arms, "train_mean")
        canonical = mse[canonical_indices]
        random = mse[random_indices]
        shifted = mse[shifted_indices]
        selected = mse[selected_index]
        train_mean = mse[mean_index]
        canonical_oracle = canonical.min(axis=0)
        random_oracle = random.min(axis=0)
        shifted_oracle = shifted.min(axis=0)
        carrier_gain = relative_gain(float(train_mean.mean()), float(selected.mean()))
        oracle_gain = relative_gain(float(selected.mean()), float(canonical_oracle.mean()))
        contiguity_gain = relative_gain(
            float(random_oracle.mean()), float(canonical_oracle.mean())
        )
        shifted_gap = relative_gain(
            float(shifted_oracle.mean()), float(canonical_oracle.mean())
        )
        worst_canonical = float(canonical.mean(axis=(1, 2)).max())
        severe_degradation = (
            relative_gain(float(train_mean.mean()), worst_canonical)
            < -float(design["gates"]["severe_degradation_threshold"])
        )
        pairs, pair_rows = crossing_pairs(
            canonical.mean(axis=1),
            float(design["gates"]["crossing_relative_margin"]),
        )
        pair_counts.update(pairs)
        for row in pair_rows:
            crossing_rows.append({"dataset": dataset, "fold": fold, **row})
        fold_rows.append(
            {
                "dataset": dataset,
                "fold": fold,
                "selected_canonical_arm": fold_meta["selected_canonical_arm"],
                "carrier_skill_relative_gain": carrier_gain,
                "canonical_oracle_relative_gain": oracle_gain,
                "canonical_vs_random_oracle_relative_gain": contiguity_gain,
                "canonical_vs_shifted_oracle_relative_gain": shifted_gap,
                "crossing_pair_count": len(pairs),
                "crossing_pairs": ";".join(pairs),
                "severe_degradation": severe_degradation,
            }
        )
        all_finite = all_finite and bool(fold_meta["all_finite"]) and bool(np.isfinite(mse).all())
        max_orthogonality_gap = max(
            max_orthogonality_gap, float(fold_meta["pca_orthogonality_max_abs"])
        )
        max_condition_number = max(
            max_condition_number, float(fold_meta["feature_condition_number"])
        )
    stable_pairs = sorted(
        pair
        for pair, count in pair_counts.items()
        if count >= int(design["gates"]["fold_direction_required"])
    )
    dataset_row = {
        "dataset": dataset,
        "carrier_skill_relative_gain": float(
            np.mean([row["carrier_skill_relative_gain"] for row in fold_rows])
        ),
        "canonical_oracle_relative_gain": float(
            np.mean([row["canonical_oracle_relative_gain"] for row in fold_rows])
        ),
        "canonical_vs_random_oracle_relative_gain": float(
            np.mean(
                [row["canonical_vs_random_oracle_relative_gain"] for row in fold_rows]
            )
        ),
        "canonical_vs_shifted_oracle_relative_gain": float(
            np.mean(
                [row["canonical_vs_shifted_oracle_relative_gain"] for row in fold_rows]
            )
        ),
        "stable_crossing": bool(stable_pairs),
        "stable_crossing_pairs": ";".join(stable_pairs),
        "severe_degradation": any(row["severe_degradation"] for row in fold_rows),
        "max_pca_orthogonality_gap": max_orthogonality_gap,
        "max_feature_condition_number": max_condition_number,
        "all_finite": all_finite,
    }
    invariants = {
        "uses_test_split": bool(metadata["uses_test_split"]),
        "forecast_model_trained": bool(metadata["forecast_model_trained"]),
        "fold_count": len(metadata["folds"]),
    }
    return fold_rows, crossing_rows, dataset_row, invariants


def parameter_gate(input_root: Path, datasets: list[str], maximum_gap: float) -> tuple[bool, float]:
    maximum_observed_gap = 0.0
    for dataset in datasets:
        with (input_root / dataset / "parameter_budget.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        by_fold: dict[int, list[int]] = {}
        for row in rows:
            if row["family"] != "canonical":
                continue
            by_fold.setdefault(int(row["fold"]), []).append(
                int(row["factor_parameters"])
            )
        for values in by_fold.values():
            reference = values[0]
            maximum_observed_gap = max(
                maximum_observed_gap,
                max(abs(value - reference) / reference for value in values),
            )
    return maximum_observed_gap <= maximum_gap, maximum_observed_gap


def apply_gates(
    dataset_rows: list[dict[str, Any]],
    invariants: list[dict[str, Any]],
    parameter_ok: bool,
    maximum_parameter_gap: float,
    design: dict[str, Any],
) -> dict[str, Any]:
    gates = design["gates"]
    carrier_count = sum(
        row["carrier_skill_relative_gain"] >= float(gates["carrier_skill_gain_min"])
        for row in dataset_rows
    )
    crossing_count = sum(bool(row["stable_crossing"]) for row in dataset_rows)
    contiguity_count = sum(
        row["canonical_vs_random_oracle_relative_gain"] > 0.0
        for row in dataset_rows
    )
    oracle_macro = float(
        np.mean([row["canonical_oracle_relative_gain"] for row in dataset_rows])
    )
    contiguity_macro = float(
        np.mean(
            [row["canonical_vs_random_oracle_relative_gain"] for row in dataset_rows]
        )
    )
    invariant_ok = all(
        not item["uses_test_split"]
        and not item["forecast_model_trained"]
        and item["fold_count"] == len(design["folds"])
        for item in invariants
    )
    invariant_ok = invariant_ok and all(
        row["all_finite"]
        and not row["severe_degradation"]
        and row["max_pca_orthogonality_gap"]
        <= float(gates["max_pca_orthogonality_gap"])
        and row["max_feature_condition_number"]
        <= float(gates["max_feature_condition_number"])
        for row in dataset_rows
    )
    gate_values = {
        "parameter_budget_pass": parameter_ok,
        "maximum_parameter_relative_gap": maximum_parameter_gap,
        "invariants_pass": invariant_ok,
        "carrier_skill_dataset_count": carrier_count,
        "carrier_skill_pass": carrier_count
        >= int(gates["carrier_skill_dataset_required"]),
        "stable_crossing_dataset_count": crossing_count,
        "stable_crossing_pass": crossing_count >= int(gates["dataset_required"]),
        "canonical_oracle_macro_gain": oracle_macro,
        "canonical_oracle_pass": oracle_macro
        >= float(gates["oracle_macro_gain_min"]),
        "contiguity_dataset_count": contiguity_count,
        "contiguity_macro_gain": contiguity_macro,
        "contiguity_pass": contiguity_count
        >= int(gates["contiguity_dataset_required"])
        and contiguity_macro >= float(gates["contiguity_macro_gain_min"]),
    }
    diagnostic_valid = (
        gate_values["parameter_budget_pass"]
        and gate_values["invariants_pass"]
        and gate_values["carrier_skill_pass"]
    )
    all_problem_gates = (
        gate_values["stable_crossing_pass"]
        and gate_values["canonical_oracle_pass"]
        and gate_values["contiguity_pass"]
    )
    if not diagnostic_valid:
        decision = "diagnostic_invalid_for_direction_rejection"
        rollback_step = "Step 2-3 diagnostic redesign"
    elif all_problem_gates:
        decision = "d14_a0_neutral_rrr_pass_return_step4_6_only"
        rollback_step = "none; return to Step 4-6 method design"
    else:
        decision = "d14_a0_neutral_rrr_fail_requires_failure_attribution"
        rollback_step = "Step 2-3 failure-attribution audit"
    return {
        "diagnostic_id": design["diagnostic_id"],
        "role": design["role"],
        "test_used": False,
        "paper_method_implemented": False,
        **gate_values,
        "diagnostic_valid": diagnostic_valid,
        "problem_evidence_pass": bool(diagnostic_valid and all_problem_gates),
        "decision": decision,
        "rollback_step": rollback_step,
    }


def write_interpretation(
    path: Path, gate: dict[str, Any], dataset_rows: list[dict[str, Any]]
) -> None:
    lines = [
        "# StageC D14-A0 Research Interpretation",
        "",
        "## Decision",
        "",
        f"- `decision`: `{gate['decision']}`",
        f"- `diagnostic_valid`: `{str(gate['diagnostic_valid']).lower()}`",
        f"- `problem_evidence_pass`: `{str(gate['problem_evidence_pass']).lower()}`",
        f"- `rollback_step`: {gate['rollback_step']}",
        "- 本诊断没有使用test split、没有训练forecast model，正结果最多授权返回Step 4-6。",
        "",
        "## Gate Summary",
        "",
        f"- carrier skill datasets: {gate['carrier_skill_dataset_count']}/5",
        f"- stable crossing datasets: {gate['stable_crossing_dataset_count']}/5",
        f"- canonical oracle macro gain: {gate['canonical_oracle_macro_gain']:.6f}",
        f"- contiguity datasets: {gate['contiguity_dataset_count']}/5",
        f"- contiguity macro gain: {gate['contiguity_macro_gain']:.6f}",
        "",
        "## Dataset Summary",
        "",
        "| Dataset | Carrier gain | Oracle gain | Contiguity gain | Stable crossing |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in dataset_rows:
        lines.append(
            f"| {row['dataset']} | {row['carrier_skill_relative_gain']:.4%} | "
            f"{row['canonical_oracle_relative_gain']:.4%} | "
            f"{row['canonical_vs_random_oracle_relative_gain']:.4%} | "
            f"{row['stable_crossing']} |"
        )
    lines.extend(
        [
            "",
            "## Failure Attribution Boundary",
            "",
            "carrier或numeric invariant失败只能判定诊断无效；问题gate失败只否定当前neutral PCA64 + linear RRR证据，不能自动否定所有nonlinear E2E coupling机制。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace, design: dict[str, Any]) -> None:
    datasets = [str(value) for value in design["datasets"]]
    fold_rows: list[dict[str, Any]] = []
    crossing_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    invariants: list[dict[str, Any]] = []
    for dataset in datasets:
        fold, crossing, dataset_row, invariant = analyze_dataset(
            args.input_root, dataset, design
        )
        fold_rows.extend(fold)
        crossing_rows.extend(crossing)
        dataset_rows.append(dataset_row)
        invariants.append(invariant)
    parameter_ok, maximum_parameter_gap = parameter_gate(
        args.input_root,
        datasets,
        float(design["gates"]["max_parameter_relative_gap"]),
    )
    gate = apply_gates(
        dataset_rows, invariants, parameter_ok, maximum_parameter_gap, design
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "fold_gate_metrics.csv", fold_rows)
    write_csv(args.output_dir / "crossing_metrics.csv", crossing_rows)
    write_csv(args.output_dir / "dataset_metrics.csv", dataset_rows)
    (args.output_dir / "gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_interpretation(
        args.output_dir / "research_interpretation.md", gate, dataset_rows
    )
    print(
        f"d14a_analysis_done decision={gate['decision']} "
        f"crossing={gate['stable_crossing_dataset_count']}/5",
        flush=True,
    )


def synthetic_smoke() -> None:
    canonical = np.asarray(
        [
            [1.00, 1.20, 1.30],
            [1.05, 1.00, 1.25],
            [1.10, 0.95, 1.10],
            [1.20, 0.90, 1.00],
            [1.30, 0.88, 0.90],
        ]
    )
    pairs, rows = crossing_pairs(canonical, 0.001)
    if not pairs or len(rows) != 10:
        raise RuntimeError("synthetic crossing calculation failed")
    dataset_rows = [
        {
            "carrier_skill_relative_gain": 0.01,
            "stable_crossing": True,
            "canonical_oracle_relative_gain": 0.01,
            "canonical_vs_random_oracle_relative_gain": 0.01,
            "all_finite": True,
            "severe_degradation": False,
            "max_pca_orthogonality_gap": 1e-12,
            "max_feature_condition_number": 1.1,
        }
        for _ in range(5)
    ]
    design = {
        "diagnostic_id": "synthetic",
        "role": "diagnostic_only",
        "folds": [{}, {}, {}],
        "gates": {
            "carrier_skill_gain_min": 0.005,
            "carrier_skill_dataset_required": 3,
            "dataset_required": 3,
            "oracle_macro_gain_min": 0.005,
            "contiguity_dataset_required": 3,
            "contiguity_macro_gain_min": 0.001,
            "max_pca_orthogonality_gap": 1e-8,
            "max_feature_condition_number": 1e6,
        },
    }
    gate = apply_gates(
        dataset_rows,
        [{"uses_test_split": False, "forecast_model_trained": False, "fold_count": 3}] * 5,
        True,
        0.005,
        design,
    )
    if not gate["problem_evidence_pass"]:
        raise RuntimeError("synthetic gate calculation failed")
    print("stage_c_d14a_analyzer_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    design = json.loads(args.design.read_text(encoding="utf-8"))
    analyze(args, design)


if __name__ == "__main__":
    main()
