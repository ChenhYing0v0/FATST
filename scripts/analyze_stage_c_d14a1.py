#!/usr/bin/env python3
"""Analyze one carrier/seed of the StageC D14-A1 diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
SCALES = (1, 48, 144, 360, 720)
CANONICAL_ARMS = tuple(f"c_s{scale}" for scale in SCALES)
RANDOM_ARMS = ("r_s48", "r_s144", "r_s360")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--carrier", choices=["neutral_raw", "a6_natural"], default="neutral_raw"
    )
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_dir(root: Path, carrier: str, arm: str, dataset: str, seed: int) -> Path:
    return root / carrier / arm / dataset / "h720_full" / f"seed{seed}"


def train_score(directory: Path) -> float:
    rows = read_csv(directory / "training_log.csv")
    values = [float(row["train_prediction_full_l1"]) for row in rows]
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"invalid training log: {directory}")
    return min(values)


def horizon_mse(directory: Path, horizon: int = 720) -> float:
    rows = read_csv(directory / "metrics_by_target_horizon.csv")
    matches = [
        float(row["mse"])
        for row in rows
        if int(row["target_horizon"]) == horizon
    ]
    if len(matches) != 1 or not math.isfinite(matches[0]):
        raise RuntimeError(f"invalid horizon metric: {directory} H={horizon}")
    return matches[0]


def relative_gain(reference: float, candidate: float) -> float:
    return (reference - candidate) / max(reference, np.finfo(np.float64).tiny)


def load_arm(
    root: Path, carrier: str, arm: str, dataset: str, seed: int
) -> dict[str, Any]:
    directory = run_dir(root, carrier, arm, dataset, seed)
    required = [
        directory / "training_log.csv",
        directory / "metrics_by_target_horizon.csv",
        directory / "validation_diagnostics.npz",
        directory / "trained_invariants.json",
        directory / "effective_config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing D14-A1 artifacts: " + ", ".join(missing))
    archive = np.load(directory / "validation_diagnostics.npz")
    invariant = json.loads(
        (directory / "trained_invariants.json").read_text(encoding="utf-8")
    )
    config = json.loads((directory / "effective_config.json").read_text(encoding="utf-8"))
    return {
        "directory": directory,
        "train_score": train_score(directory),
        "mse": archive["row_bin_mse"].astype(np.float64),
        "mae": archive["row_bin_mae"].astype(np.float64),
        "persistence_mse": archive["persistence_row_bin_mse"].astype(np.float64),
        "probe": archive["probe_predictions"].astype(np.float64),
        "probe_target": archive["probe_targets"].astype(np.float64),
        "invariant": invariant,
        "config": config,
    }


def crossing_pairs(canonical_bin_mse: np.ndarray, margin: float) -> list[str]:
    pairs: list[str] = []
    for left in range(len(SCALES)):
        for right in range(left + 1, len(SCALES)):
            gains = (canonical_bin_mse[right] - canonical_bin_mse[left]) / np.maximum(
                canonical_bin_mse[right], np.finfo(np.float64).tiny
            )
            if gains.max() >= margin and gains.min() <= -margin:
                pairs.append(f"s{SCALES[left]}_vs_s{SCALES[right]}")
    return pairs


def prediction_disagreement(arms: list[dict[str, Any]]) -> tuple[float, float, float]:
    targets = arms[0]["probe_target"]
    if any(
        arm["probe"].shape != targets.shape
        or not np.array_equal(arm["probe_target"], targets)
        for arm in arms
    ):
        raise RuntimeError("prediction probes are not row-aligned")
    target_centered = targets - targets.mean(axis=1, keepdims=True)
    denominator = math.sqrt(float(np.mean(target_centered**2)))
    denominator = max(denominator, np.finfo(np.float64).tiny)
    values = []
    for left in range(len(arms)):
        for right in range(left + 1, len(arms)):
            difference = arms[left]["probe"] - arms[right]["probe"]
            values.append(math.sqrt(float(np.mean(difference**2))) / denominator)
    return min(values), median(values), max(values)


def dataset_metrics(
    raw_root: Path,
    carrier: str,
    dataset: str,
    seed: int,
    design: dict[str, Any],
) -> dict[str, Any]:
    by_arm = {
        arm: load_arm(raw_root, carrier, arm, dataset, seed)
        for arm in (*CANONICAL_ARMS, *RANDOM_ARMS)
    }
    canonical = [by_arm[arm] for arm in CANONICAL_ARMS]
    shapes = {arm["mse"].shape for arm in by_arm.values()}
    if len(shapes) != 1:
        raise RuntimeError(f"row loss shapes differ for {dataset}: {shapes}")
    persistence = canonical[0]["persistence_mse"]
    if any(
        not np.array_equal(arm["persistence_mse"], persistence)
        for arm in canonical[1:]
    ):
        raise RuntimeError("persistence rows are not aligned")
    selected_index = min(range(len(canonical)), key=lambda index: canonical[index]["train_score"])
    selected = canonical[selected_index]["mse"]
    canonical_stack = np.stack([arm["mse"] for arm in canonical])
    random_stack = np.stack(
        [
            canonical[0]["mse"],
            by_arm["r_s48"]["mse"],
            by_arm["r_s144"]["mse"],
            by_arm["r_s360"]["mse"],
            canonical[-1]["mse"],
        ]
    )
    canonical_oracle = canonical_stack.min(axis=0)
    random_oracle = random_stack.min(axis=0)
    canonical_arm_risk = canonical_stack.mean(axis=(1, 2))
    validation_best_index = int(canonical_arm_risk.argmin())
    validation_best_risk = float(canonical_arm_risk[validation_best_index])
    validation_bin_policy_risk = float(
        canonical_stack.mean(axis=1).min(axis=0).mean()
    )
    canonical_oracle_risk = float(canonical_oracle.mean())
    min_disagreement, median_disagreement, max_disagreement = prediction_disagreement(
        canonical
    )
    mean_bin = canonical_stack.mean(axis=1)
    pairs = crossing_pairs(
        mean_bin, float(design["remote_gates"]["crossing_relative_margin"])
    )
    all_invariants = all(
        arm["invariant"]["pass"]
        and not arm["invariant"]["uses_test_split"]
        and arm["invariant"]["frozen_parameter_tensors"] == 0
        and arm["invariant"].get("row_order") == "dataset_sequential"
        for arm in by_arm.values()
    )
    maximum_parameter_gap = max(
        float(arm["invariant"]["grouped_mlp_parameter_relative_gap"])
        for arm in by_arm.values()
    )
    carrier_gain = relative_gain(float(persistence.mean()), float(selected.mean()))
    oracle_gain = relative_gain(float(selected.mean()), float(canonical_oracle.mean()))
    strict_oracle_gain = relative_gain(
        validation_best_risk,
        canonical_oracle_risk,
    )
    bin_policy_gain = relative_gain(
        validation_best_risk,
        validation_bin_policy_risk,
    )
    sample_over_bin_gain = relative_gain(
        validation_bin_policy_risk,
        canonical_oracle_risk,
    )
    contiguity_gain = relative_gain(float(random_oracle.mean()), float(canonical_oracle.mean()))
    worst = float(canonical_stack.mean(axis=(1, 2)).max())
    severe = relative_gain(float(persistence.mean()), worst) < -float(
        design["remote_gates"]["severe_degradation_threshold"]
    )
    a6_lbf_h720_mse: float | None = None
    train_selected_vs_a6_lbf_h720_gain: float | None = None
    validation_best_vs_a6_lbf_h720_gain: float | None = None
    if carrier == "a6_natural":
        control_directory = run_dir(root=raw_root, carrier=carrier, arm="a6_lbf", dataset=dataset, seed=seed)
        a6_lbf_h720_mse = horizon_mse(control_directory)
        train_selected_h720 = horizon_mse(canonical[selected_index]["directory"])
        validation_best_h720 = min(
            horizon_mse(arm["directory"]) for arm in canonical
        )
        train_selected_vs_a6_lbf_h720_gain = relative_gain(
            a6_lbf_h720_mse,
            train_selected_h720,
        )
        validation_best_vs_a6_lbf_h720_gain = relative_gain(
            a6_lbf_h720_mse,
            validation_best_h720,
        )
    return {
        "dataset": dataset,
        "carrier": carrier,
        "seed": seed,
        "train_selected_arm": CANONICAL_ARMS[selected_index],
        "carrier_skill_relative_gain": carrier_gain,
        "prediction_disagreement_min": min_disagreement,
        "prediction_disagreement_median": median_disagreement,
        "prediction_disagreement_max": max_disagreement,
        "crossing": bool(pairs),
        "crossing_pairs": ";".join(pairs),
        "canonical_oracle_relative_gain": oracle_gain,
        "validation_best_fixed_arm": CANONICAL_ARMS[validation_best_index],
        "oracle_vs_validation_best_fixed_gain": strict_oracle_gain,
        "validation_bin_policy_vs_fixed_gain": bin_policy_gain,
        "sample_oracle_vs_validation_bin_policy_gain": sample_over_bin_gain,
        "canonical_vs_random_oracle_relative_gain": contiguity_gain,
        "a6_lbf_h720_mse": a6_lbf_h720_mse,
        "train_selected_vs_a6_lbf_h720_gain": train_selected_vs_a6_lbf_h720_gain,
        "validation_best_vs_a6_lbf_h720_gain": validation_best_vs_a6_lbf_h720_gain,
        "maximum_parameter_relative_gap": maximum_parameter_gap,
        "severe_degradation": severe,
        "invariants_pass": all_invariants,
    }


def apply_gate(
    rows: list[dict[str, Any]], carrier: str, design: dict[str, Any]
) -> dict[str, Any]:
    gates = design["remote_gates"]
    separation_count = sum(
        row["prediction_disagreement_median"]
        >= float(gates["minimum_prediction_disagreement"])
        for row in rows
    )
    skill_count = sum(
        row["carrier_skill_relative_gain"] >= float(gates["carrier_skill_gain_min"])
        for row in rows
    )
    crossing_count = sum(bool(row["crossing"]) for row in rows)
    oracle_macro = mean(row["canonical_oracle_relative_gain"] for row in rows)
    strict_oracle_macro = mean(
        row["oracle_vs_validation_best_fixed_gain"] for row in rows
    )
    sample_over_bin_macro = mean(
        row["sample_oracle_vs_validation_bin_policy_gain"] for row in rows
    )
    contiguity_count = sum(
        row["canonical_vs_random_oracle_relative_gain"] > 0.0 for row in rows
    )
    contiguity_macro = mean(
        row["canonical_vs_random_oracle_relative_gain"] for row in rows
    )
    invariants_pass = all(
        row["invariants_pass"]
        and not row["severe_degradation"]
        and row["maximum_parameter_relative_gap"]
        <= float(design["local_gates"]["max_parameter_relative_gap"])
        for row in rows
    )
    diagnostic_valid = (
        invariants_pass
        and separation_count >= int(gates["function_separation_dataset_required"])
        and skill_count >= int(gates["carrier_skill_dataset_required"])
    )
    problem_pass = (
        diagnostic_valid
        and crossing_count >= int(gates["stable_crossing_dataset_required"])
        and oracle_macro >= float(gates["oracle_macro_gain_min"])
        and contiguity_count >= int(gates["contiguity_dataset_required"])
        and contiguity_macro >= float(gates["contiguity_macro_gain_min"])
    )
    if carrier == "neutral_raw":
        if not diagnostic_valid:
            decision = "neutral_invalid_for_direction_rejection"
            next_action = "audit carrier skill or trained function separation"
        elif problem_pass:
            decision = "neutral_problem_pass_authorize_a6_sensitivity"
            next_action = "run A6-natural sensitivity without changing neutral decision"
        else:
            decision = "neutral_valid_fail_close_pcsd_ccrl_pair"
            next_action = "rollback Step 2 and reconstruct paper mainline"
    else:
        if problem_pass:
            decision = "a6_sensitivity_confirming"
        else:
            decision = "a6_sensitivity_nonconfirming_no_direction_rejection"
        next_action = "combine only through the preregistered interpretation matrix"
    return {
        "diagnostic_id": design["diagnostic_id"],
        "carrier": carrier,
        "dataset_count": len(rows),
        "function_separation_dataset_count": separation_count,
        "carrier_skill_dataset_count": skill_count,
        "crossing_dataset_count": crossing_count,
        "canonical_oracle_macro_gain": oracle_macro,
        "strict_oracle_vs_validation_best_fixed_macro_gain": strict_oracle_macro,
        "sample_oracle_vs_validation_bin_policy_macro_gain": sample_over_bin_macro,
        "contiguity_dataset_count": contiguity_count,
        "contiguity_macro_gain": contiguity_macro,
        "invariants_pass": invariants_pass,
        "diagnostic_valid": diagnostic_valid,
        "problem_pass": problem_pass,
        "a6_failure_can_reject_scale_hypothesis": False,
        "a6_train_selected_vs_lbf_h720_macro_gain": (
            mean(row["train_selected_vs_a6_lbf_h720_gain"] for row in rows)
            if carrier == "a6_natural"
            else None
        ),
        "a6_validation_best_vs_lbf_h720_macro_gain": (
            mean(row["validation_best_vs_a6_lbf_h720_gain"] for row in rows)
            if carrier == "a6_natural"
            else None
        ),
        "test_used": False,
        "decision": decision,
        "next_action": next_action,
    }


def analyze(args: argparse.Namespace) -> None:
    if args.raw_root is None or args.design is None or args.output_dir is None:
        raise ValueError("raw-root, design, and output-dir are required")
    design = json.loads(args.design.read_text(encoding="utf-8"))
    rows = [
        dataset_metrics(
            args.raw_root, args.carrier, dataset, args.seed, design
        )
        for dataset in DATASETS
    ]
    gate = apply_gate(rows, args.carrier, design)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "dataset_metrics.csv", rows)
    (args.output_dir / "gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# StageC D14-A1 Carrier Result",
        "",
        f"- `carrier`: `{args.carrier}`",
        f"- `decision`: `{gate['decision']}`",
        f"- function separation: {gate['function_separation_dataset_count']}/5",
        f"- carrier skill: {gate['carrier_skill_dataset_count']}/5",
        f"- crossing: {gate['crossing_dataset_count']}/5",
        f"- oracle macro gain: {gate['canonical_oracle_macro_gain']:.6%}",
        f"- strict oracle vs validation-best fixed: {gate['strict_oracle_vs_validation_best_fixed_macro_gain']:.6%}",
        f"- sample oracle vs validation-bin policy: {gate['sample_oracle_vs_validation_bin_policy_macro_gain']:.6%}",
        f"- contiguity macro gain: {gate['contiguity_macro_gain']:.6%}",
        "- test=false；A6 failure cannot reject the scale hypothesis.",
        "",
    ]
    (args.output_dir / "research_interpretation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"d14a1_analysis=pass decision={gate['decision']}")


def synthetic_smoke() -> None:
    design = {
        "diagnostic_id": "synthetic",
        "local_gates": {"max_parameter_relative_gap": 0.01},
        "remote_gates": {
            "minimum_prediction_disagreement": 0.005,
            "function_separation_dataset_required": 3,
            "carrier_skill_gain_min": 0.005,
            "carrier_skill_dataset_required": 3,
            "stable_crossing_dataset_required": 3,
            "oracle_macro_gain_min": 0.005,
            "contiguity_dataset_required": 3,
            "contiguity_macro_gain_min": 0.001,
        },
    }
    rows = [
        {
            "prediction_disagreement_median": 0.01,
            "carrier_skill_relative_gain": 0.01,
            "crossing": True,
            "canonical_oracle_relative_gain": 0.01,
            "oracle_vs_validation_best_fixed_gain": 0.009,
            "sample_oracle_vs_validation_bin_policy_gain": 0.008,
            "canonical_vs_random_oracle_relative_gain": 0.01,
            "train_selected_vs_a6_lbf_h720_gain": -0.01,
            "validation_best_vs_a6_lbf_h720_gain": -0.005,
            "invariants_pass": True,
            "severe_degradation": False,
            "maximum_parameter_relative_gap": 0.001,
        }
        for _ in DATASETS
    ]
    gate = apply_gate(rows, "neutral_raw", design)
    if gate["decision"] != "neutral_problem_pass_authorize_a6_sensitivity":
        raise RuntimeError(f"synthetic gate failed: {gate}")
    negative_rows = [dict(row, crossing=False) for row in rows]
    a6_gate = apply_gate(negative_rows, "a6_natural", design)
    if (
        a6_gate["decision"]
        != "a6_sensitivity_nonconfirming_no_direction_rejection"
        or a6_gate["a6_failure_can_reject_scale_hypothesis"]
    ):
        raise RuntimeError(f"synthetic A6 causal boundary failed: {a6_gate}")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "gate.json"
        path.write_text(json.dumps(gate), encoding="utf-8")
        json.loads(path.read_text(encoding="utf-8"))
    print("stage_c_d14a1_analyzer_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    analyze(args)


if __name__ == "__main__":
    main()
