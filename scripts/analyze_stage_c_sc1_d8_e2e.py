#!/usr/bin/env python3
"""Analyze the validation-only SC1-D8 end-to-end PAF screening matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path
from statistics import mean, median
from typing import Any


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
ARMS = (
    "a6_e2e",
    "geo_c256_e2e",
    "perm_c256_e2e",
    "random_c256_e2e",
    "geo_m694_e2e",
    "perm_m694_e2e",
    "random_m694_e2e",
)
READOUTS = {
    "a6_e2e": "learned-basis-forecast-operator",
    "geo_c256_e2e": "plgo-paf-geo-c256",
    "perm_c256_e2e": "plgo-paf-perm-c256",
    "random_c256_e2e": "plgo-paf-random-c256",
    "geo_m694_e2e": "plgo-paf-geo-m694",
    "perm_m694_e2e": "plgo-paf-perm-m694",
    "random_m694_e2e": "plgo-paf-random-m694",
}
PRIMARY = "geo_c256_e2e"
COMPACT_CONTROLS = ("perm_c256_e2e", "random_c256_e2e")
STANDARD_HORIZONS = (48, 96, 192, 336, 720)
SEGMENTS = (
    ("H1-48", 1, 48),
    ("H49-96", 49, 96),
    ("H97-192", 97, 192),
    ("H193-336", 193, 336),
    ("H337-720", 337, 720),
)
SEED = 2021


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_dir(root: Path, arm: str, dataset: str) -> Path:
    return root / arm / dataset / "h720_full" / f"seed{SEED}"


def finite_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value: {value}")
    return number


def relative_improvement(candidate: float, reference: float) -> float:
    return (1.0 - candidate / reference) * 100.0


def load_run(root: Path, arm: str, dataset: str) -> dict[str, Any]:
    directory = run_dir(root, arm, dataset)
    required = {
        "metrics": directory / "metrics_by_target_horizon.csv",
        "training": directory / "training_log.csv",
        "config": directory / "effective_config.json",
        "diagnostics": directory / "model_diagnostics.json",
        "invariants": directory / "trained_invariants.json",
        "patch": directory / "patch_diagnostics.json",
        "patch_rows": directory / "patch_diagnostics_by_patch.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if arm != "a6_e2e" and not (
        directory / "atom_patch_jacobian_norm.npz"
    ).is_file():
        missing.append("atom_patch_jacobian")
    if missing:
        return {
            "dataset": dataset,
            "arm": arm,
            "status": "missing",
            "missing": ",".join(missing),
            "run_dir": str(directory),
        }

    metric_rows = read_csv(required["metrics"])
    by_horizon = {int(row["target_horizon"]): row for row in metric_rows}
    if sorted(by_horizon) != list(range(1, 721)):
        return {
            "dataset": dataset,
            "arm": arm,
            "status": "incomplete_dense_horizons",
            "dense_horizon_count": len(by_horizon),
            "run_dir": str(directory),
        }
    mse = [finite_float(by_horizon[h]["mse"]) for h in range(1, 721)]
    mae = [finite_float(by_horizon[h]["mae"]) for h in range(1, 721)]
    training = read_csv(required["training"])
    if not training:
        raise ValueError(f"empty training log: {required['training']}")
    for row in training:
        for field in ("train_loss", "val_mean_mse", "lr"):
            finite_float(row[field])
    config = json.loads(required["config"].read_text(encoding="utf-8"))
    diagnostics = json.loads(
        required["diagnostics"].read_text(encoding="utf-8")
    )
    invariants = json.loads(required["invariants"].read_text(encoding="utf-8"))
    patch = json.loads(required["patch"].read_text(encoding="utf-8"))
    patch_rows = read_csv(required["patch_rows"])
    adapter = config["adapter"]
    contract = config.get("training_contract", {})
    protocol_ok = (
        adapter["dataset"] == dataset
        and adapter["mode"] == "unified"
        and adapter["pred_len"] == 720
        and adapter["target_horizons"] == [720]
        and adapter["validation_horizons"] == [720]
        and adapter["checkpoint_policy"] == "best-val"
        and adapter["pred_loss_mode"] == "full"
        and adapter["protocol_class"] == "method_screening"
        and adapter["readout_mode"] == READOUTS[arm]
        and adapter["evaluation_prefix_mode"] == "full-crop"
        and adapter["final_evaluation_split"] == "val"
        and contract.get("initialization") == "from_scratch"
        and contract.get("checkpoint_input") is None
        and contract.get("encoder_trainable") is True
        and contract.get("decoder_trainable") is True
        and diagnostics.get("frozen_parameter_tensors") == 0
    )
    patch_ok = (
        patch.get("finite") is True
        and len(patch_rows) == int(patch["patch_num"])
        and finite_float(patch["flatten_block_sum_max_abs"]) <= 1e-5
        and 0.0 <= finite_float(patch["patch_contribution_entropy"]) <= 1.0 + 1e-6
    )
    invariants_ok = invariants.get("pass") is True
    status = "ok" if protocol_ok and patch_ok and invariants_ok else "audit_fail"
    result: dict[str, Any] = {
        "dataset": dataset,
        "arm": arm,
        "status": status,
        "protocol_pass": protocol_ok,
        "invariant_pass": invariants_ok,
        "patch_audit_pass": patch_ok,
        "readout_mode": adapter["readout_mode"],
        "profile_hash": adapter["profile_hash"],
        "dense_horizon_count": len(by_horizon),
        "dense_mse_auc": mean(mse),
        "dense_mae_auc": mean(mae),
        "h720_mse": mse[-1],
        "h720_mae": mae[-1],
        "epochs_ran": len(training),
        "epoch_limit": int(adapter["epochs"]),
        "hit_epoch_limit": len(training) >= int(adapter["epochs"]),
        "best_epoch": int(training[-1]["best_epoch_so_far"]),
        "best_val_mse": min(finite_float(row["val_mean_mse"]) for row in training),
        "full_prefix_max_abs": finite_float(invariants["full_prefix_max_abs"]),
        "patch_contribution_entropy": finite_float(
            patch["patch_contribution_entropy"]
        ),
        "atom_patch_profile_diversity": patch.get(
            "atom_patch_profile_diversity", ""
        ),
        "total_parameters": diagnostics.get("total_parameters", ""),
        "active_forward_parameters": diagnostics.get(
            "active_forward_parameters", ""
        ),
        "decoder_parameters": diagnostics.get(
            "plgo_paf_decoder_parameters",
            diagnostics.get("active_forward_parameters", ""),
        ),
        "run_dir": str(directory),
    }
    for horizon in STANDARD_HORIZONS:
        result[f"h{horizon}_mse"] = finite_float(by_horizon[horizon]["mse"])
        result[f"h{horizon}_mae"] = finite_float(by_horizon[horizon]["mae"])
    return result


def comparison_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["dataset"], row["arm"]): row
        for row in summary
        if row["status"] == "ok"
    }
    rows = []
    for dataset in DATASETS:
        baseline = lookup.get((dataset, "a6_e2e"))
        if baseline is None:
            continue
        for arm in ARMS[1:]:
            candidate = lookup.get((dataset, arm))
            if candidate is None:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "candidate_arm": arm,
                    "reference_arm": "a6_e2e",
                    "dense_mse_improvement_pct": relative_improvement(
                        candidate["dense_mse_auc"], baseline["dense_mse_auc"]
                    ),
                    "dense_mae_improvement_pct": relative_improvement(
                        candidate["dense_mae_auc"], baseline["dense_mae_auc"]
                    ),
                }
            )
    return rows


def segment_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        metrics: dict[str, dict[int, float]] = {}
        for arm in ARMS:
            path = run_dir(root, arm, dataset) / "metrics_by_target_horizon.csv"
            if not path.is_file():
                metrics = {}
                break
            metrics[arm] = {
                int(row["target_horizon"]): finite_float(row["mse"])
                for row in read_csv(path)
            }
        if not metrics:
            continue
        for arm in ARMS[1:]:
            for label, start, end in SEGMENTS:
                candidate = mean(metrics[arm][h] for h in range(start, end + 1))
                reference = mean(
                    metrics["a6_e2e"][h] for h in range(start, end + 1)
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "candidate_arm": arm,
                        "segment": label,
                        "mse_improvement_pct": relative_improvement(
                            candidate, reference
                        ),
                    }
                )
    return rows


def decide_gate(summary: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in summary if row.get("status") == "ok"]
    expected = len(DATASETS) * len(ARMS)
    if len(valid) != expected:
        return {
            "candidate": "SC1-D8-E2E",
            "decision": "incomplete_or_invalid",
            "valid_runs": len(valid),
            "expected_runs": expected,
            "pass": False,
        }
    lookup = {(row["dataset"], row["arm"]): row for row in valid}
    versus_a6 = [
        relative_improvement(
            lookup[(dataset, PRIMARY)]["dense_mse_auc"],
            lookup[(dataset, "a6_e2e")]["dense_mse_auc"],
        )
        for dataset in DATASETS
    ]
    versus_matched = []
    for dataset in DATASETS:
        control = median(
            lookup[(dataset, arm)]["dense_mse_auc"]
            for arm in COMPACT_CONTROLS
        )
        versus_matched.append(
            relative_improvement(
                lookup[(dataset, PRIMARY)]["dense_mse_auc"], control
            )
        )
    mae = [
        relative_improvement(
            lookup[(dataset, PRIMARY)]["dense_mae_auc"],
            lookup[(dataset, "a6_e2e")]["dense_mae_auc"],
        )
        for dataset in DATASETS
    ]
    epoch_cap_count = sum(
        bool(lookup[(dataset, PRIMARY)]["hit_epoch_limit"])
        for dataset in DATASETS
    )
    degradation_over_100 = any(value < -100.0 for value in versus_a6)
    checks = {
        "primary_vs_a6_macro": mean(versus_a6) >= 1.0,
        "primary_vs_a6_worst_dataset": min(versus_a6) >= -0.5,
        "primary_vs_matched_macro": mean(versus_matched) >= 0.5,
        "primary_vs_matched_positive_datasets": sum(
            value > 0.0 for value in versus_matched
        )
        >= 4,
        "primary_mae_macro_nonnegative": mean(mae) >= 0.0,
        "no_over_100pct_degradation": not degradation_over_100,
        "not_epoch_cap_dominated": epoch_cap_count < 4,
    }
    passed = all(checks.values())
    return {
        "candidate": "SC1-D8-E2E",
        "decision": "partial_pass" if passed else "stable_screen_fail",
        "pass": passed,
        "valid_runs": len(valid),
        "expected_runs": expected,
        "checks": checks,
        "primary_vs_a6_by_dataset_pct": dict(zip(DATASETS, versus_a6, strict=True)),
        "primary_vs_a6_macro_pct": mean(versus_a6),
        "primary_vs_a6_worst_pct": min(versus_a6),
        "primary_vs_matched_by_dataset_pct": dict(
            zip(DATASETS, versus_matched, strict=True)
        ),
        "primary_vs_matched_macro_pct": mean(versus_matched),
        "primary_vs_matched_positive_datasets": sum(
            value > 0.0 for value in versus_matched
        ),
        "primary_vs_a6_mae_macro_pct": mean(mae),
        "primary_epoch_cap_count": epoch_cap_count,
        "rollback_if_fail": "Step 4 failure attribution audit",
    }


def render_report(gate: dict[str, Any]) -> str:
    return (
        "# SC1-D8-E2E validation screen\n\n"
        f"- decision: `{gate['decision']}`\n"
        f"- valid runs: `{gate['valid_runs']}/{gate['expected_runs']}`\n"
        "- evaluation split: `validation` only\n"
        "- test used: `false`\n\n"
        "该报告只给出 Step 7B effectiveness gate；失败时必须结合 patch diagnostics "
        "区分 exact shared-latent PAF、intervention point 与 optimization pathology。\n"
    )


def synthetic_smoke() -> None:
    rows = []
    for dataset in DATASETS:
        for arm in ARMS:
            mse = {
                "a6_e2e": 1.0,
                "geo_c256_e2e": 0.98,
                "perm_c256_e2e": 0.99,
                "random_c256_e2e": 0.995,
                "geo_m694_e2e": 0.981,
                "perm_m694_e2e": 0.991,
                "random_m694_e2e": 0.996,
            }[arm]
            rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "status": "ok",
                    "dense_mse_auc": mse,
                    "dense_mae_auc": mse,
                    "hit_epoch_limit": False,
                }
            )
    gate = decide_gate(rows)
    if gate["decision"] != "partial_pass":
        raise AssertionError(gate)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "gate.json"
        path.write_text(json.dumps(gate), encoding="utf-8")
        json.loads(path.read_text(encoding="utf-8"))
    print("stage_c_sc1_d8_analyzer_synthetic=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    if args.raw_root is None or args.output_dir is None:
        raise ValueError("--raw-root and --output-dir are required")
    summary = [
        load_run(args.raw_root, arm, dataset)
        for dataset in DATASETS
        for arm in ARMS
    ]
    comparisons = comparison_rows(summary)
    segments = segment_rows(args.raw_root)
    gate = decide_gate(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "run_summary.csv", summary)
    if comparisons:
        write_csv(args.output_dir / "a6_comparisons.csv", comparisons)
    if segments:
        write_csv(args.output_dir / "horizon_segment_comparisons.csv", segments)
    (args.output_dir / "gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "research_interpretation.md").write_text(
        render_report(gate), encoding="utf-8"
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
