#!/usr/bin/env python3
"""Analyze the SC2-PCC-v1-TI validation-only Phase-A matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
MODES = (
    "measure_only",
    "equal_skill",
    "pointwise_route_only",
    "pointwise_capability_skill_only",
    "pointwise_prior_composed",
    "pointwise_pcc_v0",
    "transport_skill_only",
    "transport_route_only",
    "pcc_transport_full",
)
PRIMARY = "pcc_transport_full"
REFERENCE_ARMS = ("a6", "pcsd_direct", "dense_matched")
SCALES = (1, 48, 144, 360, 720)
REQUIRED_PCC_LOG_COLUMNS = {
    "train_pcc_total_loss",
    "train_pcc_fused_measure_l1",
    "train_pcc_skill_loss",
    "train_pcc_route_kl",
    "train_pcc_policy_normalized_entropy",
    "train_pcc_policy_usage_max",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_sc2_pcc_step7b.json"),
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


def run_dir(root: Path, arm: str, dataset: str, seed: int) -> Path:
    return root / arm / dataset / "h720_full" / f"seed{seed}"


def finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value: {value}")
    return result


def relative_gain(candidate: float, reference: float) -> float:
    return 1.0 - candidate / reference


def dense_metrics(path: Path) -> tuple[float, float, float, float]:
    rows = read_csv(path)
    by_horizon = {int(row["target_horizon"]): row for row in rows}
    if sorted(by_horizon) != list(range(1, 721)):
        raise ValueError(f"incomplete dense horizons: {path}")
    mse = [finite_float(by_horizon[value]["mse"]) for value in range(1, 721)]
    mae = [finite_float(by_horizon[value]["mae"]) for value in range(1, 721)]
    return mean(mse), mean(mae), mse[-1], mae[-1]


def mechanism_statistics(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        arm_losses = payload["arm_row_bin_mse"].astype(np.float64)
        fused_losses = payload["fused_row_bin_mse"].astype(np.float64)
        usage = payload["policy_row_bin_usage"].astype(np.float64)
        probe_arms = payload["probe_arms"].astype(np.float64)
    oracle = arm_losses.min(axis=-1)
    denominator = max(float(np.sqrt(np.mean(probe_arms**2))), 1e-12)
    pairwise = []
    for left in range(probe_arms.shape[1]):
        for right in range(left + 1, probe_arms.shape[1]):
            pairwise.append(
                float(
                    np.sqrt(
                        np.mean((probe_arms[:, left] - probe_arms[:, right]) ** 2)
                    )
                    / denominator
                )
            )
    entropy = -(
        usage * np.log(np.clip(usage, 1e-12, None))
    ).sum(axis=-1) / math.log(usage.shape[-1])
    policy_usage = usage.mean(axis=(0, 1))
    return {
        "same_run_oracle_headroom": relative_gain(
            float(oracle.mean()),
            float(fused_losses.mean()),
        ),
        "minimum_pairwise_probe_nrmse": min(pairwise),
        "mean_pairwise_probe_nrmse": mean(pairwise),
        "policy_normalized_entropy": float(entropy.mean()),
        "policy_usage_max": float(policy_usage.max()),
        "arm_mse": arm_losses.mean(axis=(0, 1)).tolist(),
    }


def load_pcc_run(
    root: Path,
    mode: str,
    dataset: str,
    seed: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    directory = run_dir(root, mode, dataset, seed)
    required = {
        "metrics": directory / "metrics_by_target_horizon.csv",
        "training": directory / "training_log.csv",
        "config": directory / "effective_config.json",
        "invariants": directory / "trained_invariants.json",
        "diagnostics": directory / "pcsd_validation_diagnostics.npz",
        "gradient": directory / "pcc_shared_gradient_diagnostics.json",
        "initialization": directory / "initialization_contract.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {
            "dataset": dataset,
            "arm": mode,
            "status": "missing",
            "missing": ",".join(missing),
            "run_dir": str(directory),
        }
    adapter = json.loads(required["config"].read_text(encoding="utf-8"))[
        "adapter"
    ]
    invariant = json.loads(required["invariants"].read_text(encoding="utf-8"))
    gradient = json.loads(required["gradient"].read_text(encoding="utf-8"))
    initialization = json.loads(
        required["initialization"].read_text(encoding="utf-8")
    )
    training = read_csv(required["training"])
    log_columns = set(training[0]) if training else set()
    protocol = config["training"]
    protocol_pass = bool(
        adapter["dataset"] == dataset
        and adapter["seed"] == seed
        and adapter["mode"] == "unified"
        and adapter["pred_len"] == 720
        and adapter["target_horizons"] == [720]
        and adapter["validation_horizons"] == [720]
        and adapter["readout_mode"] == "pcsd-coupling-field"
        and adapter["pcsd_policy_mode"] == "direct"
        and adapter["pcsd_partition"] == "canonical"
        and adapter["pcc_objective_mode"] == mode
        and adapter["checkpoint_policy"] == "best-val"
        and adapter["evaluation_prefix_mode"] == "full-crop"
        and adapter["final_evaluation_split"] == "val"
        and adapter["protocol_class"] == "method_screening"
        and adapter["epochs"] == protocol["epochs"]
        and adapter["patience"] == protocol["patience"]
        and adapter["batch_size"] == protocol["batch_size"]
        and adapter["learning_rate"] == protocol["learning_rate"]
        and REQUIRED_PCC_LOG_COLUMNS.issubset(log_columns)
        and invariant.get("uses_test_split") is False
        and invariant.get("pass") is True
        and gradient.get("pass") is True
        and gradient.get("gradient_surgery_applied") is False
    )
    mse_auc, mae_auc, h720_mse, h720_mae = dense_metrics(required["metrics"])
    row = {
        "dataset": dataset,
        "arm": mode,
        "status": "ok" if protocol_pass else "audit_fail",
        "protocol_pass": protocol_pass,
        "dense_mse_auc": mse_auc,
        "dense_mae_auc": mae_auc,
        "h720_mse": h720_mse,
        "h720_mae": h720_mae,
        "epochs_ran": len(training),
        "best_val_h720_mse": min(
            finite_float(value["val_mean_mse"]) for value in training
        ),
        "encoder_initialization_hash": initialization.get(
            "encoder_initialization_hash", ""
        ),
        "pcsd_initialization_hash": initialization.get(
            "pcsd_initialization_hash", ""
        ),
        "shared_gradient_cosine_mean": gradient["pairwise_cosine_mean"],
        "shared_gradient_cosine_min": gradient["pairwise_cosine_min"],
        "shared_gradient_cosine_max": gradient["pairwise_cosine_max"],
        "run_dir": str(directory),
    }
    row.update(mechanism_statistics(required["diagnostics"]))
    return row


def load_reference(
    root: Path,
    arm: str,
    dataset: str,
    seed: int,
) -> dict[str, Any]:
    directory = run_dir(root, arm, dataset, seed)
    metrics = directory / "metrics_by_target_horizon.csv"
    diagnostics = directory / "pcsd_validation_diagnostics.npz"
    if not metrics.is_file():
        return {
            "dataset": dataset,
            "arm": arm,
            "status": "missing",
            "missing": "metrics",
            "run_dir": str(directory),
        }
    mse_auc, mae_auc, h720_mse, h720_mae = dense_metrics(metrics)
    row = {
        "dataset": dataset,
        "arm": arm,
        "status": "ok",
        "dense_mse_auc": mse_auc,
        "dense_mae_auc": mae_auc,
        "h720_mse": h720_mse,
        "h720_mae": h720_mae,
        "run_dir": str(directory),
    }
    if diagnostics.is_file() and arm == "pcsd_direct":
        row.update(mechanism_statistics(diagnostics))
    return row


def comparison_rows(
    lookup: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    references = (*REFERENCE_ARMS, *(mode for mode in MODES if mode != PRIMARY))
    rows = []
    for reference in references:
        gains = []
        for dataset in DATASETS:
            candidate = float(lookup[(dataset, PRIMARY)]["dense_mse_auc"])
            baseline = float(lookup[(dataset, reference)]["dense_mse_auc"])
            gain = relative_gain(candidate, baseline)
            gains.append(gain)
            rows.append(
                {
                    "reference": reference,
                    "dataset": dataset,
                    "pcc_gain_fraction": gain,
                    "pcc_gain_percent": 100.0 * gain,
                }
            )
        rows.append(
            {
                "reference": reference,
                "dataset": "macro",
                "pcc_gain_fraction": mean(gains),
                "pcc_gain_percent": 100.0 * mean(gains),
                "dataset_wins": sum(value > 0.0 for value in gains),
            }
        )
    return rows


def arm_recovery_rows(
    lookup: dict[tuple[str, str], dict[str, Any]],
    reference_root: Path,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        candidate_arm_mse = lookup[(dataset, PRIMARY)]["arm_mse"]
        plain_arm_mse = lookup[(dataset, "pcsd_direct")]["arm_mse"]
        for index, scope in enumerate(SCALES):
            fixed_path = (
                run_dir(reference_root, f"pcsd_fixed_{scope}", dataset, seed)
                / "pcsd_validation_diagnostics.npz"
            )
            with np.load(fixed_path, allow_pickle=False) as payload:
                fixed_mse = float(
                    payload["fused_row_bin_mse"].astype(np.float64).mean()
                )
            plain_degradation = float(plain_arm_mse[index]) / fixed_mse - 1.0
            candidate_degradation = (
                float(candidate_arm_mse[index]) / fixed_mse - 1.0
            )
            relative_reduction = (
                plain_degradation - candidate_degradation
            ) / max(abs(plain_degradation), 1e-12)
            rows.append(
                {
                    "dataset": dataset,
                    "scope": scope,
                    "fixed_scope_mse": fixed_mse,
                    "plain_arm_mse": plain_arm_mse[index],
                    "pcc_arm_mse": candidate_arm_mse[index],
                    "plain_degradation_fraction": plain_degradation,
                    "pcc_degradation_fraction": candidate_degradation,
                    "relative_degradation_reduction": relative_reduction,
                    "arm_pair_improved": candidate_degradation
                    < plain_degradation,
                }
            )
    return rows


def gate_result(
    comparisons: list[dict[str, Any]],
    recovery: list[dict[str, Any]],
    lookup: dict[tuple[str, str], dict[str, Any]],
    gates: dict[str, Any],
) -> dict[str, Any]:
    macro = {
        row["reference"]: row
        for row in comparisons
        if row["dataset"] == "macro"
    }

    def performance_gate(reference: str, wins_key: str, gain_key: str) -> bool:
        return bool(
            int(macro[reference]["dataset_wins"]) >= int(gates[wins_key])
            and float(macro[reference]["pcc_gain_fraction"])
            >= float(gates[gain_key])
        )

    reduction_median = median(
        float(row["relative_degradation_reduction"]) for row in recovery
    )
    improved_pairs = sum(bool(row["arm_pair_improved"]) for row in recovery)
    retention = {
        dataset: float(
            lookup[(dataset, PRIMARY)]["minimum_pairwise_probe_nrmse"]
        )
        / max(
            float(
                lookup[(dataset, "pcsd_direct")][
                    "minimum_pairwise_probe_nrmse"
                ]
            ),
            1e-12,
        )
        for dataset in DATASETS
    }
    entropy_min = min(
        float(lookup[(dataset, PRIMARY)]["policy_normalized_entropy"])
        for dataset in DATASETS
    )
    usage_max = max(
        float(lookup[(dataset, PRIMARY)]["policy_usage_max"])
        for dataset in DATASETS
    )
    named_gates = {
        "pcc_over_a6": performance_gate(
            "a6",
            "pcc_over_a6_dataset_wins_min",
            "pcc_over_a6_macro_gain_min",
        ),
        "pcc_over_plain": performance_gate(
            "pcsd_direct",
            "pcc_over_plain_dataset_wins_min",
            "pcc_over_plain_macro_gain_min",
        ),
        "pcc_over_pointwise_pcc": performance_gate(
            "pointwise_pcc_v0",
            "pcc_over_pointwise_pcc_dataset_wins_min",
            "pcc_over_pointwise_pcc_macro_gain_min",
        ),
        "pcc_over_prior_composed": performance_gate(
            "pointwise_prior_composed",
            "pcc_over_prior_composed_dataset_wins_min",
            "pcc_over_prior_composed_macro_gain_min",
        ),
        "arm_degradation_recovery": bool(
            reduction_median
            >= gates["arm_degradation_median_relative_reduction_min"]
            and improved_pairs >= gates["arm_pairs_improved_min"]
        ),
        "pairwise_nrmse_retention": min(retention.values())
        >= gates["pairwise_nrmse_retention_fraction_min"],
        "policy_not_collapsed": bool(
            entropy_min >= gates["policy_normalized_entropy_min"]
            and usage_max <= gates["policy_usage_max"]
        ),
    }
    method_pass = all(named_gates.values())
    if method_pass:
        decision = "phase_a_pass_authorize_conditional_phase_b_review"
    elif not (
        named_gates["pcc_over_pointwise_pcc"]
        and named_gates["pcc_over_prior_composed"]
    ):
        decision = "generic_or_pointwise_control_explains_return_step4"
    elif named_gates["arm_degradation_recovery"] and not named_gates["pcc_over_a6"]:
        decision = "arms_recover_but_a6_not_beaten_return_sc1_step4"
    elif not named_gates["arm_degradation_recovery"]:
        decision = "no_arm_skill_recovery_return_step4_gradient_audit"
    else:
        decision = "phase_a_method_gate_fail_return_step4"
    return {
        "gates": named_gates,
        "method_pass": method_pass,
        "decision": decision,
        "macro_comparisons": {
            key: {
                "gain_fraction": float(value["pcc_gain_fraction"]),
                "gain_percent": float(value["pcc_gain_percent"]),
                "dataset_wins": int(value["dataset_wins"]),
            }
            for key, value in macro.items()
        },
        "arm_degradation_median_relative_reduction": reduction_median,
        "arm_pairs_improved": improved_pairs,
        "pairwise_nrmse_retention_by_dataset": retention,
        "pairwise_nrmse_retention_min": min(retention.values()),
        "policy_normalized_entropy_min": entropy_min,
        "policy_usage_max": usage_max,
    }


def synthetic_smoke(config: dict[str, Any]) -> None:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in DATASETS:
        for arm in (*REFERENCE_ARMS, *MODES):
            metric = 1.0
            if arm == PRIMARY:
                metric = 0.98
            elif arm in {"pointwise_pcc_v0", "pointwise_prior_composed"}:
                metric = 0.99
            lookup[(dataset, arm)] = {
                "dense_mse_auc": metric,
                "minimum_pairwise_probe_nrmse": 0.1,
                "policy_normalized_entropy": 0.8,
                "policy_usage_max": 0.3,
            }
    comparisons = comparison_rows(lookup)
    recovery = [
        {
            "relative_degradation_reduction": 0.5,
            "arm_pair_improved": True,
        }
        for _ in range(25)
    ]
    result = gate_result(comparisons, recovery, lookup, config["gates"])
    if not result["method_pass"]:
        raise RuntimeError(f"PCC analyzer synthetic smoke failed: {result}")


def report_text(result: dict[str, Any]) -> str:
    gate_rows = "\n".join(
        f"| `{name}` | `{str(value).lower()}` |"
        for name, value in result["gates"].items()
    )
    return f"""# SC2-PCC-v1-TI Step7B Validation Result

| Field | Value |
| --- | --- |
| `expected_runs` | `{result['expected_runs']}` |
| `valid_runs` | `{result['valid_runs']}` |
| `valid_reference_runs` | `{result['valid_reference_runs']}` |
| `complete` | `{str(result['complete']).lower()}` |
| `method_pass` | `{str(result['method_pass']).lower()}` |
| `decision` | `{result['decision']}` |
| `test_used` | `false` |

## Frozen Gates

| Gate | Pass |
| --- | --- |
{gate_rows}

该报告由validation-only artifacts自动生成。performance、arm recovery、diversity、policy与shared-gradient statistics
分别保存在同目录CSV/JSON中；在45/45 complete前不得作partial method judgment。
"""


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.synthetic_smoke:
        synthetic_smoke(config)
        print("pcc_step7b_analyzer_synthetic_smoke=pass")
        return
    if args.raw_root is None or args.reference_root is None or args.output_dir is None:
        raise ValueError("raw-root, reference-root and output-dir are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pcc_rows = [
        load_pcc_run(args.raw_root, mode, dataset, args.seed, config)
        for dataset in DATASETS
        for mode in MODES
    ]
    reference_rows = [
        load_reference(args.reference_root, arm, dataset, args.seed)
        for dataset in DATASETS
        for arm in REFERENCE_ARMS
    ]
    summary = [*pcc_rows, *reference_rows]
    write_csv(args.output_dir / "run_summary.csv", summary)
    valid = [row for row in summary if row.get("status") == "ok"]
    complete = len(valid) == len(DATASETS) * (len(MODES) + len(REFERENCE_ARMS))
    paired_initialization = all(
        len(
            {
                row["encoder_initialization_hash"]
                for row in pcc_rows
                if row["dataset"] == dataset and row.get("status") == "ok"
            }
        )
        == 1
        and len(
            {
                row["pcsd_initialization_hash"]
                for row in pcc_rows
                if row["dataset"] == dataset and row.get("status") == "ok"
            }
        )
        == 1
        for dataset in DATASETS
    )
    if complete and paired_initialization:
        lookup = {(row["dataset"], row["arm"]): row for row in valid}
        comparisons = comparison_rows(lookup)
        recovery = arm_recovery_rows(
            lookup,
            args.reference_root,
            args.seed,
        )
        gate = gate_result(comparisons, recovery, lookup, config["gates"])
        write_csv(args.output_dir / "pcc_comparisons.csv", comparisons)
        write_csv(args.output_dir / "arm_recovery.csv", recovery)
        write_csv(
            args.output_dir / "mechanism_by_dataset.csv",
            [
                {
                    "dataset": dataset,
                    **{
                        key: lookup[(dataset, PRIMARY)][key]
                        for key in (
                            "same_run_oracle_headroom",
                            "minimum_pairwise_probe_nrmse",
                            "mean_pairwise_probe_nrmse",
                            "policy_normalized_entropy",
                            "policy_usage_max",
                            "shared_gradient_cosine_mean",
                            "shared_gradient_cosine_min",
                            "shared_gradient_cosine_max",
                        )
                    },
                }
                for dataset in DATASETS
            ],
        )
    else:
        gate = {
            "gates": {},
            "method_pass": False,
            "decision": "artifact_or_protocol_audit_fail",
        }
    result = {
        "candidate": config["candidate"],
        "seed": args.seed,
        "expected_runs": config["expected_runs"],
        "valid_runs": sum(row.get("status") == "ok" for row in pcc_rows),
        "valid_reference_runs": sum(
            row.get("status") == "ok" for row in reference_rows
        ),
        "complete": complete,
        "paired_pcc_initialization": paired_initialization,
        "test_used": False,
        **gate,
    }
    (args.output_dir / "gate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "step7b_result_report.md").write_text(
        report_text(result),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not complete or not paired_initialization:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
