#!/usr/bin/env python3
"""Analyze a PCSD-CF validation screen or frozen-checkpoint test audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
SCALES = (1, 48, 144, 360, 720)
ARM_SPECS: dict[str, dict[str, Any]] = {
    "a6": {"readout": "learned-basis-forecast-operator"},
    "pcsd_m0": {"readout": "pcsd-coupling-field-m0"},
    **{
        f"pcsd_fixed_{scale}": {
            "readout": "pcsd-coupling-field",
            "policy": "fixed",
            "fixed_scale": scale,
            "partition": "canonical",
        }
        for scale in SCALES
    },
    "pcsd_equal": {
        "readout": "pcsd-coupling-field",
        "policy": "equal",
        "partition": "canonical",
    },
    "pcsd_static": {
        "readout": "pcsd-coupling-field",
        "policy": "static-target",
        "partition": "canonical",
    },
    "pcsd_direct": {
        "readout": "pcsd-coupling-field",
        "policy": "direct",
        "partition": "canonical",
    },
    "pcsd_random": {
        "readout": "pcsd-coupling-field",
        "policy": "direct",
        "partition": "random",
    },
    "dense_matched": {"readout": "pcsd-dense-nonlinear-matched"},
}
ARMS = tuple(ARM_SPECS)
PRIMARY = "pcsd_direct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--design",
        type=Path,
        default=Path("configs/stage_c_pcsd_cf_native_direct.json"),
    )
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--evaluation-split",
        choices=("validation", "test-audit"),
        default="validation",
    )
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


def run_dir(root: Path, arm: str, dataset: str, seed: int) -> Path:
    return root / arm / dataset / "h720_full" / f"seed{seed}"


def finite_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value: {value}")
    return number


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_gain(candidate: float, reference: float) -> float:
    return 1.0 - candidate / reference


def mechanism_statistics(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        if "arm_row_bin_mse" not in payload.files:
            return {}
        arm_losses = payload["arm_row_bin_mse"].astype(np.float64)
        fused_losses = payload["fused_row_bin_mse"].astype(np.float64)
        persistence = payload["persistence_row_bin_mse"].astype(np.float64)
        usage = payload["policy_row_bin_usage"].astype(np.float64)
        probe_arms = payload["probe_arms"].astype(np.float64)
    oracle = arm_losses.min(axis=-1)
    arm_means = arm_losses.mean(axis=(0, 1))
    fused_mean = float(fused_losses.mean())
    persistence_mean = float(persistence.mean())
    policy_usage = usage.mean(axis=(0, 1))
    entropy = -(
        usage * np.log(np.clip(usage, 1e-12, None))
    ).sum(axis=-1) / math.log(usage.shape[-1])
    denominator = max(float(np.sqrt(np.mean(probe_arms**2))), 1e-12)
    pairwise = []
    for left in range(probe_arms.shape[1]):
        for right in range(left + 1, probe_arms.shape[1]):
            pairwise.append(
                float(
                    np.sqrt(
                        np.mean(
                            (probe_arms[:, left] - probe_arms[:, right]) ** 2
                        )
                    )
                    / denominator
                )
            )
    return {
        "same_run_oracle_headroom": relative_gain(
            float(oracle.mean()),
            fused_mean,
        ),
        "best_same_run_arm_gain_over_fused": relative_gain(
            float(arm_means.min()),
            fused_mean,
        ),
        "best_same_run_arm_gain_over_persistence": relative_gain(
            float(arm_means.min()),
            persistence_mean,
        ),
        "minimum_pairwise_probe_nrmse": min(pairwise),
        "mean_pairwise_probe_nrmse": mean(pairwise),
        "policy_entropy": float(entropy.mean()),
        "policy_usage_max": float(policy_usage.max()),
        "policy_usage": json.dumps(policy_usage.tolist(), separators=(",", ":")),
    }


def load_run(
    root: Path,
    arm: str,
    dataset: str,
    seed: int,
    design: dict[str, Any],
    evaluation_split: str,
) -> dict[str, Any]:
    directory = run_dir(root, arm, dataset, seed)
    is_test_audit = evaluation_split == "test-audit"
    required = {
        "metrics": directory
        / (
            "test_audit_metrics_by_target_horizon.csv"
            if is_test_audit
            else "metrics_by_target_horizon.csv"
        ),
        "training": directory / "training_log.csv",
        "config": directory / "effective_config.json",
        "diagnostics": directory / "model_diagnostics.json",
        "initialization": directory / "initialization_contract.json",
        "invariants": directory
        / (
            "test_audit_invariants.json"
            if is_test_audit
            else "trained_invariants.json"
        ),
        "pcsd_evidence": directory
        / (
            "pcsd_test_audit_diagnostics.npz"
            if is_test_audit
            else "pcsd_validation_diagnostics.npz"
        ),
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {
            "seed": seed,
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
            "seed": seed,
            "dataset": dataset,
            "arm": arm,
            "status": "incomplete_dense_horizons",
            "dense_horizon_count": len(by_horizon),
            "run_dir": str(directory),
        }
    mse = [finite_float(by_horizon[h]["mse"]) for h in range(1, 721)]
    mae = [finite_float(by_horizon[h]["mae"]) for h in range(1, 721)]
    training = read_csv(required["training"])
    config = json.loads(required["config"].read_text(encoding="utf-8"))
    diagnostics = json.loads(
        required["diagnostics"].read_text(encoding="utf-8")
    )
    initialization = json.loads(
        required["initialization"].read_text(encoding="utf-8")
    )
    invariants = json.loads(
        required["invariants"].read_text(encoding="utf-8")
    )
    adapter = config["adapter"]
    contract = config["training_contract"]
    spec = ARM_SPECS[arm]
    protocol = design["step7b_protocol"]
    protocol_ok = bool(
        adapter["dataset"] == dataset
        and adapter["mode"] == "unified"
        and adapter["pred_len"] == 720
        and adapter["target_horizons"] == [720]
        and adapter["validation_horizons"] == [720]
        and adapter["checkpoint_policy"] == "best-val"
        and adapter["pred_loss_mode"] == "full"
        and adapter["protocol_class"] == "method_screening"
        and adapter["readout_mode"] == spec["readout"]
        and adapter["evaluation_prefix_mode"] == "full-crop"
        and adapter["final_evaluation_split"] == "val"
        and adapter["seed"] == seed
        and adapter["epochs"] == protocol["epochs"]
        and adapter["patience"] == protocol["patience"]
        and adapter["batch_size"] == protocol["batch_size"]
        and adapter["learning_rate"] == protocol["learning_rate"]
        and contract["initialization"] == "from_scratch"
        and contract["checkpoint_input"] is None
        and diagnostics["frozen_parameter_tensors"] == 0
    )
    if is_test_audit:
        protocol_ok = protocol_ok and bool(
            invariants.get("evaluation_split") == "test"
            and invariants.get("uses_test_split") is True
            and invariants.get("test_access_authorized") is True
            and invariants.get("checkpoint_retrained") is False
            and invariants.get("checkpoint_sha256")
            == file_sha256(directory / "checkpoint.pt")
        )
    else:
        protocol_ok = protocol_ok and bool(
            invariants.get("uses_test_split") is False
        )
    if "policy" in spec:
        protocol_ok = protocol_ok and (
            adapter["pcsd_policy_mode"] == spec["policy"]
            and adapter["pcsd_partition"] == spec["partition"]
        )
    if "fixed_scale" in spec:
        protocol_ok = protocol_ok and (
            adapter["pcsd_fixed_scale"] == spec["fixed_scale"]
        )
    if arm == "dense_matched":
        protocol_ok = protocol_ok and (
            diagnostics.get("pcsd_dense_parameter_relative_gap", 1.0)
            <= protocol["dense_parameter_gap_max"]
        )
    status = (
        "ok"
        if protocol_ok and invariants.get("pass") is True
        else "audit_fail"
    )
    row: dict[str, Any] = {
        "seed": seed,
        "dataset": dataset,
        "arm": arm,
        "status": status,
        "protocol_pass": protocol_ok,
        "invariant_pass": invariants.get("pass") is True,
        "readout_mode": adapter["readout_mode"],
        "profile_hash": adapter["profile_hash"],
        "encoder_initialization_hash": initialization.get(
            "encoder_initialization_hash", ""
        ),
        "operator_initialization_hash": initialization.get(
            "operator_initialization_hash", ""
        ),
        "pcsd_initialization_hash": initialization.get(
            "pcsd_initialization_hash", ""
        ),
        "dense_mse_auc": mean(mse),
        "dense_mae_auc": mean(mae),
        "h720_mse": mse[-1],
        "h720_mae": mae[-1],
        "epochs_ran": len(training),
        "best_val_mse": min(
            finite_float(item["val_mean_mse"]) for item in training
        ),
        "total_parameters": diagnostics.get("total_parameters", ""),
        "active_forward_parameters": diagnostics.get(
            "active_forward_parameters", ""
        ),
        "decoder_parameters": diagnostics.get(
            "pcsd_decoder_parameters",
            diagnostics.get(
                "pcsd_m0_decoder_parameters",
                diagnostics.get(
                    "pcsd_dense_decoder_parameters",
                    diagnostics.get("active_forward_parameters", ""),
                ),
            ),
        ),
        "dense_parameter_gap": diagnostics.get(
            "pcsd_dense_parameter_relative_gap", ""
        ),
        "run_dir": str(directory),
        "evaluation_split": evaluation_split,
        "checkpoint_sha256": invariants.get("checkpoint_sha256", ""),
    }
    row.update(mechanism_statistics(required["pcsd_evidence"]))
    return row


def comparison(
    summary: list[dict[str, Any]],
    design: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid = [row for row in summary if row.get("status") == "ok"]
    lookup = {(row["dataset"], row["arm"]): row for row in valid}
    comparisons = []
    for reference in (
        "a6",
        "pcsd_equal",
        "pcsd_static",
        "dense_matched",
        "pcsd_random",
    ):
        gains = {}
        for dataset in DATASETS:
            gains[dataset] = relative_gain(
                float(lookup[(dataset, PRIMARY)]["dense_mse_auc"]),
                float(lookup[(dataset, reference)]["dense_mse_auc"]),
            )
            comparisons.append(
                {
                    "reference": reference,
                    "dataset": dataset,
                    "direct_gain_fraction": gains[dataset],
                    "direct_gain_percent": 100.0 * gains[dataset],
                }
            )
        comparisons.append(
            {
                "reference": reference,
                "dataset": "macro",
                "direct_gain_fraction": mean(gains.values()),
                "direct_gain_percent": 100.0 * mean(gains.values()),
                "dataset_wins": sum(value > 0.0 for value in gains.values()),
            }
        )

    macro = {
        row["reference"]: row
        for row in comparisons
        if row["dataset"] == "macro"
    }
    gate = design["effectiveness_gate"]
    specificity = design["step7b_protocol"]["specificity_gates"]
    mechanism_gate = design["step7b_protocol"]["trained_mechanism_gates"]
    direct_rows = [lookup[(dataset, PRIMARY)] for dataset in DATASETS]
    arm_separation_count = sum(
        float(row["minimum_pairwise_probe_nrmse"])
        >= mechanism_gate["minimum_pairwise_probe_nrmse"]
        for row in direct_rows
    )
    skilled_count = sum(
        float(row["best_same_run_arm_gain_over_persistence"]) > 0.0
        for row in direct_rows
    )
    oracle_headroom = mean(
        float(row["same_run_oracle_headroom"]) for row in direct_rows
    )
    policy_entropy_min = min(float(row["policy_entropy"]) for row in direct_rows)
    policy_usage_max = max(float(row["policy_usage_max"]) for row in direct_rows)
    gates = {
        "direct_over_a6": bool(
            macro["a6"]["dataset_wins"]
            >= gate["direct_over_a6_dataset_required"]
            and macro["a6"]["direct_gain_fraction"]
            >= gate["direct_over_a6_macro_gain_min"]
        ),
        "direct_over_equal": bool(
            macro["pcsd_equal"]["dataset_wins"]
            >= gate["direct_over_equal_dataset_required"]
            and macro["pcsd_equal"]["direct_gain_fraction"]
            >= gate["direct_over_equal_macro_gain_min"]
        ),
        "direct_over_static": bool(
            macro["pcsd_static"]["dataset_wins"]
            >= gate["direct_over_static_dataset_required"]
            and macro["pcsd_static"]["direct_gain_fraction"]
            >= gate["direct_over_static_macro_gain_min"]
        ),
        "dense_capacity_control": bool(
            macro["dense_matched"]["dataset_wins"]
            >= specificity["direct_over_dense_dataset_required"]
            and macro["dense_matched"]["direct_gain_fraction"]
            >= specificity["direct_over_dense_macro_gain_min"]
        ),
        "random_partition_specificity": bool(
            macro["pcsd_random"]["dataset_wins"]
            >= specificity["direct_over_random_dataset_required"]
            and macro["pcsd_random"]["direct_gain_fraction"]
            >= specificity["direct_over_random_macro_gain_min"]
        ),
        "same_run_arm_separation": bool(
            arm_separation_count
            >= mechanism_gate["arm_separation_dataset_required"]
        ),
        "same_run_arm_skill": skilled_count >= 4,
        "policy_not_collapsed": bool(
            policy_entropy_min >= mechanism_gate["policy_entropy_min"]
            and policy_usage_max <= mechanism_gate["policy_max_usage_max"]
        ),
    }
    audit = {
        "gates": gates,
        "macro_comparisons": {
            key: {
                "gain_fraction": value["direct_gain_fraction"],
                "gain_percent": value["direct_gain_percent"],
                "dataset_wins": value["dataset_wins"],
            }
            for key, value in macro.items()
        },
        "arm_separation_dataset_count": arm_separation_count,
        "same_run_arm_skill_dataset_count": skilled_count,
        "same_run_oracle_headroom_macro": oracle_headroom,
        "policy_entropy_min": policy_entropy_min,
        "policy_usage_max": policy_usage_max,
    }
    audit["method_pass"] = all(gates.values())
    audit["credit_problem_supported"] = bool(
        not gates["direct_over_equal"]
        and gates["same_run_arm_separation"]
        and gates["same_run_arm_skill"]
        and oracle_headroom
        >= mechanism_gate["same_run_oracle_headroom_min"]
        and gates["policy_not_collapsed"]
    )
    if audit["method_pass"]:
        audit["decision"] = "provisional_method_pass_authorize_confirmation_design"
    elif not gates["same_run_arm_skill"]:
        audit["decision"] = "readout_or_head_design_wrong_rollback_step4"
    elif not gates["dense_capacity_control"]:
        audit["decision"] = "capacity_control_explains_no_method_pass"
    elif not gates["random_partition_specificity"]:
        audit["decision"] = "partition_specificity_fail_no_method_pass"
    elif audit["credit_problem_supported"]:
        audit["decision"] = "direct_credit_problem_supported_sc2_step2_4_only"
    else:
        audit["decision"] = "pcsd_v1_effectiveness_fail_no_sc2_evidence"
    return comparisons, audit


def initialization_audit(summary: list[dict[str, Any]]) -> dict[str, bool]:
    valid = [row for row in summary if row.get("status") == "ok"]
    paired_encoder = all(
        len(
            {
                row["encoder_initialization_hash"]
                for row in valid
                if row["dataset"] == dataset
            }
        )
        == 1
        for dataset in DATASETS
    )
    paired_pcsd = all(
        len(
            {
                row["pcsd_initialization_hash"]
                for row in valid
                if row["dataset"] == dataset
                and row["arm"].startswith("pcsd_")
                and row["arm"] != "pcsd_m0"
            }
        )
        == 1
        for dataset in DATASETS
    )
    a6_m0_exact = all(
        next(
            row["operator_initialization_hash"]
            for row in valid
            if row["dataset"] == dataset and row["arm"] == "a6"
        )
        == next(
            row["operator_initialization_hash"]
            for row in valid
            if row["dataset"] == dataset and row["arm"] == "pcsd_m0"
        )
        for dataset in DATASETS
    )
    return {
        "paired_encoder_initialization": paired_encoder,
        "paired_pcsd_initialization": paired_pcsd,
        "a6_m0_exact_initialization": a6_m0_exact,
    }


def synthetic_smoke(design: dict[str, Any]) -> None:
    summary = []
    for dataset in DATASETS:
        for arm in ARMS:
            metric = 1.0
            if arm == PRIMARY:
                metric = 0.98
            elif arm in {"pcsd_equal", "pcsd_static", "dense_matched", "pcsd_random"}:
                metric = 0.99
            summary.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "status": "ok",
                    "dense_mse_auc": metric,
                    "minimum_pairwise_probe_nrmse": 0.1,
                    "best_same_run_arm_gain_over_persistence": 0.2,
                    "same_run_oracle_headroom": 0.05,
                    "policy_entropy": 0.8,
                    "policy_usage_max": 0.4,
                }
            )
    _rows, audit = comparison(summary, design)
    if not audit["method_pass"]:
        raise RuntimeError(f"synthetic method-pass smoke failed: {audit}")


def main() -> None:
    args = parse_args()
    design = json.loads(args.design.read_text(encoding="utf-8"))
    if args.synthetic_smoke:
        synthetic_smoke(design)
        print("pcsd_step7b_analysis_synthetic_smoke=pass")
        return
    if args.raw_root is None or args.output_dir is None:
        raise ValueError("raw-root and output-dir are required outside synthetic smoke")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = [
        load_run(
            args.raw_root,
            arm,
            dataset,
            args.seed,
            design,
            args.evaluation_split,
        )
        for arm in ARMS
        for dataset in DATASETS
    ]
    write_csv(args.output_dir / "run_summary.csv", summary)
    valid = [row for row in summary if row.get("status") == "ok"]
    complete = len(valid) == len(ARMS) * len(DATASETS)
    initialization = initialization_audit(summary) if complete else {}
    if complete and all(initialization.values()):
        comparisons, audit = comparison(summary, design)
        write_csv(args.output_dir / "direct_comparisons.csv", comparisons)
    else:
        audit = {
            "method_pass": False,
            "credit_problem_supported": False,
            "decision": "artifact_or_protocol_audit_fail",
        }
    result = {
        "candidate_id": design["candidate_id"],
        "diagnostic_id": design["diagnostic_id"],
        "seed": args.seed,
        "evaluation_split": args.evaluation_split,
        "expected_runs": len(ARMS) * len(DATASETS),
        "valid_runs": len(valid),
        "complete": complete,
        "initialization_audit": initialization,
        **audit,
    }
    (args.output_dir / "gate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not complete or not all(initialization.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
