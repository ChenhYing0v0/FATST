#!/usr/bin/env python3
"""Check the frozen PCSD-CF-v1 milestone test-audit contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
EXPECTED_ARMS = (
    "a6",
    "pcsd_m0",
    "pcsd_fixed_1",
    "pcsd_fixed_48",
    "pcsd_fixed_144",
    "pcsd_fixed_360",
    "pcsd_fixed_720",
    "pcsd_equal",
    "pcsd_static",
    "pcsd_direct",
    "pcsd_random",
    "dense_matched",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_pcsd_cf_test_audit.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/stage_c_pcsd_cf_test_audit_prelaunch_20260716"
        ),
    )
    return parser.parse_args()


def case(
    rows: list[dict[str, Any]],
    name: str,
    passed: bool,
    value: Any,
) -> None:
    rows.append({"case": name, "pass": bool(passed), "value": value})


def prefix_metric_check(rows: list[dict[str, Any]]) -> None:
    rng = np.random.default_rng(20260716)
    errors = rng.normal(size=(11, 720, 7))
    step_sse = np.square(errors).sum(axis=(0, 2))
    step_sae = np.abs(errors).sum(axis=(0, 2))
    row_channels = errors.shape[0] * errors.shape[2]
    cumulative_mse = np.cumsum(step_sse) / (
        row_channels * np.arange(1, 721)
    )
    cumulative_mae = np.cumsum(step_sae) / (
        row_channels * np.arange(1, 721)
    )
    direct_mse = np.asarray(
        [np.square(errors[:, :horizon]).mean() for horizon in range(1, 721)]
    )
    direct_mae = np.asarray(
        [np.abs(errors[:, :horizon]).mean() for horizon in range(1, 721)]
    )
    mse_gap = float(np.max(np.abs(cumulative_mse - direct_mse)))
    mae_gap = float(np.max(np.abs(cumulative_mae - direct_mae)))
    case(rows, "dense_prefix_mse_exact", mse_gap <= 1e-12, mse_gap)
    case(rows, "dense_prefix_mae_exact", mae_gap <= 1e-12, mae_gap)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    matrix = config["matrix"]
    authorization = config["authorization"]
    gates = config["gates"]
    rows: list[dict[str, Any]] = []

    datasets = tuple(matrix["datasets"])
    arms = tuple(matrix["arms"])
    case(
        rows,
        "candidate_version_frozen",
        config["candidate_version"] == "SC1-PCSD-CF-v1",
        config["candidate_version"],
    )
    case(rows, "dataset_matrix_exact", set(datasets) == set(EXPECTED_DATASETS), len(datasets))
    case(rows, "arm_matrix_exact", set(arms) == set(EXPECTED_ARMS), len(arms))
    case(
        rows,
        "matrix_unique",
        len(set(datasets)) == 5 and len(set(arms)) == 12,
        len(set(datasets)) * len(set(arms)),
    )
    case(rows, "expected_runs_exact", matrix["expected_runs"] == 60, matrix["expected_runs"])
    case(rows, "dense_horizon_contract", matrix["horizons"] == "dense_h1_h720", matrix["horizons"])
    case(
        rows,
        "explicit_user_authorization",
        authorization["user_authorized"] is True,
        authorization["authorization_date"],
    )
    case(
        rows,
        "test_primary_milestone_role",
        authorization["test_role"]
        == "primary-milestone-effectiveness-gate",
        authorization["test_role"],
    )
    case(
        rows,
        "historical_best_val_checkpoint",
        authorization["checkpoint_selection"]
        == "historical-best-validation-h720-mse",
        authorization["checkpoint_selection"],
    )
    case(
        rows,
        "retraining_forbidden",
        authorization["checkpoint_retraining_allowed"] is False,
        authorization["checkpoint_retraining_allowed"],
    )
    case(
        rows,
        "checkpoint_mutation_forbidden",
        authorization["checkpoint_mutation_allowed"] is False,
        authorization["checkpoint_mutation_allowed"],
    )
    case(
        rows,
        "per_dataset_tuning_forbidden",
        authorization["per_dataset_or_horizon_tuning_allowed"] is False,
        authorization["per_dataset_or_horizon_tuning_allowed"],
    )
    case(
        rows,
        "partial_reporting_forbidden",
        authorization["partial_matrix_reporting_allowed"] is False,
        authorization["partial_matrix_reporting_allowed"],
    )
    case(
        rows,
        "single_formal_access",
        authorization["formal_test_access_count_for_version"] == 1,
        authorization["formal_test_access_count_for_version"],
    )
    case(
        rows,
        "full_matrix_gate",
        gates["full_matrix_required"] is True,
        gates["full_matrix_required"],
    )
    case(
        rows,
        "checkpoint_hash_gate",
        gates["checkpoint_hash_unchanged_required"] is True,
        gates["checkpoint_hash_unchanged_required"],
    )

    runner = Path(
        "scripts/remote/run_stage_c_pcsd_cf_test_audit.sh"
    ).read_text(encoding="utf-8")
    case(
        rows,
        "runner_has_no_training_entrypoint",
        "train_repo.py" not in runner,
        "train_repo.py" in runner,
    )
    case(
        rows,
        "runner_checks_checkpoint_hash",
        runner.count("sha256sum") >= 2,
        runner.count("sha256sum"),
    )
    case(
        rows,
        "runner_uses_test_evaluator",
        "--evaluation-split test" in runner,
        "--evaluation-split test" in runner,
    )
    prefix_metric_check(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overall_pass = all(bool(row["pass"]) for row in rows)
    result = {
        "audit_id": config["audit_id"],
        "candidate_version": config["candidate_version"],
        "case_count": len(rows),
        "passed_cases": sum(bool(row["pass"]) for row in rows),
        "overall_pass": overall_pass,
        "remote_test_audit_authorized": overall_pass,
        "checkpoint_retraining_authorized": False,
        "pcc_step6_authorized": False,
        "decision": (
            "prelaunch_pass_remote_test_audit_only"
            if overall_pass
            else "prelaunch_fail_repair_before_test_access"
        ),
        "cases": rows,
    }
    (args.output_dir / "local_gate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
