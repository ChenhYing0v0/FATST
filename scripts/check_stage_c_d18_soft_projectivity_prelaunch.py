#!/usr/bin/env python3
"""Check the frozen SC-D18-SPC diagnostic and remote prelaunch contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_stage_c_pcsd_cf_checkpoint import (  # noqa: E402
    test_audit_authorized,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_d18_soft_projectivity_cost.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/stage_c_post_ccsf_step24_reset_20260719/"
            "d18_prelaunch"
        ),
    )
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = json.loads(
        Path(config["profiles"]["path"]).read_text(encoding="utf-8")
    )["dataset_profiles"]
    jobs = []
    for dataset in config["datasets"]:
        profile = profiles[dataset]
        for arm in config["arms"]:
            jobs.append(
                {
                    "dataset": dataset,
                    "arm": arm["id"],
                    "job_type": (
                        "train" if arm["training_new"] else "reuse"
                    ),
                    "own_horizon": arm.get("own_horizon", 720),
                    "target_horizons": ",".join(
                        map(str, arm["target_horizons"])
                    ),
                    "validation_horizons": ",".join(
                        map(str, arm["validation_horizons"])
                    ),
                    "pred_loss_mode": arm["pred_loss_mode"],
                    "pcc_objective_mode": arm["pcc_objective_mode"],
                    "profile": profile["profile"],
                    "patch_num": profile["patch_num"],
                    "d_model": profile["d_model"],
                    "d_ff": profile["d_ff"],
                    "seed": config["seeds"][0],
                    "test_role": config["authorization"]["test_role"],
                }
            )
    return jobs


def local_control_audit(config: dict[str, Any]) -> bool:
    remote_root = Path(config["control_source"]["remote_root"])
    root = (
        remote_root
        if remote_root.is_dir()
        else Path(config["control_source"]["local_audit_root"])
    )
    for dataset in config["datasets"]:
        hashes: dict[str, set[str]] = {
            "encoder": set(),
            "operator": set(),
        }
        parameter_counts = set()
        for arm in ("a6_measure", "a6_full"):
            directory = root / arm / dataset / "h720_full" / "seed2021"
            required = [
                directory / "effective_config.json",
                directory / "initialization_contract.json",
                directory / "model_diagnostics.json",
                directory / "test_audit_invariants.json",
                directory / "test_audit_metrics_by_target_horizon.csv",
            ]
            if not all(path.is_file() for path in required):
                return False
            initialization = json.loads(required[1].read_text(encoding="utf-8"))
            diagnostics = json.loads(required[2].read_text(encoding="utf-8"))
            invariant = json.loads(required[3].read_text(encoding="utf-8"))
            hashes["encoder"].add(
                initialization["encoder_initialization_hash"]
            )
            hashes["operator"].add(
                initialization["operator_initialization_hash"]
            )
            parameter_counts.add(
                (
                    diagnostics["total_parameters"],
                    diagnostics["active_forward_parameters"],
                )
            )
            if invariant.get("pass") is not True:
                return False
        if (
            len(hashes["encoder"]) != 1
            or len(hashes["operator"]) != 1
            or len(parameter_counts) != 1
        ):
            return False
    return True


def runner_pass() -> bool:
    runner = ROOT / (
        "scripts/remote/run_stage_c_d18_soft_projectivity_cost.sh"
    )
    result = subprocess.run(
        ["bash", "-n", str(runner)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.returncode == 0 and os.access(runner, os.X_OK))


def analyzer_smoke_pass(config_path: Path) -> bool:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_stage_c_d18_soft_projectivity_cost.py",
            "--config",
            str(config_path),
            "--synthetic-smoke",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def gradient_contract_pass(config_path: Path, output_dir: Path) -> bool:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_stage_c_d18_gradient_contract.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir / "gradient_contract"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    jobs = build_jobs(config)
    authorization = config["authorization"]
    specialist_arms = [
        arm for arm in config["arms"] if arm["training_new"]
    ]
    control_arms = [
        arm for arm in config["arms"] if not arm["training_new"]
    ]
    architecture_contract = bool(
        all(
            arm["readout_mode"] == "learned-basis-forecast-operator"
            for arm in config["arms"]
        )
        and [arm["own_horizon"] for arm in specialist_arms]
        == config["matrix"]["own_horizons"]
        and all(
            arm["pred_loss_mode"] == "multi-prefix"
            and arm["target_horizons"] == [arm["own_horizon"]]
            and arm["validation_horizons"] == [arm["own_horizon"]]
            for arm in specialist_arms
        )
        and {arm["id"] for arm in control_arms}
        == {"a6_measure", "a6_full"}
        and config["training"]["output_domain"] == 720
        and config["training"]["same_architecture_parameter_count_required"]
        is True
    )
    authorization_contract = bool(
        config["status"] == "authorized_prelaunch"
        and authorization["user_authorized"] is True
        and authorization["remote_training_authorized"] is True
        and authorization["formal_test_access_authorized"] is True
        and authorization["test_role"]
        == "primary-problem-existence-diagnostic"
        and authorization["checkpoint_selection"]
        == config["checkpoint_selection_contract"]
        and authorization["checkpoint_mutation_during_test_allowed"] is False
        and authorization["per_dataset_horizon_or_cell_tuning_allowed"]
        is False
        and authorization["method_implementation_authorized"] is False
    )
    categories = {
        "diagnostic_identity_and_scope": bool(
            config["candidate_version"] == "SC-D18-SPC-v1"
            and config["role"].endswith("diagnostic_only")
            and config["test_informed"] is True
        ),
        "profile_hash": file_hash(Path(config["profiles"]["path"]))
        == config["profiles"]["sha256"],
        "matrix": bool(
            len(jobs) == config["matrix"]["expected_runs"] == 25
            and sum(row["job_type"] == "train" for row in jobs)
            == config["matrix"]["new_training_runs"]
            == 15
            and sum(row["job_type"] == "reuse" for row in jobs)
            == config["matrix"]["reused_control_runs"]
            == 10
            and config["matrix"]["primary_own_horizon_cells"] == 15
        ),
        "same_architecture_loss_selector_only": architecture_contract,
        "local_reused_control_lineage": local_control_audit(config),
        "frozen_gates_and_rollback": bool(
            len(config["gates"]) == 8
            and len(config["decision_map"]) == 4
        ),
        "formal_authorization": authorization_contract,
        "checkpoint_evaluator_authorization": test_audit_authorized(config),
        "shape_projectivity_and_gradient_contract": gradient_contract_pass(
            args.config,
            args.output_dir,
        ),
        "analyzer_synthetic_smoke": analyzer_smoke_pass(args.config),
        "runner_syntax_and_mode": runner_pass(),
    }
    overall_pass = all(categories.values())
    report = {
        "candidate_version": config["candidate_version"],
        "current_step": "Step3 frozen diagnostic prelaunch",
        "categories": categories,
        "categories_passed": sum(categories.values()),
        "categories_total": len(categories),
        "jobs": len(jobs),
        "new_training_runs": config["matrix"]["new_training_runs"],
        "reused_control_runs": config["matrix"]["reused_control_runs"],
        "primary_own_horizon_cells": config["matrix"][
            "primary_own_horizon_cells"
        ],
        "overall_pass": overall_pass,
        "method_implementation_authorized": False,
        "remote_training_authorized": bool(
            overall_pass and authorization["remote_training_authorized"]
        ),
        "formal_test_access_authorized": bool(
            overall_pass and authorization["formal_test_access_authorized"]
        ),
        "rollback_step": "Step2",
        "next_step": (
            "remote dry-run resource smoke and 15-run problem diagnostic"
            if overall_pass
            else "repair Step3 prelaunch"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "jobs_seed2021.csv", jobs)
    (args.output_dir / "prelaunch_gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not overall_pass:
        raise RuntimeError(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
