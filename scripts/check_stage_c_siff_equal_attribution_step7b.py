#!/usr/bin/env python3
"""Check the SIFF_EQUAL attribution Phase-A prelaunch contract."""

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


STEP7A_GATE = ROOT / (
    "analysis/stage_c_siff_equal_attribution_step7a_20260718/"
    "local_gate.json"
)
STEP7A_JOBS = ROOT / (
    "analysis/stage_c_siff_equal_attribution_step7a_20260718/"
    "jobs_seed2021.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_siff_equal_attribution_v2.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/stage_c_siff_equal_attribution_step7b_prelaunch_20260718"
        ),
    )
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def runner_syntax_pass() -> bool:
    runner = ROOT / "scripts/remote/run_stage_c_siff_equal_attribution_v2.sh"
    result = subprocess.run(
        ["bash", "-n", str(runner)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.returncode == 0 and os.access(runner, os.X_OK))


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    step7a = json.loads(STEP7A_GATE.read_text(encoding="utf-8"))
    jobs = read_csv(STEP7A_JOBS)
    profile_path = Path(config["profiles"]["path"])
    authorization = config["authorization"]

    job_contract = bool(
        len(jobs) == config["matrix"]["phase_a_expected_runs"] == 50
        and len({row["arm"] for row in jobs}) == len(config["arms"]) == 10
        and {int(row["seed"]) for row in jobs} == {2021}
        and all(
            row["checkpoint_score"]
            == "mean_mse_h96_h192_h336_h720"
            and row["formal_evaluation_split"] == "test"
            for row in jobs
        )
    )
    authorization_contract = bool(
        config["status"] == "authorized_prelaunch"
        and authorization["step7b_user_authorized"] is True
        and authorization["user_authorized"] is True
        and authorization["remote_training_authorized"] is True
        and authorization["formal_test_access_authorized"] is True
        and authorization["test_role"]
        == "primary-mechanism-effectiveness-and-paper-benchmark"
        and authorization["checkpoint_selection"]
        == "best-validation-mean-mse-h96-h192-h336-h720"
        and authorization["checkpoint_retraining_allowed"] is True
        and authorization["checkpoint_mutation_during_test_allowed"] is False
        and authorization["formal_test_access_count_for_version"] == 1
        and authorization["per_dataset_horizon_or_cell_tuning_allowed"]
        is False
    )
    confirmation_held = bool(
        authorization["phase_a_seed"] == 2021
        and authorization["confirmation_authorized"] is False
        and config["confirmation_seeds"] == [2022, 2023]
    )
    categories = {
        "candidate_identity": (
            config["candidate_version"] == "SC1-SIFF-v2-EQ-ATTR-v1"
            and config["test_informed"] is True
        ),
        "step7a_evidence": bool(
            step7a["overall_pass"]
            and step7a["categories_passed"] == step7a["categories_total"] == 13
            and step7a["jobs"] == 50
        ),
        "profile_hash": file_hash(profile_path)
        == config["profiles"]["sha256"],
        "phase_a_job_contract": job_contract,
        "paper_facing_matrix": bool(
            config["matrix"]["phase_a_expected_test_cells"] == 200
            and config["matrix"]["horizons"] == [96, 192, 336, 720]
            and config["matrix"]["metrics"] == ["mse", "mae"]
        ),
        "formal_authorization": authorization_contract,
        "checkpoint_evaluator_authorization": test_audit_authorized(config),
        "confirmation_held": confirmation_held,
        "runner_syntax_and_mode": runner_syntax_pass(),
    }
    overall_pass = all(categories.values())
    report = {
        "candidate_version": config["candidate_version"],
        "current_step": "Step7B prelaunch gate",
        "categories": categories,
        "categories_passed": sum(categories.values()),
        "categories_total": len(categories),
        "jobs": len(jobs),
        "test_cells": config["matrix"]["phase_a_expected_test_cells"],
        "profile_hash": config["profiles"]["sha256"],
        "overall_pass": overall_pass,
        "remote_training_authorized": bool(
            overall_pass and authorization["remote_training_authorized"]
        ),
        "formal_test_access_authorized": bool(
            overall_pass and authorization["formal_test_access_authorized"]
        ),
        "confirmation_authorized": False,
        "next_step": (
            "remote dry-run resource smoke and Phase-A launch"
            if overall_pass
            else "repair Step7B prelaunch"
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
