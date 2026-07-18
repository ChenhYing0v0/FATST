#!/usr/bin/env python3
"""Run the formal CCSF tau0.25 Phase-A Step 7B prelaunch gate."""

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
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_repo as training_adapter  # noqa: E402
from evaluate_stage_c_pcsd_cf_checkpoint import (  # noqa: E402
    test_audit_authorized,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/stage_c_siff_ccsf_v1_tau25_formal_candidate.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/stage_c_siff_ccsf_v1_tau25_phase_a_20260718/prelaunch"
        ),
    )
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def manifest_rows(
    config: dict[str, Any], profiles: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for dataset in config["datasets"]:
        profile = profiles[dataset]
        for arm in config["arms"]:
            rank = (
                256
                if arm["rank_rule"] == "fixed_256"
                else config["matched_ranks"][dataset]
            )
            rows.append(
                {
                    "job_index": len(rows) + 1,
                    "dataset": dataset,
                    "arm": arm["id"],
                    "readout_mode": arm["readout_mode"],
                    "objective_mode": arm["objective_mode"],
                    "mode_rank": rank,
                    "profile": profile["profile"],
                    "patch_num": profile["patch_num"],
                    "d_model": profile["d_model"],
                    "d_ff": profile["d_ff"],
                    "seed": config["seeds"][0],
                    "temperature": config["selected_objective"][
                        "calibration_temperature"
                    ],
                    "checkpoint_score": config["training"][
                        "validation_checkpoint_score"
                    ],
                    "formal_evaluation_split": "test",
                }
            )
    return rows


def training_argv(row: dict[str, Any], config: dict[str, Any]) -> list[str]:
    training = config["training"]
    return [
        "train_repo.py",
        "--dataset-root",
        "/home/yingch/dataset",
        "--dataset",
        row["dataset"],
        "--mode",
        "unified",
        "--seq-len",
        "720",
        "--pred-len",
        "720",
        "--target-horizons",
        "720",
        "--validation-horizons",
        "96,192,336,720",
        "--evaluation-horizons",
        "96,192,336,720",
        "--segment-horizons",
        "96,192,336,720",
        "--evaluation-prefix-mode",
        "full-crop",
        "--e-layers",
        "2",
        "--batch-size",
        str(training["batch_size"]),
        "--epochs",
        str(training["epochs"]),
        "--patience",
        str(training["patience"]),
        "--enable-early-stopping",
        "--seed",
        str(row["seed"]),
        "--run-name",
        f"CCSF_TAU25_{row['arm']}",
        "--output-dir",
        f"/tmp/{row['arm']}_{row['dataset']}",
        "--device",
        "cuda",
        "--checkpoint-policy",
        "best-val",
        "--protocol-class",
        "method_screening",
        "--protocol-profile",
        "stage_c_siff_ccsf_v1_tau25_phase_a",
        "--profile-hash",
        config["profiles"]["sha256"],
        "--legacy-patch-num",
        str(row["patch_num"]),
        "--legacy-d-model",
        str(row["d_model"]),
        "--legacy-d-ff",
        str(row["d_ff"]),
        "--legacy-dropout",
        "0.1",
        "--legacy-layer-norm",
        "1",
        "--learning-rate",
        str(training["learning_rate"]),
        "--readout-mode",
        row["readout_mode"],
        "--basis-rank",
        "256",
        "--pcsd-coordinate-dim",
        "4",
        "--pcsd-mode-rank",
        str(row["mode_rank"]),
        "--pcsd-policy-history-dim",
        "32",
        "--pcsd-policy-hidden-dim",
        "64",
        "--pcsd-policy-mode",
        "direct",
        "--pcsd-fixed-scale",
        "720",
        "--pcsd-partition",
        "canonical",
        "--pcsd-partition-seed",
        "15101",
        "--ccsf-correction-hidden-dim",
        "64",
        "--ccsf-calibration-temperature",
        str(config["selected_objective"]["calibration_temperature"]),
        "--ccsf-calibration-weight",
        str(config["selected_objective"]["calibration_weight"]),
        "--pcc-objective-mode",
        row["objective_mode"],
        "--pred-loss-mode",
        "full",
        "--final-evaluation-split",
        "val",
        "--no-save-predictions",
    ]


def cli_audit(rows: list[dict[str, Any]], config: dict[str, Any]) -> bool:
    original = sys.argv
    try:
        for row in rows:
            sys.argv = training_argv(row, config)
            parsed = training_adapter.parse_args()
            if not (
                parsed.dataset == row["dataset"]
                and parsed.readout_mode == row["readout_mode"]
                and parsed.pcc_objective_mode == row["objective_mode"]
                and parsed.pcsd_mode_rank == row["mode_rank"]
                and parsed.validation_horizons == [96, 192, 336, 720]
                and parsed.final_evaluation_split == "val"
                and parsed.evaluation_prefix_mode == "full-crop"
                and parsed.ccsf_calibration_temperature == 0.25
                and parsed.ccsf_calibration_weight == 0.1
            ):
                return False
    finally:
        sys.argv = original
    return True


def run(command: list[str], env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return completed.stdout


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    profile_path = Path(config["profiles"]["path"])
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))[
        "dataset_profiles"
    ]
    rows = manifest_rows(config, profiles)
    contract_hashes = {
        key: file_hash(Path(value["path"])) == value["sha256"]
        for key, value in config["contracts"].items()
    }
    pilot = json.loads(
        Path(config["contracts"]["pilot_selection"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    runner = Path("scripts/remote/run_stage_c_siff_ccsf_tau25_phase_a.sh")
    analyzer = Path("scripts/analyze_stage_c_siff_ccsf_tau25_phase_a.py")
    evaluator = Path("scripts/evaluate_stage_c_pcsd_cf_checkpoint.py")
    run(["bash", "-n", str(runner)])
    evaluator_output = run([sys.executable, str(evaluator), "--synthetic-smoke"])
    analyzer_output = run(
        [
            sys.executable,
            str(analyzer),
            "--config",
            str(args.config),
            "--synthetic-smoke",
        ]
    )
    dry_env = os.environ.copy()
    dry_env["DRY_RUN"] = "1"
    dry_output = run(["bash", str(runner)], env=dry_env)
    dry_jobs = sum(
        line.count("\t") == 8 for line in dry_output.splitlines()
    )
    runner_source = runner.read_text(encoding="utf-8")
    ccsf_source = Path(
        "baselines/timealign_official/layers/CCSF.py"
    ).read_text(encoding="utf-8")
    categories = {
        "contract_hashes": all(contract_hashes.values()),
        "profile_hash": file_hash(profile_path) == config["profiles"]["sha256"],
        "pilot_gate_passed": bool(
            pilot.get("selected_temperature") == 0.25
            and config["selected_objective"]["calibration_temperature"] == 0.25
            and config["selected_objective"]["pilot_checkpoints_reused"] is False
        ),
        "matrix_50_runs": len(rows)
        == config["matrix"]["phase_a_expected_runs"]
        == 50,
        "matrix_200_test_cells": len(rows)
        * len(config["matrix"]["horizons"])
        == config["matrix"]["phase_a_expected_test_cells"]
        == 200,
        "ten_hard_comparisons": config["gates"]["hard_comparisons"] == 10,
        "cli_contract_all_jobs": cli_audit(rows, config),
        "test_audit_authorized": test_audit_authorized(config),
        "runner_dry_run_50": dry_jobs == 50,
        "evaluator_ccsf_diagnostics": (
            "ccsf_checkpoint_evaluator_synthetic_smoke=pass"
            in evaluator_output
        ),
        "four_layer_analyzer": (
            "ccsf_tau25_phase_a_analyzer_synthetic_smoke=pass"
            in analyzer_output
        ),
        "runtime_repair_preserved": (
            "values.square().mean(dim=-1) + self.contrast_epsilon"
            in ccsf_source
        ),
        "three_batch_resource_smoke_contract": (
            "--max-train-batches 3" in runner_source
            and "ccsf_relcal" in runner_source
        ),
        "checkpoint_nonmutation_check": (
            "checkpoint_before" in runner_source
            and "checkpoint_after" in runner_source
        ),
        "formal_test_full_matrix_only": bool(
            config["matrix"]["full_matrix_required"]
            and not config["matrix"]["partial_reporting_allowed"]
            and config["authorization"][
                "per_dataset_horizon_or_cell_tuning_allowed"
            ]
            is False
        ),
    }
    payload = {
        "gate_id": "SC1-SIFF-v2-CCSF-v1-tau25-PhaseA-Step7B",
        "current_step": "Step7B-formal-prelaunch",
        "candidate_version": config["candidate_version"],
        "config_sha256": file_hash(args.config),
        "contract_hashes": contract_hashes,
        "categories": categories,
        "categories_passed": sum(categories.values()),
        "categories_total": len(categories),
        "expected_runs": len(rows),
        "expected_test_cells": len(rows) * len(config["matrix"]["horizons"]),
        "overall_pass": all(categories.values()),
        "remote_phase_a_authorized": bool(
            all(categories.values())
            and config["authorization"]["remote_training_authorized"]
            and config["authorization"]["formal_phase_a_authorized"]
            and config["authorization"]["formal_test_access_authorized"]
        ),
        "confirmation_authorized": False,
        "rollback_if_fail": "Step7A for tooling/runtime; Step6 for contract conflict",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "formal_phase_a_manifest.csv", rows)
    (args.output_dir / "prelaunch_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    if not payload["overall_pass"]:
        raise RuntimeError(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
