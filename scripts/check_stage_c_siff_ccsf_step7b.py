#!/usr/bin/env python3
"""Audit the CCSF shared-temperature validation pilot prelaunch."""

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

import train_repo as training_adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_siff_ccsf_temperature_pilot_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/stage_c_siff_ccsf_step7b_prelaunch_20260718/local_gate"),
    )
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def manifest_rows(
    config: dict[str, Any],
    profiles: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for dataset in config["datasets"]:
        profile = profiles[dataset]
        for temperature in config["temperatures"]:
            rows.append(
                {
                    "job_index": len(rows) + 1,
                    "dataset": dataset,
                    "temperature": float(temperature),
                    "readout_mode": config["pilot_arm"]["readout_mode"],
                    "objective_mode": config["pilot_arm"]["objective_mode"],
                    "mode_rank": config["pilot_arm"]["mode_rank"],
                    "profile": profile["profile"],
                    "patch_num": profile["patch_num"],
                    "d_model": profile["d_model"],
                    "d_ff": profile["d_ff"],
                    "seed": config["seed"],
                    "evaluation_split": "val",
                    "checkpoint_policy": "best-val",
                    "test_access": False,
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
        f"CCSF_TEMP_tau{row['temperature']}",
        "--output-dir",
        f"/tmp/ccsf_tau{row['temperature']}_{row['dataset']}",
        "--device",
        "cuda",
        "--checkpoint-policy",
        "best-val",
        "--no-evaluate-dual-checkpoints",
        "--protocol-class",
        "method_screening",
        "--protocol-profile",
        "stage_c_siff_ccsf_temperature_pilot_v1",
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
        str(row["temperature"]),
        "--ccsf-calibration-weight",
        "0.1",
        "--pcc-objective-mode",
        row["objective_mode"],
        "--pred-loss-mode",
        "full",
        "--final-evaluation-split",
        "val",
        "--no-official-test-mode",
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
                and parsed.readout_mode == "ccsf-coupling-field"
                and parsed.pcc_objective_mode
                == "ccsf_relative_calibration"
                and parsed.ccsf_calibration_temperature
                == row["temperature"]
                and parsed.ccsf_calibration_weight == 0.1
                and parsed.validation_horizons == [96, 192, 336, 720]
                and parsed.evaluation_horizons == [96, 192, 336, 720]
                and parsed.final_evaluation_split == "val"
                and parsed.checkpoint_policy == "best-val"
                and parsed.save_predictions is False
                and parsed.official_test_mode is False
            ):
                return False
    finally:
        sys.argv = original
    return True


def runner_audit(config_path: Path) -> tuple[bool, int]:
    runner = ROOT / "scripts" / "remote" / "run_stage_c_siff_ccsf_v1.sh"
    syntax = subprocess.run(
        ["bash", "-n", str(runner)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    environment = dict(os.environ)
    environment.update({"CONFIG": str(config_path), "DRY_RUN": "1"})
    dry = subprocess.run(
        ["bash", str(runner)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    job_count = sum(line.count("\t") == 5 for line in dry.stdout.splitlines())
    passed = bool(
        syntax.returncode == 0
        and os.access(runner, os.X_OK)
        and dry.returncode == 0
        and job_count == 15
        and "validation_only=true" in dry.stdout
        and "formal_test_authorized=false" in dry.stdout
    )
    return passed, job_count


def analyzer_smoke(config_path: Path, output_dir: Path) -> bool:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_stage_c_siff_ccsf_temperature_pilot.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--synthetic-smoke",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    selected_path = output_dir / "selected_temperature.json"
    if result.returncode != 0 or not selected_path.is_file():
        return False
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    return bool(
        selected["selected_temperature"] == 0.25
        and selected["validation_only"] is True
        and selected["test_accessed"] is False
        and selected["pilot_checkpoints_reused"] is False
        and selected["formal_phase_a_authorized"] is False
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    step7a_path = Path(config["step7a_contract"]["path"])
    step7a = json.loads(step7a_path.read_text(encoding="utf-8"))
    local_gate_path = Path(config["step7a_contract"]["local_gate_path"])
    local_gate = json.loads(local_gate_path.read_text(encoding="utf-8"))
    profile_path = Path(config["profiles"]["path"])
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))[
        "dataset_profiles"
    ]
    manifest = manifest_rows(config, profiles)
    runner_pass, dry_jobs = runner_audit(args.config)
    synthetic_dir = args.output_dir / "synthetic_selection"
    categories = {
        "step7a_config_hash": (
            file_hash(step7a_path) == config["step7a_contract"]["sha256"]
        ),
        "step7a_local_gate_hash": (
            file_hash(local_gate_path)
            == config["step7a_contract"]["local_gate_sha256"]
        ),
        "step7a_gate_passed": bool(
            local_gate["overall_pass"]
            and local_gate["categories_passed"] == 18
            and local_gate["categories_total"] == 18
        ),
        "profile_hash": file_hash(profile_path) == config["profiles"]["sha256"],
        "temperature_grid": config["temperatures"] == [0.05, 0.1, 0.25],
        "matrix_size": len(manifest) == config["matrix"]["expected_runs"] == 15,
        "workload_order": [row["dataset"] for row in manifest[:6]]
        == ["Weather"] * 3 + ["ETTm1"] * 3,
        "cli_contracts": cli_audit(manifest, config),
        "validation_only_selection": bool(
            config["training"]["final_evaluation_split"] == "val"
            and config["training"]["test_labels_allowed"] is False
            and config["selection"]["test_labels_allowed"] is False
            and config["selection"]["one_temperature_shared_by_all_datasets"]
            and config["selection"]["selected_temperature_reuses_pilot_checkpoint"]
            is False
        ),
        "authorization_boundary": bool(
            config["authorization"]["validation_temperature_pilot_authorized"]
            and config["authorization"]["pilot_remote_training_authorized"]
            and not config["authorization"]["remote_training_authorized"]
            and not config["authorization"]["formal_phase_a_authorized"]
            and not config["authorization"]["formal_test_access_authorized"]
            and not config["authorization"]["confirmation_authorized"]
        ),
        "formal_temperature_unselected": bool(
            step7a["objective"]["formal_temperature_selected"] is False
        ),
        "runner_contract": runner_pass and dry_jobs == 15,
        "selection_analyzer_smoke": analyzer_smoke(args.config, synthetic_dir),
        "external_output_root": str(config["output"]["remote_root"]).startswith(
            "/home/yingch/exp_outputs/r-2026-fatst/"
        ),
    }
    overall_pass = all(categories.values())
    payload = {
        "pilot_id": config["pilot_id"],
        "current_step": "Step7B validation-pilot prelaunch",
        "categories": categories,
        "categories_passed": sum(categories.values()),
        "categories_total": len(categories),
        "pilot_jobs": len(manifest),
        "validation_cells": config["matrix"]["expected_validation_cells"],
        "overall_pass": overall_pass,
        "decision": (
            "step7b_temperature_pilot_prelaunch_pass"
            if overall_pass
            else "step7b_fail_return_step7a_or_protocol_repair"
        ),
        "pilot_remote_training_authorized": bool(
            overall_pass
            and config["authorization"]["pilot_remote_training_authorized"]
        ),
        "formal_phase_a_authorized": False,
        "formal_test_access_authorized": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "pilot_jobs.csv", manifest)
    write_csv(
        args.output_dir / "selection_contract.csv",
        [
            {
                "temperature": temperature,
                "datasets": len(config["datasets"]),
                "horizons": len(config["training"]["validation_horizons"]),
                "validation_cells": len(config["datasets"])
                * len(config["training"]["validation_horizons"]),
                "shared_selection": True,
                "test_access": False,
                "pilot_checkpoint_reuse": False,
            }
            for temperature in config["temperatures"]
        ],
    )
    (args.output_dir / "prelaunch_gate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if not overall_pass:
        raise RuntimeError(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
