#!/usr/bin/env python3
"""Verify SC0-R1 stopping, carrier, and multi-seed selection semantics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage_c_mechanism_control_r1.json"
for path in (TIMEALIGN_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_stage_c_sc0_r1_carrier_calibration import select  # noqa: E402
from train_repo import early_stopping_update  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT.parent / "datasets",
    )
    parser.add_argument("--skip-end-to-end", action="store_true")
    return parser.parse_args()


def config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stopping_unit_gate() -> None:
    best = float("inf")
    counter = 0
    stop_epoch = None
    for epoch, value in enumerate((1.0, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95), 1):
        improved, counter = early_stopping_update(value, best, counter, 0.0)
        if improved:
            best = value
        if counter >= 5:
            stop_epoch = epoch
            break
    if stop_epoch != 7 or best != 0.9:
        raise AssertionError(f"unexpected stopping state: {stop_epoch=} {best=}")


def selection_unit_gate(config: dict) -> None:
    metrics = []
    diagnostics = []
    arm_factors = {
        "sc0_p12_d128": 1.02,
        "sc0_p24_d64": 1.00,
        "sc0_p48_d32": 1.03,
    }
    for seed in config["common"]["seeds"]:
        for dataset in config["datasets"]:
            for arm, factor in arm_factors.items():
                metrics.append(
                    {
                        "seed": seed,
                        "dataset": dataset,
                        "arm": arm,
                        "target_horizon": 720,
                        "mse": factor * (1.0 + (int(seed) - 2021) * 0.001),
                        "mae": 0.5,
                    }
                )
                diagnostics.append(
                    {
                        "seed": seed,
                        "dataset": dataset,
                        "arm": arm,
                        "status": "ok",
                        "mean_epoch_seconds": 1.0 + factor,
                    }
                )
    _seed_rows, _aggregate_rows, gate = select(metrics, diagnostics, config)
    if gate["selected_arm"] != "sc0_p24_d64":
        raise AssertionError(f"unexpected synthetic winner: {gate}")
    if gate["selected_seed_winner_count"] != 3:
        raise AssertionError(f"unexpected seed winner count: {gate}")


def analyzer_integration_gate(config_path: Path, config: dict) -> None:
    config_digest = config_hash(config_path)
    factors = {
        "sc0_p12_d128": 1.02,
        "sc0_p24_d64": 1.00,
        "sc0_p48_d32": 1.03,
    }
    with tempfile.TemporaryDirectory(prefix="stage_c_sc0_r1_analyzer_") as temp_dir:
        root = Path(temp_dir)
        raw_root = root / "raw"
        output_dir = root / "analysis"
        for seed in config["common"]["seeds"]:
            for dataset in config["datasets"]:
                for arm_name, arm in config["arms"].items():
                    run_dir = (
                        raw_root / f"SC0R1_{arm_name}_validation_only" / dataset
                        / "h720_full" / f"seed{seed}"
                    )
                    run_dir.mkdir(parents=True)
                    effective = {
                        "adapter": {
                            "protocol_class": "mechanism_control",
                            "protocol_profile": config["protocol_profile"],
                            "profile_hash": config_digest,
                            "final_evaluation_split": "val",
                            "checkpoint_policy": "best-val",
                            "enable_early_stopping": True,
                            "patience": 5,
                            "early_stopping_min_delta": 0.0,
                            "seed": seed,
                        },
                        "official_args": {
                            "patch_num": arm["patch_num"],
                            "d_model": arm["d_model"],
                            "d_ff": arm["d_ff"],
                        },
                    }
                    (run_dir / "effective_config.json").write_text(
                        json.dumps(effective), encoding="utf-8"
                    )
                    (run_dir / "model_diagnostics.json").write_text(
                        json.dumps(
                            {
                                "active_forward_parameters": arm[
                                    "active_forward_parameters"
                                ],
                                "unused_proj_x_parameters": arm[
                                    "unused_proj_x_parameters"
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )
                    with (run_dir / "training_log.csv").open(
                        "w", encoding="utf-8", newline=""
                    ) as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=[
                                "train_loss", "val_mean_mse", "lr", "epoch_seconds",
                                "best_epoch_so_far", "stop_triggered",
                            ],
                            lineterminator="\n",
                        )
                        writer.writeheader()
                        writer.writerow(
                            {
                                "train_loss": 0.5,
                                "val_mean_mse": factors[arm_name],
                                "lr": 0.0001,
                                "epoch_seconds": factors[arm_name],
                                "best_epoch_so_far": 1,
                                "stop_triggered": 1,
                            }
                        )
                    with (run_dir / "metrics_by_target_horizon.csv").open(
                        "w", encoding="utf-8", newline=""
                    ) as handle:
                        fields = [
                            "target_horizon", "mse", "mae", "evaluation_split",
                            "protocol_class",
                        ]
                        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                        writer.writeheader()
                        for horizon in config["common"]["evaluation_horizons"]:
                            writer.writerow(
                                {
                                    "target_horizon": horizon,
                                    "mse": factors[arm_name],
                                    "mae": 0.5,
                                    "evaluation_split": "val",
                                    "protocol_class": "mechanism_control",
                                }
                            )
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "analyze_stage_c_sc0_r1_carrier_calibration.py"),
                "--raw-root", str(raw_root),
                "--output-dir", str(output_dir),
                "--config", str(config_path),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        summary = json.loads((output_dir / "sc0_r1_summary.json").read_text())
        if summary["decision"] != "global_profile_selected_and_frozen":
            raise AssertionError(f"analyzer integration gate failed: {summary}")
        if summary["selected_arm"] != "sc0_p24_d64":
            raise AssertionError(f"unexpected analyzer winner: {summary}")


def end_to_end_gate(config_path: Path, config: dict, dataset_root: Path) -> None:
    common = config["common"]
    arm = config["arms"]["sc0_p24_d64"]
    with tempfile.TemporaryDirectory(prefix="stage_c_sc0_r1_") as temp_dir:
        output_dir = Path(temp_dir) / "run"
        command = [
            sys.executable,
            str(TIMEALIGN_ROOT / "train_repo.py"),
            "--dataset-root", str(dataset_root),
            "--dataset", "ETTh2",
            "--mode", "unified",
            "--seq-len", "720",
            "--pred-len", "720",
            "--target-horizons", "720",
            "--validation-horizons", "720",
            "--evaluation-horizons", "48,720",
            "--batch-size", "2",
            "--epochs", "3",
            "--patience", "1",
            "--enable-early-stopping",
            "--early-stopping-min-delta", "1000000000",
            "--seed", "2021",
            "--max-train-batches", "1",
            "--max-eval-batches", "1",
            "--num-workers", "0",
            "--run-name", "SC0R1_local_semantic_gate",
            "--output-dir", str(output_dir),
            "--device", "cpu",
            "--checkpoint-policy", "best-val",
            "--no-evaluate-dual-checkpoints",
            "--final-evaluation-split", "val",
            "--protocol-class", "mechanism_control",
            "--protocol-profile", config["protocol_profile"],
            "--profile-hash", config_hash(config_path),
            "--legacy-patch-num", str(arm["patch_num"]),
            "--legacy-d-model", str(arm["d_model"]),
            "--legacy-d-ff", str(arm["d_ff"]),
            "--legacy-dropout", str(common["dropout"]),
            "--legacy-layer-norm", str(common["layer_norm"]),
            "--learning-rate", str(common["learning_rate"]),
            "--readout-mode", common["readout_mode"],
            "--basis-rank", str(common["basis_rank"]),
            "--pred-loss-mode", common["pred_loss_mode"],
        ]
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"SC0-R1 smoke failed\n{result.stdout}\n{result.stderr}")
        with (output_dir / "training_log.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 2 or rows[-1]["stop_triggered"] != "1":
            raise AssertionError(f"early stop was not triggered at epoch 2: {rows}")
        if rows[-1]["best_epoch_so_far"] != "1":
            raise AssertionError(f"best checkpoint tracking failed: {rows[-1]}")
        required = (
            "checkpoint.pt",
            "effective_config.json",
            "metrics_by_target_horizon.csv",
            "predictions_val.npz",
        )
        if any(not (output_dir / name).exists() for name in required):
            raise AssertionError("SC0-R1 end-to-end artifacts are incomplete")
        if (output_dir / "predictions_test.npz").exists():
            raise AssertionError("test predictions are forbidden")
        effective = json.loads((output_dir / "effective_config.json").read_text())
        if not effective["adapter"]["enable_early_stopping"]:
            raise AssertionError("effective config did not record early stopping")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["common"]["early_stopping_patience"] != 5:
        raise AssertionError("SC0-R1 patience must remain preregistered at 5")
    stopping_unit_gate()
    selection_unit_gate(config)
    analyzer_integration_gate(args.config, config)
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "check_stage_c_sc0_carrier_local.py"),
            "--config", str(args.config),
            "--dataset-root", str(args.dataset_root),
            "--skip-end-to-end",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    if not args.skip_end_to_end:
        end_to_end_gate(args.config, config, args.dataset_root)
    print(f"stage_c_sc0_r1_local_check=pass profile_hash={config_hash(args.config)}")


if __name__ == "__main__":
    main()
