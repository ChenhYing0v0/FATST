#!/usr/bin/env python3
"""Verify StageC SC0-DAP-R2 Phase A semantics before remote launch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage_c_dataset_profile_calibration_r2.json"
for path in (TIMEALIGN_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_stage_c_dap_r2a_patch_screen import select  # noqa: E402
from models.TimeAlign import Model  # noqa: E402
from train_repo import model_diagnostics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT.parent / "datasets")
    parser.add_argument("--skip-end-to-end", action="store_true")
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_args(common: dict, profile: dict) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=common["seq_len"], pred_len=common["pred_len"],
        patch_num=profile["patch_num"], d_model=profile["d_model"],
        n_heads=8, e_layers=common["e_layers"], d_ff=profile["d_ff"],
        dropout=common["dropout"], layer_norm=common["layer_norm"], pos=1,
        enc_in=7, readout_mode=common["readout_mode"],
        encoder_mode=common["encoder_mode"], basis_rank=common["basis_rank"],
        target_horizons=common["target_horizons"],
    )


def structural_gate(config: dict) -> None:
    common = config["common"]
    profiles = config["phase_a_patch_screen"]["profiles"]
    observed_params = []
    for name, profile in profiles.items():
        if profile["d_model"] != 64 or profile["d_ff"] != 128:
            raise AssertionError(f"{name}: Phase A must fix natural width D64/ff128")
        if common["seq_len"] % profile["patch_num"]:
            raise AssertionError(f"{name}: patch_num must divide seq_len")
        model = Model(model_args(common, profile)).eval()
        info = model_diagnostics(model)
        observed_params.append(int(info["active_forward_parameters"]))
        x = torch.randn(2, common["seq_len"], 7)
        y = torch.randn(2, common["pred_len"], 7)
        with torch.no_grad():
            output = model(x, y, is_training=False, target_prefix=720)[0]
        if tuple(output.shape) != (2, 720, 7):
            raise AssertionError(f"{name}: output shape mismatch")
    if len(set(observed_params)) == 1:
        raise AssertionError("R2A must not force profiles to equal parameter counts")


def selector_gate(config: dict) -> None:
    profiles = list(config["phase_a_patch_screen"]["profiles"])
    expected = {
        "Weather": profiles[0], "ETTm1": profiles[1], "ETTh2": profiles[2]
    }
    metrics = []
    for dataset in config["datasets"]:
        for profile in profiles:
            factor = 1.0 if profile == expected[dataset] else 1.1
            for horizon in config["common"]["evaluation_horizons"]:
                metrics.append(
                    {"dataset": dataset, "profile": profile, "target_horizon": horizon,
                     "mse": factor * (1.0 + horizon / 10000.0), "mae": 0.5}
                )
    _rows, winners = select(metrics, config)
    if winners != expected:
        raise AssertionError(f"dense selector mismatch: {winners} != {expected}")


def end_to_end_gate(config_path: Path, config: dict, dataset_root: Path) -> None:
    common = config["common"]
    profile = config["phase_a_patch_screen"]["profiles"]["r2a_p24_d64_ff128"]
    with tempfile.TemporaryDirectory(prefix="stage_c_dap_r2a_") as temp_dir:
        output_dir = Path(temp_dir) / "run"
        command = [
            sys.executable, str(TIMEALIGN_ROOT / "train_repo.py"),
            "--dataset-root", str(dataset_root), "--dataset", "ETTh2", "--mode", "unified",
            "--seq-len", "720", "--pred-len", "720", "--target-horizons", "720",
            "--validation-horizons", "720", "--evaluation-horizons", "48,720",
            "--batch-size", "2", "--epochs", "3", "--patience", "1",
            "--enable-early-stopping", "--early-stopping-min-delta", "1000000000",
            "--seed", "2021", "--max-train-batches", "1", "--max-eval-batches", "1",
            "--num-workers", "0", "--run-name", "SC0DAP_R2A_local",
            "--output-dir", str(output_dir), "--device", "cpu",
            "--checkpoint-policy", "best-val", "--no-evaluate-dual-checkpoints",
            "--final-evaluation-split", "val", "--protocol-class", "mechanism_control",
            "--protocol-profile", config["protocol_profile"],
            "--profile-hash", file_hash(config_path),
            "--legacy-patch-num", str(profile["patch_num"]),
            "--legacy-d-model", str(profile["d_model"]),
            "--legacy-d-ff", str(profile["d_ff"]),
            "--legacy-dropout", str(common["dropout"]),
            "--legacy-layer-norm", str(common["layer_norm"]),
            "--learning-rate", str(common["learning_rate"]),
            "--readout-mode", common["readout_mode"],
            "--basis-rank", str(common["basis_rank"]), "--pred-loss-mode", "full",
        ]
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"R2A local smoke failed\n{result.stdout}\n{result.stderr}")
        with (output_dir / "training_log.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 2 or rows[-1]["stop_triggered"] != "1":
            raise AssertionError("early stopping semantic gate failed")
        if not (output_dir / "metrics_by_target_horizon.csv").exists():
            raise AssertionError("validation metrics missing")
        if (output_dir / "predictions_test.npz").exists():
            raise AssertionError("test artifacts are forbidden")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["selection_policy"]["parameter_count_used_for_selection"]:
        raise AssertionError("parameter count must not participate in R2 selection")
    structural_gate(config)
    selector_gate(config)
    if not args.skip_end_to_end:
        end_to_end_gate(args.config, config, args.dataset_root)
    print(f"stage_c_dap_r2a_local_check=pass profile_hash={file_hash(args.config)}")


if __name__ == "__main__":
    main()
