#!/usr/bin/env python3
"""Run the local Decoder-Transfer prelaunch gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baselines" / "timealign_official"
sys.path.insert(0, str(BASELINE))
from models import TimeAlign  # noqa: E402
import train_repo  # noqa: E402


def add(rows: list[dict[str, Any]], category: str, case: str, passed: bool, value: Any, expected: Any) -> None:
    rows.append({"category": category, "case": case, "value": value, "expected": expected, "pass": bool(passed)})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_model(
    config: dict[str, Any],
    profiles: dict[str, Any],
    backbone: str,
    dataset: str,
    arm: dict[str, Any],
    output: Path,
) -> TimeAlign.Model:
    family = profiles["backbones"][backbone]
    profile = family["dataset_profiles"][dataset]
    argv = [
        "train_repo.py", "--dataset-root", str(output), "--dataset", dataset,
        "--mode", "unified", "--seq-len", "336", "--pred-len", "720",
        "--target-horizons", "720", "--validation-horizons", "96,192,336,720",
        "--evaluation-horizons", "96,192,336,720", "--segment-horizons", "96,192,336,720",
        "--evaluation-prefix-mode", "full-crop", "--encoder-mode", family["encoder_mode"],
        "--readout-mode", arm["readout_mode"], "--batch-size", "2", "--learning-rate", str(profile["learning_rate"]),
        "--epochs", "1", "--patience", "1", "--enable-early-stopping", "--checkpoint-policy", "best-val",
        "--protocol-class", "method_screening", "--protocol-profile", "iscf_bsca_decoder_transfer_20260814",
        "--profile-hash", config["profiles"]["sha256"], "--seed", "2021", "--num-workers", "0",
        "--run-name", "TRANSFER_PRELAUNCH", "--output-dir", str(output / arm["id"]), "--device", "cpu",
        "--pcsd-mode-rank", str(
            config["matched_ranks"][dataset]
            if arm["readout_mode"] == "siff-independent-scope-control"
            else 256
        ), "--pcsd-policy-mode", arm["policy_mode"],
        "--pcc-objective-mode", arm["objective_mode"], "--pred-loss-mode", "full", "--final-evaluation-split", "val",
    ]
    if backbone == "dlinear_style":
        argv += ["--dlinear-moving-avg", str(family["moving_average"])]
    else:
        argv += [
            "--history-patch-len", str(family["history_patch_len"]),
            "--history-patch-stride", str(family["history_patch_stride"]),
            "--history-d-model", str(profile["d_model"]), "--history-n-heads", str(profile["n_heads"]),
            "--history-d-ff", str(profile["d_ff"]), "--history-e-layers", str(family["history_e_layers"]),
            "--history-dropout", str(profile["dropout"]),
        ]
    original = sys.argv
    try:
        sys.argv = argv
        parsed = train_repo.parse_args()
    finally:
        sys.argv = original
    preset = train_repo.OFFICIAL_PRESETS[dataset][720]
    data_path = output / preset.data_path
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.touch()
    official = train_repo.build_official_args(parsed, preset)
    torch.manual_seed(2021)
    return TimeAlign.Model(official).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/iscf_bsca_decoder_transfer_protocol.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    profile_path = ROOT / config["profiles"]["path"]
    profiles = json.loads(profile_path.read_text())
    rows: list[dict[str, Any]] = []
    add(rows, "contract", "profile_hash", sha256(profile_path) == config["profiles"]["sha256"], sha256(profile_path), config["profiles"]["sha256"])
    add(rows, "contract", "matrix_30", len(config["launch_order"]) == 30 and len({tuple(x) for x in config["launch_order"]}) == 30, len(config["launch_order"]), 30)
    add(rows, "governance", "joint_training", config["training"]["from_scratch"] and config["training"]["joint_encoder_decoder"] and not config["training"]["frozen_replacement_or_warm_start"], "joint_from_scratch", "joint_from_scratch")
    arms = {row["id"]: row for row in config["arms"]}
    with tempfile.TemporaryDirectory(prefix="fatst_transfer_") as temp:
        root = Path(temp)
        for backbone in ("dlinear_style", "patchtst_style"):
            subset = [arms[f"{'dlinear' if backbone == 'dlinear_style' else 'patchtst'}_{suffix}"] for suffix in ("original", "iscf", "iscf_bsca")]
            hashes = []
            for arm in subset:
                model = parse_model(config, profiles, backbone, "ETTh1", arm, root)
                hashes.append(train_repo.initialization_contract(model)["encoder_initialization_hash"])
                x = torch.randn(1, 336, 7)
                y = torch.zeros(1, 720, 7)
                with torch.no_grad():
                    full = model(x, y, is_training=False, target_prefix=720)[0]
                    prefix = model(x, y, is_training=False, target_prefix=96)[0]
                passed = full.shape == (1, 720, 7) and prefix.shape == (1, 96, 7) and torch.isfinite(full).all() and torch.equal(prefix, full[:, :96])
                add(rows, "model", arm["id"], bool(passed), f"full={tuple(full.shape)},prefix={tuple(prefix.shape)}", "finite_and_exact_prefix")
            add(rows, "model", f"{backbone}_matched_encoder_init", len(set(hashes)) == 1, len(set(hashes)), 1)
    syntax = subprocess.run(["bash", "-n", "scripts/remote/run_iscf_bsca_decoder_transfer.sh"], cwd=ROOT, check=False)
    dry = subprocess.run(["bash", "scripts/remote/run_iscf_bsca_decoder_transfer.sh"], cwd=ROOT, env={**os.environ, "DRY_RUN": "1"}, text=True, capture_output=True, check=False)
    add(rows, "execution", "runner_syntax", syntax.returncode == 0, syntax.returncode, 0)
    add(rows, "execution", "runner_dry_run", dry.returncode == 0 and "jobs=30" in dry.stdout, dry.stdout.strip().splitlines()[-1] if dry.stdout.strip() else dry.stderr, "jobs=30")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "prelaunch_checks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    passed = all(row["pass"] for row in rows)
    (args.output_dir / "prelaunch_summary.json").write_text(json.dumps({"pass": passed, "checks": len(rows), "protocol_sha256": sha256(args.config)}, indent=2) + "\n")
    if not passed:
        raise SystemExit("decoder transfer prelaunch failed")
    print(f"decoder_transfer_prelaunch=pass checks={len(rows)}")


if __name__ == "__main__":
    main()
