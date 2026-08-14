#!/usr/bin/env python3
"""Run the local PatchTST decoder-HPO v2 prelaunch gate."""

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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add(
    rows: list[dict[str, Any]],
    category: str,
    case: str,
    passed: bool,
    value: Any,
    expected: Any,
) -> None:
    rows.append(
        {
            "category": category,
            "case": case,
            "value": value,
            "expected": expected,
            "pass": bool(passed),
        }
    )


def parse_trial(
    config: dict[str, Any],
    source: dict[str, Any],
    profile: dict[str, Any],
    dataset: str,
    output: Path,
) -> tuple[argparse.Namespace, argparse.Namespace, TimeAlign.Model]:
    family = source["backbones"]["patchtst_style"]
    dataset_profile = family["dataset_profiles"][dataset]
    reference_rank = config["reference_profile"]["mode_rank_by_dataset"][dataset]
    rank = max(1, round(reference_rank * profile["rank_scale"]))
    argv = [
        "train_repo.py",
        "--dataset-root", str(output),
        "--dataset", dataset,
        "--mode", "unified",
        "--seq-len", "336",
        "--pred-len", "720",
        "--target-horizons", "720",
        "--validation-horizons", "96,192,336,720",
        "--evaluation-horizons", "96,192,336,720",
        "--segment-horizons", "96,192,336,720",
        "--evaluation-prefix-mode", "full-crop",
        "--encoder-mode", "contextual-patch-transformer",
        "--readout-mode", "siff-independent-scope-control",
        "--batch-size", "2",
        "--learning-rate", str(dataset_profile["learning_rate"]),
        "--weight-decay", "0",
        "--readout-learning-rate-multiplier", str(profile["readout_learning_rate_multiplier"]),
        "--readout-weight-decay", str(profile["readout_weight_decay"]),
        "--epochs", "1",
        "--patience", "1",
        "--enable-early-stopping",
        "--checkpoint-policy", "best-val",
        "--allow-archived-research-modes",
        "--protocol-class", "method_screening",
        "--protocol-profile", "iscf_bsca_decoder_transfer_patchtst_hpo_v2_20260815",
        "--profile-hash", config["backbone"]["profile_sha256"],
        "--seed", "2021",
        "--num-workers", "0",
        "--run-name", "PATCHTST_HPO_V2_PRELAUNCH",
        "--output-dir", str(output / profile["id"]),
        "--device", "cpu",
        "--no-official-test-mode",
        "--history-patch-len", str(family["history_patch_len"]),
        "--history-patch-stride", str(family["history_patch_stride"]),
        "--history-d-model", str(dataset_profile["d_model"]),
        "--history-n-heads", str(dataset_profile["n_heads"]),
        "--history-d-ff", str(dataset_profile["d_ff"]),
        "--history-e-layers", str(family["history_e_layers"]),
        "--history-dropout", str(dataset_profile["dropout"]),
        "--pcsd-mode-rank", str(rank),
        "--pcsd-policy-mode", "direct",
        "--pcc-objective-mode", "equal_uniform_scope_anchor",
        "--pred-loss-mode", "full",
        "--final-evaluation-split", "val",
    ]
    original = sys.argv
    try:
        sys.argv = argv
        args = train_repo.parse_args()
    finally:
        sys.argv = original
    preset = train_repo.OFFICIAL_PRESETS[dataset][720]
    data_path = output / preset.data_path
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.touch()
    official = train_repo.build_official_args(args, preset)
    torch.manual_seed(2021)
    model = TimeAlign.Model(official).eval()
    return args, official, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/iscf_bsca_decoder_transfer_patchtst_hpo_v2.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source_path = ROOT / config["backbone"]["profile_path"]
    source = json.loads(source_path.read_text())
    rows: list[dict[str, Any]] = []

    add(
        rows,
        "contract",
        "source_profile_hash",
        sha256(source_path) == config["backbone"]["profile_sha256"],
        sha256(source_path),
        config["backbone"]["profile_sha256"],
    )
    profiles = config["search_profiles"]
    datasets = config["datasets"]
    jobs = [(dataset, profile["id"]) for dataset in datasets for profile in profiles]
    add(rows, "contract", "ten_profiles", len(profiles) == 10, len(profiles), 10)
    add(rows, "contract", "matrix_50_unique", len(jobs) == 50 and len(set(jobs)) == 50, len(set(jobs)), 50)
    add(rows, "governance", "no_formal_test", not config["authorization"]["formal_test_access_authorized"] and config["matrix"]["formal_test_jobs"] == 0, config["matrix"]["formal_test_jobs"], 0)
    add(rows, "governance", "joint_from_scratch", config["training"]["from_scratch"] and config["arm"]["joint_encoder_decoder_training"] and not config["arm"]["frozen_replacement"], "joint_from_scratch", "joint_from_scratch")

    with tempfile.TemporaryDirectory(prefix="fatst_patchtst_hpo_v2_") as temp:
        temp_root = Path(temp)
        for profile in profiles:
            parsed, official, model = parse_trial(config, source, profile, "ETTh1", temp_root)
            optimizer = train_repo.build_optimizer(model, official, parsed)
            readout_group = next(
                (group for group in optimizer.param_groups if group.get("group_name") == "readout"),
                None,
            )
            expected_lr = official.learning_rate * profile["readout_learning_rate_multiplier"]
            group_pass = (
                readout_group is not None
                and abs(readout_group["lr"] - expected_lr) < 1e-15
                and abs(readout_group["weight_decay"] - profile["readout_weight_decay"]) < 1e-15
            )
            x = torch.randn(1, 336, 7)
            y = torch.zeros(1, 720, 7)
            with torch.inference_mode():
                full = model(x, y, is_training=False, target_prefix=720)[0]
                prefix = model(x, y, is_training=False, target_prefix=96)[0]
            shape_pass = full.shape == (1, 720, 7) and torch.equal(prefix, full[:, :96]) and torch.isfinite(full).all()
            add(rows, "optimizer", profile["id"], group_pass, f"lr={readout_group['lr'] if readout_group else None},wd={readout_group['weight_decay'] if readout_group else None}", f"lr={expected_lr},wd={profile['readout_weight_decay']}")
            add(rows, "model", profile["id"], bool(shape_pass), tuple(full.shape), "finite_exact_prefix")

    runner = "scripts/remote/run_iscf_bsca_decoder_transfer_patchtst_hpo_v2.sh"
    syntax = subprocess.run(["bash", "-n", runner], cwd=ROOT, check=False)
    dry = subprocess.run(
        ["bash", runner],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    add(rows, "execution", "runner_syntax", syntax.returncode == 0, syntax.returncode, 0)
    add(rows, "execution", "runner_dry_run", dry.returncode == 0 and "jobs=50" in dry.stdout, dry.stdout.strip().splitlines()[-1] if dry.stdout.strip() else dry.stderr, "jobs=50")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "prelaunch_checks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    passed = all(row["pass"] for row in rows)
    (args.output_dir / "prelaunch_summary.json").write_text(
        json.dumps(
            {
                "pass": passed,
                "checks": len(rows),
                "protocol_sha256": sha256(args.config),
            },
            indent=2,
        )
        + "\n"
    )
    if not passed:
        raise SystemExit("PatchTST decoder-HPO v2 prelaunch failed")
    print(f"patchtst_decoder_hpo_v2_prelaunch=pass checks={len(rows)}")


if __name__ == "__main__":
    main()
