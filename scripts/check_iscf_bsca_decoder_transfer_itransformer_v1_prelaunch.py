#!/usr/bin/env python3
"""Run the local iTransformer-style Decoder-Transfer prelaunch gate."""

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
from layers.SIFF import siff_parameter_count  # noqa: E402
from models import TimeAlign  # noqa: E402
import train_repo  # noqa: E402


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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_model(
    config: dict[str, Any],
    profiles: dict[str, Any],
    dataset: str,
    arm: dict[str, Any],
    output: Path,
) -> TimeAlign.Model:
    common = profiles["common"]
    profile = profiles["dataset_profiles"][dataset]
    argv = [
        "train_repo.py",
        "--dataset-root",
        str(output),
        "--dataset",
        dataset,
        "--mode",
        "unified",
        "--seq-len",
        str(common["seq_len"]),
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
        "--encoder-mode",
        common["encoder_mode"],
        "--readout-mode",
        arm["readout_mode"],
        "--history-d-model",
        str(profile["d_model"]),
        "--history-n-heads",
        str(common["n_heads"]),
        "--history-d-ff",
        str(profile["d_ff"]),
        "--history-e-layers",
        str(profile["e_layers"]),
        "--history-dropout",
        str(common["dropout"]),
        "--batch-size",
        "2",
        "--learning-rate",
        str(common["learning_rate"]),
        "--epochs",
        "1",
        "--patience",
        "1",
        "--enable-early-stopping",
        "--checkpoint-policy",
        "best-val",
        "--protocol-class",
        "method_screening",
        "--protocol-profile",
        "iscf_bsca_decoder_transfer_itransformer_v1_20260815",
        "--profile-hash",
        config["profiles"]["sha256"],
        "--seed",
        "2021",
        "--num-workers",
        "0",
        "--run-name",
        "ITRANSFORMER_TRANSFER_PRELAUNCH",
        "--output-dir",
        str(output / arm["id"] / dataset),
        "--device",
        "cpu",
        "--pcsd-mode-rank",
        str(
            profile["mode_rank"]
            if arm["readout_mode"] == "siff-independent-scope-control"
            else 256
        ),
        "--pcsd-policy-mode",
        arm["policy_mode"],
        "--pcc-objective-mode",
        arm["objective_mode"],
        "--pred-loss-mode",
        "full",
        "--final-evaluation-split",
        "val",
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
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/iscf_bsca_decoder_transfer_itransformer_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    profile_path = ROOT / config["profiles"]["path"]
    profiles = json.loads(profile_path.read_text())
    rows: list[dict[str, Any]] = []

    add(
        rows,
        "contract",
        "profile_hash",
        sha256(profile_path) == config["profiles"]["sha256"],
        sha256(profile_path),
        config["profiles"]["sha256"],
    )
    launch_order = [tuple(row) for row in config["launch_order"]]
    add(
        rows,
        "contract",
        "matrix_15",
        len(launch_order) == 15 and len(set(launch_order)) == 15,
        len(launch_order),
        15,
    )
    authorization = config["authorization"]
    add(
        rows,
        "governance",
        "formal_test_blocked",
        authorization["formal_test"] is False
        and authorization["table_mutation"] is False,
        f"test={authorization['formal_test']},table={authorization['table_mutation']}",
        "test=False,table=False",
    )
    add(
        rows,
        "governance",
        "joint_training",
        config["training"]["from_scratch"]
        and config["training"]["joint_encoder_decoder"]
        and not config["training"]["frozen_replacement_or_warm_start"],
        "joint_from_scratch",
        "joint_from_scratch",
    )

    for dimension, rank, expected in ((128, 21, 91961), (512, 30, 370829)):
        actual = siff_parameter_count(
            readout_dim=dimension,
            mode_rank=rank,
            scale_components=5,
        )
        add(
            rows,
            "capacity",
            f"d{dimension}_rank{rank}",
            actual == expected,
            actual,
            expected,
        )

    arms = {row["id"]: row for row in config["arms"]}
    with tempfile.TemporaryDirectory(prefix="fatst_itransformer_transfer_") as temp:
        root = Path(temp)
        for dataset in config["datasets"]:
            encoder_hashes = []
            for arm_id in (
                "itransformer_original",
                "itransformer_iscf",
                "itransformer_iscf_bsca",
            ):
                model = parse_model(config, profiles, dataset, arms[arm_id], root)
                contract = train_repo.initialization_contract(model)
                encoder_hashes.append(contract["encoder_initialization_hash"])
                channels = train_repo.OFFICIAL_PRESETS[dataset][720].enc_in
                x = torch.randn(1, profiles["common"]["seq_len"], channels)
                y = torch.zeros(1, 720, channels)
                with torch.inference_mode():
                    full = model(
                        x,
                        y,
                        is_training=False,
                        target_prefix=720,
                    )[0]
                    prefix = model(
                        x,
                        y,
                        is_training=False,
                        target_prefix=96,
                    )[0]
                passed = (
                    full.shape == (1, 720, channels)
                    and prefix.shape == (1, 96, channels)
                    and torch.isfinite(full).all()
                    and torch.equal(prefix, full[:, :96])
                )
                add(
                    rows,
                    "model",
                    f"{dataset}_{arm_id}",
                    bool(passed),
                    f"full={tuple(full.shape)},prefix={tuple(prefix.shape)}",
                    "finite_and_exact_prefix",
                )
            add(
                rows,
                "model",
                f"{dataset}_matched_encoder_init",
                len(set(encoder_hashes)) == 1,
                len(set(encoder_hashes)),
                1,
            )

    runner = "scripts/remote/run_iscf_bsca_decoder_transfer_itransformer_v1.sh"
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
    add(
        rows,
        "execution",
        "runner_dry_run",
        dry.returncode == 0
        and "jobs=15" in dry.stdout
        and "test" not in dry.stdout.lower(),
        dry.stdout.strip().splitlines()[-1] if dry.stdout.strip() else dry.stderr,
        "jobs=15",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "prelaunch_checks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    passed = all(row["pass"] for row in rows)
    summary = {
        "pass": passed,
        "checks": len(rows),
        "protocol_sha256": sha256(args.config),
        "formal_test_authorized": False,
    }
    (args.output_dir / "prelaunch_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if not passed:
        failed = [row["case"] for row in rows if not row["pass"]]
        raise SystemExit(f"iTransformer transfer prelaunch failed: {failed}")
    print(f"itransformer_transfer_prelaunch=pass checks={len(rows)}")


if __name__ == "__main__":
    main()
