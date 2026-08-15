#!/usr/bin/env python3
"""Run the local iTransformer decoder-HPO v2 prelaunch gate."""

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


def build_model(
    config: dict[str, Any],
    source: dict[str, Any],
    profile: dict[str, Any],
    output: Path,
) -> tuple[argparse.Namespace, argparse.Namespace, TimeAlign.Model, int]:
    dataset = "ETTh1"
    common = source["common"]
    dataset_profile = source["dataset_profiles"][dataset]
    reference_rank = config["reference_profile"]["mode_rank_by_dataset"][dataset]
    rank = max(1, round(reference_rank * profile["rank_scale"]))
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
        "itransformer-variate-attention",
        "--readout-mode",
        "siff-independent-scope-control",
        "--history-d-model",
        str(dataset_profile["d_model"]),
        "--history-n-heads",
        str(common["n_heads"]),
        "--history-d-ff",
        str(dataset_profile["d_ff"]),
        "--history-e-layers",
        str(dataset_profile["e_layers"]),
        "--history-dropout",
        str(common["dropout"]),
        "--batch-size",
        "2",
        "--learning-rate",
        str(common["learning_rate"]),
        "--weight-decay",
        "0",
        "--readout-learning-rate-multiplier",
        str(profile["readout_learning_rate_multiplier"]),
        "--readout-weight-decay",
        str(profile["readout_weight_decay"]),
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
        "iscf_bsca_decoder_transfer_itransformer_hpo_v2_20260816",
        "--profile-hash",
        config["backbone"]["profile_sha256"],
        "--seed",
        "2021",
        "--num-workers",
        "0",
        "--run-name",
        "ITRANSFORMER_HPO_V2_PRELAUNCH",
        "--output-dir",
        str(output / profile["id"]),
        "--device",
        "cpu",
        "--no-official-test-mode",
        "--pcsd-coordinate-dim",
        str(profile["coordinate_dim"]),
        "--pcsd-mode-rank",
        str(rank),
        "--pcsd-scales",
        ",".join(map(str, profile["scales"])),
        "--pcsd-policy-history-dim",
        str(profile["policy_history_dim"]),
        "--pcsd-policy-hidden-dim",
        str(profile["policy_hidden_dim"]),
        "--pcsd-policy-mode",
        "direct",
        "--pcsd-fixed-scale",
        "720",
        "--pcc-objective-mode",
        "equal_uniform_scope_anchor",
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
    model = TimeAlign.Model(official).eval()
    return parsed, official, model, rank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/iscf_bsca_decoder_transfer_itransformer_hpo_v2.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source_path = ROOT / config["backbone"]["profile_path"]
    source = json.loads(source_path.read_text())
    profiles = config["search_profiles"]
    datasets = config["datasets"]
    rows: list[dict[str, Any]] = []

    add(
        rows,
        "contract",
        "source_profile_hash",
        sha256(source_path) == config["backbone"]["profile_sha256"],
        sha256(source_path),
        config["backbone"]["profile_sha256"],
    )
    jobs = [(dataset, profile["id"]) for dataset in datasets for profile in profiles]
    add(rows, "contract", "profiles", len(profiles) == 14, len(profiles), 14)
    add(
        rows,
        "contract",
        "matrix",
        len(jobs) == 70 and len(set(jobs)) == 70,
        len(set(jobs)),
        70,
    )
    fingerprints = {
        (
            profile["rank_scale"],
            profile["coordinate_dim"],
            profile["policy_history_dim"],
            profile["policy_hidden_dim"],
            tuple(profile["scales"]),
            profile["readout_learning_rate_multiplier"],
            profile["readout_weight_decay"],
        )
        for profile in profiles
    }
    add(
        rows,
        "contract",
        "unique_effective_profiles",
        len(fingerprints) == 14,
        len(fingerprints),
        14,
    )
    selection = config["test_tuned_selection"]
    add(
        rows,
        "governance",
        "dataset_level_test_tuning_only",
        bool(
            config["test_tuned"]
            and selection["unit"]
            == "one_dataset_profile_shared_by_all_four_horizons"
            and selection["per_horizon_seed_metric_or_cell_selection_allowed"]
            is False
        ),
        selection["unit"],
        "one profile per dataset and four horizons",
    )
    add(
        rows,
        "governance",
        "joint_from_scratch",
        bool(
            config["training"]["from_scratch"]
            and config["arm"]["joint_encoder_decoder_training"]
            and not config["arm"]["frozen_replacement"]
        ),
        "joint_from_scratch",
        "joint_from_scratch",
    )

    with tempfile.TemporaryDirectory(prefix="fatst_itransformer_hpo_v2_") as temp:
        temp_root = Path(temp)
        for profile in profiles:
            parsed, official, model, rank = build_model(
                config, source, profile, temp_root
            )
            readout = model.pcsd_readout
            optimizer = train_repo.build_optimizer(model, official, parsed)
            readout_group = next(
                group
                for group in optimizer.param_groups
                if group.get("group_name") == "readout"
            )
            expected_lr = (
                official.learning_rate
                * profile["readout_learning_rate_multiplier"]
            )
            contract_pass = bool(
                readout.mode_rank == rank
                and readout.coordinate_dim == profile["coordinate_dim"]
                and readout.policy_history_dim == profile["policy_history_dim"]
                and readout.policy_hidden_dim == profile["policy_hidden_dim"]
                and list(readout.scales) == profile["scales"]
                and abs(readout_group["lr"] - expected_lr) < 1e-15
                and abs(
                    readout_group["weight_decay"]
                    - profile["readout_weight_decay"]
                )
                < 1e-15
            )
            x = torch.randn(1, source["common"]["seq_len"], 7)
            y = torch.zeros(1, 720, 7)
            with torch.inference_mode():
                full = model(x, y, is_training=False, target_prefix=720)[0]
                prefix = model(x, y, is_training=False, target_prefix=96)[0]
            forward_pass = bool(
                full.shape == (1, 720, 7)
                and torch.equal(prefix, full[:, :96])
                and torch.isfinite(full).all()
            )
            add(
                rows,
                "model",
                profile["id"],
                contract_pass and forward_pass,
                f"rank={rank},params={sum(p.numel() for p in readout.parameters())}",
                "effective_contract_and_exact_prefix",
            )

    runner = "scripts/remote/run_iscf_bsca_decoder_transfer_itransformer_hpo_v2.sh"
    syntax = subprocess.run(["bash", "-n", runner], cwd=ROOT, check=False)
    dry = subprocess.run(
        ["bash", runner],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    add(
        rows,
        "execution",
        "runner_syntax",
        syntax.returncode == 0,
        syntax.returncode,
        0,
    )
    add(
        rows,
        "execution",
        "runner_dry_run",
        dry.returncode == 0 and "jobs=70" in dry.stdout,
        dry.stdout.strip().splitlines()[-1] if dry.stdout.strip() else dry.stderr,
        "jobs=70",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "prelaunch_checks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    passed = all(row["pass"] for row in rows)
    (args.output_dir / "prelaunch_summary.json").write_text(
        json.dumps(
            {
                "pass": passed,
                "checks": len(rows),
                "protocol_sha256": sha256(args.config),
                "training_jobs": 70,
                "formal_test_jobs_before_manifest": 0,
            },
            indent=2,
        )
        + "\n"
    )
    if not passed:
        raise SystemExit("iTransformer decoder-HPO v2 prelaunch failed")
    print(f"itransformer_decoder_hpo_v2_prelaunch=pass checks={len(rows)}")


if __name__ == "__main__":
    main()
