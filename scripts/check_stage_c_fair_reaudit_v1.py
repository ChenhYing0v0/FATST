#!/usr/bin/env python3
"""Validate the frozen StageC fair paper-facing re-audit matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from models import TimeAlign  # noqa: E402
import train_repo as training_adapter  # noqa: E402
from train_repo import initialization_contract  # noqa: E402


CHANNELS = {"ETTh1": 7, "ETTh2": 7, "ETTm1": 7, "ETTm2": 7, "Weather": 21}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_fair_reaudit_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/stage_c_fair_reaudit_v1_20260717/prelaunch"),
    )
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def manifest_rows(
    config: dict[str, Any],
    profiles: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for dataset in config["datasets"]:
        profile = profiles[dataset]
        for arm in config["arms"]:
            rule = arm["rank_rule"]
            rank = (
                256
                if rule == "fixed_256"
                else config["matched_ranks"][dataset][rule]
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
        f"FAIR_REAUDIT_{row['arm']}",
        "--output-dir",
        f"/tmp/{row['arm']}_{row['dataset']}",
        "--device",
        "cuda",
        "--checkpoint-policy",
        "best-val",
        "--protocol-class",
        "method_screening",
        "--protocol-profile",
        "stage_c_fair_reaudit_v1",
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
        "--pcsd-mode-rank",
        str(row["mode_rank"]),
        "--pcsd-policy-mode",
        "direct",
        "--pcsd-partition",
        "canonical",
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
            ):
                return False
    finally:
        sys.argv = original
    return True


def model_config(row: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        encoder_mode="timealign-token-mlp",
        readout_mode=row["readout_mode"],
        e_layers=2,
        patch_num=int(row["patch_num"]),
        d_model=int(row["d_model"]),
        d_ff=int(row["d_ff"]),
        dropout=0.1,
        pos=1,
        layer_norm=1,
        enc_in=CHANNELS[row["dataset"]],
        basis_rank=256,
        pcsd_coordinate_dim=4,
        pcsd_mode_rank=int(row["mode_rank"]),
        pcsd_policy_history_dim=32,
        pcsd_policy_hidden_dim=64,
        pcsd_policy_mode="direct",
        pcsd_fixed_scale=720,
        pcsd_partition="canonical",
        pcsd_partition_seed=15101,
        pcsd_group_chunk_size=64,
        pcsd_target_chunk_size=128,
    )


def construction_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    representatives = {
        (row["dataset"], row["readout_mode"], row["mode_rank"]): row
        for row in rows
    }
    results = []
    for row in representatives.values():
        torch.manual_seed(int(row["seed"]))
        model = TimeAlign.Model(model_config(row)).float().eval()
        contract = initialization_contract(model)
        x = torch.randn(1, 720, CHANNELS[row["dataset"]])
        y = torch.zeros(1, 720, CHANNELS[row["dataset"]])
        with torch.no_grad():
            full = model(x, y, is_training=False, target_prefix=720)[0]
            prefix = model(x, y, is_training=False, target_prefix=96)[0]
        prefix_gap = float((prefix - full[:, :96]).abs().max())
        results.append(
            {
                "dataset": row["dataset"],
                "readout_mode": row["readout_mode"],
                "mode_rank": row["mode_rank"],
                "encoder_hash": contract["encoder_initialization_hash"],
                "prefix_gap": prefix_gap,
                "pass": prefix_gap <= 2e-5,
            }
        )
        del model
    return results


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    profile_path = Path(config["profiles"]["path"])
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))[
        "dataset_profiles"
    ]
    manifest = manifest_rows(config, profiles)
    construction = construction_audit(manifest)
    arm_ids = {arm["id"] for arm in config["arms"]}
    comparison_ids_valid = all(
        entry["candidate"] in arm_ids and entry["reference"] in arm_ids
        for entry in config["comparisons"]
    )
    encoder_hashes: dict[str, set[str]] = {}
    for row in construction:
        encoder_hashes.setdefault(row["dataset"], set()).add(row["encoder_hash"])
    categories = {
        "profile_hash": file_hash(profile_path) == config["profiles"]["sha256"],
        "matrix_size": len(manifest) == config["matrix"]["expected_runs"] == 70,
        "unique_arms": len(arm_ids) == len(config["arms"]) == 14,
        "comparison_references": comparison_ids_valid,
        "checkpoint_rule": config["training"]["validation_horizons"]
        == [96, 192, 336, 720]
        and config["training"]["validation_checkpoint_score"]
        == "mean_mse_h96_h192_h336_h720",
        "test_authorization": config["authorization"]["user_authorized"] is True
        and config["authorization"]["test_role"]
        == "primary-mechanism-effectiveness-and-paper-benchmark"
        and config["authorization"]["per_dataset_horizon_or_cell_tuning_allowed"]
        is False,
        "cli_contracts": cli_audit(manifest, config),
        "model_construction": all(row["pass"] for row in construction),
        "paired_encoder_initialization": all(
            len(values) == 1 for values in encoder_hashes.values()
        ),
    }
    overall_pass = all(categories.values())
    report = {
        "audit_id": config["audit_id"],
        "candidate_version": config["candidate_version"],
        "categories": categories,
        "jobs": len(manifest),
        "construction_cases": len(construction),
        "overall_pass": overall_pass,
        "remote_authorized": overall_pass,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "jobs_seed2021.csv", manifest)
    write_csv(args.output_dir / "model_construction.csv", construction)
    (args.output_dir / "local_gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not overall_pass:
        raise RuntimeError(json.dumps(report, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
