#!/usr/bin/env python3
"""Run the SIFF/MCCA Step 7B prelaunch contract gate."""

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
        default=Path("configs/stage_c_post_pcc_step7b.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/stage_c_post_pcc_step7b_prelaunch_20260717"),
    )
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def manifest_rows(
    config: dict[str, Any], profiles: dict[str, Any]
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
                    "seed": config["seed"],
                    "evaluation_split": "val",
                    "test_used": False,
                }
            )
    return rows


def training_argv(row: dict[str, Any], config: dict[str, Any]) -> list[str]:
    training = config["training"]
    return [
        "train_repo.py",
        "--dataset-root", "/home/yingch/dataset",
        "--dataset", row["dataset"],
        "--mode", "unified",
        "--seq-len", "720",
        "--pred-len", "720",
        "--target-horizons", "720",
        "--validation-horizons", "720",
        "--evaluation-horizons", "1,2,3,720",
        "--segment-horizons", "48,96,144,192,288,336,512,720",
        "--evaluation-prefix-mode", "full-crop",
        "--e-layers", "2",
        "--batch-size", str(training["batch_size"]),
        "--epochs", str(training["epochs"]),
        "--patience", str(training["patience"]),
        "--enable-early-stopping",
        "--seed", str(config["seed"]),
        "--run-name", f"POST_PCC_STEP7B_{row['arm']}",
        "--output-dir", f"/tmp/{row['arm']}_{row['dataset']}",
        "--device", "cuda",
        "--checkpoint-policy", "best-val",
        "--protocol-class", "method_screening",
        "--protocol-profile", "stage_c_post_pcc_step7b_validation_screen_v1",
        "--profile-hash", config["profiles"]["sha256"],
        "--legacy-patch-num", str(row["patch_num"]),
        "--legacy-d-model", str(row["d_model"]),
        "--legacy-d-ff", str(row["d_ff"]),
        "--legacy-dropout", "0.1",
        "--legacy-layer-norm", "1",
        "--learning-rate", str(training["learning_rate"]),
        "--readout-mode", row["readout_mode"],
        "--pcsd-mode-rank", str(row["mode_rank"]),
        "--pcsd-policy-mode", "direct",
        "--pcsd-partition", "canonical",
        "--pcc-objective-mode", row["objective_mode"],
        "--pred-loss-mode", "full",
        "--final-evaluation-split", "val",
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
    results = []
    representative = {
        (row["dataset"], row["readout_mode"], row["mode_rank"]): row
        for row in rows
    }
    for row in representative.values():
        torch.manual_seed(2021)
        model = TimeAlign.Model(model_config(row))
        contract = initialization_contract(model)
        decoder_hash = contract.get(
            "pcsd_initialization_hash",
            contract.get("pcsd_dense_initialization_hash", ""),
        )
        results.append(
            {
                "dataset": row["dataset"],
                "readout_mode": row["readout_mode"],
                "mode_rank": row["mode_rank"],
                "encoder_hash": contract["encoder_initialization_hash"],
                "decoder_hash": decoder_hash,
                "pass": bool(decoder_hash),
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_rows(config, profiles)
    construction = construction_audit(manifest)
    write_csv(args.output_dir / "jobs_seed2021.csv", manifest)
    write_csv(args.output_dir / "model_construction.csv", construction)

    frozen_hashes = all(
        file_hash(Path(config[key]["path"])) == config[key]["sha256"]
        for key in ("profiles", "step6_design", "step7a_gate")
    )
    step7a = json.loads(
        Path(config["step7a_gate"]["path"]).read_text(encoding="utf-8")
    )
    source_paths = (
        Path("scripts/remote/run_stage_c_post_pcc_step7b.sh"),
        Path("scripts/evaluate_stage_c_pcsd_cf_checkpoint.py"),
        Path("scripts/evaluate_stage_c_sc2_pcc_gradient.py"),
        Path("scripts/analyze_stage_c_post_pcc_step7b.py"),
    )
    categories = {
        "frozen_hashes": frozen_hashes,
        "step7a_gate": bool(
            step7a.get("all_pass") is True
            and step7a.get("cases_passed") == config["step7a_gate"]["cases"]
        ),
        "new_run_matrix": bool(
            len(manifest) == config["expected_runs"] == 55
            and len({row["arm"] for row in manifest}) == 11
            and all(row["test_used"] is False for row in manifest)
        ),
        "dataset_major_workload_order": [
            row["dataset"] for row in manifest[: len(config["arms"])]
        ]
        == [config["datasets"][0]] * len(config["arms"]),
        "cli_contracts": cli_audit(manifest, config),
        "model_construction": all(row["pass"] for row in construction),
        "validation_only_authorization": bool(
            config["training"]["final_evaluation_split"] == "val"
            and config["training"]["test_used"] is False
            and config["authorization"]["remote_training_authorized"] is True
            and config["authorization"]["test_access_authorized"] is False
            and config["authorization"]["confirmation_seeds_authorized"] is False
            and config["authorization"]["conditional_phase_b_authorized"] is False
        ),
        "tooling_present": all(path.is_file() for path in source_paths),
    }
    result = {
        "candidate": config["candidate"],
        "current_step": "Step7B prelaunch gate",
        "expected_jobs": config["expected_runs"],
        "config_sha256": file_hash(args.config),
        "profile_sha256": file_hash(profile_path),
        "categories": categories,
        "overall_pass": all(categories.values()),
        "test_used": False,
        "decision": (
            "step7b_prelaunch_pass_remote_seed2021_authorized"
            if all(categories.values())
            else "step7b_prelaunch_fail_hold_remote"
        ),
    }
    (args.output_dir / "prelaunch_gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    if not result["overall_pass"]:
        raise RuntimeError(f"Step7B prelaunch failed: {categories}")


if __name__ == "__main__":
    main()
