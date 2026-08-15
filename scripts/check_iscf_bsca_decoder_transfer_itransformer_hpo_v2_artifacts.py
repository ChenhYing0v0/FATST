#!/usr/bin/env python3
"""Audit iTransformer decoder-HPO v2 training artifacts without test access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/iscf_bsca_decoder_transfer_itransformer_hpo_v2.json"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    required = config["artifact_contract"]["required_training_files"]
    reference_ranks = config["reference_profile"]["mode_rank_by_dataset"]
    scorecard: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for dataset in config["datasets"]:
        for profile in config["search_profiles"]:
            run_dir = args.output_root / dataset / profile["id"] / "seed2021"
            missing = [name for name in required if not (run_dir / name).is_file()]
            expected_rank = max(
                1, round(reference_ranks[dataset] * profile["rank_scale"])
            )
            best_epoch = -1
            validation_mean_mse = float("nan")
            metric_mean_mse = float("nan")
            checkpoint_hash = ""
            encoder_hash = ""
            effective_ok = False
            selector_ok = False
            if not missing:
                effective = json.loads((run_dir / "effective_config.json").read_text())
                adapter = effective["adapter"]
                effective_ok = bool(
                    adapter["dataset"] == dataset
                    and adapter["hpo_profile_id"] == profile["id"]
                    and adapter["protocol_profile"]
                    == "iscf_bsca_decoder_transfer_itransformer_hpo_v2_20260816"
                    and adapter["final_evaluation_split"] == "val"
                    and adapter["official_test_mode"] is False
                    and adapter["pcsd_mode_rank"] == expected_rank
                    and adapter["pcsd_coordinate_dim"]
                    == profile["coordinate_dim"]
                    and adapter["pcsd_policy_history_dim"]
                    == profile["policy_history_dim"]
                    and adapter["pcsd_policy_hidden_dim"]
                    == profile["policy_hidden_dim"]
                    and adapter["pcsd_scales"] == profile["scales"]
                    and abs(
                        adapter["readout_learning_rate_multiplier"]
                        - profile["readout_learning_rate_multiplier"]
                    )
                    < 1e-15
                    and abs(
                        adapter["readout_weight_decay"]
                        - profile["readout_weight_decay"]
                    )
                    < 1e-15
                    and adapter["epochs"] == config["training"]["max_epochs"]
                    and adapter["patience"] == config["training"]["patience"]
                )
                training_rows = read_rows(run_dir / "training_log.csv")
                validation_values = [
                    float(row["val_mean_mse"]) for row in training_rows
                ]
                metric_rows = read_rows(run_dir / "metrics_by_target_horizon.csv")
                metric_values = [
                    float(row[key])
                    for row in metric_rows
                    for key in ("mse", "mae")
                ]
                if (
                    validation_values
                    and all(math.isfinite(value) for value in validation_values)
                    and len(metric_rows) == 4
                    and [int(row["target_horizon"]) for row in metric_rows]
                    == [96, 192, 336, 720]
                    and all(math.isfinite(value) for value in metric_values)
                ):
                    index = min(
                        range(len(validation_values)),
                        key=validation_values.__getitem__,
                    )
                    best_epoch = int(training_rows[index]["epoch"])
                    validation_mean_mse = validation_values[index]
                    metric_mean_mse = sum(
                        float(row["mse"]) for row in metric_rows
                    ) / 4
                    selector_ok = abs(validation_mean_mse - metric_mean_mse) <= 1e-6
                checkpoint_hash = sha256(run_dir / "checkpoint.pt")
                initialization = json.loads(
                    (run_dir / "initialization_contract.json").read_text()
                )
                encoder_hash = initialization["encoder_initialization_hash"]

            passed = bool(
                not missing
                and effective_ok
                and selector_ok
                and math.isfinite(validation_mean_mse)
            )
            row = {
                "dataset": dataset,
                "profile_id": profile["id"],
                "run_dir": str(run_dir),
                "mode_rank": expected_rank,
                "coordinate_dim": profile["coordinate_dim"],
                "policy_history_dim": profile["policy_history_dim"],
                "policy_hidden_dim": profile["policy_hidden_dim"],
                "scales": ",".join(map(str, profile["scales"])),
                "readout_lr_multiplier": profile[
                    "readout_learning_rate_multiplier"
                ],
                "readout_weight_decay": profile["readout_weight_decay"],
                "best_epoch": best_epoch,
                "validation_mean_mse": validation_mean_mse,
                "artifact_validation_mean_mse": metric_mean_mse,
                "checkpoint_sha256": checkpoint_hash,
                "encoder_initialization_hash": encoder_hash,
                "missing_files": ";".join(missing),
                "effective_config_ok": effective_ok,
                "validation_selector_ok": selector_ok,
                "pass": passed,
            }
            scorecard.append(row)
            if passed:
                manifest.append(
                    {
                        "dataset": dataset,
                        "profile_id": profile["id"],
                        "checkpoint": str(run_dir / "checkpoint.pt"),
                        "checkpoint_sha256": checkpoint_hash,
                        "checkpoint_selector": (
                            "mean_validation_mse_h96_h192_h336_h720"
                        ),
                        "seed": 2021,
                    }
                )

    args.audit_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.audit_dir / "training_artifact_scorecard.csv", scorecard)
    complete = sum(bool(row["pass"]) for row in scorecard)
    unique_hashes = len(
        {row["checkpoint_sha256"] for row in scorecard if row["pass"]}
    )
    matched_encoder_datasets = 0
    for dataset in config["datasets"]:
        hashes = {
            row["encoder_initialization_hash"]
            for row in scorecard
            if row["dataset"] == dataset and row["pass"]
        }
        matched_encoder_datasets += len(hashes) == 1
    passed = bool(
        complete == 70
        and unique_hashes == 70
        and matched_encoder_datasets == 5
    )
    summary = {
        "pass": passed,
        "complete_runs": complete,
        "expected_runs": 70,
        "unique_checkpoint_hashes": unique_hashes,
        "expected_unique_checkpoint_hashes": 70,
        "datasets_with_matched_encoder_initialization": matched_encoder_datasets,
        "formal_test_jobs": 0,
    }
    (args.audit_dir / "training_artifact_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    if args.manifest is not None and passed:
        write_rows(args.manifest, manifest)
    if not passed:
        raise SystemExit(
            "artifact gate failed: "
            f"complete={complete}/70 hashes={unique_hashes}/70 "
            f"encoder_sets={matched_encoder_datasets}/5"
        )
    print(
        "itransformer_decoder_hpo_v2_artifacts=pass "
        "runs=70 hashes=70 encoder_sets=5 test=0"
    )


if __name__ == "__main__":
    main()
