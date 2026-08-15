#!/usr/bin/env python3
"""Audit iTransformer-style transfer training artifacts without test access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


HORIZONS = [96, 192, 336, 720]
REQUIRED_FILES = [
    "checkpoint.pt",
    "effective_config.json",
    "environment.json",
    "initialization_contract.json",
    "metrics_by_target_horizon.csv",
    "model_diagnostics.json",
    "training_log.csv",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def finite_metrics(rows: list[dict[str, str]]) -> bool:
    try:
        values = [float(row[key]) for row in rows for key in ("mse", "mae")]
    except (KeyError, ValueError):
        return False
    return len(rows) == len(HORIZONS) and all(math.isfinite(value) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/iscf_bsca_decoder_transfer_itransformer_v1.json"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    config = load_json(args.config)
    profile_path = Path(config["profiles"]["path"])
    profiles = load_json(profile_path)
    profile_hash = sha256(profile_path)
    arm_by_id = {arm["id"]: arm for arm in config["arms"]}

    scorecard: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    encoder_hashes: dict[str, dict[str, str]] = {
        dataset: {} for dataset in config["datasets"]
    }
    for dataset in config["datasets"]:
        dataset_profile = profiles["dataset_profiles"][dataset]
        common = profiles["common"]
        for arm_id, arm in arm_by_id.items():
            run_dir = args.output_root / arm_id / dataset / "seed2021"
            missing = [
                name for name in REQUIRED_FILES if not (run_dir / name).is_file()
            ]
            checkpoint_hash = ""
            encoder_hash = ""
            best_epoch = -1
            best_validation_mse = float("nan")
            metric_mean_mse = float("nan")
            metrics_ok = False
            selector_ok = False
            effective_ok = False
            diagnostics_ok = False
            if not missing:
                effective = load_json(run_dir / "effective_config.json")["adapter"]
                expected_rank = (
                    dataset_profile["mode_rank"]
                    if arm["readout_mode"] == "siff-independent-scope-control"
                    else 256
                )
                effective_ok = (
                    effective["dataset"] == dataset
                    and effective["seed"] == config["primary_seed"]
                    and effective["mode"] == "unified"
                    and effective["seq_len"] == common["seq_len"]
                    and effective["pred_len"] == 720
                    and effective["target_horizons"] == [720]
                    and effective["validation_horizons"] == HORIZONS
                    and effective["evaluation_horizons"] == HORIZONS
                    and effective["final_evaluation_split"] == "val"
                    and effective["checkpoint_policy"] == "best-val"
                    and effective["encoder_mode"] == "itransformer-variate-attention"
                    and effective["readout_mode"] == arm["readout_mode"]
                    and effective["pcsd_policy_mode"] == arm["policy_mode"]
                    and effective["pcc_objective_mode"] == arm["objective_mode"]
                    and effective["pcsd_mode_rank"] == expected_rank
                    and effective["history_d_model"] == dataset_profile["d_model"]
                    and effective["history_d_ff"] == dataset_profile["d_ff"]
                    and effective["history_e_layers"] == dataset_profile["e_layers"]
                    and effective["history_n_heads"] == common["n_heads"]
                    and effective["batch_size"] == common["batch_size"]
                    and effective["epochs"] == common["max_epochs"]
                    and effective["patience"] == common["patience"]
                    and effective["protocol_class"] == "method_screening"
                    and effective["protocol_profile"]
                    == "iscf_bsca_decoder_transfer_itransformer_v1_20260815"
                    and effective["profile_hash"] == profile_hash
                    and effective["pred_loss_mode"] == "full"
                    and effective["save_predictions"] is False
                )

                metric_rows = read_rows(run_dir / "metrics_by_target_horizon.csv")
                metrics_ok = (
                    finite_metrics(metric_rows)
                    and [int(row["target_horizon"]) for row in metric_rows] == HORIZONS
                    and all(row["dataset"] == dataset for row in metric_rows)
                    and all(row["evaluation_split"] == "val" for row in metric_rows)
                    and all(
                        row["checkpoint_policy"] == "best-val"
                        for row in metric_rows
                    )
                    and all(row["profile_hash"] == profile_hash for row in metric_rows)
                )
                if metrics_ok:
                    metric_mean_mse = sum(float(row["mse"]) for row in metric_rows) / 4

                training_rows = read_rows(run_dir / "training_log.csv")
                try:
                    validation_values = [
                        float(row["val_mean_mse"]) for row in training_rows
                    ]
                    training_finite = validation_values and all(
                        math.isfinite(value) for value in validation_values
                    )
                except (KeyError, ValueError):
                    training_finite = False
                    validation_values = []
                if training_finite:
                    best_index = min(
                        range(len(validation_values)),
                        key=validation_values.__getitem__,
                    )
                    best_epoch = int(training_rows[best_index]["epoch"])
                    best_validation_mse = validation_values[best_index]
                    selector_ok = (
                        metrics_ok
                        and abs(metric_mean_mse - best_validation_mse) <= 1e-9
                    )

                initialization = load_json(run_dir / "initialization_contract.json")
                encoder_hash = initialization.get("encoder_initialization_hash", "")
                encoder_hashes[dataset][arm_id] = encoder_hash
                diagnostics = load_json(run_dir / "model_diagnostics.json")
                diagnostics_ok = (
                    diagnostics.get("encoder_mode")
                    == "itransformer-variate-attention"
                    and diagnostics.get("readout_mode") == arm["readout_mode"]
                    and diagnostics.get("frozen_parameter_tensors") == 0
                    and diagnostics.get("trainable_parameters")
                    == diagnostics.get("total_parameters")
                )
                checkpoint_hash = sha256(run_dir / "checkpoint.pt")

            passed = (
                not missing
                and effective_ok
                and metrics_ok
                and selector_ok
                and diagnostics_ok
                and bool(encoder_hash)
                and bool(checkpoint_hash)
            )
            scorecard.append(
                {
                    "dataset": dataset,
                    "arm": arm_id,
                    "run_dir": str(run_dir),
                    "best_epoch": best_epoch,
                    "best_validation_mean_mse": best_validation_mse,
                    "artifact_validation_mean_mse": metric_mean_mse,
                    "checkpoint_sha256": checkpoint_hash,
                    "encoder_initialization_hash": encoder_hash,
                    "missing_files": ";".join(missing),
                    "effective_config_ok": effective_ok,
                    "four_h_metrics_ok": metrics_ok,
                    "validation_selector_ok": selector_ok,
                    "diagnostics_ok": diagnostics_ok,
                    "pass": passed,
                }
            )
            if passed:
                manifest.append(
                    {
                        "dataset": dataset,
                        "arm": arm_id,
                        "seed": config["primary_seed"],
                        "checkpoint": str(run_dir / "checkpoint.pt"),
                        "checkpoint_sha256": checkpoint_hash,
                        "encoder_initialization_hash": encoder_hash,
                        "checkpoint_selector": "mean_validation_mse_h96_h192_h336_h720",
                        "best_epoch": best_epoch,
                        "validation_mean_mse": best_validation_mse,
                        "test_role": config["test_role"],
                        "formal_test_accessed": False,
                    }
                )

    pairing_rows: list[dict[str, Any]] = []
    for dataset, hashes in encoder_hashes.items():
        unique = {value for value in hashes.values() if value}
        pairing_rows.append(
            {
                "dataset": dataset,
                **{f"{arm}_encoder_hash": hashes.get(arm, "") for arm in arm_by_id},
                "unique_encoder_initialization_hashes": len(unique),
                "matched_triplet": len(hashes) == 3 and len(unique) == 1,
            }
        )

    args.audit_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.audit_dir / "artifact_scorecard.csv", scorecard)
    write_csv(args.audit_dir / "matched_initialization_triplets.csv", pairing_rows)
    completed = sum(bool(row["pass"]) for row in scorecard)
    unique_checkpoints = len(
        {row["checkpoint_sha256"] for row in scorecard if row["pass"]}
    )
    matched_triplets = sum(bool(row["matched_triplet"]) for row in pairing_rows)
    forbidden_test_artifacts = sorted(
        str(path.relative_to(args.output_root))
        for pattern in ("*test_audit*", "predictions_test.npz", "*formal_test*")
        for path in args.output_root.rglob(pattern)
    )
    summary = {
        "pass": (
            completed == 15
            and unique_checkpoints == 15
            and matched_triplets == 5
            and not forbidden_test_artifacts
        ),
        "complete_training_runs": completed,
        "expected_training_runs": 15,
        "unique_checkpoint_hashes": unique_checkpoints,
        "matched_encoder_initialization_triplets": matched_triplets,
        "forbidden_test_artifacts": forbidden_test_artifacts,
        "formal_test_jobs": 0,
        "profile_sha256": profile_hash,
    }
    (args.audit_dir / "artifact_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    if not summary["pass"]:
        raise SystemExit(
            "artifact gate failed: "
            f"complete={completed}/15 hashes={unique_checkpoints}/15 "
            f"triplets={matched_triplets}/5 "
            f"forbidden_test={len(forbidden_test_artifacts)}"
        )
    write_csv(args.manifest, manifest)
    print("itransformer_transfer_artifacts=pass runs=15 hashes=15 triplets=5 test=0")


if __name__ == "__main__":
    main()
