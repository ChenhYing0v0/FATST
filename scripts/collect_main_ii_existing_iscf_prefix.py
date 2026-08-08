#!/usr/bin/env python3
"""Collect the seven completed ISCF full-crop audits for Main II."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DATASET_CHANNELS = {
    "ETTh1": 7,
    "ETTh2": 7,
    "ETTm1": 7,
    "ETTm2": 7,
    "Weather": 21,
    "ECL": 321,
    "Solar": 137,
}
HORIZONS = (96, 192, 336, 720)


def sha256(path: Path) -> str:
    """Return the SHA256 of one immutable artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV artifact as dictionaries."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = read_csv(args.selected_manifest)
    selected = [row for row in manifest_rows if row["dataset"] in DATASET_CHANNELS]
    if len(selected) != 7 or {row["dataset"] for row in selected} != set(
        DATASET_CHANNELS
    ):
        raise RuntimeError("selected ISCF manifest is not the frozen seven-dataset set")

    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for manifest_row in selected:
        dataset = manifest_row["dataset"]
        training_dir = Path(manifest_row["training_artifact_dir"])
        test_dir = Path(manifest_row["test_artifact_dir"])
        checkpoint = training_dir / "checkpoint.pt"
        metrics_path = test_dir / "test_audit_metrics_by_target_horizon.csv"
        invariants_path = test_dir / "test_audit_invariants.json"
        diagnostics_path = test_dir / "pcsd_test_audit_diagnostics.npz"
        for path in (checkpoint, metrics_path, invariants_path, diagnostics_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        checkpoint_sha = sha256(checkpoint)
        expected_sha = manifest_row["checkpoint_sha256_before_test"]
        if checkpoint_sha != expected_sha:
            raise RuntimeError(f"checkpoint hash mismatch: {dataset}")
        if expected_sha != manifest_row["checkpoint_sha256_after_test"]:
            raise RuntimeError(f"checkpoint mutation recorded: {dataset}")

        invariants = json.loads(invariants_path.read_text(encoding="utf-8"))
        if not invariants.get("pass") or invariants.get("full_prefix_max_abs") != 0.0:
            raise RuntimeError(f"full-crop invariant failed: {dataset}")
        invariant_horizons = {
            int(row["horizon"])
            for row in invariants.get("prefix_rows", [])
            if int(row["horizon"]) in HORIZONS
        }
        if invariant_horizons != set(HORIZONS):
            raise RuntimeError(f"missing prefix invariant: {dataset}")

        metric_map = {
            int(row["target_horizon"]): row for row in read_csv(metrics_path)
        }
        if set(metric_map) != set(range(1, 721)):
            raise RuntimeError(f"dense H1-H720 metric surface incomplete: {dataset}")
        channels = DATASET_CHANNELS[dataset]
        dataset_rows: list[dict[str, object]] = []
        for horizon in HORIZONS:
            source = metric_map[horizon]
            row_channels = int(source["num_rows_channels"])
            if row_channels % channels:
                raise RuntimeError(f"row/channel count mismatch: {dataset}")
            dataset_rows.append(
                {
                    "system": "ISCF-BSCA-MAIN-v1",
                    "dataset": dataset,
                    "repeat": 0,
                    "horizon": horizon,
                    "mse": float(source["mse"]),
                    "mae": float(source["mae"]),
                    "origin_count": row_channels // channels,
                    "channel_count": channels,
                    "checkpoint_sha256": checkpoint_sha,
                    "prefix_identity": True,
                    "prediction_prefix_sha256": "not_retained_prior_completed_audit",
                    "target_prefix_sha256": "not_retained_prior_completed_audit",
                    "test_role": "reused_completed_formal_H720_full_crop_dense_prefix",
                    "test_tuned": True,
                    "test_access_date": manifest_row["test_access_date"],
                    "matrix_complete": False,
                }
            )
        mean_mse = sum(float(row["mse"]) for row in dataset_rows) / 4
        mean_mae = sum(float(row["mae"]) for row in dataset_rows) / 4
        if abs(mean_mse - float(manifest_row["test_mean_mse_4h"])) > 1e-12:
            raise RuntimeError(f"four-H MSE mismatch: {dataset}")
        if abs(mean_mae - float(manifest_row["test_mean_mae_4h"])) > 1e-12:
            raise RuntimeError(f"four-H MAE mismatch: {dataset}")
        rows.extend(dataset_rows)
        audits.append(
            {
                "dataset": dataset,
                "trial_id": manifest_row["trial_id"],
                "profile_id": manifest_row["profile_id"],
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "metrics_path": str(metrics_path),
                "metrics_sha256": sha256(metrics_path),
                "invariants_path": str(invariants_path),
                "invariants_sha256": sha256(invariants_path),
                "diagnostics_path": str(diagnostics_path),
                "diagnostics_sha256": sha256(diagnostics_path),
                "dense_metric_rows": len(metric_map),
                "full_prefix_max_abs": invariants["full_prefix_max_abs"],
                "checkpoint_immutable": True,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "prefix_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "gate": "pass",
        "system": "ISCF-BSCA-MAIN-v1",
        "checkpoint_evaluations": 7,
        "raw_prefix_rows": 28,
        "new_test_access": False,
        "reuse_basis": "completed_H720_full_crop_dense_prefix_formal_audit",
        "prior_array_retention": "not_retained",
        "artifact_hash_substitute": True,
        "audits": audits,
        "rows": rows,
    }
    (args.output_dir / "prefix_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("main_ii_iscf_reuse=pass checkpoints=7 raw_rows=28 new_test_access=false")


if __name__ == "__main__":
    main()
