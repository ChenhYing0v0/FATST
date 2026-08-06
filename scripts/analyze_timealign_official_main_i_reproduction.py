#!/usr/bin/env python3
"""Audit the combined 32-run TimeAlign Main I reproduction matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (96, 192, 336, 720)
FAILURE_PATTERN = re.compile(
    r"Traceback|CUDA out of memory|(^|[^A-Za-z0-9_])(nan|inf)([^A-Za-z0-9_]|$)",
    re.IGNORECASE | re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "timealign_official_main_i_reproduction.json",
    )
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--reuse-root", type=Path, required=True)
    parser.add_argument("--checkpoint-hashes", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_hashes(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, checkpoint = line.split(maxsplit=1)
        run_id = Path(checkpoint).parent.name
        if len(digest) != 64 or run_id in hashes:
            raise ValueError(f"invalid checkpoint hash row: {line}")
        hashes[run_id] = digest
    return hashes


def expanded_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    reuse = set(config["reuse_contract"]["run_ids"])
    for dataset in config["matrix"]["datasets"]:
        for horizon in HORIZONS:
            raw = config["profile_contract"][dataset]
            profile = raw.get(str(horizon), raw)
            run_id = f"TimeAlign__{dataset}__H{horizon}__seed2021"
            jobs.append(
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "horizon": horizon,
                    "seed": 2021,
                    "epochs": 1 if dataset == "ETTh1" and horizon == 96 else 10,
                    "batch_size": 16 if dataset == "ECL" else 32,
                    "artifact_role": "reusable" if run_id in reuse else "new",
                    **profile,
                }
            )
    return jobs


def published_timealign() -> dict[tuple[str, int], tuple[float, float]]:
    path = (
        ROOT
        / "analysis"
        / "iscf_bsca_paper_experiment_consolidation_20260731"
        / "timealign_table6_main_i_published.csv"
    )
    result = {}
    for row in read_csv(path):
        if row["model"] == "TimeAlign":
            result[(row["dataset"], int(row["horizon"]))] = (
                float(row["mse"]),
                float(row["mae"]),
            )
    return result


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    jobs = expanded_jobs(config)
    hashes = load_hashes(args.checkpoint_hashes)
    if len(jobs) != 32 or set(hashes) != {job["run_id"] for job in jobs}:
        raise ValueError("checkpoint hash list does not match the 32-run matrix")
    if len(set(hashes.values())) != 32:
        raise ValueError("checkpoint hashes are not unique")

    source = config["source_contract"]
    for name in ("adapter", "executed_model", "data_loader", "metric"):
        if sha256(ROOT / source[f"{name}_path"]) != source[f"{name}_sha256"]:
            raise ValueError(f"source hash mismatch: {name}")
    for dataset, item in source["dataset_scripts"].items():
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"dataset script hash mismatch: {dataset}")

    published = published_timealign()
    prior_manifest = {
        row["run_id"]: row
        for row in read_csv(ROOT / config["reuse_contract"]["prior_artifact_manifest_path"])
    }
    if set(prior_manifest) != set(config["reuse_contract"]["run_ids"]):
        raise ValueError("prior reusable artifact manifest is incomplete")
    metric_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for job in jobs:
        root = (
            args.reuse_root
            if job["artifact_role"] == "reusable"
            else args.new_root
        )
        directory = root / job["run_id"]
        if (
            job["artifact_role"] == "reusable"
            and prior_manifest[job["run_id"]]["checkpoint_sha256"]
            != hashes[job["run_id"]]
        ):
            raise ValueError(f"reusable checkpoint hash mismatch: {job['run_id']}")
        required = [
            filename
            for filename in config["artifact_gate"]["required_all_runs"]
            if filename != "checkpoint.pt"
        ]
        for filename in required:
            path = directory / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
        prediction = directory / "predictions_test.npz"
        if job["artifact_role"] == "new" and prediction.exists():
            raise ValueError(f"unexpected quota-heavy prediction artifact: {prediction}")
        log_text = (directory / "run.log").read_text(encoding="utf-8")
        if FAILURE_PATTERN.search(log_text):
            raise ValueError(f"failure pattern in {job['run_id']}")

        effective = json.loads(
            (directory / "effective_config.json").read_text(encoding="utf-8")
        )
        adapter = effective["adapter"]
        preset = effective["official_preset"]
        expected_adapter = {
            "dataset": job["dataset"],
            "mode": "fixed",
            "seq_len": 720,
            "label_len": 48,
            "pred_len": job["horizon"],
            "target_horizons": [job["horizon"]],
            "validation_horizons": [job["horizon"]],
            "evaluation_horizons": [job["horizon"]],
            "encoder_mode": "timealign-token-mlp",
            "readout_mode": "official",
            "pred_loss_mode": "full",
            "checkpoint_policy": "official-last",
            "final_evaluation_split": "test",
            "official_test_mode": True,
            "protocol_class": "native_external",
            "seed": 2021,
            "epochs": job["epochs"],
            "batch_size": job["batch_size"],
            "enable_early_stopping": False,
        }
        for field, value in expected_adapter.items():
            if adapter[field] != value:
                raise ValueError((job["run_id"], field, adapter[field], value))
        for field in (
            "d_model",
            "d_ff",
            "dropout",
            "learning_rate",
            "w_align",
            "patch_num",
            "layer_norm",
        ):
            if preset[field] != job[field]:
                raise ValueError((job["run_id"], field, preset[field], job[field]))

        training_rows = read_csv(directory / "training_log.csv")
        if len(training_rows) != job["epochs"]:
            raise ValueError(f"epoch count mismatch: {job['run_id']}")
        if not all(int(row["early_stopping_enabled"]) == 0 for row in training_rows):
            raise ValueError(f"early stopping enabled: {job['run_id']}")
        if not all(math.isfinite(float(row["train_loss"])) for row in training_rows):
            raise ValueError(f"non-finite train loss: {job['run_id']}")

        metrics = read_csv(directory / "metrics_by_target_horizon.csv")
        if len(metrics) != 1:
            raise ValueError(f"expected one fixed-H metric row: {job['run_id']}")
        row = metrics[0]
        mse, mae = float(row["mse"]), float(row["mae"])
        if not math.isfinite(mse) or not math.isfinite(mae):
            raise ValueError(f"non-finite metric: {job['run_id']}")
        key = (job["dataset"], job["horizon"])
        pub_mse, pub_mae = published.get(key, (math.nan, math.nan))
        metric_rows.append(
            {
                "model": "TimeAlign",
                "dataset": job["dataset"],
                "horizon": job["horizon"],
                "seed": job["seed"],
                "mse": mse,
                "mae": mae,
                "published_mse": pub_mse,
                "published_mae": pub_mae,
                "mse_vs_published_pct": (
                    100 * (mse / pub_mse - 1)
                    if math.isfinite(pub_mse)
                    else math.nan
                ),
                "mae_vs_published_pct": (
                    100 * (mae / pub_mae - 1)
                    if math.isfinite(pub_mae)
                    else math.nan
                ),
                "artifact_role": job["artifact_role"],
                "preset_role": source["dataset_scripts"][job["dataset"]][
                    "preset_role"
                ],
                "value_origin": (
                    "local_timealign_official_native_seed2021"
                    if job["dataset"] != "Exchange"
                    else "local_timealign_source_informed_exchange_seed2021"
                ),
            }
        )
        manifest_rows.append(
            {
                "run_id": job["run_id"],
                "dataset": job["dataset"],
                "horizon": job["horizon"],
                "seed": job["seed"],
                "epochs": job["epochs"],
                "batch_size": job["batch_size"],
                "artifact_role": job["artifact_role"],
                "preset_role": source["dataset_scripts"][job["dataset"]][
                    "preset_role"
                ],
                "checkpoint_sha256": hashes[job["run_id"]],
                "effective_config_sha256": sha256(directory / "effective_config.json"),
                "metrics_sha256": sha256(directory / "metrics_by_target_horizon.csv"),
                "training_log_sha256": sha256(directory / "training_log.csv"),
                "predictions_retained": (
                    True
                    if job["artifact_role"] == "reusable"
                    else prediction.is_file()
                ),
                "status": "artifact_complete_test_complete",
            }
        )

    metric_rows.sort(key=lambda row: (row["dataset"], int(row["horizon"])))
    manifest_rows.sort(key=lambda row: (row["dataset"], int(row["horizon"])))
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.analysis_dir / "timealign_main_i_local_metrics.csv"
    manifest_path = args.analysis_dir / "timealign_main_i_artifact_manifest.csv"
    write_csv(metrics_path, metric_rows)
    write_csv(manifest_path, manifest_rows)

    dataset_summary = {}
    for dataset in config["matrix"]["datasets"]:
        rows = [row for row in metric_rows if row["dataset"] == dataset]
        dataset_summary[dataset] = {
            "mse": sum(float(row["mse"]) for row in rows) / 4,
            "mae": sum(float(row["mae"]) for row in rows) / 4,
            "published_comparison_available": all(
                math.isfinite(float(row["published_mse"])) for row in rows
            ),
            "preset_role": rows[0]["preset_role"],
        }
    summary = {
        "protocol_id": config["protocol_id"],
        "runs_complete": len(metric_rows),
        "test_rows_complete": len(metric_rows),
        "unique_checkpoint_hashes": len(set(hashes.values())),
        "reusable_runs": sum(row["artifact_role"] == "reusable" for row in manifest_rows),
        "new_runs": sum(row["artifact_role"] == "new" for row in manifest_rows),
        "dataset_summary": dataset_summary,
        "metrics_sha256": sha256(metrics_path),
        "artifact_manifest_sha256": sha256(manifest_path),
        "artifact_complete": True,
        "matrix_complete": True,
        "decision": "TimeAlign_Main_I_32_of_32_local_reproduction_complete",
    }
    (args.analysis_dir / "timealign_main_i_reproduction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
