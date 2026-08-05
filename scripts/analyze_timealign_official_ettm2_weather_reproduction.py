#!/usr/bin/env python3
"""Audit and compare the artifact-complete TimeAlign reproduction."""

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
FAILURE_PATTERN = re.compile(
    r"Traceback|CUDA out of memory|(^|[^A-Za-z0-9_])(nan|inf)([^A-Za-z0-9_]|$)",
    re.IGNORECASE | re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "timealign_official_ettm2_weather_reproduction.json",
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--checkpoint-hashes", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_hashes(path: Path) -> dict[str, str]:
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, checkpoint = line.split(maxsplit=1)
        run_id = Path(checkpoint).parent.name
        if len(digest) != 64:
            raise ValueError(f"invalid checkpoint hash for {run_id}")
        output[run_id] = digest
    return output


def published_metrics(path: Path) -> dict[tuple[str, int], tuple[float, float]]:
    output = {}
    for row in read_csv(path):
        if row["model"] == "TimeAlign" and row["dataset"] in {"ETTm2", "Weather"}:
            output[(row["dataset"], int(row["horizon"]))] = (
                float(row["mse"]),
                float(row["mae"]),
            )
    return output


def historical_metrics(path: Path) -> dict[tuple[str, int], tuple[float, float]]:
    output = {}
    for row in read_csv(path):
        if row["dataset"] in {"ETTm2", "Weather"}:
            output[(row["dataset"], int(row["target_horizon"]))] = (
                float(row["mse"]),
                float(row["mae"]),
            )
    return output


def relative_pct(value: float, reference: float) -> float:
    return 100.0 * (value / reference - 1.0)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    jobs = config["jobs"]
    hashes = checkpoint_hashes(args.checkpoint_hashes)
    assert len(jobs) == len(hashes) == 8
    assert len(set(hashes.values())) == 8

    source = config["source_contract"]
    for name in (
        "adapter",
        "ettm2_script",
        "weather_script",
        "executed_model",
        "data_loader",
        "metric",
    ):
        assert sha256(ROOT / source[f"{name}_path"]) == source[f"{name}_sha256"]

    published_path = (
        ROOT
        / "analysis"
        / "iscf_bsca_paper_experiment_consolidation_20260731"
        / "timealign_table6_main_i_published.csv"
    )
    historical_path = ROOT / config["historical_reference"]["path"]
    published = published_metrics(published_path)
    historical = historical_metrics(historical_path)
    expected_cells = {
        (dataset, horizon)
        for dataset in ("ETTm2", "Weather")
        for horizon in (96, 192, 336, 720)
    }
    assert set(published) == set(historical) == expected_cells

    metric_rows = []
    manifest_rows = []
    for job in jobs:
        run_id = job["run_id"]
        directory = args.input_root / run_id
        required = (
            "effective_config.json",
            "environment.json",
            "training_log.csv",
            "metrics_by_target_horizon.csv",
            "metrics_by_segment.csv",
            "model_diagnostics.json",
            "run.log",
        )
        for filename in required:
            path = directory / filename
            assert path.is_file() and path.stat().st_size > 0, path
        assert FAILURE_PATTERN.search(
            (directory / "run.log").read_text(encoding="utf-8")
        ) is None

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
            "epochs": 10,
            "enable_early_stopping": False,
        }
        for field, value in expected_adapter.items():
            assert adapter[field] == value, (run_id, field, adapter[field], value)
        for field in (
            "d_model",
            "d_ff",
            "dropout",
            "learning_rate",
            "w_align",
            "patch_num",
            "layer_norm",
        ):
            assert preset[field] == job[field], (run_id, field, preset[field], job[field])

        training = read_csv(directory / "training_log.csv")
        assert len(training) == 10
        assert all(int(row["early_stopping_enabled"]) == 0 for row in training)
        assert all(math.isfinite(float(row["train_loss"])) for row in training)

        metrics = read_csv(directory / "metrics_by_target_horizon.csv")
        assert len(metrics) == 1
        row = metrics[0]
        assert int(row["target_horizon"]) == job["horizon"]
        assert row["dataset"] == job["dataset"]
        assert row["checkpoint_policy"] == "official-last"
        assert row["evaluation_split"] == "test"
        assert row["protocol_class"] == "native_external"
        mse, mae = float(row["mse"]), float(row["mae"])
        assert math.isfinite(mse) and math.isfinite(mae)
        cell = (job["dataset"], job["horizon"])
        pub_mse, pub_mae = published[cell]
        hist_mse, hist_mae = historical[cell]
        metric_rows.append(
            {
                "dataset": job["dataset"],
                "horizon": job["horizon"],
                "seed": job["seed"],
                "mse": mse,
                "mae": mae,
                "published_mse": pub_mse,
                "published_mae": pub_mae,
                "mse_vs_published_pct": relative_pct(mse, pub_mse),
                "mae_vs_published_pct": relative_pct(mae, pub_mae),
                "historical_repro_mse": hist_mse,
                "historical_repro_mae": hist_mae,
                "mse_vs_historical_pct": relative_pct(mse, hist_mse),
                "mae_vs_historical_pct": relative_pct(mae, hist_mae),
            }
        )
        manifest_rows.append(
            {
                "run_id": run_id,
                "dataset": job["dataset"],
                "horizon": job["horizon"],
                "seed": job["seed"],
                "checkpoint_sha256": hashes[run_id],
                "effective_config_sha256": sha256(directory / "effective_config.json"),
                "metrics_sha256": sha256(directory / "metrics_by_target_horizon.csv"),
                "training_log_sha256": sha256(directory / "training_log.csv"),
                "status": "artifact_complete_test_complete",
            }
        )

    metric_rows.sort(key=lambda row: (row["dataset"], row["horizon"]))
    manifest_rows.sort(key=lambda row: (row["dataset"], row["horizon"]))
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.analysis_dir / "reproduced_metrics_and_comparison.csv", metric_rows)
    write_csv(args.analysis_dir / "artifact_manifest.csv", manifest_rows)

    dataset_summary = {}
    for dataset in ("ETTm2", "Weather"):
        rows = [row for row in metric_rows if row["dataset"] == dataset]
        means = {
            key: sum(float(row[key]) for row in rows) / len(rows)
            for key in ("mse", "mae", "published_mse", "published_mae")
        }
        dataset_summary[dataset] = {
            **means,
            "mean_mse_vs_published_pct": relative_pct(
                means["mse"], means["published_mse"]
            ),
            "mean_mae_vs_published_pct": relative_pct(
                means["mae"], means["published_mae"]
            ),
        }
    summary = {
        "protocol_id": config["protocol_id"],
        "execution_label": source["execution_label"],
        "license_status": source["license_status"],
        "runs_complete": 8,
        "test_rows_complete": 8,
        "unique_checkpoint_hashes": 8,
        "dataset_summary": dataset_summary,
        "max_abs_cell_mse_vs_historical_pct": max(
            abs(float(row["mse_vs_historical_pct"])) for row in metric_rows
        ),
        "max_abs_cell_mae_vs_historical_pct": max(
            abs(float(row["mae_vs_historical_pct"])) for row in metric_rows
        ),
        "artifact_complete": True,
        "decision": "artifact_complete_close_single_seed_official_native_reproduction",
    }
    (args.analysis_dir / "reproduction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
