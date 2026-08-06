#!/usr/bin/env python3
"""Consolidate a complete QDF L336 eight-dataset reproduction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL", "Solar", "Exchange")
HORIZONS = (96, 192, 336, 720)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exactly_one(root: Path, name: str) -> Path:
    paths = list(root.rglob(name))
    if len(paths) != 1:
        raise ValueError(f"expected exactly one {name} under {root}, got {len(paths)}")
    return paths[0]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            run_root = args.results_root / "runs" / f"QDF__{dataset}__H{horizon}__seed2023"
            metrics_path = exactly_one(run_root / "results", "metrics.npy")
            config_path = exactly_one(run_root / "results", "config.yaml")
            checkpoint_path = exactly_one(run_root / "checkpoints", "checkpoint.pth")
            loss_path = exactly_one(run_root / "checkpoints", "A.pth")
            stdout_path = run_root / "stdout.log"
            if not stdout_path.is_file() or stdout_path.stat().st_size == 0:
                raise ValueError(f"missing stdout: {dataset} H{horizon}")
            stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
            if any(token in stdout for token in ("Traceback", "CUDA out of memory", "nan")):
                raise ValueError(f"failure marker: {dataset} H{horizon}")
            values = np.load(metrics_path, allow_pickle=False)
            if values.shape != (7,) or not np.isfinite(values).all():
                raise ValueError(f"invalid metrics: {dataset} H{horizon}: {values}")
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            expected = {"model": "TQNet", "seq_len": 336, "label_len": 48, "pred_len": horizon, "fix_seed": 2023, "final_evaluation_split": "test"}
            for key, expected_value in expected.items():
                if config.get(key) != expected_value:
                    raise ValueError(f"config mismatch {dataset} H{horizon} {key}: {config.get(key)!r}")
            source_role = "official_released_profile" if dataset not in {"Solar", "Exchange"} else ("source_informed_ecl_profile" if dataset == "Solar" else "source_informed_etth1_profile")
            metric_rows.append({"model": "QDF", "dataset": dataset, "horizon": horizon, "seed": 2023, "seq_len": 336, "mse": float(values[1]), "mae": float(values[0]), "value_origin": "official_code_local_single_seed_l336", "profile_role": source_role, "system_role": "horizon_specific_official_native", "test_tuned": "false"})
            for role, path in (("checkpoint", checkpoint_path), ("learned_qdf_loss", loss_path), ("metrics", metrics_path), ("effective_config", config_path), ("stdout", stdout_path)):
                artifact_rows.append({"dataset": dataset, "horizon": horizon, "seed": 2023, "artifact_role": role, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
            retry_log = run_root / "training_stdout_before_test_retry.log"
            if retry_log.is_file() and retry_log.stat().st_size > 0:
                artifact_rows.append({"dataset": dataset, "horizon": horizon, "seed": 2023, "artifact_role": "training_stdout_before_test_retry", "path": str(retry_log), "sha256": sha256(retry_log), "bytes": retry_log.stat().st_size})

    with (args.output_dir / "qdf_main_i_l336_local_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(metric_rows)
    with (args.output_dir / "qdf_main_i_l336_artifact_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(artifact_rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(artifact_rows)
    summary = {"protocol_id": "QDF-OFFICIAL-MAIN-I-L336-REPRODUCTION-20260806", "matrix_complete": True, "test_rows": len(metric_rows), "artifact_rows": len(artifact_rows), "datasets": list(DATASETS), "horizons": list(HORIZONS), "seq_len": 336, "seed": 2023, "mean_mse": sum(float(row["mse"]) for row in metric_rows) / len(metric_rows), "mean_mae": sum(float(row["mae"]) for row in metric_rows) / len(metric_rows), "claim_boundary": "six released profiles plus disclosed ECL-derived Solar and ETTh1-derived Exchange profiles; all values are local single-seed reproductions"}
    (args.output_dir / "qdf_main_i_l336_reproduction_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
