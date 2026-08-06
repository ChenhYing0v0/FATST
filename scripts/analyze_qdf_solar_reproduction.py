#!/usr/bin/env python3
"""Consolidate a completed QDF Solar four-horizon reproduction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


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
    metrics_rows: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        run_root = args.results_root / "runs" / f"QDF__Solar__H{horizon}__seed2023"
        metrics_path = exactly_one(run_root / "results", "metrics.npy")
        config_path = exactly_one(run_root / "results", "config.yaml")
        checkpoint_path = exactly_one(run_root / "checkpoints", "checkpoint.pth")
        a_path = exactly_one(run_root / "checkpoints", "A.pth")
        stdout_path = run_root / "stdout.log"
        if not stdout_path.is_file() or stdout_path.stat().st_size == 0:
            raise ValueError(f"missing stdout for H{horizon}")
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        if any(token in stdout for token in ("Traceback", "CUDA out of memory", "nan")):
            raise ValueError(f"failure marker in H{horizon} stdout")

        values = np.load(metrics_path, allow_pickle=False)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError(f"invalid metrics array for H{horizon}: {values}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        expected = {
            "data": "Solar",
            "data_id": "Solar",
            "model": "TQNet",
            "seq_len": 96,
            "pred_len": horizon,
            "enc_in": 137,
            "cycle": 144,
            "fix_seed": 2023,
            "final_evaluation_split": "test",
        }
        for key, expected_value in expected.items():
            if config.get(key) != expected_value:
                raise ValueError(
                    f"H{horizon} config mismatch for {key}: "
                    f"{config.get(key)!r} != {expected_value!r}"
                )
        metrics_rows.append(
            {
                "model": "QDF",
                "dataset": "Solar",
                "horizon": horizon,
                "seed": 2023,
                "mse": float(values[1]),
                "mae": float(values[0]),
                "value_origin": "official_native_source_informed_solar_single_seed",
                "system_role": "horizon_specific_official_native_source_informed",
                "test_tuned": "false",
            }
        )
        for role, path in (
            ("checkpoint", checkpoint_path),
            ("learned_qdf_loss", a_path),
            ("metrics", metrics_path),
            ("effective_config", config_path),
            ("stdout", stdout_path),
        ):
            artifact_rows.append(
                {
                    "dataset": "Solar",
                    "horizon": horizon,
                    "seed": 2023,
                    "artifact_role": role,
                    "path": str(path),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )

    metric_fields = list(metrics_rows[0])
    with (args.output_dir / "qdf_solar_local_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics_rows)
    artifact_fields = list(artifact_rows[0])
    with (args.output_dir / "qdf_solar_artifact_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=artifact_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(artifact_rows)
    summary = {
        "protocol_id": "QDF-OFFICIAL-SOLAR-REPRODUCTION-20260806",
        "matrix_complete": True,
        "test_rows": 4,
        "artifact_rows": len(artifact_rows),
        "mean_mse": sum(float(row["mse"]) for row in metrics_rows) / 4,
        "mean_mae": sum(float(row["mae"]) for row in metrics_rows) / 4,
        "test_role": "official_native_source_informed_solar_reproduction",
        "claim_boundary": "Solar uses an ECL-derived released-code preset; it is not a published QDF value",
    }
    (args.output_dir / "qdf_solar_reproduction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
