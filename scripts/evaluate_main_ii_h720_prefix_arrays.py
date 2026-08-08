#!/usr/bin/env python3
"""Compute the frozen Main II prefix metrics from one H720 tensor pair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HORIZONS = (96, 192, 336, 720)


def sha256_array(array: np.ndarray) -> str:
    """Hash shape, dtype, and contiguous bytes for an auditable tensor id."""
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def load_canonical(path: Path, layout: str) -> np.ndarray:
    """Load an upstream tensor and convert it to [origin, time, channel]."""
    array = np.load(path, allow_pickle=False)
    if array.ndim != 3:
        raise ValueError(f"expected rank-3 tensor at {path}, got {array.shape}")
    if layout == "NTC":
        canonical = array
    elif layout == "NCT":
        canonical = np.transpose(array, (0, 2, 1))
    else:
        raise ValueError(f"unsupported layout: {layout}")
    if canonical.shape[1] != 720:
        raise ValueError(f"expected H720 tensor at {path}, got {canonical.shape}")
    if not np.isfinite(canonical).all():
        raise ValueError(f"non-finite values in {path}")
    return np.ascontiguousarray(canonical)


def evaluate(prediction: np.ndarray, target: np.ndarray) -> list[dict[str, object]]:
    """Evaluate all frozen prefixes using views of the same H720 tensors."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}"
        )
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        pred_prefix = prediction[:, :horizon, :]
        true_prefix = target[:, :horizon, :]
        if not np.array_equal(pred_prefix, prediction[:, 0:horizon, :]):
            raise AssertionError(f"H{horizon} prediction prefix identity failed")
        error = pred_prefix.astype(np.float64) - true_prefix.astype(np.float64)
        rows.append(
            {
                "horizon": horizon,
                "mse": float(np.mean(np.square(error), dtype=np.float64)),
                "mae": float(np.mean(np.abs(error), dtype=np.float64)),
                "origin_count": int(pred_prefix.shape[0]),
                "channel_count": int(pred_prefix.shape[2]),
                "prediction_prefix_sha256": sha256_array(pred_prefix),
                "target_prefix_sha256": sha256_array(true_prefix),
                "prefix_identity": True,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--layout", choices=["NTC", "NCT"], default="NTC")
    parser.add_argument("--system", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction = load_canonical(args.prediction, args.layout)
    target = load_canonical(args.target, args.layout)
    rows = evaluate(prediction, target)
    for row in rows:
        row.update(
            {
                "system": args.system,
                "dataset": args.dataset,
                "repeat": args.repeat,
                "checkpoint_sha256": args.checkpoint_sha256,
                "test_role": "formal_H720_native_loader_exact_prefix",
                "test_tuned": False,
                "matrix_complete": False,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (args.output_dir / "prefix_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "system": args.system,
        "dataset": args.dataset,
        "repeat": args.repeat,
        "checkpoint_sha256": args.checkpoint_sha256,
        "prediction_shape": list(prediction.shape),
        "target_shape": list(target.shape),
        "prediction_h720_sha256": sha256_array(prediction),
        "target_h720_sha256": sha256_array(target),
        "canonical_layout": "NTC",
        "source_layout": args.layout,
        "rows": rows,
    }
    (args.output_dir / "prefix_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "main_ii_prefix_eval=pass "
        f"system={args.system} dataset={args.dataset} rows={len(rows)} "
        f"shape={prediction.shape}"
    )


if __name__ == "__main__":
    main()
