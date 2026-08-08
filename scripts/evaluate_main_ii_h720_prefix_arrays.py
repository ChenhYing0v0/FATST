#!/usr/bin/env python3
"""Compute frozen Main II prefix metrics from one H720 tensor pair.

The evaluator memory-maps upstream arrays and processes origin chunks so large
ECL tensors do not need to be copied into host memory as a whole.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HORIZONS = (96, 192, 336, 720)


def load_array(path: Path, layout: str) -> np.ndarray:
    """Memory-map an upstream tensor and validate its canonical NTC shape."""
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    if array.ndim != 3:
        raise ValueError(f"expected rank-3 tensor at {path}, got {array.shape}")
    if layout == "NTC":
        canonical_shape = array.shape
    elif layout == "NCT":
        canonical_shape = (array.shape[0], array.shape[2], array.shape[1])
    else:
        raise ValueError(f"unsupported layout: {layout}")
    if canonical_shape[1] != 720:
        raise ValueError(f"expected H720 tensor at {path}, got {canonical_shape}")
    return array


def canonical_chunk(
    array: np.ndarray, layout: str, start: int, stop: int
) -> np.ndarray:
    """Return one contiguous [origin, time, channel] chunk."""
    chunk = array[start:stop]
    if layout == "NCT":
        chunk = np.transpose(chunk, (0, 2, 1))
    return np.ascontiguousarray(chunk)


def canonical_shape(array: np.ndarray, layout: str) -> tuple[int, int, int]:
    """Return the logical NTC shape without materializing a transpose."""
    if layout == "NTC":
        return tuple(int(value) for value in array.shape)
    return (int(array.shape[0]), int(array.shape[2]), int(array.shape[1]))


def initialize_digest(shape: tuple[int, ...], dtype: np.dtype) -> hashlib._Hash:
    """Initialize the canonical tensor hash used by the original evaluator."""
    digest = hashlib.sha256()
    digest.update(str(shape).encode("utf-8"))
    digest.update(str(dtype).encode("utf-8"))
    return digest


def evaluate(
    prediction: np.ndarray,
    target: np.ndarray,
    layout: str,
    chunk_origins: int,
) -> tuple[list[dict[str, object]], tuple[int, int, int]]:
    """Evaluate all prefixes in float64 from the same streamed H720 batches."""
    prediction_shape = canonical_shape(prediction, layout)
    target_shape = canonical_shape(target, layout)
    if prediction_shape != target_shape:
        raise ValueError(
            f"prediction/target shape mismatch: {prediction_shape} vs {target_shape}"
        )
    if prediction.dtype != target.dtype:
        raise ValueError(
            f"prediction/target dtype mismatch: {prediction.dtype} vs {target.dtype}"
        )

    accumulators: dict[int, dict[str, object]] = {}
    for horizon in HORIZONS:
        prefix_shape = (prediction_shape[0], horizon, prediction_shape[2])
        accumulators[horizon] = {
            "squared_error_sum": 0.0,
            "absolute_error_sum": 0.0,
            "element_count": 0,
            "prediction_digest": initialize_digest(prefix_shape, prediction.dtype),
            "target_digest": initialize_digest(prefix_shape, target.dtype),
        }

    for start in range(0, prediction_shape[0], chunk_origins):
        stop = min(start + chunk_origins, prediction_shape[0])
        prediction_chunk = canonical_chunk(prediction, layout, start, stop)
        target_chunk = canonical_chunk(target, layout, start, stop)
        if not np.isfinite(prediction_chunk).all():
            raise ValueError(f"non-finite predictions in origins [{start}, {stop})")
        if not np.isfinite(target_chunk).all():
            raise ValueError(f"non-finite targets in origins [{start}, {stop})")
        for horizon in HORIZONS:
            pred_prefix = np.ascontiguousarray(prediction_chunk[:, :horizon, :])
            true_prefix = np.ascontiguousarray(target_chunk[:, :horizon, :])
            error = pred_prefix.astype(np.float64) - true_prefix.astype(np.float64)
            accumulator = accumulators[horizon]
            accumulator["squared_error_sum"] += float(
                np.sum(np.square(error), dtype=np.float64)
            )
            accumulator["absolute_error_sum"] += float(
                np.sum(np.abs(error), dtype=np.float64)
            )
            accumulator["element_count"] += int(error.size)
            accumulator["prediction_digest"].update(pred_prefix.tobytes())
            accumulator["target_digest"].update(true_prefix.tobytes())

    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        accumulator = accumulators[horizon]
        element_count = int(accumulator["element_count"])
        rows.append(
            {
                "horizon": horizon,
                "mse": float(accumulator["squared_error_sum"]) / element_count,
                "mae": float(accumulator["absolute_error_sum"]) / element_count,
                "origin_count": prediction_shape[0],
                "channel_count": prediction_shape[2],
                "prediction_prefix_sha256": accumulator[
                    "prediction_digest"
                ].hexdigest(),
                "target_prefix_sha256": accumulator["target_digest"].hexdigest(),
                "prefix_identity": True,
            }
        )
    return rows, prediction_shape


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
    parser.add_argument("--chunk-origins", type=int, default=8)
    parser.add_argument("--remove-input-arrays-after-success", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_origins <= 0:
        raise ValueError("--chunk-origins must be positive")
    prediction = load_array(args.prediction, args.layout)
    target = load_array(args.target, args.layout)
    rows, tensor_shape = evaluate(
        prediction, target, args.layout, args.chunk_origins
    )
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
        "prediction_shape": list(tensor_shape),
        "target_shape": list(tensor_shape),
        "prediction_h720_sha256": rows[-1]["prediction_prefix_sha256"],
        "target_h720_sha256": rows[-1]["target_prefix_sha256"],
        "canonical_layout": "NTC",
        "source_layout": args.layout,
        "metric_accumulation": "origin_chunked_float64_global_elementwise",
        "chunk_origins": args.chunk_origins,
        "input_array_retention": "retained",
        "rows": rows,
    }
    output_json = args.output_dir / "prefix_metrics.json"
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.remove_input_arrays_after_success:
        removed_paths = []
        for path in (args.prediction, args.target):
            resolved = path.resolve()
            if resolved.suffix != ".npy" or resolved.name not in {
                "pred.npy",
                "true.npy",
            }:
                raise ValueError(f"refusing to remove non-ephemeral array: {resolved}")
            resolved.unlink()
            removed_paths.append(str(resolved))
        payload["input_array_retention"] = "removed_after_successful_metric_hash_audit"
        payload["removed_input_arrays"] = removed_paths
        output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(
        "main_ii_prefix_eval=pass "
        f"system={args.system} dataset={args.dataset} rows={len(rows)} "
        f"shape={tensor_shape} retention={payload['input_array_retention']}"
    )


if __name__ == "__main__":
    main()
