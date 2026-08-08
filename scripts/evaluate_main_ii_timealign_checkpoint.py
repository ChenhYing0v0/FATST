#!/usr/bin/env python3
"""Stream Main II prefixes from one frozen TimeAlign H720 checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
if str(TIMEALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEALIGN_ROOT))

import train_repo  # noqa: E402
from models import TimeAlign  # noqa: E402


HORIZONS = (96, 192, 336, 720)


def sha256(path: Path) -> str:
    """Return the SHA256 of one checkpoint or audit artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_digest(shape: tuple[int, ...], dtype: np.dtype) -> object:
    """Initialize a canonical tensor hash with shape and dtype."""
    digest = hashlib.sha256()
    digest.update(str(shape).encode("utf-8"))
    digest.update(str(dtype).encode("utf-8"))
    return digest


def read_anchor(path: Path) -> tuple[float, float]:
    """Read the exact frozen Main I H720 MSE/MAE anchor."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidates = [
        row for row in rows if int(row.get("target_horizon", row.get("horizon", 0))) == 720
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one H720 anchor row: {path}")
    return float(candidates[0]["mse"]), float(candidates[0]["mae"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--effective-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-metrics", type=Path, required=True)
    parser.add_argument("--system", default="TimeAlign")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.effective_config.read_text(encoding="utf-8"))
    official_args = SimpleNamespace(**payload["official_args"])
    if int(official_args.pred_len) != 720:
        raise RuntimeError("TimeAlign checkpoint is not an H720 model")
    if str(official_args.readout_mode) != "official":
        raise RuntimeError("this evaluator is restricted to official TimeAlign")
    official_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not Path(official_args.root_path).is_dir():
        raise FileNotFoundError(official_args.root_path)

    checkpoint_before = sha256(args.checkpoint)
    model = TimeAlign.Model(official_args).float().to(official_args.device)
    state = torch.load(args.checkpoint, map_location=official_args.device)
    model.load_state_dict(state)
    model.eval()
    dataset, loader = train_repo.data_provider(official_args, "test")
    del dataset
    if loader.drop_last:
        expected_origins = len(loader) * int(loader.batch_size)
    else:
        expected_origins = len(loader.dataset)
    channels = int(official_args.c_out)
    dtype = np.dtype(np.float32)
    accumulators: dict[int, dict[str, object]] = {}
    for horizon in HORIZONS:
        shape = (expected_origins, horizon, channels)
        accumulators[horizon] = {
            "squared_error_sum": 0.0,
            "absolute_error_sum": 0.0,
            "element_count": 0,
            "prediction_digest": initialize_digest(shape, dtype),
            "target_digest": initialize_digest(shape, dtype),
        }

    observed_origins = 0
    f_dim = -1 if official_args.features == "MS" else 0
    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
            batch_x = batch_x.float().to(official_args.device)
            batch_y = batch_y.float().to(official_args.device)
            outputs, _reconstruction, _alignment = model(
                batch_x,
                batch_y[:, -official_args.pred_len :, :],
                is_training=True,
            )
            prediction = np.ascontiguousarray(
                outputs[:, -official_args.pred_len :, f_dim:].detach().cpu().numpy()
            )
            target = np.ascontiguousarray(
                batch_y[:, -official_args.pred_len :, f_dim:].detach().cpu().numpy()
            )
            if prediction.shape != target.shape or prediction.shape[1:] != (
                720,
                channels,
            ):
                raise RuntimeError(
                    f"unexpected TimeAlign tensor shapes: {prediction.shape}, {target.shape}"
                )
            if not np.isfinite(prediction).all() or not np.isfinite(target).all():
                raise RuntimeError("non-finite TimeAlign formal-test tensor")
            observed_origins += int(prediction.shape[0])
            for horizon in HORIZONS:
                pred_prefix = np.ascontiguousarray(prediction[:, :horizon, :])
                true_prefix = np.ascontiguousarray(target[:, :horizon, :])
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
    if observed_origins != expected_origins:
        raise RuntimeError(
            f"origin count mismatch: observed={observed_origins}, expected={expected_origins}"
        )

    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        accumulator = accumulators[horizon]
        element_count = int(accumulator["element_count"])
        rows.append(
            {
                "system": args.system,
                "dataset": args.dataset,
                "repeat": args.repeat,
                "horizon": horizon,
                "mse": float(accumulator["squared_error_sum"]) / element_count,
                "mae": float(accumulator["absolute_error_sum"]) / element_count,
                "origin_count": observed_origins,
                "channel_count": channels,
                "checkpoint_sha256": checkpoint_before,
                "prediction_prefix_sha256": accumulator[
                    "prediction_digest"
                ].hexdigest(),
                "target_prefix_sha256": accumulator["target_digest"].hexdigest(),
                "prefix_identity": True,
                "test_role": "formal_H720_native_loader_exact_prefix_stream",
                "test_tuned": False,
                "matrix_complete": False,
            }
        )

    anchor_mse, anchor_mae = read_anchor(args.anchor_metrics)
    h720 = rows[-1]
    mse_delta = float(h720["mse"]) - anchor_mse
    mae_delta = float(h720["mae"]) - anchor_mae
    if abs(mse_delta) > args.tolerance or abs(mae_delta) > args.tolerance:
        raise RuntimeError(
            f"H720 anchor mismatch: mse_delta={mse_delta}, mae_delta={mae_delta}"
        )
    checkpoint_after = sha256(args.checkpoint)
    if checkpoint_before != checkpoint_after:
        raise RuntimeError("checkpoint mutated during TimeAlign formal test")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "prefix_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "gate": "pass",
        "system": args.system,
        "dataset": args.dataset,
        "repeat": args.repeat,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "checkpoint_immutable": True,
        "effective_config_path": str(args.effective_config),
        "effective_config_sha256": sha256(args.effective_config),
        "anchor_metrics_path": str(args.anchor_metrics),
        "anchor_metrics_sha256": sha256(args.anchor_metrics),
        "h720_anchor_mse": anchor_mse,
        "h720_anchor_mae": anchor_mae,
        "h720_mse_delta": mse_delta,
        "h720_mae_delta": mae_delta,
        "array_retention": "streamed_not_saved",
        "rows": rows,
    }
    (args.output_dir / "prefix_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "main_ii_timealign_eval=pass "
        f"dataset={args.dataset} origins={observed_origins} rows=4 "
        f"h720_mse_delta={mse_delta:.3e} h720_mae_delta={mae_delta:.3e}"
    )


if __name__ == "__main__":
    main()
