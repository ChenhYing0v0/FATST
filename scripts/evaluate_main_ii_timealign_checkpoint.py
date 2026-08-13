#!/usr/bin/env python3
"""Stream Main II prefixes from one frozen TimeAlign H720 checkpoint."""

from __future__ import annotations

import argparse
import copy
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


def numeric_tolerance(reference: float, absolute_floor: float) -> float:
    """Allow the frozen floor or four float32 ULPs, whichever is larger."""
    return max(
        absolute_floor,
        4.0 * abs(float(np.spacing(np.float32(reference)))),
    )


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
    parser.add_argument("--loader-horizon", type=int, choices=HORIZONS)
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
    evaluation_horizons = (
        (args.loader_horizon,) if args.loader_horizon is not None else HORIZONS
    )
    official_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not Path(official_args.root_path).is_dir():
        raise FileNotFoundError(official_args.root_path)

    checkpoint_before = sha256(args.checkpoint)
    model = TimeAlign.Model(official_args).float().to(official_args.device)
    state = torch.load(args.checkpoint, map_location=official_args.device)
    model.load_state_dict(state)
    model.eval()
    loader_args = copy.deepcopy(official_args)
    if args.loader_horizon is not None:
        loader_args.pred_len = args.loader_horizon
    dataset, loader = train_repo.data_provider(loader_args, "test")
    del dataset
    if loader.drop_last:
        expected_origins = len(loader) * int(loader.batch_size)
    else:
        expected_origins = len(loader.dataset)
    channels = int(official_args.c_out)
    dtype = np.dtype(np.float32)
    accumulators: dict[int, dict[str, object]] = {}
    for horizon in evaluation_horizons:
        shape = (expected_origins, horizon, channels)
        accumulators[horizon] = {
            "squared_error_sum": 0.0,
            "absolute_error_sum": 0.0,
            "element_count": 0,
            "prediction_digest": initialize_digest(shape, dtype),
            "target_digest": initialize_digest(shape, dtype),
        }

    observed_origins = 0
    native_predictions: list[np.ndarray] = []
    native_targets: list[np.ndarray] = []
    f_dim = -1 if official_args.features == "MS" else 0
    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
            batch_x = batch_x.float().to(official_args.device)
            batch_y = batch_y.float().to(official_args.device)
            if args.loader_horizon is None:
                model_target = batch_y[:, -official_args.pred_len :, :]
                is_training = True
            else:
                model_target = torch.zeros(
                    batch_x.shape[0],
                    official_args.pred_len,
                    batch_x.shape[-1],
                    dtype=batch_x.dtype,
                    device=batch_x.device,
                )
                is_training = False
            outputs, _reconstruction, _alignment = model(
                batch_x, model_target, is_training=is_training
            )
            prediction = np.ascontiguousarray(
                outputs[:, : loader_args.pred_len, f_dim:].detach().cpu().numpy()
            )
            target = np.ascontiguousarray(
                batch_y[:, -loader_args.pred_len :, f_dim:].detach().cpu().numpy()
            )
            if prediction.shape != target.shape or prediction.shape[1:] != (
                loader_args.pred_len,
                channels,
            ):
                raise RuntimeError(
                    f"unexpected TimeAlign tensor shapes: {prediction.shape}, {target.shape}"
                )
            if not np.isfinite(prediction).all() or not np.isfinite(target).all():
                raise RuntimeError("non-finite TimeAlign formal-test tensor")
            native_predictions.append(prediction)
            native_targets.append(target)
            observed_origins += int(prediction.shape[0])
            for horizon in evaluation_horizons:
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

    native_prediction = np.concatenate(native_predictions, axis=0)
    del native_predictions
    native_target = np.concatenate(native_targets, axis=0)
    del native_targets
    native_rows = train_repo.metric_rows(
        native_prediction,
        native_target,
        list(evaluation_horizons),
    )
    native_metrics = {
        int(row["target_horizon"]): {
            "mse": float(row["mse"]),
            "mae": float(row["mae"]),
        }
        for row in native_rows
    }
    del native_prediction, native_target, native_rows

    rows: list[dict[str, object]] = []
    for horizon in evaluation_horizons:
        accumulator = accumulators[horizon]
        element_count = int(accumulator["element_count"])
        rows.append(
            {
                "system": args.system,
                "dataset": args.dataset,
                "repeat": args.repeat,
                "horizon": horizon,
                "mse": native_metrics[horizon]["mse"],
                "mae": native_metrics[horizon]["mae"],
                "global_elementwise_float64_mse": (
                    float(accumulator["squared_error_sum"]) / element_count
                ),
                "global_elementwise_float64_mae": (
                    float(accumulator["absolute_error_sum"]) / element_count
                ),
                "metric_semantics": "official_numpy_step_mean_float32_cumsum_float64",
                "origin_count": observed_origins,
                "channel_count": channels,
                "checkpoint_sha256": checkpoint_before,
                "prediction_prefix_sha256": accumulator[
                    "prediction_digest"
                ].hexdigest(),
                "target_prefix_sha256": accumulator["target_digest"].hexdigest(),
                "prefix_identity": args.loader_horizon is None,
                "test_role": (
                    "formal_horizon_specific_loader_H720_checkpoint_prefix"
                    if args.loader_horizon is not None
                    else "formal_H720_native_loader_exact_prefix_stream"
                ),
                "loader_horizon": loader_args.pred_len,
                "loader_drop_last": bool(loader.drop_last),
                "input_only_inference": args.loader_horizon is not None,
                "test_tuned": False,
                "matrix_complete": False,
            }
        )

    anchor_mse, anchor_mae = read_anchor(args.anchor_metrics)
    mse_delta = mae_delta = mse_tolerance = mae_tolerance = None
    if loader_args.pred_len == 720:
        h720 = rows[-1]
        mse_delta = float(h720["mse"]) - anchor_mse
        mae_delta = float(h720["mae"]) - anchor_mae
        mse_tolerance = numeric_tolerance(anchor_mse, args.tolerance)
        mae_tolerance = numeric_tolerance(anchor_mae, args.tolerance)
        if abs(mse_delta) > mse_tolerance or abs(mae_delta) > mae_tolerance:
            raise RuntimeError(
                "H720 anchor mismatch: "
                f"mse_delta={mse_delta}, mse_tolerance={mse_tolerance}, "
                f"mae_delta={mae_delta}, mae_tolerance={mae_tolerance}"
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
        "h720_mse_tolerance": mse_tolerance,
        "h720_mae_tolerance": mae_tolerance,
        "numeric_tolerance_rule": "max(1e-8, four_float32_ULPs_of_anchor)",
        "loader_horizon": loader_args.pred_len,
        "loader_drop_last": bool(loader.drop_last),
        "loader_dataset_size": len(loader.dataset),
        "model_horizon": int(official_args.pred_len),
        "input_only_inference": args.loader_horizon is not None,
        "future_label_used_as_model_input": False,
        "array_retention": "streamed_not_saved",
        "rows": rows,
    }
    (args.output_dir / "prefix_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "main_ii_timealign_eval=pass "
        f"dataset={args.dataset} loader_horizon={loader_args.pred_len} "
        f"origins={observed_origins} rows={len(rows)} "
        f"h720_mse_delta={mse_delta if mse_delta is not None else 'n/a'} "
        f"h720_mae_delta={mae_delta if mae_delta is not None else 'n/a'}"
    )


if __name__ == "__main__":
    main()
