#!/usr/bin/env python3
"""Stream Main II prefixes from one frozen official QDF H720 checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
QDF_ROOT = REPO_ROOT / "baselines" / "qdf_official"
if str(QDF_ROOT) not in sys.path:
    sys.path.insert(0, str(QDF_ROOT))

from exp import EXP_DICT  # noqa: E402


HORIZONS = (96, 192, 336, 720)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_digest(shape: tuple[int, ...], dtype: np.dtype) -> object:
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
    values = np.load(path, allow_pickle=False)
    if values.size < 2:
        raise RuntimeError(f"invalid QDF metric anchor: {path}")
    return float(values[1]), float(values[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-yaml", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-metrics", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    raw_config = yaml.safe_load(cli.config_yaml.read_text(encoding="utf-8"))
    args = SimpleNamespace(**raw_config)
    if int(args.pred_len) != 720 or args.task_name != "long_term_forecast_meta_ml3":
        raise RuntimeError("QDF config is not the frozen H720 ML3 task")
    args.use_gpu = bool(torch.cuda.is_available())
    args.gpu = 0
    args.use_multi_gpu = False
    args.num_workers = 0

    fix_seed = int(args.fix_seed)
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)

    checkpoint_before = sha256(cli.checkpoint)
    if checkpoint_before != cli.expected_checkpoint_sha256:
        raise RuntimeError("QDF checkpoint hash differs from the frozen manifest")
    experiment = EXP_DICT[args.task_name](args)
    experiment.model.load_state_dict(
        torch.load(cli.checkpoint, map_location=experiment.device, weights_only=True)
    )
    experiment.model.eval()
    test_data, test_loader = experiment._get_data(flag="test")
    if test_loader.drop_last:
        expected_origins = len(test_loader) * int(test_loader.batch_size)
    else:
        expected_origins = len(test_loader.dataset)
    channels = int(args.c_out)
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
    native_predictions: list[torch.Tensor] = []
    native_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle in test_loader:
            outputs, batch_y, _ = experiment.forward_step(
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle
            )
            outputs = outputs.detach()
            batch_y = batch_y.detach()
            if test_data.scale and args.inverse:
                output_np = outputs.cpu().numpy()
                target_np = batch_y.cpu().numpy()
                shape = output_np.shape
                output_np = test_data.inverse_transform(
                    output_np.reshape(-1, shape[-1])
                ).reshape(shape)
                target_np = test_data.inverse_transform(
                    target_np.reshape(-1, shape[-1])
                ).reshape(shape)
            else:
                output_np = outputs.cpu().numpy()
                target_np = batch_y.cpu().numpy()
            prediction = np.ascontiguousarray(output_np, dtype=np.float32)
            target = np.ascontiguousarray(target_np, dtype=np.float32)
            if prediction.shape != target.shape or prediction.shape[1:] != (
                720,
                channels,
            ):
                raise RuntimeError(
                    f"unexpected QDF tensor shapes: {prediction.shape}, {target.shape}"
                )
            if not np.isfinite(prediction).all() or not np.isfinite(target).all():
                raise RuntimeError("non-finite QDF formal-test tensor")
            native_predictions.append(torch.from_numpy(prediction))
            native_targets.append(torch.from_numpy(target))
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
            f"QDF origin mismatch: {observed_origins} vs {expected_origins}"
        )

    native_prediction = torch.cat(native_predictions, dim=0)
    native_target = torch.cat(native_targets, dim=0)
    native_metrics = {}
    for horizon in HORIZONS:
        difference = (
            native_prediction[:, :horizon, :] - native_target[:, :horizon, :]
        )
        native_metrics[horizon] = {
            "mse": float(torch.mean(difference**2).item()),
            "mae": float(torch.mean(torch.abs(difference)).item()),
        }
    del native_prediction, native_target, native_predictions, native_targets

    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        accumulator = accumulators[horizon]
        count = int(accumulator["element_count"])
        rows.append(
            {
                "system": "QDF",
                "dataset": cli.dataset,
                "repeat": 0,
                "horizon": horizon,
                "mse": native_metrics[horizon]["mse"],
                "mae": native_metrics[horizon]["mae"],
                "global_elementwise_float64_mse": (
                    float(accumulator["squared_error_sum"]) / count
                ),
                "global_elementwise_float64_mae": (
                    float(accumulator["absolute_error_sum"]) / count
                ),
                "metric_semantics": "official_torch_float32_global_mean",
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

    anchor_mse, anchor_mae = read_anchor(cli.anchor_metrics)
    mse_delta = float(rows[-1]["mse"]) - anchor_mse
    mae_delta = float(rows[-1]["mae"]) - anchor_mae
    mse_tolerance = numeric_tolerance(anchor_mse, cli.tolerance)
    mae_tolerance = numeric_tolerance(anchor_mae, cli.tolerance)
    if abs(mse_delta) > mse_tolerance or abs(mae_delta) > mae_tolerance:
        raise RuntimeError(
            "QDF H720 anchor mismatch: "
            f"mse_delta={mse_delta}, mse_tolerance={mse_tolerance}, "
            f"mae_delta={mae_delta}, mae_tolerance={mae_tolerance}"
        )
    checkpoint_after = sha256(cli.checkpoint)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("QDF checkpoint mutated during formal test")

    cli.output_dir.mkdir(parents=True, exist_ok=True)
    with (cli.output_dir / "prefix_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "gate": "pass",
        "system": "QDF",
        "dataset": cli.dataset,
        "checkpoint_path": str(cli.checkpoint),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "checkpoint_immutable": True,
        "config_yaml_path": str(cli.config_yaml),
        "config_yaml_sha256": sha256(cli.config_yaml),
        "anchor_metrics_path": str(cli.anchor_metrics),
        "anchor_metrics_sha256": sha256(cli.anchor_metrics),
        "h720_anchor_mse": anchor_mse,
        "h720_anchor_mae": anchor_mae,
        "h720_mse_delta": mse_delta,
        "h720_mae_delta": mae_delta,
        "h720_mse_tolerance": mse_tolerance,
        "h720_mae_tolerance": mae_tolerance,
        "numeric_tolerance_rule": "max(1e-8, four_float32_ULPs_of_anchor)",
        "array_retention": "streamed_not_saved",
        "rows": rows,
    }
    (cli.output_dir / "prefix_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "main_ii_qdf_eval=pass "
        f"dataset={cli.dataset} origins={observed_origins} rows=4 "
        f"h720_mse_delta={mse_delta:.3e} h720_mae_delta={mae_delta:.3e}"
    )


if __name__ == "__main__":
    main()
