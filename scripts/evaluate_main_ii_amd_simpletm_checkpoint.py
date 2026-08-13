#!/usr/bin/env python3
"""Stream Main II prefixes from frozen AMD or SimpleTM H720 checkpoints."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import runpy
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


HORIZONS = (96, 192, 336, 720)


def numeric_tolerance(reference: float, absolute_floor: float) -> float:
    """Allow the frozen floor or four float32 ULPs, whichever is larger."""
    return max(
        absolute_floor,
        4.0 * abs(float(np.spacing(np.float32(reference)))),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def initialize_accumulators(
    origins: int,
    channels: int,
    dtype: np.dtype,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[int, dict[str, Any]]:
    accumulators: dict[int, dict[str, Any]] = {}
    for horizon in horizons:
        shape = (origins, horizon, channels)
        prediction_digest = hashlib.sha256()
        prediction_digest.update(str(shape).encode("utf-8"))
        prediction_digest.update(str(dtype).encode("utf-8"))
        target_digest = hashlib.sha256()
        target_digest.update(str(shape).encode("utf-8"))
        target_digest.update(str(dtype).encode("utf-8"))
        accumulators[horizon] = {
            "squared_error_sum": 0.0,
            "absolute_error_sum": 0.0,
            "element_count": 0,
            "native_batch_mse_sum": 0.0,
            "native_batch_mae_sum": 0.0,
            "native_batch_count": 0,
            "prediction_digest": prediction_digest,
            "target_digest": target_digest,
        }
    return accumulators


def update_accumulators(
    accumulators: dict[int, dict[str, Any]],
    prediction: np.ndarray,
    target: np.ndarray,
    horizons: tuple[int, ...] = HORIZONS,
) -> None:
    maximum = max(horizons)
    if prediction.shape != target.shape or prediction.shape[1] != maximum:
        raise RuntimeError(f"unexpected tensor shapes: {prediction.shape}, {target.shape}")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise RuntimeError("non-finite formal-test tensor")
    for horizon in horizons:
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
        accumulator["native_batch_mse_sum"] += float(np.mean(np.square(error)))
        accumulator["native_batch_mae_sum"] += float(np.mean(np.abs(error)))
        accumulator["native_batch_count"] += 1
        accumulator["prediction_digest"].update(pred_prefix.tobytes())
        accumulator["target_digest"].update(true_prefix.tobytes())


def finalize_rows(
    baseline: str,
    dataset: str,
    repeat: int,
    checkpoint_sha: str,
    origins: int,
    channels: int,
    accumulators: dict[int, dict[str, Any]],
    native_overrides: dict[int, tuple[float, float]] | None = None,
    horizons: tuple[int, ...] = HORIZONS,
    loader_drop_last: bool | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        accumulator = accumulators[horizon]
        count = int(accumulator["element_count"])
        global_mse = float(accumulator["squared_error_sum"]) / count
        global_mae = float(accumulator["absolute_error_sum"]) / count
        native_count = int(accumulator["native_batch_count"])
        if native_overrides is None:
            native_mse = float(accumulator["native_batch_mse_sum"]) / native_count
            native_mae = float(accumulator["native_batch_mae_sum"]) / native_count
        else:
            native_mse, native_mae = native_overrides[horizon]
        use_native_batch_mean = baseline == "AMD"
        rows.append(
            {
                "system": baseline,
                "dataset": dataset,
                "repeat": repeat,
                "horizon": horizon,
                "mse": native_mse if use_native_batch_mean else global_mse,
                "mae": native_mae if use_native_batch_mean else global_mae,
                "global_elementwise_mse": global_mse,
                "global_elementwise_mae": global_mae,
                "native_batch_mean_mse": native_mse,
                "native_batch_mean_mae": native_mae,
                "metric_semantics": (
                    "official_unweighted_batch_mean"
                    if use_native_batch_mean
                    else "global_elementwise"
                ),
                "origin_count": origins,
                "channel_count": channels,
                "checkpoint_sha256": checkpoint_sha,
                "prediction_prefix_sha256": accumulator[
                    "prediction_digest"
                ].hexdigest(),
                "target_prefix_sha256": accumulator["target_digest"].hexdigest(),
                "prefix_identity": len(horizons) > 1,
                "test_role": (
                    "formal_horizon_specific_loader_H720_checkpoint_prefix"
                    if len(horizons) == 1
                    else "formal_H720_native_loader_exact_prefix_stream"
                ),
                "loader_horizon": max(horizons),
                "loader_drop_last": loader_drop_last,
                "input_only_inference": len(horizons) == 1,
                "test_tuned": False,
                "matrix_complete": False,
            }
        )
    return rows


def h720_record(run_dir: Path, repeat: int) -> dict[str, str]:
    matches = [
        row
        for row in read_rows(run_dir / "metrics.csv")
        if int(row["horizon"]) == 720 and int(row["repeat"]) == repeat
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one H720 repeat={repeat} record")
    return matches[0]


def capture_amd_command(
    source: Path, script_rel: str, horizon: int
) -> list[str]:
    shim = (
        'python(){ printf "CMD"; printf " %q" "$@"; printf "\\n"; }; '
        'export -f python; bash "$1"'
    )
    result = subprocess.run(
        ["bash", "-c", shim, "capture", script_rel],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    commands = [shlex.split(line[4:]) for line in result.stdout.splitlines() if line.startswith("CMD ")]
    matches = [
        command
        for command in commands
        if "--pred_len" in command
        and command[command.index("--pred_len") + 1] == str(horizon)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"could not isolate the AMD H{horizon} official command")
    return matches[0]


def evaluate_amd(
    run_dir: Path,
    dataset: str,
    record: dict[str, str],
    loader_horizon: int | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    source = run_dir / "workspace" / "source"
    completion = json.loads((run_dir / "complete.json").read_text(encoding="utf-8"))
    command = capture_amd_command(source, completion["official_script"], 720)
    horizon = 720 if loader_horizon is None else loader_horizon
    loader_command = capture_amd_command(
        source, completion["official_script"], horizon
    )
    checkpoint = Path(record["checkpoint"])
    checkpoint_before = sha256(checkpoint)
    if checkpoint_before != record["checkpoint_sha256"]:
        raise RuntimeError("AMD checkpoint hash mismatch")

    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    sys.path.insert(0, str(source))
    os.chdir(source)
    try:
        import main as amd_main
        from models.tsAMD import AMD
        from utils.dataloader import CustomDataLoader

        sys.argv = ["main.py", *loader_command[2:]]
        loader_args = amd_main.parse_args()
        sys.argv = ["main.py", *command[2:]]
        args = amd_main.parse_args()
        data_loader = CustomDataLoader(
            loader_args.data,
            loader_args.batch_size,
            loader_args.seq_len,
            loader_args.pred_len,
            loader_args.feature_type,
            loader_args.target,
        )
        test_loader = data_loader.get_test()
        model = AMD(
            input_shape=(args.seq_len, data_loader.n_feature),
            pred_len=args.pred_len,
            dropout=args.dropout,
            n_block=args.n_block,
            patch=args.patch,
            k=args.mix_layer_num,
            c=args.mix_layer_scale,
            alpha=args.alpha,
            target_slice=data_loader.target_slice,
            norm=args.norm,
            layernorm=args.layernorm,
        ).cuda()
        model.load_state_dict(torch.load(checkpoint, map_location="cuda"))
        model.eval()
        expected_origins = (
            len(test_loader) * int(test_loader.batch_size)
            if test_loader.drop_last
            else len(test_loader.dataset)
        )
        channels = int(data_loader.n_feature)
        evaluation_horizons = (horizon,) if loader_horizon is not None else HORIZONS
        accumulators = initialize_accumulators(
            expected_origins, channels, np.dtype(np.float32), evaluation_horizons
        )
        observed_origins = 0
        native_running = {
            horizon: {
                "mse": torch.zeros(1, device="cuda"),
                "mae": torch.zeros(1, device="cuda"),
            }
            for horizon in evaluation_horizons
        }
        with torch.no_grad():
            for batch_index, (batch_x, batch_y) in enumerate(test_loader):
                batch_y_cuda = batch_y.cuda()
                prediction, _moe_loss = model(batch_x.cuda())
                for horizon in evaluation_horizons:
                    difference = (
                        prediction[:, :horizon, :] - batch_y_cuda[:, :horizon, :]
                    )
                    batch_mse = torch.mean(difference**2)
                    batch_mae = torch.mean(torch.abs(difference))
                    running = native_running[horizon]
                    running["mse"] = (
                        running["mse"] * batch_index + batch_mse.detach()
                    ) / (batch_index + 1)
                    running["mae"] = (
                        running["mae"] * batch_index + batch_mae.detach()
                    ) / (batch_index + 1)
                pred_np = np.ascontiguousarray(
                    prediction.detach().cpu().numpy(), dtype=np.float32
                )
                pred_np = np.ascontiguousarray(
                    pred_np[:, :horizon, :], dtype=np.float32
                )
                true_np = np.ascontiguousarray(
                    batch_y.numpy()[:, -horizon:, :], dtype=np.float32
                )
                observed_origins += int(pred_np.shape[0])
                update_accumulators(
                    accumulators, pred_np, true_np, evaluation_horizons
                )
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        sys.path.remove(str(source))
    if observed_origins != expected_origins:
        raise RuntimeError("AMD origin-count mismatch")
    native_overrides = {
        horizon: (
            float(native_running[horizon]["mse"].item()),
            float(native_running[horizon]["mae"].item()),
        )
        for horizon in evaluation_horizons
    }
    rows = finalize_rows(
        "AMD",
        dataset,
        0,
        checkpoint_before,
        observed_origins,
        channels,
        accumulators,
        native_overrides=native_overrides,
        horizons=evaluation_horizons,
        loader_drop_last=bool(test_loader.drop_last),
    )
    checkpoint_after = sha256(checkpoint)
    return rows, {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "checkpoint_immutable": checkpoint_before == checkpoint_after,
        "official_command": command,
        "loader_command": loader_command,
        "loader_horizon": horizon,
        "loader_drop_last": bool(test_loader.drop_last),
        "loader_dataset_size": len(test_loader.dataset),
        "model_horizon": 720,
        "input_only_inference": loader_horizon is not None,
        "future_label_used_as_model_input": False,
    }


def simpletm_command(run_dir: Path, horizon: int) -> list[str]:
    commands = []
    with (run_dir / "run.log").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("COMMAND "):
                commands.append(shlex.split(line[len("COMMAND ") :]))
    matches = [
        command
        for command in commands
        if "--pred_len" in command
        and command[command.index("--pred_len") + 1] == str(horizon)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"could not isolate the SimpleTM H{horizon} official command")
    return matches[0]


def evaluate_simpletm(
    run_dir: Path,
    dataset: str,
    repeat: int,
    record: dict[str, str],
    loader_horizon: int | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    source = run_dir / "workspace" / "source"
    checkpoint = Path(record["checkpoint"])
    checkpoint_before = sha256(checkpoint)
    if checkpoint_before != record["checkpoint_sha256"]:
        raise RuntimeError("SimpleTM checkpoint hash mismatch")
    command = simpletm_command(run_dir, 720)
    horizon = 720 if loader_horizon is None else loader_horizon
    loader_command = simpletm_command(run_dir, horizon)
    result: dict[str, object] = {}

    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    sys.path.insert(0, str(source))
    os.chdir(source)
    try:
        from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
        from data_provider.data_factory import data_provider

        def stream_test(self: object, _setting: str, test: int = 0) -> None:
            del test
            loader_args = copy.deepcopy(self.args)
            loader_options = {
                loader_command[index][2:].replace("-", "_"): loader_command[index + 1]
                for index in range(len(loader_command) - 1)
                if loader_command[index].startswith("--")
                and not loader_command[index + 1].startswith("--")
            }
            for name in (
                "root_path", "data_path", "data", "features", "target", "freq",
                "seq_len", "label_len", "pred_len", "batch_size", "test_batch_size",
            ):
                if name not in loader_options or not hasattr(loader_args, name):
                    continue
                current = getattr(loader_args, name)
                value = loader_options[name]
                if isinstance(current, bool):
                    value = value.lower() in {"1", "true", "yes"}
                elif isinstance(current, int):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
                setattr(loader_args, name, value)
            loader_args.pred_len = horizon
            test_data, test_loader = data_provider(loader_args, flag="test")
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
            self.model.eval()
            expected_origins = (
                len(test_loader) * int(test_loader.batch_size)
                if test_loader.drop_last
                else len(test_loader.dataset)
            )
            channels = int(self.args.c_out)
            evaluation_horizons = (
                (horizon,) if loader_horizon is not None else HORIZONS
            )
            accumulators = initialize_accumulators(
                expected_origins,
                channels,
                np.dtype(np.float32),
                evaluation_horizons,
            )
            observed_origins = 0
            with torch.no_grad():
                for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    if "PEMS" in self.args.data or "Solar" in self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)
                    dec_inp = torch.zeros(
                        batch_y.shape[0],
                        self.args.pred_len,
                        batch_y.shape[-1],
                        dtype=batch_y.dtype,
                        device=self.device,
                    )
                    dec_inp = torch.cat(
                        [batch_y[:, : self.args.label_len, :], dec_inp], dim=1
                    ).float().to(self.device)
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                    else:
                        outputs, _attention = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                    f_dim = -1 if self.args.features == "MS" else 0
                    prediction = np.ascontiguousarray(
                        outputs[:, :horizon, f_dim:]
                        .detach()
                        .cpu()
                        .numpy(),
                        dtype=np.float32,
                    )
                    target = np.ascontiguousarray(
                        batch_y[:, -horizon:, f_dim:]
                        .detach()
                        .cpu()
                        .numpy(),
                        dtype=np.float32,
                    )
                    observed_origins += int(prediction.shape[0])
                    update_accumulators(
                        accumulators, prediction, target, evaluation_horizons
                    )
            if observed_origins != expected_origins:
                raise RuntimeError("SimpleTM origin-count mismatch")
            result["rows"] = finalize_rows(
                "SimpleTM",
                dataset,
                repeat,
                checkpoint_before,
                observed_origins,
                channels,
                accumulators,
                horizons=evaluation_horizons,
                loader_drop_last=bool(test_loader.drop_last),
            )
            result["origin_count"] = observed_origins
            result["loader_dataset_size"] = len(test_loader.dataset)
            result["loader_drop_last"] = bool(test_loader.drop_last)
            result["loader_batch_size"] = int(test_loader.batch_size)

        Exp_Long_Term_Forecast.test = stream_test
        effective = command[:]
        effective[effective.index("--is_training") + 1] = "0"
        effective[effective.index("--itr") + 1] = "1"
        sys.argv = ["run.py", *effective[3:]]
        runpy.run_path(str(source / "run.py"), run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        sys.path.remove(str(source))
    rows = result.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("SimpleTM streaming test did not produce rows")
    checkpoint_after = sha256(checkpoint)
    return rows, {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "checkpoint_immutable": checkpoint_before == checkpoint_after,
        "official_command": command,
        "loader_command": loader_command,
        "loader_horizon": horizon,
        "loader_dataset_size": result["loader_dataset_size"],
        "loader_drop_last": result["loader_drop_last"],
        "loader_batch_size": result["loader_batch_size"],
        "model_horizon": 720,
        "input_only_inference": loader_horizon is not None,
        "future_label_used_as_model_input": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=("AMD", "SimpleTM"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--loader-horizon", type=int, choices=HORIZONS)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = h720_record(args.run_dir, args.repeat)
    if record["baseline"] != args.baseline or record["dataset"] != args.dataset:
        raise RuntimeError("Main I record identity mismatch")
    if args.baseline == "AMD":
        if args.repeat != 0:
            raise RuntimeError("AMD has only repeat 0")
        rows, audit = evaluate_amd(
            args.run_dir, args.dataset, record, args.loader_horizon
        )
    else:
        rows, audit = evaluate_simpletm(
            args.run_dir,
            args.dataset,
            args.repeat,
            record,
            args.loader_horizon,
        )
    if not audit["checkpoint_immutable"]:
        raise RuntimeError("checkpoint mutated during formal test")
    mse_delta = mae_delta = mse_tolerance = mae_tolerance = None
    if args.loader_horizon in {None, 720}:
        mse_delta = float(rows[-1]["mse"]) - float(record["mse"])
        mae_delta = float(rows[-1]["mae"]) - float(record["mae"])
        mse_tolerance = numeric_tolerance(float(record["mse"]), args.tolerance)
        mae_tolerance = numeric_tolerance(float(record["mae"]), args.tolerance)
        if abs(mse_delta) > mse_tolerance or abs(mae_delta) > mae_tolerance:
            raise RuntimeError(
                "H720 anchor mismatch: "
                f"mse_delta={mse_delta}, mse_tolerance={mse_tolerance}, "
                f"mae_delta={mae_delta}, mae_tolerance={mae_tolerance}"
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
        "system": args.baseline,
        "dataset": args.dataset,
        "repeat": args.repeat,
        "h720_anchor_mse": float(record["mse"]),
        "h720_anchor_mae": float(record["mae"]),
        "h720_mse_delta": mse_delta,
        "h720_mae_delta": mae_delta,
        "h720_mse_tolerance": mse_tolerance,
        "h720_mae_tolerance": mae_tolerance,
        "numeric_tolerance_rule": "max(1e-8, four_float32_ULPs_of_anchor)",
        "array_retention": "streamed_not_saved",
        "run_dir": str(args.run_dir),
        "metrics_csv_sha256": sha256(args.run_dir / "metrics.csv"),
        **audit,
        "rows": rows,
    }
    (args.output_dir / "prefix_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"main_ii_{args.baseline.lower()}_eval=pass dataset={args.dataset} "
        f"repeat={args.repeat} loader_horizon={audit['loader_horizon']} "
        f"rows={len(rows)} h720_mse_delta={mse_delta if mse_delta is not None else 'n/a'} "
        f"h720_mae_delta={mae_delta if mae_delta is not None else 'n/a'}"
    )


if __name__ == "__main__":
    main()
