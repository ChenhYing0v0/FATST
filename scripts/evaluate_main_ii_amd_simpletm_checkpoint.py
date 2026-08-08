#!/usr/bin/env python3
"""Stream Main II prefixes from frozen AMD or SimpleTM H720 checkpoints."""

from __future__ import annotations

import argparse
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
    origins: int, channels: int, dtype: np.dtype
) -> dict[int, dict[str, Any]]:
    accumulators: dict[int, dict[str, Any]] = {}
    for horizon in HORIZONS:
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
) -> None:
    if prediction.shape != target.shape or prediction.shape[1] != 720:
        raise RuntimeError(f"unexpected tensor shapes: {prediction.shape}, {target.shape}")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise RuntimeError("non-finite formal-test tensor")
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
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        accumulator = accumulators[horizon]
        count = int(accumulator["element_count"])
        global_mse = float(accumulator["squared_error_sum"]) / count
        global_mae = float(accumulator["absolute_error_sum"]) / count
        native_count = int(accumulator["native_batch_count"])
        native_mse = float(accumulator["native_batch_mse_sum"]) / native_count
        native_mae = float(accumulator["native_batch_mae_sum"]) / native_count
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
                "prefix_identity": True,
                "test_role": "formal_H720_native_loader_exact_prefix_stream",
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


def capture_amd_h720_command(source: Path, script_rel: str) -> list[str]:
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
        and command[command.index("--pred_len") + 1] == "720"
    ]
    if len(matches) != 1:
        raise RuntimeError("could not isolate the AMD H720 official command")
    return matches[0]


def evaluate_amd(
    run_dir: Path, dataset: str, record: dict[str, str]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    source = run_dir / "workspace" / "source"
    completion = json.loads((run_dir / "complete.json").read_text(encoding="utf-8"))
    command = capture_amd_h720_command(source, completion["official_script"])
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

        sys.argv = ["main.py", *command[2:]]
        args = amd_main.parse_args()
        data_loader = CustomDataLoader(
            args.data,
            args.batch_size,
            args.seq_len,
            args.pred_len,
            args.feature_type,
            args.target,
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
        accumulators = initialize_accumulators(
            expected_origins, channels, np.dtype(np.float32)
        )
        observed_origins = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                prediction, _moe_loss = model(batch_x.cuda())
                pred_np = np.ascontiguousarray(
                    prediction.detach().cpu().numpy(), dtype=np.float32
                )
                true_np = np.ascontiguousarray(batch_y.numpy(), dtype=np.float32)
                observed_origins += int(pred_np.shape[0])
                update_accumulators(accumulators, pred_np, true_np)
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        sys.path.remove(str(source))
    if observed_origins != expected_origins:
        raise RuntimeError("AMD origin-count mismatch")
    rows = finalize_rows(
        "AMD", dataset, 0, checkpoint_before, observed_origins, channels, accumulators
    )
    checkpoint_after = sha256(checkpoint)
    return rows, {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "checkpoint_immutable": checkpoint_before == checkpoint_after,
        "official_command": command,
    }


def simpletm_h720_command(run_dir: Path) -> list[str]:
    commands = []
    with (run_dir / "run.log").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("COMMAND "):
                commands.append(shlex.split(line[len("COMMAND ") :]))
    matches = [
        command
        for command in commands
        if "--pred_len" in command
        and command[command.index("--pred_len") + 1] == "720"
    ]
    if len(matches) != 1:
        raise RuntimeError("could not isolate the SimpleTM H720 official command")
    return matches[0]


def evaluate_simpletm(
    run_dir: Path, dataset: str, repeat: int, record: dict[str, str]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    source = run_dir / "workspace" / "source"
    checkpoint = Path(record["checkpoint"])
    checkpoint_before = sha256(checkpoint)
    if checkpoint_before != record["checkpoint_sha256"]:
        raise RuntimeError("SimpleTM checkpoint hash mismatch")
    command = simpletm_h720_command(run_dir)
    result: dict[str, object] = {}

    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    sys.path.insert(0, str(source))
    os.chdir(source)
    try:
        from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast

        def stream_test(self: object, _setting: str, test: int = 0) -> None:
            del test
            test_data, test_loader = self._get_data(flag="test")
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
            self.model.eval()
            expected_origins = (
                len(test_loader) * int(test_loader.batch_size)
                if test_loader.drop_last
                else len(test_loader.dataset)
            )
            channels = int(self.args.c_out)
            accumulators = initialize_accumulators(
                expected_origins, channels, np.dtype(np.float32)
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
                    dec_inp = torch.zeros_like(
                        batch_y[:, -self.args.pred_len :, :]
                    ).float()
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
                        outputs[:, -self.args.pred_len :, f_dim:]
                        .detach()
                        .cpu()
                        .numpy(),
                        dtype=np.float32,
                    )
                    target = np.ascontiguousarray(
                        batch_y[:, -self.args.pred_len :, f_dim:]
                        .detach()
                        .cpu()
                        .numpy(),
                        dtype=np.float32,
                    )
                    observed_origins += int(prediction.shape[0])
                    update_accumulators(accumulators, prediction, target)
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
            )
            result["origin_count"] = observed_origins

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
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=("AMD", "SimpleTM"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
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
        rows, audit = evaluate_amd(args.run_dir, args.dataset, record)
    else:
        rows, audit = evaluate_simpletm(
            args.run_dir, args.dataset, args.repeat, record
        )
    if not audit["checkpoint_immutable"]:
        raise RuntimeError("checkpoint mutated during formal test")
    mse_delta = float(rows[-1]["mse"]) - float(record["mse"])
    mae_delta = float(rows[-1]["mae"]) - float(record["mae"])
    if abs(mse_delta) > args.tolerance or abs(mae_delta) > args.tolerance:
        raise RuntimeError(
            f"H720 anchor mismatch: mse_delta={mse_delta}, mae_delta={mae_delta}"
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
        f"repeat={args.repeat} rows=4 h720_mse_delta={mse_delta:.3e} "
        f"h720_mae_delta={mae_delta:.3e}"
    )


if __name__ == "__main__":
    main()
