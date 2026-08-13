#!/usr/bin/env python3
"""Evaluate one upstream H720 checkpoint on one fixed-H official test loader."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import runpy
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


HORIZONS = (96, 192, 336, 720)
BASELINES = ("iTransformer", "PatchTST", "DLinear")
LOADER_OPTIONS = (
    "seq_len",
    "label_len",
    "pred_len",
    "batch_size",
    "test_batch_size",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_option(command: list[str], name: str, value: str) -> None:
    if name in command:
        command[command.index(name) + 1] = value
    else:
        command.extend([name, value])


def extract_horizon_command(
    workspace: Path, script_relative: str, horizon: int
) -> list[str]:
    script = workspace / script_relative
    with tempfile.TemporaryDirectory(prefix="fatst-main-ii-loader-command-") as temp:
        temp_path = Path(temp)
        capture = temp_path / "commands.jsonl"
        wrapper = temp_path / "python"
        wrapper.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "with open(os.environ['FATST_COMMAND_CAPTURE'], 'a', encoding='utf-8') as f:\n"
            "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n",
            encoding="utf-8",
        )
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
        env = dict(os.environ)
        env["PATH"] = str(temp_path) + os.pathsep + env["PATH"]
        env["FATST_COMMAND_CAPTURE"] = str(capture)
        subprocess.run(["bash", str(script)], cwd=workspace, env=env, check=True)
        commands = [json.loads(line) for line in capture.read_text().splitlines()]
    matches = []
    for command in commands:
        if command and command[0] == "-u":
            command = command[1:]
        if (
            "--pred_len" in command
            and command[command.index("--pred_len") + 1] == str(horizon)
        ):
            matches.append(command)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one H{horizon} command in {script_relative}, found {len(matches)}"
        )
    return matches[0]


def option_values(command: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for index, token in enumerate(command[:-1]):
        if token.startswith("--") and not command[index + 1].startswith("--"):
            values[token[2:].replace("-", "_")] = command[index + 1]
    return values


def cast_like(value: str, current: object) -> object:
    if isinstance(current, bool):
        return value.lower() in {"1", "true", "yes"}
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=BASELINES, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--loader-horizon", type=int, choices=HORIZONS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    config = json.loads(cli.config.read_text(encoding="utf-8"))
    spec = config["training_baselines"][cli.baseline]
    command_record = json.loads(
        (cli.training_dir / "effective_command.json").read_text(encoding="utf-8")
    )
    artifact = json.loads(
        (cli.training_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    model_command = list(command_record["command"])
    if int(model_command[model_command.index("--pred_len") + 1]) != 720:
        raise RuntimeError("training command is not H720")
    checkpoint = Path(artifact["checkpoint"])
    checkpoint_before = sha256(checkpoint)
    if checkpoint_before != artifact["checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash differs from training artifact")

    loader_command = extract_horizon_command(
        cli.workspace, spec["scripts"][cli.dataset], cli.loader_horizon
    )
    loader_values = option_values(loader_command)
    result: dict[str, Any] = {}

    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    sys.path.insert(0, str(cli.workspace))
    os.chdir(cli.workspace)
    try:
        if cli.baseline == "iTransformer":
            from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
        else:
            from exp.exp_main import Exp_Main as Exp_Long_Term_Forecast
        from data_provider.data_factory import data_provider

        def horizon_loader_test(
            self: object, _setting: str, test: int = 0
        ) -> None:
            del test
            loader_args = copy.deepcopy(self.args)
            for name in LOADER_OPTIONS:
                if name in loader_values and hasattr(loader_args, name):
                    setattr(
                        loader_args,
                        name,
                        cast_like(loader_values[name], getattr(loader_args, name)),
                    )
            loader_args.pred_len = cli.loader_horizon
            loader_args.num_workers = 0
            test_data, test_loader = data_provider(loader_args, "test")
            self.model.load_state_dict(
                torch.load(checkpoint, map_location=self.device)
            )
            self.model.eval()
            expected_origins = (
                len(test_loader) * int(test_loader.batch_size)
                if test_loader.drop_last
                else len(test_loader.dataset)
            )
            squared_error_sum = 0.0
            absolute_error_sum = 0.0
            element_count = 0
            observed_origins = 0
            prediction_digest = hashlib.sha256()
            target_digest = hashlib.sha256()
            logical_shape = (
                expected_origins,
                cli.loader_horizon,
                int(self.args.c_out),
            )
            for digest in (prediction_digest, target_digest):
                digest.update(str(logical_shape).encode("utf-8"))
                digest.update(str(np.dtype(np.float32)).encode("utf-8"))
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
                    dec_zeros = torch.zeros(
                        batch_y.shape[0],
                        int(self.args.pred_len),
                        batch_y.shape[-1],
                        dtype=batch_y.dtype,
                        device=self.device,
                    )
                    dec_inp = torch.cat(
                        [batch_y[:, : self.args.label_len, :], dec_zeros], dim=1
                    )
                    outputs = self.model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark
                    )
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    f_dim = -1 if self.args.features == "MS" else 0
                    prediction = np.ascontiguousarray(
                        outputs[:, : cli.loader_horizon, f_dim:]
                        .detach()
                        .cpu()
                        .numpy(),
                        dtype=np.float32,
                    )
                    target = np.ascontiguousarray(
                        batch_y[:, -cli.loader_horizon :, f_dim:]
                        .detach()
                        .cpu()
                        .numpy(),
                        dtype=np.float32,
                    )
                    if test_data.scale and getattr(self.args, "inverse", False):
                        shape = prediction.shape
                        prediction = test_data.inverse_transform(
                            prediction.reshape(-1, shape[-1])
                        ).reshape(shape)
                        target = test_data.inverse_transform(
                            target.reshape(-1, shape[-1])
                        ).reshape(shape)
                    if prediction.shape != target.shape:
                        raise RuntimeError("prediction/target shape mismatch")
                    error = prediction.astype(np.float64) - target.astype(np.float64)
                    squared_error_sum += float(
                        np.sum(np.square(error), dtype=np.float64)
                    )
                    absolute_error_sum += float(
                        np.sum(np.abs(error), dtype=np.float64)
                    )
                    element_count += int(error.size)
                    observed_origins += int(prediction.shape[0])
                    prediction_digest.update(prediction.tobytes())
                    target_digest.update(target.tobytes())
            if observed_origins != expected_origins:
                raise RuntimeError("origin-count mismatch")
            result.update(
                {
                    "origin_count": observed_origins,
                    "loader_dataset_size": len(test_loader.dataset),
                    "loader_drop_last": bool(test_loader.drop_last),
                    "loader_batch_size": int(test_loader.batch_size),
                    "channel_count": int(self.args.c_out),
                    "mse": squared_error_sum / element_count,
                    "mae": absolute_error_sum / element_count,
                    "prediction_sha256": prediction_digest.hexdigest(),
                    "target_sha256": target_digest.hexdigest(),
                }
            )

        Exp_Long_Term_Forecast.test = horizon_loader_test
        normalized = model_command[:]
        entrypoint_index = next(
            index
            for index, token in enumerate(normalized)
            if token.endswith(("run.py", "run_longExp.py"))
        )
        effective = normalized[entrypoint_index + 1 :]
        set_option(effective, "--is_training", "0")
        set_option(effective, "--itr", "1")
        sys.argv = [spec["entrypoint"], *effective]
        runpy.run_path(str(cli.workspace / spec["entrypoint"]), run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        sys.path.remove(str(cli.workspace))

    if not result:
        raise RuntimeError("patched formal test produced no result")
    checkpoint_after = sha256(checkpoint)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("checkpoint mutated during formal test")
    row = {
        "system": cli.baseline,
        "dataset": cli.dataset,
        "repeat": 0,
        "horizon": cli.loader_horizon,
        "mse": result["mse"],
        "mae": result["mae"],
        "metric_semantics": "global_elementwise_float64",
        "origin_count": result["origin_count"],
        "channel_count": result["channel_count"],
        "checkpoint_sha256": checkpoint_before,
        "prediction_prefix_sha256": result["prediction_sha256"],
        "target_prefix_sha256": result["target_sha256"],
        "prefix_identity": False,
        "test_role": "formal_horizon_specific_loader_H720_checkpoint_prefix",
        "loader_horizon": cli.loader_horizon,
        "loader_drop_last": result["loader_drop_last"],
        "input_only_inference": True,
        "test_tuned": False,
        "matrix_complete": False,
    }
    cli.output_dir.mkdir(parents=True, exist_ok=False)
    with (cli.output_dir / "prefix_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    payload = {
        "gate": "pass",
        "baseline": cli.baseline,
        "dataset": cli.dataset,
        "loader_horizon": cli.loader_horizon,
        "model_horizon": 720,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "checkpoint_immutable": True,
        "model_command": model_command,
        "loader_command": loader_command,
        "loader_dataset_size": result["loader_dataset_size"],
        "loader_origin_count": result["origin_count"],
        "loader_batch_size": result["loader_batch_size"],
        "loader_drop_last": result["loader_drop_last"],
        "input_only_inference": True,
        "future_label_used_as_model_input": False,
        "row": row,
    }
    (cli.output_dir / "prefix_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "main_ii_horizon_loader_upstream=pass "
        f"baseline={cli.baseline} dataset={cli.dataset} "
        f"horizon={cli.loader_horizon} origins={result['origin_count']}"
    )


if __name__ == "__main__":
    main()
