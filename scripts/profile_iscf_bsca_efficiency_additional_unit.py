#!/usr/bin/env python3
"""Profile one AMD or SimpleTM four-horizon resident service unit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import torch


HORIZONS = (96, 192, 336, 720)
CHANNELS = {
    "ETTm1": 7,
    "ETTm2": 7,
    "ETTh1": 7,
    "ETTh2": 7,
    "Weather": 21,
    "ECL": 321,
    "Solar": 137,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=("AMD", "SimpleTM"), required=True)
    parser.add_argument("--dataset", choices=tuple(CHANNELS), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metrics(run_dir: Path, baseline: str) -> list[dict[str, str]]:
    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row["baseline"] == baseline and int(row["repeat"]) == 0
    ]
    selected.sort(key=lambda row: int(row["horizon"]))
    if [int(row["horizon"]) for row in selected] != list(HORIZONS):
        raise RuntimeError("expected repeat-0 checkpoints for all four horizons")
    for row in selected:
        checkpoint = Path(row["checkpoint"])
        if sha256(checkpoint) != row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
    return selected


def option_values(command: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for index, token in enumerate(command[:-1]):
        if token.startswith("--") and not command[index + 1].startswith("--"):
            values[token[2:].replace("-", "_")] = command[index + 1]
    return values


def capture_amd_command(source: Path, script_rel: str, horizon: int) -> list[str]:
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
    commands = [
        shlex.split(line[4:])
        for line in result.stdout.splitlines()
        if line.startswith("CMD ")
    ]
    matches = [
        command
        for command in commands
        if "--pred_len" in command
        and command[command.index("--pred_len") + 1] == str(horizon)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"could not isolate AMD H{horizon} command")
    return matches[0]


def simpletm_commands(run_dir: Path) -> dict[int, list[str]]:
    commands: dict[int, list[str]] = {}
    with (run_dir / "run.log").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("COMMAND "):
                continue
            command = shlex.split(line[len("COMMAND ") :])
            horizon = int(command[command.index("--pred_len") + 1])
            commands[horizon] = command
    if set(commands) != set(HORIZONS):
        raise RuntimeError("could not isolate all SimpleTM commands")
    return commands


def load_state(model: torch.nn.Module, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(state)
    model.float().cuda().eval()


def amd_service(
    run_dir: Path, dataset: str, rows: list[dict[str, str]]
) -> tuple[list[torch.nn.Module], list[Callable[[], torch.Tensor]]]:
    source = run_dir / "workspace" / "source"
    completion = json.loads((run_dir / "complete.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(source))
    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    os.chdir(source)
    try:
        import main as amd_main
        from models.tsAMD import AMD

        models: list[torch.nn.Module] = []
        calls: list[Callable[[], torch.Tensor]] = []
        for row in rows:
            horizon = int(row["horizon"])
            command = capture_amd_command(
                source, completion["official_script"], horizon
            )
            sys.argv = ["main.py", *command[2:]]
            config = amd_main.parse_args()
            model = AMD(
                input_shape=(int(config.seq_len), CHANNELS[dataset]),
                pred_len=int(config.pred_len),
                dropout=float(config.dropout),
                n_block=int(config.n_block),
                patch=int(config.patch),
                k=int(config.mix_layer_num),
                c=int(config.mix_layer_scale),
                alpha=float(config.alpha),
                target_slice=None,
                norm=bool(config.norm),
                layernorm=bool(config.layernorm),
            )
            load_state(model, Path(row["checkpoint"]))
            x = torch.randn(
                1, int(config.seq_len), CHANNELS[dataset], device="cuda"
            )

            def call(
                current_model: torch.nn.Module = model,
                current_x: torch.Tensor = x,
            ) -> torch.Tensor:
                output, _ = current_model(current_x)
                return output

            models.append(model)
            calls.append(call)
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        sys.path.remove(str(source))
    return models, calls


def simpletm_config(command: list[str]) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "seq_len": 96,
        "pred_len": 96,
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        "d_model": 32,
        "d_ff": 32,
        "e_layers": 1,
        "factor": 1,
        "dropout": 0.1,
        "geomattn_dropout": 0.5,
        "embed": "timeF",
        "freq": "h",
        "activation": "gelu",
        "requires_grad": True,
        "wv": "db1",
        "m": 3,
        "kernel_size": None,
        "alpha": 1.0,
        "output_attention": False,
        "use_norm": 1,
    }
    integer_fields = {
        "seq_len", "pred_len", "enc_in", "dec_in", "c_out", "d_model",
        "d_ff", "e_layers", "factor", "m", "use_norm",
    }
    float_fields = {"dropout", "geomattn_dropout", "alpha"}
    for name, value in option_values(command).items():
        if name in integer_fields:
            defaults[name] = int(value)
        elif name in float_fields:
            defaults[name] = float(value)
        elif name in defaults:
            defaults[name] = value
    return SimpleNamespace(**defaults)


def simpletm_service(
    run_dir: Path, rows: list[dict[str, str]]
) -> tuple[list[torch.nn.Module], list[Callable[[], torch.Tensor]]]:
    source = run_dir / "workspace" / "source"
    commands = simpletm_commands(run_dir)
    sys.path.insert(0, str(source))
    old_cwd = Path.cwd()
    os.chdir(source)
    try:
        from model.SimpleTM import Model

        models: list[torch.nn.Module] = []
        calls: list[Callable[[], torch.Tensor]] = []
        for row in rows:
            horizon = int(row["horizon"])
            config = simpletm_config(commands[horizon])
            model = Model(config)
            load_state(model, Path(row["checkpoint"]))
            x = torch.randn(
                1, int(config.seq_len), int(config.enc_in), device="cuda"
            )

            def call(
                current_model: torch.nn.Module = model,
                current_x: torch.Tensor = x,
            ) -> torch.Tensor:
                output, _ = current_model(current_x, None, None, None)
                return output

            models.append(model)
            calls.append(call)
    finally:
        os.chdir(old_cwd)
        sys.path.remove(str(source))
    return models, calls


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = read_metrics(args.run_dir, args.baseline)
    torch.manual_seed(20260817)
    torch.cuda.manual_seed_all(20260817)
    torch.cuda.empty_cache()
    if args.baseline == "AMD":
        models, calls = amd_service(args.run_dir, args.dataset, rows)
    else:
        models, calls = simpletm_service(args.run_dir, rows)
    if len(models) != 4 or len(calls) != 4:
        raise RuntimeError("service must load four native models")
    with torch.inference_mode():
        outputs = [call() for call in calls]
        for output, horizon in zip(outputs, HORIZONS, strict=True):
            if (
                output.shape[-2] != horizon
                or output.shape[-1] != CHANNELS[args.dataset]
                or any(size != 1 for size in output.shape[:-2])
            ):
                raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")
            if not bool(torch.isfinite(output).all().item()):
                raise RuntimeError("non-finite synthetic output")
    del outputs
    torch.cuda.synchronize()
    baseline_allocated = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        outputs = [call() for call in calls]
    torch.cuda.synchronize()
    peak = int(torch.cuda.max_memory_allocated())
    payload = {
        "gate": "pass",
        "system": args.baseline,
        "dataset": args.dataset,
        "protocol_id": "ISCF-BSCA-EFFICIENCY-ACCURACY-MEMORY-STORAGE-20260817",
        "trained_model_count": 4,
        "deployment_repeat": 0,
        "total_stored_parameters": sum(
            parameter.numel()
            for model in models
            for parameter in model.parameters()
        ),
        "actual_checkpoint_bytes": sum(
            Path(row["checkpoint"]).stat().st_size for row in rows
        ),
        "peak_inference_memory_bytes": peak,
        "incremental_activation_peak_bytes": max(0, peak - baseline_allocated),
        "resident_allocated_bytes_before_service": baseline_allocated,
        "checkpoint_paths": [row["checkpoint"] for row in rows],
        "checkpoint_sha256": [row["checkpoint_sha256"] for row in rows],
        "hardware": torch.cuda.get_device_name(),
        "precision": "fp32",
        "batch_size": 1,
        "input_role": "synthetic_standardized_no_test_loader_or_labels",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"additional_efficiency_unit=pass system={args.baseline} "
        f"dataset={args.dataset} peak_mib={peak / (1024**2):.3f}"
    )


if __name__ == "__main__":
    main()
