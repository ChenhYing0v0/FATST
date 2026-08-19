#!/usr/bin/env python3
"""Profile one four-horizon baseline service from official architecture configs."""

from __future__ import annotations

import argparse
import hashlib
import io
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
SOURCE_ROOTS = {
    "DLinear": Path(
        "/home/yingch/exp_outputs/r-2026-fatst/main_ii_h720_prefix_20260808/"
        "_workspaces_v3/DLinear"
    ),
    "PatchTST": Path(
        "/home/yingch/exp_outputs/r-2026-fatst/main_ii_h720_prefix_20260808/"
        "_workspaces_v3/PatchTST"
    ),
    "iTransformer": Path("/home/yingch/iTransformer"),
    "TimeMixer": Path("/home/yingch/TimeMixer"),
}
SCRIPT_PATHS = {
    "DLinear": {
        "ETTm1": "scripts/EXP-LongForecasting/Linear/ettm1.sh",
        "ETTm2": "scripts/EXP-LongForecasting/Linear/ettm2.sh",
        "ETTh1": "scripts/EXP-LongForecasting/Linear/etth1.sh",
        "ETTh2": "scripts/EXP-LongForecasting/Linear/etth2.sh",
        "Weather": "scripts/EXP-LongForecasting/Linear/weather.sh",
        "ECL": "scripts/EXP-LongForecasting/Linear/electricity.sh",
        "Solar": "scripts/EXP-LongForecasting/Linear/electricity.sh",
    },
    "PatchTST": {
        "ETTm1": "scripts/PatchTST/ettm1.sh",
        "ETTm2": "scripts/PatchTST/ettm2.sh",
        "ETTh1": "scripts/PatchTST/etth1.sh",
        "ETTh2": "scripts/PatchTST/etth2.sh",
        "Weather": "scripts/PatchTST/weather.sh",
        "ECL": "scripts/PatchTST/electricity.sh",
        "Solar": "scripts/PatchTST/electricity.sh",
    },
    "iTransformer": {
        "ETTm1": "scripts/multivariate_forecasting/ETT/iTransformer_ETTm1.sh",
        "ETTm2": "scripts/multivariate_forecasting/ETT/iTransformer_ETTm2.sh",
        "ETTh1": "scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh",
        "ETTh2": "scripts/multivariate_forecasting/ETT/iTransformer_ETTh2.sh",
        "Weather": "scripts/multivariate_forecasting/Weather/iTransformer.sh",
        "ECL": "scripts/multivariate_forecasting/ECL/iTransformer.sh",
        "Solar": "scripts/multivariate_forecasting/SolarEnergy/iTransformer.sh",
    },
    "TimeMixer": {
        "ETTm1": "scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1_unify.sh",
        "ETTm2": "scripts/long_term_forecast/ETT_script/TimeMixer_ETTm2_unify.sh",
        "ETTh1": "scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_unify.sh",
        "ETTh2": "scripts/long_term_forecast/ETT_script/TimeMixer_ETTh2_unify.sh",
        "Weather": "scripts/long_term_forecast/Weather_script/TimeMixer_unify.sh",
        "ECL": "scripts/long_term_forecast/ECL_script/TimeMixer_unify.sh",
        "Solar": "scripts/long_term_forecast/Solar_script/TimeMixer_unify.sh",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=tuple(SOURCE_ROOTS), required=True)
    parser.add_argument("--dataset", choices=tuple(CHANNELS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def option_values(command: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for index, token in enumerate(command[:-1]):
        if token.startswith("--") and not command[index + 1].startswith("--"):
            values[token[2:].replace("-", "_")] = command[index + 1]
    return values


def capture_commands(source: Path, script_rel: str) -> dict[int, list[str]]:
    # FD 3 bypasses redirections attached to the shell's original python calls.
    shim = (
        'exec 3>&1; python(){ printf "CMD" >&3; printf " %q" "$@" >&3; '
        'printf "\\n" >&3; }; export -f python; bash "$1"'
    )
    result = subprocess.run(
        ["bash", "-c", shim, "capture", script_rel],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    commands: dict[int, list[str]] = {}
    for line in result.stdout.splitlines():
        if not line.startswith("CMD "):
            continue
        command = shlex.split(line[4:])
        values = option_values(command)
        if "pred_len" in values and int(values["pred_len"]) in HORIZONS:
            commands[int(values["pred_len"])] = command
    if set(commands) != set(HORIZONS):
        raise RuntimeError(
            f"could not isolate four horizon commands from {source / script_rel}"
        )
    return commands


def typed_config(command: list[str], defaults: dict[str, object]) -> SimpleNamespace:
    values = option_values(command)
    for name, original in tuple(defaults.items()):
        if name not in values:
            continue
        raw = values[name]
        if isinstance(original, int):
            defaults[name] = int(raw)
        elif isinstance(original, float):
            defaults[name] = float(raw)
        else:
            defaults[name] = raw
    return SimpleNamespace(**defaults)


def config_for(
    baseline: str, dataset: str, horizon: int, command: list[str]
) -> SimpleNamespace:
    channels = CHANNELS[dataset]
    if baseline in {"DLinear", "PatchTST"}:
        defaults: dict[str, object] = {
            "seq_len": 336,
            "pred_len": horizon,
            "enc_in": channels,
            "e_layers": 2,
            "n_heads": 8,
            "d_model": 512,
            "d_ff": 2048,
            "dropout": 0.05,
            "fc_dropout": 0.05,
            "head_dropout": 0.0,
            "individual": 0,
            "patch_len": 16,
            "stride": 8,
            "padding_patch": "end",
            "revin": 1,
            "affine": 0,
            "subtract_last": 0,
            "decomposition": 0,
            "kernel_size": 25,
        }
        config = typed_config(command, defaults)
        # The upstream repositories have no Solar script. Preserve the frozen
        # ECL-adjacent architecture while applying Solar's channel contract.
        if dataset == "Solar":
            config.enc_in = channels
            config.seq_len = 336
            config.pred_len = horizon
        return config
    if baseline == "iTransformer":
        return typed_config(
            command,
            {
                "seq_len": 96,
                "pred_len": horizon,
                "enc_in": channels,
                "output_attention": 0,
                "use_norm": 1,
                "d_model": 512,
                "embed": "timeF",
                "freq": "h",
                "dropout": 0.1,
                "factor": 1,
                "n_heads": 8,
                "d_ff": 2048,
                "activation": "gelu",
                "e_layers": 2,
                "class_strategy": "projection",
            },
        )
    return typed_config(
        command,
        {
            "task_name": "long_term_forecast",
            "seq_len": 96,
            "label_len": 0,
            "pred_len": horizon,
            "down_sampling_window": 2,
            "down_sampling_layers": 3,
            "down_sampling_method": "avg",
            "channel_independence": 1,
            "e_layers": 2,
            "d_model": 16,
            "d_ff": 32,
            "dropout": 0.1,
            "decomp_method": "moving_avg",
            "moving_avg": 25,
            "top_k": 5,
            "enc_in": channels,
            "c_out": channels,
            "use_future_temporal_feature": 0,
            "embed": "timeF",
            "freq": "h",
            "use_norm": 1,
        },
    )


def instantiate(
    baseline: str, source: Path, config: SimpleNamespace
) -> torch.nn.Module:
    sys.path.insert(0, str(source))
    old_cwd = Path.cwd()
    os.chdir(source)
    try:
        if baseline == "DLinear":
            from models import DLinear as model_module  # type: ignore
        elif baseline == "PatchTST":
            from models import PatchTST as model_module  # type: ignore
        elif baseline == "iTransformer":
            from model import iTransformer as model_module  # type: ignore
        else:
            from models import TimeMixer as model_module  # type: ignore
        return model_module.Model(config)
    finally:
        os.chdir(old_cwd)


def serialized_state_dict_bytes(model: torch.nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    source = SOURCE_ROOTS[args.baseline]
    script_rel = SCRIPT_PATHS[args.baseline][args.dataset]
    commands = capture_commands(source, script_rel)
    torch.manual_seed(20260819)
    torch.cuda.manual_seed_all(20260819)
    torch.cuda.empty_cache()

    models: list[torch.nn.Module] = []
    calls: list[Callable[[], torch.Tensor]] = []
    configs: dict[str, dict[str, object]] = {}
    checkpoint_bytes: dict[str, int] = {}
    parameter_counts: dict[str, int] = {}
    for horizon in HORIZONS:
        config = config_for(args.baseline, args.dataset, horizon, commands[horizon])
        model = instantiate(args.baseline, source, config)
        checkpoint_bytes[str(horizon)] = serialized_state_dict_bytes(model)
        parameter_counts[str(horizon)] = sum(
            parameter.numel() for parameter in model.parameters()
        )
        model.float().cuda().eval()
        x = torch.randn(1, int(config.seq_len), int(config.enc_in), device="cuda")

        def call(
            current_model: torch.nn.Module = model,
            current_x: torch.Tensor = x,
            current_baseline: str = args.baseline,
        ) -> torch.Tensor:
            if current_baseline in {"DLinear", "PatchTST"}:
                return current_model(current_x)
            return current_model(current_x, None, None, None)

        models.append(model)
        calls.append(call)
        configs[str(horizon)] = vars(config)

    with torch.inference_mode():
        outputs = [call() for call in calls]
        for output, horizon in zip(outputs, HORIZONS, strict=True):
            if output.shape != (1, horizon, CHANNELS[args.dataset]):
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

    script = source / script_rel
    payload = {
        "gate": "pass",
        "system": args.baseline,
        "dataset": args.dataset,
        "protocol_id": "ISCF-BSCA-EFFICIENCY-ACCURACY-MEMORY-STORAGE-20260817",
        "trained_model_count": 4,
        "resident_model_count": 4,
        "resource_evidence_role": (
            "official_architecture_equivalent_untrained_state_dict_serialization"
        ),
        "checkpoint_bytes_semantics": (
            "standard_torch_save_state_dict_equivalent_not_trained_artifact"
        ),
        "total_stored_parameters": sum(parameter_counts.values()),
        "actual_checkpoint_bytes": sum(checkpoint_bytes.values()),
        "checkpoint_equivalent_bytes_by_horizon": checkpoint_bytes,
        "parameter_counts_by_horizon": parameter_counts,
        "peak_inference_memory_bytes": peak,
        "incremental_activation_peak_bytes": max(0, peak - baseline_allocated),
        "resident_allocated_bytes_before_service": baseline_allocated,
        "effective_configs": configs,
        "official_commands": {
            str(horizon): commands[horizon] for horizon in HORIZONS
        },
        "official_source_root": str(source),
        "official_script": script_rel,
        "official_script_sha256": sha256(script),
        "solar_source_patch": (
            "ECL_architecture_with_Solar_channel_contract"
            if args.dataset == "Solar" and args.baseline in {"DLinear", "PatchTST"}
            else None
        ),
        "hardware": torch.cuda.get_device_name(),
        "precision": "fp32",
        "batch_size": 1,
        "input_role": "synthetic_standardized_no_test_loader_or_labels",
        "random_seed": 20260819,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"official_architecture_unit=pass system={args.baseline} "
        f"dataset={args.dataset} peak_mib={peak / (1024**2):.3f} "
        f"checkpoint_mib={sum(checkpoint_bytes.values()) / (1024**2):.3f}"
    )


if __name__ == "__main__":
    main()
