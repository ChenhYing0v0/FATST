#!/usr/bin/env python3
"""Profile one frozen ISCF-BSCA efficiency system-dataset service unit."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (96, 192, 336, 720)
SYSTEMS = (
    "ISCF-BSCA",
    "TimeAlign",
    "QDF",
    "DLinear-H720-prefix",
    "PatchTST-H720-prefix",
)
WORKSPACE_ROOT = Path(
    "/home/yingch/exp_outputs/r-2026-fatst/main_ii_h720_prefix_20260808/"
    "_workspaces_v3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def option_values(command: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for index, token in enumerate(command[:-1]):
        if token.startswith("--") and not command[index + 1].startswith("--"):
            values[token[2:].replace("-", "_")] = command[index + 1]
    return values


def load_state(model: torch.nn.Module, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(state)
    model.float().cuda().eval()


def timealign_service(
    rows: list[dict[str, object]],
) -> tuple[list[torch.nn.Module], list[Callable[[], torch.Tensor]]]:
    root = REPO_ROOT / "baselines" / "timealign_official"
    sys.path.insert(0, str(root))
    from models import TimeAlign  # type: ignore

    models: list[torch.nn.Module] = []
    calls: list[Callable[[], torch.Tensor]] = []
    for row in rows:
        payload = json.loads(
            Path(str(row["effective_config_path"])).read_text(encoding="utf-8")
        )
        config = SimpleNamespace(**payload["official_args"])
        config.device = "cuda"
        model = TimeAlign.Model(config)
        load_state(model, Path(str(row["checkpoint_path"])))
        x = torch.randn(
            1, int(config.seq_len), int(config.enc_in), device="cuda"
        )
        y = torch.zeros(
            1, int(config.pred_len), int(config.c_out), device="cuda"
        )

        def call(
            current_model: torch.nn.Module = model,
            current_x: torch.Tensor = x,
            current_y: torch.Tensor = y,
        ) -> torch.Tensor:
            output = current_model(current_x, current_y, is_training=False)
            return output[0] if isinstance(output, tuple) else output

        models.append(model)
        calls.append(call)
    return models, calls


def qdf_service(
    rows: list[dict[str, object]],
) -> tuple[list[torch.nn.Module], list[Callable[[], torch.Tensor]]]:
    root = REPO_ROOT / "baselines" / "qdf_official"
    sys.path.insert(0, str(root))
    from models import TQNet  # type: ignore

    models: list[torch.nn.Module] = []
    calls: list[Callable[[], torch.Tensor]] = []
    for row in rows:
        raw = yaml.safe_load(
            Path(str(row["effective_config_path"])).read_text(encoding="utf-8")
        )
        config = SimpleNamespace(**raw)
        model = TQNet.Model(config)
        load_state(model, Path(str(row["checkpoint_path"])))
        x = torch.randn(
            1, int(config.seq_len), int(config.enc_in), device="cuda"
        )
        x_mark = torch.zeros(1, int(config.seq_len), 4, device="cuda")
        decoder = torch.zeros(
            1,
            int(config.label_len) + int(config.pred_len),
            int(config.dec_in),
            device="cuda",
        )
        y_mark = torch.zeros(
            1, int(config.label_len) + int(config.pred_len), 4, device="cuda"
        )
        cycle = torch.zeros(1, dtype=torch.long, device="cuda")

        def call(
            current_model: torch.nn.Module = model,
            current_x: torch.Tensor = x,
            current_x_mark: torch.Tensor = x_mark,
            current_decoder: torch.Tensor = decoder,
            current_y_mark: torch.Tensor = y_mark,
            current_cycle: torch.Tensor = cycle,
        ) -> torch.Tensor:
            return current_model(
                current_x,
                current_x_mark,
                current_decoder,
                current_y_mark,
                current_cycle,
            )

        models.append(model)
        calls.append(call)
    return models, calls


def upstream_service(
    system: str, row: dict[str, object]
) -> tuple[list[torch.nn.Module], list[Callable[[], torch.Tensor]]]:
    baseline = system.removesuffix("-H720-prefix")
    workspace = WORKSPACE_ROOT / baseline
    sys.path.insert(0, str(workspace))
    command_path = Path(str(row["training_log_path"])).parent / "effective_command.json"
    command = json.loads(command_path.read_text(encoding="utf-8"))["command"]
    values = option_values(command)
    defaults: dict[str, object] = {
        "seq_len": 96,
        "pred_len": 96,
        "enc_in": 7,
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
    integer_fields = {
        "seq_len",
        "pred_len",
        "enc_in",
        "e_layers",
        "n_heads",
        "d_model",
        "d_ff",
        "individual",
        "patch_len",
        "stride",
        "revin",
        "affine",
        "subtract_last",
        "decomposition",
        "kernel_size",
    }
    float_fields = {"dropout", "fc_dropout", "head_dropout"}
    for name, value in values.items():
        if name in integer_fields:
            defaults[name] = int(value)
        elif name in float_fields:
            defaults[name] = float(value)
        elif name in defaults:
            defaults[name] = value
    config = SimpleNamespace(**defaults)
    if baseline == "DLinear":
        from models import DLinear as model_module  # type: ignore
    else:
        from models import PatchTST as model_module  # type: ignore
    model = model_module.Model(config)
    load_state(model, Path(str(row["checkpoint_path"])))
    x = torch.randn(1, config.seq_len, config.enc_in, device="cuda")

    def call() -> torch.Tensor:
        return model(x)

    return [model], [call, call, call, call]


def load_service(
    system: str, rows: list[dict[str, object]]
) -> tuple[list[torch.nn.Module], list[Callable[[], torch.Tensor]]]:
    if system in {"ISCF-BSCA", "TimeAlign"}:
        models, calls = timealign_service(rows)
        if system == "ISCF-BSCA":
            return models, [calls[0], calls[0], calls[0], calls[0]]
        return models, calls
    if system == "QDF":
        return qdf_service(rows)
    return upstream_service(system, rows[0])


def measure(
    call: Callable[[], torch.Tensor], warmup: int, rounds: int, iterations: int
) -> dict[str, object]:
    with torch.inference_mode():
        for _ in range(warmup):
            call()
        torch.cuda.synchronize()
        round_means: list[float] = []
        samples: list[float] = []
        for _ in range(rounds):
            current: list[float] = []
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                call()
                end.record()
                end.synchronize()
                current.append(float(start.elapsed_time(end)))
            samples.extend(current)
            round_means.append(statistics.fmean(current))
    median = statistics.median(round_means)
    p95 = float(np.percentile(np.asarray(samples, dtype=np.float64), 95))
    cv = statistics.pstdev(round_means) / statistics.fmean(round_means)
    return {
        "latency_ms": median,
        "p95_iteration_latency_ms": p95,
        "round_cv": cv,
        "round_means_ms": round_means,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen efficiency protocol")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [
        row
        for row in manifest["rows"]
        if row["system"] == args.system and row["dataset"] == args.dataset
    ]
    expected = 4 if args.system in {"TimeAlign", "QDF"} else 1
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} manifest rows, got {len(rows)}")
    rows.sort(key=lambda row: 0 if row["horizon"] == "all" else int(row["horizon"]))

    torch.manual_seed(20260814)
    torch.cuda.manual_seed_all(20260814)
    torch.cuda.empty_cache()
    models, calls = load_service(args.system, rows)
    if len(calls) != 4:
        raise RuntimeError("service must expose four requested horizons")
    with torch.inference_mode():
        for call in calls:
            output = call()
            if output.ndim != 3 or not bool(torch.isfinite(output).all().item()):
                raise RuntimeError("invalid synthetic profiler output")
    torch.cuda.synchronize()

    params = sum(parameter.numel() for model in models for parameter in model.parameters())
    checkpoint_bytes = sum(int(row["checkpoint_bytes"]) for row in rows)
    baseline_allocated = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()

    def all_horizon_call() -> torch.Tensor:
        if args.system in {"TimeAlign", "QDF"}:
            outputs = [call() for call in calls]
            return outputs[-1]
        output = calls[-1]()
        _prefixes = tuple(output[:, :horizon, :] for horizon in HORIZONS)
        return _prefixes[-1]

    with torch.inference_mode():
        all_horizon_call()
    torch.cuda.synchronize()
    total_peak = int(torch.cuda.max_memory_allocated())
    incremental_peak = max(0, total_peak - baseline_allocated)

    request_results = [
        measure(call, args.warmup, args.rounds, args.iterations) for call in calls
    ]
    service_result = measure(
        all_horizon_call, args.warmup, args.rounds, args.iterations
    )
    single_latency = statistics.fmean(
        float(result["latency_ms"]) for result in request_results
    )
    if not all(
        math.isfinite(float(value))
        for value in (
            single_latency,
            service_result["latency_ms"],
            service_result["round_cv"],
        )
    ):
        raise RuntimeError("non-finite profiler statistic")
    payload = {
        "gate": (
            "pass"
            if float(service_result["round_cv"]) <= 0.10
            and all(float(item["round_cv"]) <= 0.10 for item in request_results)
            else "remeasure_required"
        ),
        "system": args.system,
        "dataset": args.dataset,
        "protocol_id": manifest["protocol_id"],
        "manifest_path": str(args.manifest),
        "checkpoint_object_ids": [row["object_id"] for row in rows],
        "checkpoint_sha256": [row["checkpoint_sha256"] for row in rows],
        "trained_model_count": len(models),
        "total_stored_parameters": params,
        "actual_checkpoint_bytes": checkpoint_bytes,
        "single_request_latency_ms": single_latency,
        "request_latency_by_horizon": {
            str(horizon): result
            for horizon, result in zip(HORIZONS, request_results, strict=True)
        },
        "all_horizon_service": service_result,
        "peak_inference_memory_bytes": total_peak,
        "incremental_activation_peak_bytes": incremental_peak,
        "resident_allocated_bytes_before_service": baseline_allocated,
        "hardware": torch.cuda.get_device_name(),
        "precision": "fp32",
        "batch_size": 1,
        "input_role": "synthetic_standardized_no_test_loader_or_labels",
        "warmup_iterations": args.warmup,
        "timed_rounds": args.rounds,
        "iterations_per_round": args.iterations,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"efficiency_unit={payload['gate']} system={args.system} "
        f"dataset={args.dataset} service_ms={service_result['latency_ms']:.6f}"
    )


if __name__ == "__main__":
    main()
