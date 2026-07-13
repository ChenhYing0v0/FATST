#!/usr/bin/env python3
"""Run the StageC PMFO-RCT Step 7A local implementation gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from layers.PMFO import (  # noqa: E402
    PMFO_BLOCK_SIZES,
    PMFO_RADICES,
    PMFORCTReadout,
)
from models import TimeAlign  # noqa: E402


HORIZONS = (1, 48, 96, 192, 336, 720)
VARIANTS = (
    "learned-basis-forecast-operator",
    "pmfo-rct",
    "pmfo-rct-no-transition",
    "pmfo-rct-no-conservation",
    "dense-mlp-matched",
)
PROFILE_CONFIGS = {
    "Weather": {
        "profile": "r2b_p12_d64_ff128_medium",
        "patch_num": 12,
        "d_model": 64,
        "d_ff": 128,
        "channels": 21,
    },
    "ETTm1": {
        "profile": "r2b_p24_d32_ff64_narrow",
        "patch_num": 24,
        "d_model": 32,
        "d_ff": 64,
        "channels": 7,
    },
    "ETTh2": {
        "profile": "r2b_p12_d64_ff128_medium",
        "patch_num": 12,
        "d_model": 64,
        "d_ff": 128,
        "channels": 7,
    },
}
TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/stage_c_step7a_pmfo_rct_local_20260713"),
    )
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def build_config(dataset: str, readout_mode: str) -> SimpleNamespace:
    profile = PROFILE_CONFIGS[dataset]
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        encoder_mode="timealign-token-mlp",
        readout_mode=readout_mode,
        e_layers=2,
        patch_num=profile["patch_num"],
        d_model=profile["d_model"],
        d_ff=profile["d_ff"],
        dropout=0.1,
        pos=1,
        layer_norm=1,
        enc_in=profile["channels"],
        basis_rank=256,
        pmfo_state_dim=32,
        pmfo_dense_hidden_dim=144,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def instantiate_models() -> dict[tuple[str, str], TimeAlign.Model]:
    models = {}
    for dataset in PROFILE_CONFIGS:
        for variant in VARIANTS:
            model = TimeAlign.Model(build_config(dataset, variant)).float().eval()
            models[(dataset, variant)] = model
    return models


def shape_and_prefix_rows(
    models: dict[tuple[str, str], TimeAlign.Model],
    generator: torch.Generator,
) -> tuple[list[dict[str, Any]], float]:
    rows = []
    maximum_gap = 0.0
    for dataset, profile in PROFILE_CONFIGS.items():
        channels = profile["channels"]
        x = torch.randn(1, 720, channels, generator=generator)
        y = torch.zeros(1, 720, channels)
        for variant in VARIANTS:
            model = models[(dataset, variant)]
            with torch.no_grad():
                full = model(x, y, is_training=False)[0]
                if full.shape != (1, 720, channels):
                    raise AssertionError(
                        f"unexpected full shape for {dataset}/{variant}: {full.shape}"
                    )
                for horizon in HORIZONS:
                    prefix = model(
                        x,
                        y,
                        is_training=False,
                        target_prefix=horizon,
                    )[0]
                    expected_shape = (1, horizon, channels)
                    if prefix.shape != expected_shape:
                        raise AssertionError(
                            f"unexpected shape for {dataset}/{variant}/H{horizon}: "
                            f"{prefix.shape}"
                        )
                    gap = float((prefix - full[:, :horizon]).abs().max())
                    maximum_gap = max(maximum_gap, gap)
                    if gap > TOLERANCE:
                        raise AssertionError(
                            f"prefix gap {gap} exceeds {TOLERANCE} for "
                            f"{dataset}/{variant}/H{horizon}"
                        )
                    rows.append(
                        {
                            "dataset": dataset,
                            "profile": profile["profile"],
                            "variant": variant,
                            "horizon": horizon,
                            "output_shape": str(list(prefix.shape)),
                            "full_prefix_max_abs": gap,
                            "pass": True,
                        }
                    )
    return rows, maximum_gap


def refinement_metrics(generator: torch.Generator) -> dict[str, float]:
    readout = PMFORCTReadout(readout_dim=768, state_dim=32, conservative=True)
    synthesis = readout.synthesis
    recovery_errors = []
    conservation_errors = []
    for level, radix in enumerate(PMFO_RADICES):
        parent = torch.randn(5, 11, generator=generator)
        detail = torch.randn(5, 11, radix - 1, generator=generator)
        children = synthesis.refine(parent, detail, level)
        scaling = getattr(synthesis, f"scaling_{level}")
        contrast = getattr(synthesis, f"contrast_{level}")
        recovered_parent = torch.einsum("bnr,r->bn", children, scaling)
        recovered_detail = torch.einsum("bnr,rd->bnd", children, contrast)
        recovery_errors.extend(
            [
                float((recovered_parent - parent).abs().max()),
                float((recovered_detail - detail).abs().max()),
            ]
        )
        perturbed = detail.clone()
        perturbed[..., 0] += 3.0
        changed_children = synthesis.refine(parent, perturbed, level)
        changed_parent = torch.einsum("bnr,r->bn", changed_children, scaling)
        conservation_errors.append(
            float((changed_parent - recovered_parent).abs().max())
        )

    refinement_error = max(recovery_errors)
    conservation_error = max(conservation_errors)
    if refinement_error > TOLERANCE:
        raise AssertionError(f"refinement recovery failed: {refinement_error}")
    if conservation_error > TOLERANCE:
        raise AssertionError(f"conservation failed: {conservation_error}")
    return {
        "refinement_recovery_max_abs": refinement_error,
        "conservation_perturbation_max_abs": conservation_error,
    }


def locality_error(generator: torch.Generator) -> float:
    synthesis = PMFORCTReadout(
        readout_dim=768,
        state_dim=32,
        conservative=True,
    ).synthesis
    coarse = torch.randn(1, 1, 8, generator=generator)
    parent_counts = [720 // block_size for block_size in PMFO_BLOCK_SIZES[:-1]]
    details = tuple(
        torch.randn(1, 1, parent_count, radix - 1, generator=generator)
        for parent_count, radix in zip(parent_counts, PMFO_RADICES, strict=True)
    )
    baseline = synthesis(coarse, details, 720)
    outside_errors = []
    for level, block_size in enumerate(PMFO_BLOCK_SIZES[:-1]):
        parent_index = parent_counts[level] // 2
        changed = [detail.clone() for detail in details]
        changed[level][0, 0, parent_index, 0] += 2.0
        perturbed = synthesis(coarse, tuple(changed), 720)
        delta = (perturbed - baseline).abs()
        start = parent_index * block_size
        end = start + block_size
        outside = torch.cat([delta[..., :start], delta[..., end:]], dim=-1)
        outside_errors.append(float(outside.max()) if outside.numel() else 0.0)
    maximum = max(outside_errors)
    if maximum > TOLERANCE:
        raise AssertionError(f"locality failed: {maximum}")
    return maximum


def horizon_path_audit(model: TimeAlign.Model) -> dict[str, Any]:
    learned_inputs_are_tensors = True
    observed_calls = 0
    hooks = []

    def inspect_inputs(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
        nonlocal learned_inputs_are_tensors, observed_calls
        observed_calls += 1
        learned_inputs_are_tensors = learned_inputs_are_tensors and all(
            isinstance(value, torch.Tensor) for value in inputs
        )

    for module in model.pmfo_readout.modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_pre_hook(inspect_inputs))
    x = torch.randn(1, 720, 7)
    y = torch.zeros(1, 720, 7)
    with torch.no_grad():
        model(x, y, is_training=False, target_prefix=48)
    for hook in hooks:
        hook.remove()

    forbidden_norms = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.GroupNorm,
        nn.LayerNorm,
        nn.SyncBatchNorm,
    )
    normalization_modules = [
        type(module).__name__
        for module in model.pmfo_readout.modules()
        if isinstance(module, forbidden_norms)
    ]
    passed = (
        observed_calls > 0
        and learned_inputs_are_tensors
        and not normalization_modules
    )
    if not passed:
        raise AssertionError(
            "horizon path audit failed: learned modules received non-tensors or "
            "node-count normalization was found"
        )
    return {
        "learned_module_calls": observed_calls,
        "learned_inputs_tensor_only": learned_inputs_are_tensors,
        "node_count_normalization_modules": normalization_modules,
        "pass": passed,
    }


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    if hasattr(model, "pmfo_readout"):
        decoder = sum(
            parameter.numel() for parameter in model.pmfo_readout.parameters()
        )
    else:
        decoder = (
            model.learned_basis_coeff.weight.numel()
            + model.learned_basis_coeff.bias.numel()
            + model.learned_temporal_basis.numel()
            + model.learned_temporal_bias.numel()
        )
    return total, decoder


def count_active_forward_parameters(
    model: TimeAlign.Model,
    channels: int,
) -> tuple[int, int]:
    model.zero_grad(set_to_none=True)
    x = torch.randn(1, 720, channels)
    y = torch.zeros(1, 720, channels)
    output = model(x, y, is_training=False, target_prefix=720)[0]
    output.square().mean().backward()
    active = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    if hasattr(model, "pmfo_readout"):
        decoder = sum(
            parameter.numel()
            for parameter in model.pmfo_readout.parameters()
            if parameter.grad is not None
        )
    else:
        decoder_names = {
            "learned_basis_coeff.weight",
            "learned_basis_coeff.bias",
            "learned_temporal_basis",
            "learned_temporal_bias",
        }
        decoder = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name in decoder_names and parameter.grad is not None
        )
    model.zero_grad(set_to_none=True)
    return active, decoder


def linear_dominant_flops(
    model: TimeAlign.Model,
    channels: int,
    horizon: int,
) -> int:
    linear_macs = 0
    hooks = []

    def count_linear(
        module: nn.Linear,
        inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        nonlocal linear_macs
        vectors = inputs[0].numel() // module.in_features
        linear_macs += vectors * module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(count_linear))
    x = torch.randn(1, 720, channels)
    y = torch.zeros(1, 720, channels)
    with torch.no_grad():
        model(x, y, is_training=False, target_prefix=horizon)
    for hook in hooks:
        hook.remove()

    per_series_flops = 2 * linear_macs // channels
    if model.readout_mode == "learned-basis-forecast-operator":
        per_series_flops += 2 * horizon * model.basis_rank
    elif model.readout_mode.startswith("pmfo-rct"):
        conservative = model.readout_mode != "pmfo-rct-no-conservation"
        for level, radix in enumerate(PMFO_RADICES):
            active_parents = math.ceil(horizon / PMFO_BLOCK_SIZES[level])
            if conservative:
                per_series_flops += active_parents * (
                    radix + 2 * radix * (radix - 1) + radix
                )
            else:
                per_series_flops += active_parents * 3 * radix
    elif model.readout_mode == "pmfo-rct-no-transition":
        for level, radix in enumerate(PMFO_RADICES):
            active_parents = math.ceil(horizon / PMFO_BLOCK_SIZES[level])
            per_series_flops += active_parents * (
                radix + 2 * radix * (radix - 1) + radix
            )
    return int(per_series_flops)


def capacity_rows(
    models: dict[tuple[str, str], TimeAlign.Model],
) -> list[dict[str, Any]]:
    rows = []
    for dataset, profile in PROFILE_CONFIGS.items():
        for variant in VARIANTS:
            model = models[(dataset, variant)]
            total, decoder = count_parameters(model)
            active, active_decoder = count_active_forward_parameters(
                model,
                profile["channels"],
            )
            rows.append(
                {
                    "dataset": dataset,
                    "profile": profile["profile"],
                    "variant": variant,
                    "total_parameters": total,
                    "decoder_parameters": decoder,
                    "active_forward_parameters_h720": active,
                    "active_decoder_parameters_h720": active_decoder,
                    "linear_dominant_flops_h1_per_series": linear_dominant_flops(
                        model,
                        profile["channels"],
                        1,
                    ),
                    "linear_dominant_flops_h720_per_series": linear_dominant_flops(
                        model,
                        profile["channels"],
                        720,
                    ),
                }
            )
    for dataset in PROFILE_CONFIGS:
        by_variant = {
            row["variant"]: row
            for row in rows
            if row["dataset"] == dataset
        }
        pmfo_params = by_variant["pmfo-rct"]["decoder_parameters"]
        dense_params = by_variant["dense-mlp-matched"]["decoder_parameters"]
        relative_gap = abs(dense_params - pmfo_params) / pmfo_params
        if relative_gap > 0.05:
            raise AssertionError(
                f"dense decoder parameter gap {relative_gap:.4f} exceeds 5% "
                f"for {dataset}"
            )
    return rows


def render_report(
    output_dir: Path,
    summary: dict[str, Any],
    capacity: list[dict[str, Any]],
) -> None:
    representative = [row for row in capacity if row["dataset"] == "ETTm1"]
    capacity_table = [
        "| Variant | Active params | Active decoder | State-dict params | FLOPs H1 | FLOPs H720 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in representative:
        capacity_table.append(
            f"| `{row['variant']}` | {row['active_forward_parameters_h720']} | "
            f"{row['active_decoder_parameters_h720']} | "
            f"{row['total_parameters']} | "
            f"{row['linear_dominant_flops_h1_per_series']} | "
            f"{row['linear_dominant_flops_h720_per_series']} |"
        )
    report = "\n".join(
        [
            "# StageC PMFO-RCT Step 7A Local Gate",
            "",
            "## Scope",
            "",
            "本轮只验证实现与algebra，不训练、不读取test split，也不评估forecast effectiveness。",
            "Step 7B固定验证集为`ETTm1`、`ETTh2`、`Weather`，但仍需单独启动。",
            "",
            "## Gate Result",
            "",
            f"- decision: `{summary['decision']}`；",
            f"- shape/prefix cases: `{summary['shape_prefix_cases']}`；",
            f"- full-prefix max abs: `{summary['full_prefix_max_abs']:.3e}`；",
            f"- refinement recovery max abs: `{summary['refinement_recovery_max_abs']:.3e}`；",
            f"- conservation perturbation max abs: `{summary['conservation_perturbation_max_abs']:.3e}`；",
            f"- locality outside-support max abs: `{summary['locality_outside_support_max_abs']:.3e}`；",
            f"- horizon path audit: `{summary['horizon_path_audit']['pass']}`。",
            "",
            "[Fact] 上述结果只证明代码满足Step 7A tensor/algebra contract；不能证明PMFO有效。",
            "",
            "## Parameter And FLOP Audit",
            "",
            "下表给出ETTm1 profile；decoder参数对三个profile相同，active参数随Encoder profile变化。",
            "`state-dict params`包含TimeAlign兼容性保留但不进入当前forward的legacy `proj_x`，",
            "因此mechanism attribution以`active params`和`active decoder`为准。",
            "FLOPs是每条univariate series的linear-dominant estimate，包含Linear与显式synthesis/basis",
            "乘加，不包含GELU、RevIN和tensor reshape，因此只用于matched-control审计。",
            "",
            *capacity_table,
            "",
            "## Decision Boundary",
            "",
            "Step 7A通过仅授权准备Step 7B。`dense-MLP-matched`或`no-transition`若在训练后解释收益，",
            "仍触发`capacity_control_explains`并回滚Step 4；本地invariant通过不能覆盖该effectiveness gate。",
        ]
    )
    (output_dir / "step7a_local_gate_report.md").write_text(
        report + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    models = instantiate_models()
    shape_rows, prefix_gap = shape_and_prefix_rows(models, generator)
    refinement = refinement_metrics(generator)
    locality = locality_error(generator)
    path_audit = horizon_path_audit(models[("ETTm1", "pmfo-rct")])
    capacity = capacity_rows(models)

    summary = {
        "current_step": "Step 7A local implementation/invariant gate",
        "trains_forecast_model": False,
        "uses_test_split": False,
        "remote_training_authorized": False,
        "step7b_fixed_datasets": ["ETTm1", "ETTh2", "Weather"],
        "shape_prefix_cases": len(shape_rows),
        "full_prefix_max_abs": prefix_gap,
        **refinement,
        "locality_outside_support_max_abs": locality,
        "horizon_path_audit": path_audit,
        "tolerance": TOLERANCE,
        "decision": "step7a_pass",
    }
    manifest = {
        "seed": args.seed,
        "profiles": PROFILE_CONFIGS,
        "variants": list(VARIANTS),
        "horizons": list(HORIZONS),
        "pmfo_block_sizes": list(PMFO_BLOCK_SIZES),
        "pmfo_radices": list(PMFO_RADICES),
        "pmfo_state_dim": 32,
        "dense_hidden_dim": 144,
        "tolerance": TOLERANCE,
    }
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": "cpu",
    }
    write_csv(args.output_dir / "shape_prefix_checks.csv", shape_rows)
    write_csv(args.output_dir / "parameter_flop_audit.csv", capacity)
    (args.output_dir / "step7a_gate.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "step7a_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    render_report(args.output_dir, summary, capacity)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
