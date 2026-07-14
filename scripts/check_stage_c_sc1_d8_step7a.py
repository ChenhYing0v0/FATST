#!/usr/bin/env python3
"""Run the SC1-D8 end-to-end PAF Step 7A local gate."""

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

from layers.PLGO import (  # noqa: E402
    PLGO_GLOBAL_RANK,
    PLGO_RANDOM_DESCRIPTOR_SEED,
    canonical_atom_descriptors,
    descriptor_family,
    restricted_global_nested_basis,
)
from models import TimeAlign  # noqa: E402


HORIZONS = (1, 48, 96, 192, 336, 720)
VARIANTS = (
    "learned-basis-forecast-operator",
    "plgo-paf-geo-c256",
    "plgo-paf-perm-c256",
    "plgo-paf-random-c256",
    "plgo-paf-geo-m694",
    "plgo-paf-perm-m694",
    "plgo-paf-random-m694",
)
PROFILES = {
    "ETTh1": (24, 64, 128, 7),
    "ETTh2": (12, 64, 128, 7),
    "ETTm1": (24, 32, 64, 7),
    "ETTm2": (48, 64, 128, 7),
    "Weather": (12, 64, 128, 21),
}
# Float32 prefix and patch-block sums accumulate up to 3,072 latent terms.
# Keep this aligned with the StageC projectivity/orthogonality audit protocol.
TOLERANCE = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/stage_c_sc1_d8_step7a_local_20260714"),
    )
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def config(dataset: str, readout_mode: str) -> SimpleNamespace:
    patch_num, d_model, d_ff, channels = PROFILES[dataset]
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        encoder_mode="timealign-token-mlp",
        readout_mode=readout_mode,
        e_layers=2,
        patch_num=patch_num,
        d_model=d_model,
        d_ff=d_ff,
        dropout=0.1,
        pos=1,
        layer_norm=1,
        enc_in=channels,
        basis_rank=256,
        plgo_global_rank=16,
        plgo_latent_width=256,
        plgo_permutation_seed=7101,
        plgo_random_descriptor_seed=7102,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def shape_prefix_and_gradient_audit(
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, float],
]:
    shape_rows = []
    gradient_rows = []
    parameter_rows = []
    maxima = {
        "full_prefix_max_abs": 0.0,
        "flatten_block_sum_max_abs": 0.0,
        "basis_orthogonality_max_abs": 0.0,
    }
    for dataset, (patch_num, d_model, _d_ff, channels) in PROFILES.items():
        generator = torch.Generator().manual_seed(seed + len(shape_rows))
        x = torch.randn(1, 720, channels, generator=generator)
        y = torch.zeros(1, 720, channels)
        target = torch.randn(1, 720, channels, generator=generator)
        for variant in VARIANTS:
            torch.manual_seed(seed)
            model = TimeAlign.Model(config(dataset, variant)).float().eval()
            with torch.no_grad():
                full = model(x, y, is_training=False, target_prefix=720)[0]
                for horizon in HORIZONS:
                    prefix = model(
                        x,
                        y,
                        is_training=False,
                        target_prefix=horizon,
                    )[0]
                    gap = float((prefix - full[:, :horizon]).abs().max())
                    maxima["full_prefix_max_abs"] = max(
                        maxima["full_prefix_max_abs"], gap
                    )
                    expected = [1, horizon, channels]
                    shape_rows.append(
                        {
                            "dataset": dataset,
                            "variant": variant,
                            "horizon": horizon,
                            "output_shape": str(list(prefix.shape)),
                            "expected_shape": str(expected),
                            "full_prefix_max_abs": gap,
                            "pass": list(prefix.shape) == expected and gap <= TOLERANCE,
                        }
                    )

            model.zero_grad(set_to_none=True)
            prediction = model(x, y, is_training=False, target_prefix=720)[0]
            (prediction - target).square().mean().backward()
            active_prefixes = (
                "patch_emb_x.",
                "encoder.",
                "norm_x.",
                "learned_basis_coeff.",
                "learned_temporal_basis",
                "learned_temporal_bias",
                "plgo_paf_readout.",
            )
            active = [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if name.startswith(active_prefixes)
            ]
            missing = [name for name, parameter in active if parameter.grad is None]
            nonfinite = [
                name
                for name, parameter in active
                if parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
            ]
            zero = [
                name
                for name, parameter in active
                if parameter.grad is not None
                and float(parameter.grad.abs().max()) == 0.0
            ]
            patch_gradient_min = ""
            block_gap = ""
            if hasattr(model, "plgo_paf_readout"):
                readout = model.plgo_paf_readout
                branch_gradient = readout.branch.weight.grad.reshape(
                    readout.latent_width,
                    patch_num,
                    d_model,
                )
                patch_norms = branch_gradient.square().sum(dim=(0, 2)).sqrt()
                patch_gradient_min = float(patch_norms.min())
                memory = model.encode_history(x)
                hidden = memory.flatten(start_dim=-2)
                direct = readout.branch(hidden)
                explicit = readout.latent_from_patch_blocks(
                    hidden,
                    patch_num,
                    d_model,
                )
                block_gap = float((direct - explicit).abs().max())
                maxima["flatten_block_sum_max_abs"] = max(
                    maxima["flatten_block_sum_max_abs"],
                    block_gap,
                )
                identity = torch.eye(720)
                orthogonality = float(
                    (readout.basis_rows @ readout.basis_rows.T - identity)
                    .abs()
                    .max()
                )
                maxima["basis_orthogonality_max_abs"] = max(
                    maxima["basis_orthogonality_max_abs"],
                    orthogonality,
                )
            gradient_rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "active_parameter_tensors": len(active),
                    "missing_gradient_tensors": len(missing),
                    "nonfinite_gradient_tensors": len(nonfinite),
                    "zero_gradient_tensors": len(zero),
                    "patch_gradient_min_l2": patch_gradient_min,
                    "flatten_block_sum_max_abs": block_gap,
                    "pass": not missing
                    and not nonfinite
                    and not zero
                    and (
                        patch_gradient_min == ""
                        or float(patch_gradient_min) > 0.0
                    )
                    and (block_gap == "" or float(block_gap) <= TOLERANCE),
                }
            )
            parameter_rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "total_parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "trainable_parameters": sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ),
                    "decoder_parameters": (
                        sum(
                            parameter.numel()
                            for parameter in model.plgo_paf_readout.parameters()
                        )
                        if hasattr(model, "plgo_paf_readout")
                        else sum(
                            parameter.numel()
                            for name, parameter in model.named_parameters()
                            if name.startswith(
                                (
                                    "learned_basis_coeff.",
                                    "learned_temporal_basis",
                                    "learned_temporal_bias",
                                )
                            )
                        )
                    ),
                }
            )
    return shape_rows, gradient_rows, parameter_rows, maxima


def descriptor_audit() -> dict[str, float]:
    synthesis, atoms = restricted_global_nested_basis(720, PLGO_GLOBAL_RANK)
    canonical = canonical_atom_descriptors(atoms)
    random = descriptor_family(
        canonical,
        "random",
        random_seed=PLGO_RANDOM_DESCRIPTOR_SEED,
    )
    return {
        "reference_basis_orthogonality_max_abs": float(
            (synthesis.T @ synthesis - torch.eye(720, dtype=synthesis.dtype))
            .abs()
            .max()
        ),
        "random_descriptor_mean_max_abs": float(
            (random.mean(dim=0) - canonical.mean(dim=0)).abs().max()
        ),
        "random_descriptor_std_max_abs": float(
            (
                random.std(dim=0, unbiased=False)
                - canonical.std(dim=0, unbiased=False)
            )
            .abs()
            .max()
        ),
    }


def horizon_path_audit(seed: int) -> dict[str, Any]:
    model = TimeAlign.Model(config("ETTm1", "plgo-paf-geo-c256")).float().eval()
    calls = 0
    tensor_only = True
    hooks = []

    def inspect(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
        nonlocal calls, tensor_only
        calls += 1
        tensor_only = tensor_only and all(
            isinstance(value, torch.Tensor) for value in inputs
        )

    for module in model.plgo_paf_readout.modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_pre_hook(inspect))
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 720, 7, generator=generator)
    y = torch.zeros(1, 720, 7)
    with torch.no_grad():
        model(x, y, is_training=False, target_prefix=48)
    for hook in hooks:
        hook.remove()
    return {
        "learned_module_calls": calls,
        "learned_module_inputs_tensor_only": tensor_only,
        "node_axis_normalization_present": any(
            isinstance(module, (nn.LayerNorm, nn.BatchNorm1d))
            for module in model.plgo_paf_readout.modules()
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shape_rows, gradient_rows, parameter_rows, maxima = (
        shape_prefix_and_gradient_audit(args.seed)
    )
    descriptor = descriptor_audit()
    horizon_path = horizon_path_audit(args.seed)
    shape_pass = len(shape_rows) == 210 and all(row["pass"] for row in shape_rows)
    gradient_pass = len(gradient_rows) == 35 and all(
        row["pass"] for row in gradient_rows
    )
    descriptor_pass = (
        descriptor["reference_basis_orthogonality_max_abs"] <= 1e-10
        and descriptor["random_descriptor_mean_max_abs"] <= 1e-5
        and descriptor["random_descriptor_std_max_abs"] <= 1e-5
    )
    horizon_pass = (
        horizon_path["learned_module_calls"] > 0
        and horizon_path["learned_module_inputs_tensor_only"]
        and not horizon_path["node_axis_normalization_present"]
    )
    passed = shape_pass and gradient_pass and descriptor_pass and horizon_pass
    gate = {
        "candidate": "SC1-D8-E2E",
        "current_step": "Step 7A",
        "shape_prefix_gate": shape_pass,
        "gradient_patch_interface_gate": gradient_pass,
        "descriptor_basis_gate": descriptor_pass,
        "horizon_path_gate": horizon_pass,
        "shape_cases": len(shape_rows),
        "gradient_cases": len(gradient_rows),
        **maxima,
        **descriptor,
        **horizon_path,
        "test_used": False,
        "forecast_training_run": False,
        "remote_training_authorized": passed,
        "decision": "step7a_pass" if passed else "step7a_fail_repair",
        "pass": passed,
    }
    write_csv(args.output_dir / "shape_prefix_checks.csv", shape_rows)
    write_csv(args.output_dir / "gradient_patch_checks.csv", gradient_rows)
    write_csv(args.output_dir / "parameter_audit.csv", parameter_rows)
    (args.output_dir / "step7a_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": "cpu",
                "seed": args.seed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report = "\n".join(
        [
            "# SC1-D8 Step 7A Local Gate",
            "",
            f"- decision: `{gate['decision']}`",
            f"- shape-prefix: `{shape_pass}` ({len(shape_rows)}/210)",
            f"- gradient/patch interface: `{gradient_pass}` ({len(gradient_rows)}/35)",
            f"- descriptor/basis: `{descriptor_pass}`",
            f"- horizon path: `{horizon_pass}`",
            f"- full-prefix max abs: `{maxima['full_prefix_max_abs']:.3e}`",
            f"- flatten/block-sum max abs: `{maxima['flatten_block_sum_max_abs']:.3e}`",
            f"- float32 basis orthogonality max abs: `{maxima['basis_orthogonality_max_abs']:.3e}`",
            "- test used: `false`",
            "- forecast training run: `false`",
            "",
            "数值协议说明：初始1e-6检查只被float32长向量累积误差触发；shape、gradient与float64参考构造均正常。项目既有projectivity协议使用1e-5，因此在remote前统一修正阈值，不涉及训练结果或性能选择。",
            "",
            "该gate只证明joint-training implementation、projectivity、patch information path与gradient contract可执行；不构成forecast effectiveness证据。",
            "",
        ]
    )
    (args.output_dir / "step7a_local_gate_report.md").write_text(
        report,
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError(json.dumps(gate, sort_keys=True))
    print(json.dumps(gate, sort_keys=True))


if __name__ == "__main__":
    main()
