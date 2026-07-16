#!/usr/bin/env python3
"""Run the SC-D15-A PCSD-CF Step 7A local implementation gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from layers.PCSD import PCSDCouplingFieldReadout  # noqa: E402
from models import TimeAlign  # noqa: E402
import train_repo as training_adapter  # noqa: E402
from train_repo import (  # noqa: E402
    OFFICIAL_PRESETS,
    PREFIX_READOUT_MODES,
    STAGE_C_ACTIVE_READOUTS,
    build_official_args,
    initialization_contract,
    model_diagnostics,
    parse_args as parse_training_args,
)


CHANNELS = {"ETTh1": 7, "ETTh2": 7, "ETTm1": 7, "ETTm2": 7, "Weather": 21}
HORIZONS = (1, 7, 48, 95, 96, 144, 192, 257, 336, 511, 512, 719, 720)
FLOAT_TOLERANCES = {"float32": 2e-5, "float64": 2e-11}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=Path("configs/stage_c_pcsd_cf_native_direct.json"),
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("configs/stage_c_five_dataset_natural_profiles.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/stage_c_pcsd_cf_step7a_local_20260716"),
    )
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_readout(
    readout_dim: int,
    design: dict[str, Any],
    **overrides: Any,
) -> PCSDCouplingFieldReadout:
    field = design["coupling_field"]
    policy = design["policy"]
    kwargs: dict[str, Any] = {
        "readout_dim": readout_dim,
        "series_length": int(design["series_length"]),
        "scales": tuple(int(value) for value in design["coupling_scales"]),
        "coordinate_dim": int(field["coordinate_dim"]),
        "mode_rank": int(field["mode_rank"]),
        "policy_history_dim": int(policy["history_projection_dim"]),
        "policy_hidden_dim": int(policy["hidden_dim"]),
        "policy_mode": "direct",
        "fixed_scale": 720,
        "partition": "canonical",
        "partition_seed": 15101,
        "group_chunk_size": 64,
        "target_chunk_size": 128,
    }
    kwargs.update(overrides)
    return PCSDCouplingFieldReadout(**kwargs)


def model_config(
    dataset: str,
    profile: dict[str, Any],
    design: dict[str, Any],
) -> SimpleNamespace:
    field = design["coupling_field"]
    policy = design["policy"]
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        encoder_mode="timealign-token-mlp",
        readout_mode="pcsd-coupling-field",
        e_layers=2,
        patch_num=int(profile["patch_num"]),
        d_model=int(profile["d_model"]),
        d_ff=int(profile["d_ff"]),
        dropout=0.1,
        pos=1,
        layer_norm=1,
        enc_in=CHANNELS[dataset],
        pcsd_coordinate_dim=int(field["coordinate_dim"]),
        pcsd_mode_rank=int(field["mode_rank"]),
        pcsd_policy_history_dim=int(policy["history_projection_dim"]),
        pcsd_policy_hidden_dim=int(policy["hidden_dim"]),
        pcsd_policy_mode="direct",
        pcsd_fixed_scale=720,
        pcsd_partition="canonical",
        pcsd_partition_seed=15101,
        pcsd_group_chunk_size=64,
        pcsd_target_chunk_size=128,
    )


def shape_and_prefix_audit(
    design: dict[str, Any],
    profiles: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prefix_rows: list[dict[str, Any]] = []
    integration_rows: list[dict[str, Any]] = []
    for dataset, profile in profiles.items():
        readout_dim = int(profile["patch_num"]) * int(profile["d_model"])
        torch.manual_seed(seed)
        readout = build_readout(readout_dim, design).eval()
        generator = torch.Generator().manual_seed(seed + readout_dim)
        hidden = torch.randn(1, 1, readout_dim, generator=generator)
        with torch.no_grad():
            full, arms, weights = readout.forward_with_diagnostics(hidden, 720)
            for horizon in HORIZONS:
                prefix = readout(hidden, horizon)
                gap = float((prefix - full[:, :horizon]).abs().max())
                expected = (1, horizon, 1)
                prefix_rows.append(
                    {
                        "dataset": dataset,
                        "readout_dim": readout_dim,
                        "horizon": horizon,
                        "output_shape": str(tuple(prefix.shape)),
                        "expected_shape": str(expected),
                        "full_prefix_max_abs": gap,
                        "pass": tuple(prefix.shape) == expected and gap == 0.0,
                    }
                )
        torch.manual_seed(seed)
        model = TimeAlign.Model(model_config(dataset, profile, design)).float().eval()
        x = torch.randn(
            1,
            720,
            CHANNELS[dataset],
            generator=torch.Generator().manual_seed(seed + CHANNELS[dataset]),
        )
        y = torch.zeros(1, 720, CHANNELS[dataset])
        with torch.no_grad():
            model_full = model(x, y, is_training=False, target_prefix=720)[0]
            model_prefix = model(x, y, is_training=False, target_prefix=336)[0]
            memory = model.encode_history(x)
            model_hidden = memory.flatten(start_dim=-2)
            _forecast, model_arms, model_weights = (
                model.pcsd_readout.forward_with_diagnostics(model_hidden, 720)
            )
        model_gap = float((model_prefix - model_full[:, :336]).abs().max())
        initialization = initialization_contract(model)
        diagnostics = model_diagnostics(model)
        initialization_pass = all(
            key in initialization
            for key in (
                "pcsd_initialization_hash",
                "pcsd_coordinate_hash",
                "pcsd_partition_hash",
                "pcsd_initial_scope_usage",
            )
        )
        diagnostics_pass = (
            diagnostics.get("pcsd_decoder_parameters")
            == sum(parameter.numel() for parameter in model.pcsd_readout.parameters())
            and diagnostics.get("pcsd_coupling_field_parameters")
            == model.pcsd_readout.coupling_field_parameters
            and diagnostics.get("pcsd_policy_parameters")
            == model.pcsd_readout.policy_parameters
        )
        integration_rows.append(
            {
                "dataset": dataset,
                "memory_shape": str(
                    (1, CHANNELS[dataset], profile["patch_num"], profile["d_model"])
                ),
                "hidden_shape": str((1, CHANNELS[dataset], readout_dim)),
                "arms_shape": str(tuple(model_arms.shape)),
                "weights_shape": str(tuple(model_weights.shape)),
                "full_shape": str(tuple(model_full.shape)),
                "prefix_shape": str(tuple(model_prefix.shape)),
                "prefix_max_abs": model_gap,
                "initialization_contract_pass": initialization_pass,
                "model_diagnostics_pass": diagnostics_pass,
                "pass": tuple(model_full.shape)
                == (1, 720, CHANNELS[dataset])
                and tuple(model_prefix.shape) == (1, 336, CHANNELS[dataset])
                and tuple(model_arms.shape)
                == (1, CHANNELS[dataset], 5, 720)
                and tuple(model_weights.shape)
                == (1, CHANNELS[dataset], 720, 5)
                and model_gap == 0.0
                and initialization_pass
                and diagnostics_pass,
            }
        )
    return prefix_rows, integration_rows


def containment_audit(
    design: dict[str, Any],
    profiles: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    widths = sorted(
        {
            int(profile["patch_num"]) * int(profile["d_model"])
            for profile in profiles.values()
        }
    )
    rank = int(design["coupling_field"]["mode_rank"])
    length = int(design["series_length"])
    for dtype in (torch.float32, torch.float64):
        dtype_name = str(dtype).split(".")[-1]
        for readout_dim in widths:
            torch.manual_seed(seed + readout_dim)
            readout = build_readout(readout_dim, design).to(dtype=dtype).eval()
            generator = torch.Generator().manual_seed(seed + 2 * readout_dim)
            weight = torch.randn(rank, readout_dim, generator=generator, dtype=dtype)
            weight *= readout_dim**-0.5
            coefficient_bias = torch.randn(rank, generator=generator, dtype=dtype)
            basis = torch.randn(length, rank, generator=generator, dtype=dtype)
            basis *= rank**-0.5
            temporal_bias = torch.randn(length, generator=generator, dtype=dtype)
            hidden = torch.randn(1, 1, readout_dim, generator=generator, dtype=dtype)
            readout.map_a6_parameters_(weight, coefficient_bias, basis, temporal_bias)
            with torch.no_grad():
                arms = readout.arm_forecasts(hidden)
                prediction = readout(hidden, 720)
                coefficients = torch.einsum("bcr,kr->bck", hidden, weight)
                coefficients = coefficients + coefficient_bias.view(1, 1, -1)
                expected = torch.einsum("tk,bck->bct", basis, coefficients)
                expected = expected + temporal_bias.view(1, 1, -1)
                expected = expected.permute(0, 2, 1)
            output_gap = float((prediction - expected).abs().max())
            arm_gap = float((arms - arms[:, :, :1]).abs().max())
            tolerance = FLOAT_TOLERANCES[dtype_name]
            rows.append(
                {
                    "dtype": dtype_name,
                    "readout_dim": readout_dim,
                    "output_max_abs": output_gap,
                    "arm_max_abs": arm_gap,
                    "tolerance": tolerance,
                    "pass": output_gap <= tolerance and arm_gap <= tolerance,
                }
            )
    return rows


def topology_audit(
    design: dict[str, Any],
) -> list[dict[str, Any]]:
    readout = build_readout(8, design, mode_rank=4).double().eval()
    rows: list[dict[str, Any]] = []
    for scale_index, scale in enumerate(readout.scales):
        indices = readout.group_indices(scale_index)
        labels = readout.target_group_labels(scale_index)
        descriptors = readout.target_scope_descriptors(scale_index).double()
        cover = torch.sort(indices.flatten()).values
        within_group_gap = 0.0
        for group in indices:
            within_group_gap = max(
                within_group_gap,
                float((descriptors[group] - descriptors[group[:1]]).abs().max()),
            )
        distinct_descriptors = torch.unique(descriptors, dim=0).shape[0]

        same_targets = indices[0, : min(2, scale)]
        same_jacobian_gap: float | str = ""
        if len(same_targets) == 2:
            same_jacobian_gap = float(
                (
                    descriptors[int(same_targets[0])]
                    - descriptors[int(same_targets[1])]
                ).abs().max()
            )
        cross_jacobian_gap: float | str = ""
        if indices.shape[0] > 1:
            cross_jacobian_gap = float(
                (descriptors[int(indices[0, 0])] - descriptors[int(indices[1, 0])])
                .abs()
                .max()
            )
        expected_distinct = indices.shape[0]
        topology_pass = (
            torch.equal(cover, torch.arange(readout.series_length))
            and bool((torch.bincount(labels) == scale).all())
            and within_group_gap == 0.0
            and distinct_descriptors == expected_distinct
            and (same_jacobian_gap == "" or same_jacobian_gap == 0.0)
            and (cross_jacobian_gap == "" or cross_jacobian_gap > 0.0)
        )
        rows.append(
            {
                "scale": scale,
                "group_count": indices.shape[0],
                "distinct_target_jacobians": distinct_descriptors,
                "expected_distinct_jacobians": expected_distinct,
                "within_group_jacobian_max_abs": within_group_gap,
                "same_group_pair_max_abs": same_jacobian_gap,
                "cross_group_pair_max_abs": cross_jacobian_gap,
                "global_nonconstant_rms": float(
                    descriptors[:, 1:].square().mean().sqrt()
                ),
                "pass": topology_pass,
            }
        )
    return rows


def separation_and_partition_audit(
    design: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    separation_rows: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    for partition in ("canonical", "random"):
        torch.manual_seed(seed)
        readout = build_readout(768, design, partition=partition).eval()
        hidden = torch.randn(
            1,
            1,
            768,
            generator=torch.Generator().manual_seed(seed + 1),
        )
        with torch.no_grad():
            arms = readout.arm_forecasts(hidden)
            weights = readout.policy_weights(hidden)
        denominator = float(arms.square().mean().sqrt().clamp_min(1e-12))
        pairwise = []
        for left in range(len(readout.scales)):
            for right in range(left + 1, len(readout.scales)):
                pairwise.append(
                    float(
                        (arms[:, :, left] - arms[:, :, right])
                        .square()
                        .mean()
                        .sqrt()
                    )
                    / denominator
                )
        uniform_gap = float((weights - 1.0 / len(readout.scales)).abs().max())
        separation_rows.append(
            {
                "partition": partition,
                "minimum_pairwise_normalized_rmse": min(pairwise),
                "mean_pairwise_normalized_rmse": sum(pairwise) / len(pairwise),
                "initial_policy_uniform_max_abs": uniform_gap,
                "pass": min(pairwise) > 1e-4 and uniform_gap <= 1e-7,
            }
        )
        parameter_shapes = sorted(
            (name, tuple(parameter.shape))
            for name, parameter in readout.named_parameters()
        )
        parameter_values = b"".join(
            parameter.detach().cpu().contiguous().numpy().tobytes()
            for _name, parameter in sorted(readout.named_parameters())
        )
        partition_rows.append(
            {
                "partition": partition,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in readout.parameters()
                ),
                "parameter_shape_hash": hashlib.sha256(
                    repr(parameter_shapes).encode("utf-8")
                ).hexdigest(),
                "parameter_value_hash": hashlib.sha256(parameter_values).hexdigest(),
            }
        )
    counts = {row["trainable_parameters"] for row in partition_rows}
    hashes = {row["parameter_shape_hash"] for row in partition_rows}
    value_hashes = {row["parameter_value_hash"] for row in partition_rows}
    for row in partition_rows:
        row["pass"] = (
            len(counts) == 1 and len(hashes) == 1 and len(value_hashes) == 1
        )
    return separation_rows, partition_rows


def gradient_audit(
    design: dict[str, Any],
    profiles: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, profile in profiles.items():
        readout_dim = int(profile["patch_num"]) * int(profile["d_model"])
        torch.manual_seed(seed)
        readout = build_readout(readout_dim, design).float().train()
        optimizer = torch.optim.SGD(readout.parameters(), lr=1e-3)
        hidden = torch.randn(
            1,
            1,
            readout_dim,
            generator=torch.Generator().manual_seed(seed + readout_dim),
        )
        target = torch.randn(
            1,
            720,
            1,
            generator=torch.Generator().manual_seed(seed + readout_dim + 1),
        )
        step_records = []
        for _step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = readout(hidden, 720)
            loss = (output - target).abs().mean()
            loss.backward()
            gradients = {
                name: parameter.grad
                for name, parameter in readout.named_parameters()
            }
            step_records.append(gradients)
            optimizer.step()
        all_finite = all(
            gradient is not None and bool(torch.isfinite(gradient).all())
            for records in step_records
            for gradient in records.values()
        )
        second_step_policy_active = all(
            step_records[1][name] is not None
            and float(step_records[1][name].abs().sum()) > 0.0
            for name in (
                "history_projection.weight",
                "policy_hidden.weight",
                "policy_output.weight",
            )
        )
        core_active = all(
            step_records[0][name] is not None
            and float(step_records[0][name].abs().sum()) > 0.0
            for name in (
                "mode_weight",
                "identity_synthesis",
                "nonlinear_synthesis",
            )
        )
        rows.append(
            {
                "dataset": dataset,
                "readout_dim": readout_dim,
                "forward_finite": bool(torch.isfinite(output).all()),
                "loss_finite": math.isfinite(float(loss)),
                "all_gradient_tensors_finite_two_steps": all_finite,
                "core_active_step1": core_active,
                "policy_active_step2": second_step_policy_active,
                "pass": bool(torch.isfinite(output).all())
                and math.isfinite(float(loss))
                and all_finite
                and core_active
                and second_step_policy_active,
            }
        )

    dataset = "ETTh2"
    profile = profiles[dataset]
    torch.manual_seed(seed)
    model = TimeAlign.Model(model_config(dataset, profile, design)).float().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed + 99)
    x = torch.randn(1, 720, CHANNELS[dataset], generator=generator)
    y = torch.zeros(1, 720, CHANNELS[dataset])
    target = torch.randn(1, 720, CHANNELS[dataset], generator=generator)
    active_prefixes = ("patch_emb_x.", "encoder.", "norm_x.", "pcsd_readout.")
    final_records: dict[str, torch.Tensor | None] = {}
    output = target.new_zeros(target.shape)
    loss = output.new_zeros(())
    for _step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(x, y, is_training=False, target_prefix=720)[0]
        loss = (output - target).abs().mean()
        loss.backward()
        final_records = {
            name: parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith(active_prefixes)
        }
        optimizer.step()
    e2e_finite = bool(final_records) and all(
        gradient is not None and bool(torch.isfinite(gradient).all())
        for gradient in final_records.values()
    )
    e2e_active = all(
        final_records[name] is not None
        and float(final_records[name].abs().sum()) > 0.0
        for name in (
            "patch_emb_x.patch_proj.weight",
            "encoder.0.0.weight",
            "pcsd_readout.mode_weight",
            "pcsd_readout.policy_output.weight",
        )
    )
    rows.append(
        {
            "dataset": dataset,
            "readout_dim": int(profile["state_width"]),
            "scope": "a6-natural-e2e-model",
            "forward_finite": bool(torch.isfinite(output).all()),
            "loss_finite": math.isfinite(float(loss)),
            "all_gradient_tensors_finite_two_steps": e2e_finite,
            "core_active_step1": "checked-in-module-rows",
            "policy_active_step2": e2e_active,
            "pass": bool(torch.isfinite(output).all())
            and math.isfinite(float(loss))
            and e2e_finite
            and e2e_active,
        }
    )
    return rows


def accounting_audit(
    design: dict[str, Any],
    profiles: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scales = [int(value) for value in design["coupling_scales"]]
    length = int(design["series_length"])
    dimension = int(design["coupling_field"]["coordinate_dim"])
    rank = int(design["coupling_field"]["mode_rank"])
    history_dim = int(design["policy"]["history_projection_dim"])
    policy_hidden = int(design["policy"]["hidden_dim"])
    scale_count = len(scales)
    for dataset, profile in profiles.items():
        readout_dim = int(profile["patch_num"]) * int(profile["d_model"])
        readout = build_readout(readout_dim, design)
        a6_parameters = (
            readout_dim * rank + rank + length * rank + length
        )
        pcsd_parameters = sum(parameter.numel() for parameter in readout.parameters())
        a6_flops = 2 * readout_dim * rank + 2 * length * rank
        group_counts = [length // scale for scale in scales]
        field_flops = 2 * dimension * readout_dim * rank
        field_flops += sum(2 * groups * dimension * rank for groups in group_counts)
        field_flops += scale_count * 4 * length * rank
        policy_flops = 2 * readout_dim * history_dim
        policy_flops += 2 * length * (history_dim + dimension) * policy_hidden
        policy_flops += 2 * length * policy_hidden * scale_count
        fusion_flops = 2 * length * scale_count
        pcsd_flops = field_flops + policy_flops + fusion_flops
        peak_elements = max(
            dimension * rank + 64 * rank + scale_count * length * 2,
            dimension * rank
            + 128 * (history_dim + dimension + policy_hidden + scale_count),
        )
        rows.append(
            {
                "dataset": dataset,
                "readout_dim": readout_dim,
                "a6_decoder_parameters": a6_parameters,
                "pcsd_field_parameters": readout.coupling_field_parameters,
                "pcsd_policy_parameters": readout.policy_parameters,
                "pcsd_total_parameters": pcsd_parameters,
                "pcsd_field_to_a6_parameter_ratio": readout.coupling_field_parameters
                / a6_parameters,
                "pcsd_to_a6_parameter_ratio": pcsd_parameters / a6_parameters,
                "a6_estimated_flops_per_channel": a6_flops,
                "pcsd_estimated_flops_per_channel": pcsd_flops,
                "pcsd_to_a6_flop_ratio": pcsd_flops / a6_flops,
                "naive_point_state_fp32_mib_per_channel": length
                * rank
                * 4
                / 2**20,
                "chunked_estimated_peak_fp32_mib_per_channel": peak_elements
                * 4
                / 2**20,
                "parameter_dof_equals_trainable_count": True,
                "pass": pcsd_parameters
                == readout.coupling_field_parameters + readout.policy_parameters,
            }
        )
    return rows


def parse_training_contract() -> argparse.Namespace:
    original = sys.argv
    sys.argv = [
        "train_repo.py",
        "--dataset-root",
        ".",
        "--dataset",
        "ETTh2",
        "--mode",
        "unified",
        "--pred-len",
        "720",
        "--target-horizons",
        "96,192,336,720",
        "--run-name",
        "pcsd_step7a_parse_only",
        "--output-dir",
        "/tmp/pcsd_step7a_parse_only",
        "--readout-mode",
        "pcsd-coupling-field",
        "--final-evaluation-split",
        "none",
    ]
    try:
        return parse_training_args()
    finally:
        sys.argv = original


def contract_audit(design: dict[str, Any]) -> list[dict[str, Any]]:
    training_args = parse_training_contract()
    original_resolver = training_adapter.resolve_dataset_root
    training_adapter.resolve_dataset_root = lambda _root, _preset: Path(".")
    try:
        official_args = build_official_args(
            training_args,
            OFFICIAL_PRESETS[training_args.dataset][720],
        )
    finally:
        training_adapter.resolve_dataset_root = original_resolver
    checks = {
        "cli_prefix_readout_registered": "pcsd-coupling-field"
        in PREFIX_READOUT_MODES,
        "cli_active_readout_registered": "pcsd-coupling-field"
        in STAGE_C_ACTIVE_READOUTS,
        "cli_native_contract_parses": training_args.readout_mode
        == "pcsd-coupling-field"
        and training_args.mode == "unified"
        and training_args.pred_len == 720,
        "cli_frozen_dimensions": training_args.pcsd_coordinate_dim == 4
        and training_args.pcsd_mode_rank == 256
        and training_args.pcsd_policy_history_dim == 32
        and training_args.pcsd_policy_hidden_dim == 64,
        "adapter_propagates_pcsd_contract": official_args.pcsd_coordinate_dim == 4
        and official_args.pcsd_mode_rank == 256
        and official_args.pcsd_policy_mode == "direct"
        and official_args.pcsd_partition == "canonical",
        "requested_horizon_feature_false": not design["policy"][
            "requested_horizon_feature"
        ],
        "future_truth_feature_false": not design["policy"]["future_truth_feature"],
        "full_domain_objective": design["training"]["objective"]
        == "full-720-pointwise-l1-of-fused-forecast",
        "joint_training_true": design["training"]["joint_encoder_decoder_training"],
        "frozen_replacement_false": not design["training"]["frozen_replacement"],
        "warm_start_false": not design["training"]["warm_start"],
        "test_access_false": not design["training"]["test_used"]
        and not design["authorization"]["test_access_authorized"],
        "remote_false": not design["authorization"]["remote_training_authorized"],
        "sc2_false": not design["authorization"][
            "contribution2_implementation_authorized"
        ],
    }
    return [
        {"contract": name, "observed": value, "pass": bool(value)}
        for name, value in checks.items()
    ]


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    separator = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(str(row[field]) for field in fields) + " |"
        for row in rows
    ]
    return "\n".join((header, separator, *body))


def main() -> None:
    args = parse_args()
    design_path = resolve(args.design)
    profile_path = resolve(args.profiles)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    design = load_json(design_path)
    profile_contract = load_json(profile_path)
    profiles = profile_contract["dataset_profiles"]

    prefix_rows, integration_rows = shape_and_prefix_audit(
        design,
        profiles,
        args.seed,
    )
    containment_rows = containment_audit(design, profiles, args.seed)
    topology_rows = topology_audit(design)
    separation_rows, partition_rows = separation_and_partition_audit(
        design,
        args.seed,
    )
    gradient_rows = gradient_audit(design, profiles, args.seed)
    accounting_rows = accounting_audit(design, profiles)
    contract_rows = contract_audit(design)

    named_rows = {
        "shape_prefix_checks.csv": prefix_rows,
        "model_integration_checks.csv": integration_rows,
        "containment_checks.csv": containment_rows,
        "topology_checks.csv": topology_rows,
        "separation_checks.csv": separation_rows,
        "partition_checks.csv": partition_rows,
        "gradient_checks.csv": gradient_rows,
        "accounting.csv": accounting_rows,
        "protocol_contract_checks.csv": contract_rows,
    }
    for filename, rows in named_rows.items():
        write_csv(output_dir / filename, rows)

    categories = {
        filename.removesuffix(".csv"): all(bool(row["pass"]) for row in rows)
        for filename, rows in named_rows.items()
    }
    overall_pass = all(categories.values())
    summary = {
        "stage": "StageC-UVHF",
        "candidate_id": design["candidate_id"],
        "diagnostic_id": design["diagnostic_id"],
        "current_step": "Step7A local implementation gate",
        "design_path": str(design_path.relative_to(ROOT)),
        "design_sha256": sha256(design_path),
        "profile_path": str(profile_path.relative_to(ROOT)),
        "profile_sha256": sha256(profile_path),
        "seed": args.seed,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "execution_device": "cpu",
        "categories": categories,
        "overall_pass": overall_pass,
        "authorization_after_gate": {
            "step7a_local": overall_pass,
            "step7b_remote": False,
            "effectiveness_claim": False,
            "contribution2": False,
            "test_access": False,
        },
        "decision": (
            "step7a_local_pass_step7b_design_only_next"
            if overall_pass
            else "step7a_local_fail_rollback_step5_or_6"
        ),
    }
    (output_dir / "step7a_local_gate.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    containment_table_rows = [
        {
            "dtype": row["dtype"],
            "R": row["readout_dim"],
            "output_gap": f"{row['output_max_abs']:.3e}",
            "arm_gap": f"{row['arm_max_abs']:.3e}",
            "pass": row["pass"],
        }
        for row in containment_rows
    ]
    separation_table_rows = [
        {
            "partition": row["partition"],
            "min_pair_nrmse": f"{row['minimum_pairwise_normalized_rmse']:.6f}",
            "mean_pair_nrmse": f"{row['mean_pairwise_normalized_rmse']:.6f}",
            "uniform_gap": f"{row['initial_policy_uniform_max_abs']:.3e}",
            "pass": row["pass"],
        }
        for row in separation_rows
    ]
    accounting_table_rows = [
        {
            "dataset": row["dataset"],
            "R": row["readout_dim"],
            "field/A6 params": f"{row['pcsd_field_to_a6_parameter_ratio']:.4f}",
            "total/A6 params": f"{row['pcsd_to_a6_parameter_ratio']:.4f}",
            "PCSD/A6 FLOPs": f"{row['pcsd_to_a6_flop_ratio']:.4f}",
        }
        for row in accounting_rows
    ]

    report = f"""# SC-D15-A PCSD-CF Step 7A Local Gate

## Reader path

本地gate测试`PCSD-CF-v1`的实现是否符合Step4-6冻结的数学与protocol contract。它不访问dataset、validation或
test，不训练模型，也不构成performance evidence。读取顺序为：shape/projectivity -> A6 containment ->
sharing topology -> arm/policy/gradient -> accounting -> decision。

## Gate result

- `overall_pass={str(overall_pass).lower()}`
- `decision={summary['decision']}`
- remote、effectiveness claim、Contribution 2与test仍为`false`。

{markdown_table([{'gate': key, 'pass': value} for key, value in categories.items()], ['gate', 'pass'])}

## Numerical evidence

### Arbitrary-A6 containment

{markdown_table(containment_table_rows, ['dtype', 'R', 'output_gap', 'arm_gap', 'pass'])}

### Arm separation and equal-logit initialization

{markdown_table(separation_table_rows, ['partition', 'min_pair_nrmse', 'mean_pair_nrmse', 'uniform_gap', 'pass'])}

### Static accounting

{markdown_table(accounting_table_rows, ['dataset', 'R', 'field/A6 params', 'total/A6 params', 'PCSD/A6 FLOPs'])}

## What each artifact means

- `shape_prefix_checks.csv`：直接readout在5个dataset-aware state widths与13个dense/arbitrary horizons下的
  shape及full-domain prefix crop equality。
- `model_integration_checks.csv`：真实A6-natural encoder路径的`[B,C,P,D_e] -> [B,C,R] -> [B,C,5,720]
  -> [B,H,C]`接线检查。
- `containment_checks.csv`：将任意A6系数映射、basis和bias构造到PCSD constant mode，检查float32/float64
  output与所有scope arms的最大绝对误差。
- `topology_checks.csv`：`target_scope_descriptors=P_sQ[g_s(tau)]`正是group state对history modes的Jacobian；
  检查同组target共享Jacobian、跨组target不同、point/global端点分别有720/1个sharing class。
- `separation_checks.csv`：random trainable parameters下不同scope arms的pairwise normalized RMSE，以及
  direct policy final logits全零产生的equal initial weights。
- `partition_checks.csv`：canonical/random只改变fixed buffers，trainable parameter count与shape hash相同。
- `gradient_checks.csv`：两步local SGD；第一步field active，zero-logit policy output更新后第二步history/target
  policy path active，所有gradient tensors有限。
- `accounting.csv`：参数/DoF、multiply-add FLOP估算与activation估算。FLOP以单channel full-T forward计，
  GELU/softmax标量代价未计；activation是chunked operator的静态上界估算，不是GPU profiler实测。
- `protocol_contract_checks.csv`：requested H/future truth/test不进入learned path，frozen/warm-start/remote/SC2关闭。

## Failure attribution boundary

若本gate失败，只能定位为`implementation_or_theory_contract_mismatch`并回滚Step5/6；若通过，只证明实现忠于
PCSD-CF-v1设计且数值可微，不证明coupling-spectrum hypothesis、性能、paper-core effectiveness或SC2价值。
真正的effectiveness gate仍需单独授权后的Step7B validation-only matched end-to-end实验。
"""
    (output_dir / "step7a_local_gate_report.md").write_text(
        report,
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
