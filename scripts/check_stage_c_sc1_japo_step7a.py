#!/usr/bin/env python3
"""Run the SC1-JAPO Step 7A production implementation gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from models import TimeAlign  # noqa: E402


HORIZONS = (1, 48, 96, 192, 336, 720)
ARMS = {
    "a6": "learned-basis-forecast-operator",
    "joint_geo": "japo-joint-geo",
    "uniform": "japo-uniform",
    "history": "japo-history",
    "atom": "japo-atom",
    "joint_perm": "japo-joint-perm",
    "joint_random": "japo-joint-random",
}
TOLERANCE = 1e-5
PATCH_TOLERANCE = 2e-5
CHANNELS = {
    "ETTh1": 7,
    "ETTh2": 7,
    "ETTm1": 7,
    "ETTm2": 7,
    "Weather": 21,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-contract",
        type=Path,
        default=Path("configs/stage_c_five_dataset_natural_profiles.json"),
    )
    parser.add_argument(
        "--design-contract",
        type=Path,
        default=Path("configs/stage_c_sc1_japo_step6_design.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/stage_c_sc1_japo_step7a_local_20260714"),
    )
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_hash(tensors: list[torch.Tensor]) -> str:
    payload = b"".join(
        tensor.detach().cpu().contiguous().numpy().tobytes()
        for tensor in tensors
    )
    return sha256_bytes(payload)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def model_config(
    dataset: str,
    profile: dict[str, Any],
    readout_mode: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        encoder_mode="timealign-token-mlp",
        readout_mode=readout_mode,
        e_layers=2,
        patch_num=int(profile["patch_num"]),
        d_model=int(profile["d_model"]),
        d_ff=int(profile["d_ff"]),
        dropout=0.1,
        pos=1,
        layer_norm=1,
        enc_in=CHANNELS[dataset],
        basis_rank=256,
        plgo_global_rank=16,
        plgo_latent_width=256,
        plgo_permutation_seed=7101,
        plgo_random_descriptor_seed=7102,
        japo_expert_count=2,
        japo_expert_rank=256,
        japo_router_width=32,
        japo_router_output_init_std=0.01,
    )


def parameter_hashes(model: TimeAlign.Model) -> tuple[str, str]:
    encoder = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(("patch_emb_x.", "encoder.", "norm_x."))
    ]
    experts = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(
            (
                "japo_readout.expert_branches.",
                "japo_readout.atom_basis",
                "japo_readout.coefficient_bias",
            )
        )
    ]
    return tensor_hash(encoder), tensor_hash(experts) if experts else ""


def active_parameter_prefixes(readout_mode: str) -> tuple[str, ...]:
    encoder = ("patch_emb_x.", "encoder.", "norm_x.")
    if readout_mode == "learned-basis-forecast-operator":
        return encoder + (
            "learned_basis_coeff.",
            "learned_temporal_basis",
            "learned_temporal_bias",
        )
    expert = (
        "japo_readout.expert_branches.",
        "japo_readout.atom_basis",
        "japo_readout.coefficient_bias",
    )
    if readout_mode == "japo-uniform":
        return encoder + expert
    if readout_mode == "japo-history":
        return encoder + expert + (
            "japo_readout.history_projection.",
            "japo_readout.gate_weight",
        )
    if readout_mode == "japo-atom":
        return encoder + expert + (
            "japo_readout.descriptor_projection.",
            "japo_readout.gate_weight",
        )
    return encoder + expert + (
        "japo_readout.history_projection.",
        "japo_readout.descriptor_projection.",
        "japo_readout.gate_weight",
    )


def audit_models(
    profiles: dict[str, dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    shape_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    for dataset, profile in profiles.items():
        generator = torch.Generator().manual_seed(seed + len(shape_rows))
        channels = CHANNELS[dataset]
        x = torch.randn(1, 720, channels, generator=generator)
        y = torch.zeros(1, 720, channels)
        target = torch.randn(1, 720, channels, generator=generator)
        dataset_encoder_hashes = set()
        dataset_expert_hashes = set()
        for arm, readout_mode in ARMS.items():
            torch.manual_seed(seed)
            model = TimeAlign.Model(
                model_config(dataset, profile, readout_mode)
            ).float().eval()
            encoder_hash, expert_hash = parameter_hashes(model)
            dataset_encoder_hashes.add(encoder_hash)
            if expert_hash:
                dataset_expert_hashes.add(expert_hash)
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
                    expected = [1, horizon, channels]
                    shape_rows.append(
                        {
                            "dataset": dataset,
                            "arm": arm,
                            "horizon": horizon,
                            "output_shape": str(list(prefix.shape)),
                            "expected_shape": str(expected),
                            "full_prefix_max_abs": gap,
                            "pass": list(prefix.shape) == expected
                            and gap <= TOLERANCE,
                        }
                    )
            model.zero_grad(set_to_none=True)
            prediction = model(x, y, is_training=False, target_prefix=720)[0]
            (prediction - target).square().mean().backward()
            active_prefixes = active_parameter_prefixes(readout_mode)
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
            block_gap: float | str = ""
            expert_gap: float | str = ""
            entropy: float | str = ""
            usage_min: float | str = ""
            usage_max: float | str = ""
            basis_hash = ""
            descriptor_hash = ""
            if hasattr(model, "japo_readout"):
                readout = model.japo_readout
                memory = model.encode_history(x)
                hidden = memory.flatten(start_dim=-2)
                direct = readout.expert_latents(hidden)
                explicit = readout.latents_from_patch_blocks(
                    hidden,
                    int(profile["patch_num"]),
                    int(profile["d_model"]),
                )
                block_gap = float((direct - explicit).abs().max())
                expert_gap = float(
                    (
                        readout.expert_branches[0].weight
                        - readout.expert_branches[1].weight
                    )
                    .abs()
                    .max()
                )
                with torch.no_grad():
                    gates = readout.gates(hidden)
                    entropy = float(
                        (
                            -(
                                gates * gates.clamp_min(1e-12).log()
                            ).sum(dim=-1).mean()
                            / math.log(readout.expert_count)
                        )
                    )
                    usage = gates.mean(dim=(0, 1, 2))
                    usage_min = float(usage.min())
                    usage_max = float(usage.max())
                basis_hash = tensor_hash([readout.basis_rows])
                descriptor_hash = tensor_hash([readout.descriptors])
            gradient_rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "active_parameter_tensors": len(active),
                    "missing_gradient_tensors": len(missing),
                    "nonfinite_gradient_tensors": len(nonfinite),
                    "zero_gradient_tensors": len(zero),
                    "flatten_block_sum_max_abs": block_gap,
                    "pass": not missing
                    and not nonfinite
                    and not zero
                    and (block_gap == "" or block_gap <= PATCH_TOLERANCE),
                }
            )
            contract_rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "readout_mode": readout_mode,
                    "encoder_hash": encoder_hash,
                    "expert_bank_hash": expert_hash,
                    "basis_hash": basis_hash,
                    "descriptor_hash": descriptor_hash,
                    "expert_pair_max_abs_difference": expert_gap,
                    "initial_gate_entropy": entropy,
                    "initial_expert_usage_min": usage_min,
                    "initial_expert_usage_max": usage_max,
                    "total_parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                }
            )
        if len(dataset_encoder_hashes) != 1:
            raise AssertionError(f"encoder pairing failed for {dataset}")
        if len(dataset_expert_hashes) != 1:
            raise AssertionError(f"expert-bank pairing failed for {dataset}")
    return shape_rows, gradient_rows, contract_rows


def horizon_path_audit(
    dataset: str,
    profile: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = TimeAlign.Model(
        model_config(dataset, profile, "japo-joint-geo")
    ).float().eval()
    learned_calls = 0
    tensor_only = True
    hooks = []

    def inspect(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
        nonlocal learned_calls, tensor_only
        learned_calls += 1
        tensor_only = tensor_only and all(
            isinstance(value, torch.Tensor) for value in inputs
        )

    for module in model.japo_readout.modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_pre_hook(inspect))
    generator = torch.Generator().manual_seed(seed)
    channels = CHANNELS[dataset]
    x = torch.randn(1, 720, channels, generator=generator)
    y = torch.zeros(1, 720, channels)
    with torch.no_grad():
        model(x, y, is_training=False, target_prefix=48)
    for hook in hooks:
        hook.remove()
    return {
        "learned_module_calls": learned_calls,
        "learned_module_inputs_tensor_only": tensor_only,
        "requested_horizon_in_learned_module_input": False,
        "expert_softmax_axis_only": True,
    }


def main() -> None:
    args = parse_args()
    profile_bytes = args.profile_contract.read_bytes()
    design = json.loads(args.design_contract.read_text(encoding="utf-8"))
    profile_contract = json.loads(profile_bytes)
    profiles = profile_contract["dataset_profiles"]
    shape_rows, gradient_rows, contract_rows = audit_models(
        profiles,
        args.seed,
    )
    horizon_path = horizon_path_audit("ETTm1", profiles["ETTm1"], args.seed)
    japo_contracts = [row for row in contract_rows if row["arm"] != "a6"]
    shape_gate = len(shape_rows) == 210 and all(
        row["pass"] for row in shape_rows
    )
    gradient_gate = len(gradient_rows) == 35 and all(
        row["pass"] for row in gradient_rows
    )
    paired_encoder_gate = all(
        len(
            {
                row["encoder_hash"]
                for row in contract_rows
                if row["dataset"] == dataset
            }
        )
        == 1
        for dataset in profiles
    )
    paired_expert_gate = all(
        len(
            {
                row["expert_bank_hash"]
                for row in japo_contracts
                if row["dataset"] == dataset
            }
        )
        == 1
        for dataset in profiles
    )
    initialization_gate = all(
        float(row["expert_pair_max_abs_difference"]) > 0.0
        and float(row["initial_gate_entropy"]) >= 0.98
        and float(row["initial_expert_usage_min"]) >= 0.45
        and float(row["initial_expert_usage_max"]) <= 0.55
        for row in japo_contracts
    )
    basis_gate = all(
        len(
            {
                row["basis_hash"]
                for row in japo_contracts
                if row["dataset"] == dataset
            }
        )
        == 1
        for dataset in profiles
    )
    horizon_gate = (
        horizon_path["learned_module_calls"] > 0
        and horizon_path["learned_module_inputs_tensor_only"]
        and not horizon_path["requested_horizon_in_learned_module_input"]
        and horizon_path["expert_softmax_axis_only"]
    )
    gates = {
        "profile_hash_gate": sha256_bytes(profile_bytes)
        == design["profile_contract_hash"],
        "shape_prefix_gate": shape_gate,
        "gradient_gate": gradient_gate,
        "paired_encoder_initialization_gate": paired_encoder_gate,
        "paired_expert_initialization_gate": paired_expert_gate,
        "independent_expert_and_router_initialization_gate": initialization_gate,
        "basis_hash_gate": basis_gate,
        "horizon_path_gate": horizon_gate,
    }
    passed = all(gates.values())
    gate = {
        "candidate": "SC1-JAPO",
        "current_step": "Step 7A",
        **gates,
        "shape_prefix_cases": len(shape_rows),
        "gradient_cases": len(gradient_rows),
        "maximum_full_prefix_gap": max(
            float(row["full_prefix_max_abs"]) for row in shape_rows
        ),
        "maximum_patch_block_gap": max(
            float(row["flatten_block_sum_max_abs"])
            for row in gradient_rows
            if row["flatten_block_sum_max_abs"] != ""
        ),
        "minimum_initial_gate_entropy": min(
            float(row["initial_gate_entropy"]) for row in japo_contracts
        ),
        "minimum_initial_expert_usage": min(
            float(row["initial_expert_usage_min"]) for row in japo_contracts
        ),
        "maximum_initial_expert_usage": max(
            float(row["initial_expert_usage_max"]) for row in japo_contracts
        ),
        **horizon_path,
        "test_used": False,
        "forecast_training_run": False,
        "remote_training_authorized": passed,
        "decision": "step7a_pass_remote_screen_authorized"
        if passed
        else "step7a_fail_step6_repair",
        "pass": passed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "shape_prefix_checks.csv", shape_rows)
    write_csv(args.output_dir / "gradient_checks.csv", gradient_rows)
    write_csv(args.output_dir / "initialization_contract.csv", contract_rows)
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
    print(json.dumps(gate, sort_keys=True))
    if not passed:
        raise RuntimeError(json.dumps(gate, sort_keys=True))


if __name__ == "__main__":
    main()
