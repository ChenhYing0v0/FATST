#!/usr/bin/env python3
"""Audit the SC1-JAPO Step 6 method and control design."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from layers.PLGO import (  # noqa: E402
    canonical_atom_descriptors,
    descriptor_family,
    restricted_global_nested_basis,
)


DEFAULT_CONFIG = Path("configs/stage_c_sc1_japo_step6_design.json")
DEFAULT_OUTPUT_DIR = Path(
    "analysis/stage_c_sc1_japo_step6_design_20260714"
)
HORIZONS = (1, 48, 96, 192, 336, 720)
TOLERANCE = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize_linear(
    output_dim: int,
    input_dim: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.empty(output_dim, input_dim, dtype=torch.float64)
    torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5), generator=generator)
    bound = 1.0 / math.sqrt(input_dim)
    bias = torch.empty(output_dim, dtype=torch.float64).uniform_(
        -bound,
        bound,
        generator=generator,
    )
    return weight, bias


def rms_normalize(value: torch.Tensor) -> torch.Tensor:
    return value / value.square().mean(dim=-1, keepdim=True).add(1e-8).sqrt()


def project_context(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    projected = torch.einsum("...r,gr->...g", value, weight) + bias
    return rms_normalize(torch.tanh(projected))


def router_gate(
    arm: str,
    history_context: torch.Tensor,
    atom_context: torch.Tensor,
    gate_weight: torch.Tensor,
) -> torch.Tensor:
    width = history_context.shape[-1]
    if arm == "JAPO-UNIFORM":
        shape = (*history_context.shape[:-1], atom_context.shape[0], 2)
        return torch.full(shape, 0.5, dtype=history_context.dtype)
    if arm == "JAPO-HISTORY":
        features = history_context.unsqueeze(-2).expand(
            *history_context.shape[:-1],
            atom_context.shape[0],
            width,
        )
    elif arm == "JAPO-ATOM":
        features = atom_context.view(
            *((1,) * (history_context.ndim - 1)),
            atom_context.shape[0],
            width,
        ).expand(*history_context.shape[:-1], atom_context.shape[0], width)
    else:
        features = history_context.unsqueeze(-2) * atom_context.view(
            *((1,) * (history_context.ndim - 1)),
            atom_context.shape[0],
            width,
        )
        features = rms_normalize(features)
    logits = torch.einsum("...jg,eg->...je", features, gate_weight)
    logits = logits / math.sqrt(width)
    return torch.softmax(logits, dim=-1)


def expert_coefficients(
    history: torch.Tensor,
    branch_weight: torch.Tensor,
    branch_bias: torch.Tensor,
    atom_basis: torch.Tensor,
    coefficient_bias: torch.Tensor,
) -> torch.Tensor:
    latent = torch.einsum(
        "ekr,bcr->bcek",
        branch_weight,
        history,
    ) + branch_bias.view(1, 1, *branch_bias.shape)
    coefficient = torch.einsum(
        "bcek,ejk->bcje",
        latent,
        atom_basis,
    )
    return coefficient + coefficient_bias.T.view(
        1,
        1,
        coefficient_bias.shape[1],
        coefficient_bias.shape[0],
    )


def parameter_count(
    readout_dim: int,
    length: int,
    rank: int,
    experts: int,
    router_width: int,
    descriptor_dim: int,
) -> tuple[int, int, int]:
    a6 = rank * readout_dim + rank + length * rank + length
    bank = experts * (
        rank * readout_dim + rank + length * rank + length
    )
    router = (
        router_width * readout_dim
        + router_width
        + router_width * descriptor_dim
        + router_width
        + experts * router_width
    )
    return a6, bank, router


def arm_contract() -> list[dict[str, Any]]:
    return [
        {
            "arm": "A6-LBF-natural",
            "expert_bank": "none",
            "gate": "none",
            "descriptor": "none",
            "primary_role": "accepted_carrier_reference",
        },
        {
            "arm": "JAPO-JOINT-GEO",
            "expert_bank": "paired_e2_k256",
            "gate": "history_times_atom",
            "descriptor": "canonical",
            "primary_role": "complete_candidate",
        },
        {
            "arm": "JAPO-UNIFORM",
            "expert_bank": "paired_e2_k256",
            "gate": "fixed_half_half",
            "descriptor": "none",
            "primary_role": "capacity_control",
        },
        {
            "arm": "JAPO-HISTORY",
            "expert_bank": "paired_e2_k256",
            "gate": "history_only",
            "descriptor": "none",
            "primary_role": "sample_conditioning_control",
        },
        {
            "arm": "JAPO-ATOM",
            "expert_bank": "paired_e2_k256",
            "gate": "atom_only",
            "descriptor": "canonical",
            "primary_role": "geometry_only_fixed_operator_control",
        },
        {
            "arm": "JAPO-JOINT-PERM",
            "expert_bank": "paired_e2_k256",
            "gate": "history_times_atom",
            "descriptor": "permuted",
            "primary_role": "geometry_alignment_control",
        },
        {
            "arm": "JAPO-JOINT-RANDOM",
            "expert_bank": "paired_e2_k256",
            "gate": "history_times_atom",
            "descriptor": "moment_matched_random",
            "primary_role": "descriptor_semantics_control",
        },
    ]


def screen_gate_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    screening = config["screening_gate"]
    confirmation = config["confirmation_gate"]
    return [
        {
            "stage": "seed2021_immediate_fail",
            "comparison": "joint_vs_a6",
            "macro_requirement": (
                f"<= {screening['immediate_fail']['joint_vs_a6_macro_max']}"
            ),
            "dataset_requirement": (
                "positive <= "
                f"{screening['immediate_fail']['joint_vs_a6_positive_datasets_max']}"
            ),
            "decision": "stop_exact_design_and_attribute",
        },
        {
            "stage": "seed2021_immediate_fail",
            "comparison": "joint_vs_same_bank_median",
            "macro_requirement": (
                "<= "
                f"{screening['immediate_fail']['joint_vs_same_bank_median_macro_max']}"
            ),
            "dataset_requirement": (
                "positive <= "
                f"{screening['immediate_fail']['joint_vs_same_bank_median_positive_datasets_max']}"
            ),
            "decision": "same_bank_controls_explain",
        },
        {
            "stage": "seed2021_provisional_pass",
            "comparison": "joint_vs_a6",
            "macro_requirement": "> 0",
            "dataset_requirement": (
                "positive >= "
                f"{screening['provisional_pass']['joint_vs_a6_positive_datasets_min']}"
            ),
            "decision": "check_all_same_bank_controls",
        },
        {
            "stage": "seed2021_provisional_pass",
            "comparison": "joint_vs_each_same_bank_control",
            "macro_requirement": "> 0 for every control",
            "dataset_requirement": (
                "positive >= "
                f"{screening['provisional_pass']['joint_vs_each_control_positive_datasets_min']}"
            ),
            "decision": "provisional_pass_or_seed2022_if_ambiguous",
        },
        {
            "stage": "three_seed_confirmation",
            "comparison": "joint_vs_a6_and_each_control",
            "macro_requirement": "> 0; median >= 1 percent",
            "dataset_requirement": (
                "A6 positive >= "
                f"{confirmation['joint_vs_a6_positive_datasets_min']}; "
                "each control positive >= "
                f"{confirmation['joint_vs_each_control_positive_datasets_min']}"
            ),
            "decision": "paper_core_effectiveness_candidate_if_guards_pass",
        },
        {
            "stage": "three_seed_confirmation",
            "comparison": "mae_and_horizon_segments",
            "macro_requirement": (
                f"MAE >= {confirmation['mae_macro_guard_min']}; "
                "short > 0; long > 0"
            ),
            "dataset_requirement": "report_all_five_datasets",
            "decision": "guard_against_auc_only_gain",
        },
    ]


def audit_profile(
    dataset: str,
    readout_dim: int,
    architecture: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    length = architecture["series_length"]
    global_rank = architecture["global_rank"]
    experts = architecture["expert_count"]
    rank = architecture["expert_rank"]
    router_width = architecture["router_width"]
    generator = torch.Generator().manual_seed(seed)

    synthesis, atoms = restricted_global_nested_basis(length, global_rank)
    synthesis = synthesis.to(torch.float64)
    canonical = canonical_atom_descriptors(atoms).to(torch.float64)
    descriptors = {
        "JAPO-JOINT-GEO": canonical,
        "JAPO-JOINT-PERM": descriptor_family(
            canonical.float(),
            "perm",
        ).to(torch.float64),
        "JAPO-JOINT-RANDOM": descriptor_family(
            canonical.float(),
            "random",
        ).to(torch.float64),
        "JAPO-ATOM": canonical,
    }

    branch_weights = []
    branch_biases = []
    for _expert in range(experts):
        weight, bias = initialize_linear(rank, readout_dim, generator)
        branch_weights.append(weight)
        branch_biases.append(bias)
    branch_weight = torch.stack(branch_weights).requires_grad_()
    branch_bias = torch.stack(branch_biases).requires_grad_()
    atom_basis = torch.randn(
        experts,
        length,
        rank,
        dtype=torch.float64,
        generator=generator,
    ) * math.sqrt(experts / rank)
    atom_basis.requires_grad_()
    coefficient_bias = torch.zeros(
        experts,
        length,
        dtype=torch.float64,
        requires_grad=True,
    )
    history_weight, history_bias = initialize_linear(
        router_width,
        readout_dim,
        generator,
    )
    descriptor_weight, descriptor_bias = initialize_linear(
        router_width,
        canonical.shape[1],
        generator,
    )
    gate_weight = torch.randn(
        experts,
        router_width,
        dtype=torch.float64,
        generator=generator,
    ) * architecture["router_output_init_std"]
    history_weight.requires_grad_()
    history_bias.requires_grad_()
    descriptor_weight.requires_grad_()
    descriptor_bias.requires_grad_()
    gate_weight.requires_grad_()

    history = torch.randn(
        2,
        3,
        readout_dim,
        dtype=torch.float64,
        generator=generator,
    )
    normalized_history = F.layer_norm(history, (readout_dim,))
    history_context = project_context(
        normalized_history,
        history_weight,
        history_bias,
    )
    coefficients = expert_coefficients(
        history,
        branch_weight,
        branch_bias,
        atom_basis,
        coefficient_bias,
    )

    gates: dict[str, torch.Tensor] = {}
    mixtures: dict[str, torch.Tensor] = {}
    japo_arms = [row["arm"] for row in arm_contract()[1:]]
    for arm in japo_arms:
        descriptor = descriptors.get(arm, canonical)
        atom_context = project_context(
            descriptor,
            descriptor_weight,
            descriptor_bias,
        )
        gate = router_gate(
            arm,
            history_context,
            atom_context,
            gate_weight,
        )
        gates[arm] = gate
        mixtures[arm] = (gate * coefficients).sum(dim=-1)

    full = torch.einsum(
        "tj,bcj->bct",
        synthesis,
        mixtures["JAPO-JOINT-GEO"],
    )
    prefix_gap = 0.0
    for horizon in HORIZONS:
        active = torch.tensor(
            [index for index, atom in enumerate(atoms) if atom.start < horizon],
            dtype=torch.long,
        )
        atom_context = project_context(
            canonical[active],
            descriptor_weight,
            descriptor_bias,
        )
        active_gate = router_gate(
            "JAPO-JOINT-GEO",
            history_context,
            atom_context,
            gate_weight,
        )
        active_mix = (
            active_gate * coefficients[:, :, active]
        ).sum(dim=-1)
        prefix = torch.einsum(
            "hj,bcj->bch",
            synthesis[:horizon, active],
            active_mix,
        )
        prefix_gap = max(
            prefix_gap,
            float((prefix - full[:, :, :horizon]).abs().max().item()),
        )

    loss = full.square().mean()
    loss.backward()
    gradient_tensors = [
        branch_weight.grad,
        branch_bias.grad,
        atom_basis.grad,
        coefficient_bias.grad,
        history_weight.grad,
        history_bias.grad,
        descriptor_weight.grad,
        descriptor_bias.grad,
        gate_weight.grad,
    ]
    gradients_finite = all(
        gradient is not None and bool(torch.isfinite(gradient).all())
        for gradient in gradient_tensors
    )
    gradients_nonzero = all(
        gradient is not None and float(gradient.abs().max().item()) > 0.0
        for gradient in gradient_tensors
    )

    joint_gate = gates["JAPO-JOINT-GEO"]
    entropy = -(
        joint_gate * joint_gate.clamp_min(1e-30).log()
    ).sum(dim=-1).mean() / math.log(experts)
    usage = joint_gate.mean(dim=(0, 1, 2))
    uniform_gap = float(
        (
            mixtures["JAPO-UNIFORM"] - coefficients.mean(dim=-1)
        )
        .abs()
        .max()
        .item()
    )
    control_effect_min = min(
        float(
            (
                mixtures["JAPO-JOINT-GEO"] - mixtures[arm]
            )
            .abs()
            .mean()
            .item()
        )
        for arm in japo_arms
        if arm != "JAPO-JOINT-GEO"
    )
    expert_difference = float(
        (branch_weight[0] - branch_weight[1]).abs().max().item()
    )
    a6_parameters, bank_parameters, router_parameters = parameter_count(
        readout_dim,
        length,
        rank,
        experts,
        router_width,
        canonical.shape[1],
    )
    return {
        "dataset": dataset,
        "readout_dim": readout_dim,
        "a6_readout_parameters": a6_parameters,
        "japo_expert_bank_parameters": bank_parameters,
        "japo_router_parameters": router_parameters,
        "japo_to_a6_readout_ratio": (
            (bank_parameters + router_parameters) / a6_parameters
        ),
        "expert_pair_max_abs_difference": expert_difference,
        "initial_gate_entropy": float(entropy.item()),
        "initial_expert_usage_min": float(usage.min().item()),
        "initial_expert_usage_max": float(usage.max().item()),
        "uniform_control_max_abs": uniform_gap,
        "minimum_control_functional_effect": control_effect_min,
        "prefix_projectivity_max_abs": prefix_gap,
        "all_joint_gradients_finite": gradients_finite,
        "all_joint_gradients_nonzero": gradients_nonzero,
        "uniform_variance_ratio_theory": (
            experts * experts * (1.0 / experts) ** 2
        ),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    profile_path = Path(config["profile_contract"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile_hash = sha256(profile_path)
    architecture = config["architecture"]

    profile_rows = []
    for offset, (dataset, values) in enumerate(
        profile["dataset_profiles"].items()
    ):
        profile_rows.append(
            audit_profile(
                dataset,
                int(values["state_width"]),
                architecture,
                args.seed + offset,
            )
        )

    arms = arm_contract()
    screen_rows = screen_gate_rows(config)
    design_gate = {
        "candidate": "SC1-JAPO",
        "candidate_status": "narrative_ready",
        "step": 6,
        "profile_hash_expected": config["profile_contract_hash"],
        "profile_hash_observed": profile_hash,
        "profile_hash_gate": profile_hash == config["profile_contract_hash"],
        "datasets": len(profile_rows),
        "arms": len(arms),
        "seed2021_screen_jobs": len(profile_rows) * len(arms),
        "maximum_prefix_projectivity_gap": max(
            row["prefix_projectivity_max_abs"] for row in profile_rows
        ),
        "minimum_initial_gate_entropy": min(
            row["initial_gate_entropy"] for row in profile_rows
        ),
        "minimum_initial_expert_usage": min(
            row["initial_expert_usage_min"] for row in profile_rows
        ),
        "maximum_initial_expert_usage": max(
            row["initial_expert_usage_max"] for row in profile_rows
        ),
        "minimum_control_functional_effect": min(
            row["minimum_control_functional_effect"] for row in profile_rows
        ),
        "maximum_uniform_control_gap": max(
            row["uniform_control_max_abs"] for row in profile_rows
        ),
        "independent_expert_initialization_gate": all(
            row["expert_pair_max_abs_difference"] > 0.0
            for row in profile_rows
        ),
        "router_initialization_gate": all(
            row["initial_gate_entropy"] >= 0.98
            and row["initial_expert_usage_min"] >= 0.45
            and row["initial_expert_usage_max"] <= 0.55
            for row in profile_rows
        ),
        "gradient_path_gate": all(
            row["all_joint_gradients_finite"]
            and row["all_joint_gradients_nonzero"]
            for row in profile_rows
        ),
        "prefix_projectivity_gate": all(
            row["prefix_projectivity_max_abs"] <= TOLERANCE
            for row in profile_rows
        ),
        "uniform_control_gate": all(
            row["uniform_control_max_abs"] <= TOLERANCE
            for row in profile_rows
        ),
        "controls_functionally_distinct_gate": all(
            row["minimum_control_functional_effect"] > TOLERANCE
            for row in profile_rows
        ),
        "requested_horizon_in_learned_path": False,
        "hard_top_k_authorized": False,
        "auxiliary_routing_loss_authorized": False,
        "test_allowed": config["training_protocol"]["test_allowed"],
        "narrative_gate": "pass_complete_contract_with_mandatory_controls",
        "step6_decision": "narrative_ready_step7a_local_implementation_only",
        "step7a_implementation_authorized": True,
        "remote_training_authorized": False,
        "sc2_authorized": False,
        "rollback_if_step7a_fails": "step6_design_repair",
        "rollback_if_effectiveness_fails": "step4_or_step2_3_by_attribution",
    }
    boolean_gates = [
        value
        for key, value in design_gate.items()
        if key.endswith("_gate") and isinstance(value, bool)
    ]
    design_gate["all_design_gates_pass"] = all(boolean_gates)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "profile_design_checks.csv", profile_rows)
    write_csv(args.output_dir / "arm_contract.csv", arms)
    write_csv(args.output_dir / "screen_gate_matrix.csv", screen_rows)
    (args.output_dir / "design_gate.json").write_text(
        json.dumps(design_gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "stage_c_japo_step6=complete "
        f"decision={design_gate['step6_decision']} "
        f"all_gates={design_gate['all_design_gates_pass']} "
        f"prefix_gap={design_gate['maximum_prefix_projectivity_gap']:.3e} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
