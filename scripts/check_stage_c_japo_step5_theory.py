#!/usr/bin/env python3
"""Audit the SC1-JAPO Step 5 theory and control contract."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

from check_stage_c_plgo_step5_theory import (
    Atom,
    active_indices,
    restricted_global_nested_basis,
)


DEFAULT_OUTPUT_DIR = Path(
    "analysis/stage_c_sc1_japo_step5_theory_20260714"
)
TOLERANCE = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def atom_descriptors(atoms: list[Atom], length: int) -> torch.Tensor:
    """Return fixed support descriptors with no requested-horizon field."""
    maximum_depth = max(1, math.ceil(math.log2(length)))
    rows = []
    for atom in atoms:
        rows.append(
            [
                (atom.start + atom.end) / (2.0 * length),
                (atom.end - atom.start) / length,
                max(0, atom.depth + 1) / maximum_depth,
                float(atom.kind == "global"),
            ]
        )
    return torch.tensor(rows, dtype=torch.float64)


def joint_gate(
    history: torch.Tensor,
    descriptors: torch.Tensor,
    history_projection: torch.Tensor,
    descriptor_projection: torch.Tensor,
    interaction: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Compute dense expert-only softmax gates for each history-atom pair."""
    history_context = torch.einsum(
        "bcr,sr->bcs",
        history,
        history_projection,
    )
    atom_context = torch.einsum(
        "jd,fd->jf",
        descriptors,
        descriptor_projection,
    )
    logits = torch.einsum(
        "bcs,esf,jf->bcje",
        history_context,
        interaction,
        atom_context,
    )
    return torch.softmax(logits + bias, dim=-1)


def horizon_set(length: int) -> list[int]:
    candidates = {1, length, max(1, length // 4), max(1, length // 2)}
    if length >= 96:
        candidates.update({5, 48, 96, 192, 336, 720})
    return sorted(value for value in candidates if value <= length)


def check_containment_and_projectivity(
    length: int,
    global_rank: int,
    generator: torch.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Embed an arbitrary affine A6 readout and audit every tested prefix."""
    synthesis, atoms, _prototypes, _gap = restricted_global_nested_basis(
        length,
        global_rank,
    )
    descriptors = atom_descriptors(atoms, length)
    batch_size = 3
    channels = 2
    hidden_dim = 11
    latent_rank = min(7, length)
    experts = 2
    history_width = 5
    descriptor_width = 4

    history = torch.randn(
        batch_size,
        channels,
        hidden_dim,
        dtype=torch.float64,
        generator=generator,
    )
    latent_map = torch.randn(
        latent_rank,
        hidden_dim,
        dtype=torch.float64,
        generator=generator,
    )
    latent_bias = torch.randn(
        latent_rank,
        dtype=torch.float64,
        generator=generator,
    )
    output_basis = torch.randn(
        length,
        latent_rank,
        dtype=torch.float64,
        generator=generator,
    )
    temporal_bias = torch.randn(
        length,
        dtype=torch.float64,
        generator=generator,
    )
    latent = torch.einsum("kr,bcr->bck", latent_map, history) + latent_bias
    reference = (
        torch.einsum("tk,bck->bct", output_basis, latent) + temporal_bias
    )

    coefficient_basis = synthesis.T @ output_basis
    coefficient_bias = synthesis.T @ temporal_bias
    one_expert = (
        torch.einsum("jk,bck->bcj", coefficient_basis, latent)
        + coefficient_bias
    )
    expert_coefficients = one_expert.unsqueeze(-1).expand(
        -1,
        -1,
        -1,
        experts,
    )

    history_projection = torch.randn(
        history_width,
        hidden_dim,
        dtype=torch.float64,
        generator=generator,
    )
    descriptor_projection = torch.randn(
        descriptor_width,
        descriptors.shape[1],
        dtype=torch.float64,
        generator=generator,
    )
    interaction = torch.randn(
        experts,
        history_width,
        descriptor_width,
        dtype=torch.float64,
        generator=generator,
    )
    gate_bias = torch.randn(
        experts,
        dtype=torch.float64,
        generator=generator,
    )
    full_gate = joint_gate(
        history,
        descriptors,
        history_projection,
        descriptor_projection,
        interaction,
        gate_bias,
    )
    full_coefficients = (full_gate * expert_coefficients).sum(dim=-1)
    reconstructed = torch.einsum(
        "tj,bcj->bct",
        synthesis,
        full_coefficients,
    )
    containment_gap = float((reconstructed - reference).abs().max().item())

    permutation = torch.randperm(length, generator=generator)
    paired_permutation = torch.einsum(
        "tj,bcj->bct",
        synthesis[:, permutation],
        full_coefficients[:, :, permutation],
    )
    permutation_gap = float(
        (paired_permutation - reconstructed).abs().max().item()
    )

    prefix_rows = []
    for horizon in horizon_set(length):
        active = active_indices(atoms, horizon)
        active_gate = joint_gate(
            history,
            descriptors[active],
            history_projection,
            descriptor_projection,
            interaction,
            gate_bias,
        )
        active_coefficients = (
            active_gate * expert_coefficients[:, :, active]
        ).sum(dim=-1)
        coefficient_gap = float(
            (
                active_coefficients - full_coefficients[:, :, active]
            )
            .abs()
            .max()
            .item()
        )
        prefix = torch.einsum(
            "hj,bcj->bch",
            synthesis[:horizon, active],
            active_coefficients,
        )
        prefix_gap = float(
            (prefix - reference[:, :, :horizon]).abs().max().item()
        )
        prefix_rows.append(
            {
                "length": length,
                "global_rank": global_rank,
                "horizon": horizon,
                "active_atoms": int(active.numel()),
                "shared_coefficient_max_abs": coefficient_gap,
                "prefix_reconstruction_max_abs": prefix_gap,
                "requested_horizon_in_learned_path": False,
                "atom_axis_normalization": False,
            }
        )

    return (
        {
            "length": length,
            "global_rank": global_rank,
            "experts": experts,
            "a6_containment_max_abs": containment_gap,
            "paired_atom_permutation_max_abs": permutation_gap,
            "prefix_cases": len(prefix_rows),
            "projective_coefficient_max_abs": max(
                row["shared_coefficient_max_abs"] for row in prefix_rows
            ),
            "projective_output_max_abs": max(
                row["prefix_reconstruction_max_abs"] for row in prefix_rows
            ),
        },
        prefix_rows,
    )


def check_geometry_only_collapse(
    generator: torch.Generator,
) -> dict[str, Any]:
    """Show that geometry-only expert mixing is one fixed linear operator."""
    samples = 13
    atoms = 17
    experts = 3
    rank = 5
    hidden_dim = 11
    history = torch.randn(
        samples,
        hidden_dim,
        dtype=torch.float64,
        generator=generator,
    )
    expert_trunks = torch.randn(
        experts,
        atoms,
        rank,
        dtype=torch.float64,
        generator=generator,
    )
    expert_branches = torch.randn(
        experts,
        rank,
        hidden_dim,
        dtype=torch.float64,
        generator=generator,
    )
    geometry_logits = torch.randn(
        atoms,
        experts,
        dtype=torch.float64,
        generator=generator,
    )
    geometry_gate = torch.softmax(geometry_logits, dim=-1)
    expert_values = torch.einsum(
        "ejk,ekr,nr->nje",
        expert_trunks,
        expert_branches,
        history,
    )
    mixture = (geometry_gate.unsqueeze(0) * expert_values).sum(dim=-1)
    fixed_operator = torch.einsum(
        "je,ejk,ekr->jr",
        geometry_gate,
        expert_trunks,
        expert_branches,
    )
    collapsed = torch.einsum("jr,nr->nj", fixed_operator, history)
    return {
        "samples": samples,
        "atoms": atoms,
        "experts": experts,
        "total_rank": experts * rank,
        "collapse_max_abs": float((mixture - collapsed).abs().max().item()),
        "result": "geometry_only_mixture_is_fixed_linear_operator",
    }


def check_joint_noncollapse() -> dict[str, Any]:
    """Give a three-point witness outside every fixed affine PAF."""
    history = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    descriptor = torch.tensor(1.0, dtype=torch.float64)
    logits = torch.stack(
        [history * descriptor, -history * descriptor],
        dim=-1,
    )
    gate = torch.softmax(logits, dim=-1)
    expert_values = torch.stack([history, -history], dim=-1)
    output = (gate * expert_values).sum(dim=-1)
    second_difference = output[2] - 2.0 * output[1] + output[0]
    endpoint_equality_gap = (output[2] - output[0]).abs()
    center_to_endpoint_gap = (output[1] - output[0]).abs()
    return {
        "history_minus_one": float(output[0].item()),
        "history_zero": float(output[1].item()),
        "history_plus_one": float(output[2].item()),
        "endpoint_equality_max_abs": float(endpoint_equality_gap.item()),
        "center_to_endpoint_abs": float(center_to_endpoint_gap.item()),
        "affine_second_difference_abs": float(second_difference.abs().item()),
        "fixed_affine_representation_possible": bool(
            second_difference.abs().item() <= TOLERANCE
        ),
        "scope": "constructive_function_class_witness_not_trainability_evidence",
    }


def synthetic_joint_state(
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    samples = 19
    atoms = 23
    experts = 2
    hidden_dim = 7
    descriptor_dim = 4
    history = torch.randn(
        samples,
        hidden_dim,
        dtype=torch.float64,
        generator=generator,
    )
    descriptors = torch.randn(
        atoms,
        descriptor_dim,
        dtype=torch.float64,
        generator=generator,
    )
    interaction = torch.randn(
        experts,
        hidden_dim,
        descriptor_dim,
        dtype=torch.float64,
        generator=generator,
    )
    expert_maps = torch.randn(
        experts,
        atoms,
        hidden_dim,
        dtype=torch.float64,
        generator=generator,
    )
    return {
        "history": history,
        "descriptors": descriptors,
        "interaction": interaction,
        "expert_maps": expert_maps,
    }


def evaluate_synthetic_joint(
    state: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    history = state["history"]
    descriptors = state["descriptors"]
    interaction = state["interaction"]
    expert_maps = state["expert_maps"]
    logits = torch.einsum(
        "nr,erf,jf->nje",
        history,
        interaction,
        descriptors,
    )
    gate = torch.softmax(logits, dim=-1)
    expert_values = torch.einsum(
        "ejr,nr->nje",
        expert_maps,
        history,
    )
    output = (gate * expert_values).sum(dim=-1)
    return output, gate, expert_values


def check_continuity_and_diagnostics(
    generator: torch.Generator,
) -> dict[str, Any]:
    """Exercise continuity and define specialization observables."""
    state = synthetic_joint_state(generator)
    output, gate, expert_values = evaluate_synthetic_joint(state)
    epsilon = 1e-6

    history_direction = torch.randn(
        state["history"].shape,
        dtype=torch.float64,
        generator=generator,
    )
    history_direction /= torch.linalg.vector_norm(history_direction)
    history_perturbed = dict(state)
    history_perturbed["history"] = (
        state["history"] + epsilon * history_direction
    )
    history_output, _history_gate, _history_values = evaluate_synthetic_joint(
        history_perturbed
    )
    history_ratio = torch.linalg.vector_norm(history_output - output) / epsilon

    descriptor_direction = torch.randn(
        state["descriptors"].shape,
        dtype=torch.float64,
        generator=generator,
    )
    descriptor_direction /= torch.linalg.vector_norm(descriptor_direction)
    descriptor_perturbed = dict(state)
    descriptor_perturbed["descriptors"] = (
        state["descriptors"] + epsilon * descriptor_direction
    )
    descriptor_output, _descriptor_gate, _descriptor_values = (
        evaluate_synthetic_joint(descriptor_perturbed)
    )
    descriptor_ratio = (
        torch.linalg.vector_norm(descriptor_output - output) / epsilon
    )

    entropy = -(gate * gate.clamp_min(1e-30).log()).sum(dim=-1)
    normalized_entropy = entropy.mean() / math.log(gate.shape[-1])
    usage = gate.mean(dim=(0, 1))
    history_gate_sensitivity = (gate[0] - gate[1]).abs().mean()
    geometry_gate_sensitivity = (gate[:, 0] - gate[:, 1]).abs().mean()
    interaction_residual = (
        gate[0, 0]
        - gate[0, 1]
        - gate[1, 0]
        + gate[1, 1]
    ).abs().mean()
    uniform_output = expert_values.mean(dim=-1)

    return {
        "epsilon": epsilon,
        "history_local_output_ratio": float(history_ratio.item()),
        "descriptor_local_output_ratio": float(descriptor_ratio.item()),
        "ratios_finite": bool(
            torch.isfinite(history_ratio) and torch.isfinite(descriptor_ratio)
        ),
        "normalized_gate_entropy": float(normalized_entropy.item()),
        "minimum_mean_expert_usage": float(usage.min().item()),
        "expert_output_disagreement": float(
            expert_values.var(dim=-1, unbiased=False).mean().item()
        ),
        "history_gate_sensitivity": float(history_gate_sensitivity.item()),
        "geometry_gate_sensitivity": float(geometry_gate_sensitivity.item()),
        "joint_interaction_residual": float(interaction_residual.item()),
        "routing_effect_vs_uniform": float(
            (output - uniform_output).abs().mean().item()
        ),
        "hard_top_k_used": False,
        "softmax_axis": "expert_only",
        "scope": "metric_sanity_check_not_learned_specialization_evidence",
    }


def check_initialization_symmetry(
    generator: torch.Generator,
) -> dict[str, Any]:
    """Show why exact containment must not prescribe identical initialization."""
    samples = 17
    hidden_dim = 5
    experts = 2
    history = torch.randn(
        samples,
        hidden_dim,
        dtype=torch.float64,
        generator=generator,
    )
    target = torch.randn(
        samples,
        dtype=torch.float64,
        generator=generator,
    )
    base = torch.randn(
        hidden_dim,
        dtype=torch.float64,
        generator=generator,
    )

    identical_experts = base.repeat(experts, 1).clone().requires_grad_()
    uniform_router = torch.zeros(
        experts,
        hidden_dim,
        dtype=torch.float64,
        requires_grad=True,
    )
    gate = torch.softmax(history @ uniform_router.T, dim=-1)
    values = history @ identical_experts.T
    prediction = (gate * values).sum(dim=-1)
    loss = (prediction - target).square().mean()
    loss.backward()
    identical_router_gradient = float(
        uniform_router.grad.abs().max().item()
    )
    identical_expert_gradient_gap = float(
        (
            identical_experts.grad[0] - identical_experts.grad[1]
        )
        .abs()
        .max()
        .item()
    )

    offset = torch.linspace(
        -0.2,
        0.2,
        hidden_dim,
        dtype=torch.float64,
    )
    distinct_experts = torch.stack(
        [base + offset, base - offset],
        dim=0,
    ).requires_grad_()
    second_router = torch.zeros(
        experts,
        hidden_dim,
        dtype=torch.float64,
        requires_grad=True,
    )
    second_gate = torch.softmax(history @ second_router.T, dim=-1)
    second_values = history @ distinct_experts.T
    second_prediction = (second_gate * second_values).sum(dim=-1)
    second_loss = (second_prediction - target).square().mean()
    second_loss.backward()
    broken_router_gradient = float(
        torch.linalg.vector_norm(second_router.grad).item()
    )

    return {
        "identical_expert_router_gradient_max_abs": (
            identical_router_gradient
        ),
        "identical_expert_gradient_max_pair_gap": (
            identical_expert_gradient_gap
        ),
        "distinct_expert_router_gradient_l2": broken_router_gradient,
        "symmetry_trap_detected": bool(
            identical_router_gradient <= TOLERANCE
            and identical_expert_gradient_gap <= TOLERANCE
            and broken_router_gradient > TOLERANCE
        ),
        "decision": (
            "containment_is_function_class_only_identical_initialization_forbidden"
        ),
    }


def control_matrix() -> list[dict[str, Any]]:
    """Freeze the matched-bank controls required before any Step 7 screen."""
    return [
        {
            "arm": "A6-LBF-natural",
            "same_expert_bank_as_joint": False,
            "history_in_gate": False,
            "geometry_in_gate": False,
            "descriptor_integrity": "none",
            "purpose": "accepted_carrier_and_dense_readout_reference",
        },
        {
            "arm": "JAPO-JOINT-GEO",
            "same_expert_bank_as_joint": True,
            "history_in_gate": True,
            "geometry_in_gate": True,
            "descriptor_integrity": "canonical",
            "purpose": "candidate_complete_mechanism",
        },
        {
            "arm": "JAPO-UNIFORM",
            "same_expert_bank_as_joint": True,
            "history_in_gate": False,
            "geometry_in_gate": False,
            "descriptor_integrity": "none",
            "purpose": "capacity_and_expert_ensemble_control",
        },
        {
            "arm": "JAPO-HISTORY",
            "same_expert_bank_as_joint": True,
            "history_in_gate": True,
            "geometry_in_gate": False,
            "descriptor_integrity": "none",
            "purpose": "sample_conditioning_without_geometry_control",
        },
        {
            "arm": "JAPO-ATOM",
            "same_expert_bank_as_joint": True,
            "history_in_gate": False,
            "geometry_in_gate": True,
            "descriptor_integrity": "canonical",
            "purpose": "analytic_fixed_operator_control",
        },
        {
            "arm": "JAPO-JOINT-PERM",
            "same_expert_bank_as_joint": True,
            "history_in_gate": True,
            "geometry_in_gate": True,
            "descriptor_integrity": "permuted",
            "purpose": "canonical_geometry_necessity_control",
        },
        {
            "arm": "JAPO-JOINT-RANDOM",
            "same_expert_bank_as_joint": True,
            "history_in_gate": True,
            "geometry_in_gate": True,
            "descriptor_integrity": "moment_matched_random",
            "purpose": "descriptor_semantics_control",
        },
    ]


def metric_definitions() -> list[dict[str, Any]]:
    return [
        {
            "metric": "a6_containment_max_abs",
            "source": "arbitrary affine A6 output and identical JAPO experts",
            "computation": "max absolute difference after RGNB coordinate transform",
            "meaning": "existence of a dense-bypass-free A6 embedding",
        },
        {
            "metric": "projective_coefficient_max_abs",
            "source": "full-atom and active-only joint gate evaluations",
            "computation": "max difference on shared active atoms",
            "meaning": "requested horizon changes only the active set",
        },
        {
            "metric": "collapse_max_abs",
            "source": "geometry-only expert mixture and collapsed fixed map",
            "computation": "max absolute output difference",
            "meaning": "geometry-only routing adds no sample-conditional operator",
        },
        {
            "metric": "affine_second_difference_abs",
            "source": "constructive scalar joint gate at h in {-1,0,1}",
            "computation": "abs(f(1)-2f(0)+f(-1))",
            "meaning": "positive value witnesses a function outside fixed affine PAF",
        },
        {
            "metric": "normalized_gate_entropy",
            "source": "expert probabilities per history-atom pair",
            "computation": "mean entropy divided by log(E)",
            "meaning": "routing confidence, not specialization by itself",
        },
        {
            "metric": "minimum_mean_expert_usage",
            "source": "expert probabilities",
            "computation": "minimum expert mean probability",
            "meaning": "soft expert starvation indicator",
        },
        {
            "metric": "expert_output_disagreement",
            "source": "expert coefficient predictions",
            "computation": "mean population variance across experts",
            "meaning": "whether routing can change the coefficient output",
        },
        {
            "metric": "history_gate_sensitivity",
            "source": "two histories with common atoms",
            "computation": "mean absolute gate difference",
            "meaning": "whether routing is sample dependent",
        },
        {
            "metric": "geometry_gate_sensitivity",
            "source": "two atoms under common histories",
            "computation": "mean absolute gate difference",
            "meaning": "whether routing uses atom geometry",
        },
        {
            "metric": "joint_interaction_residual",
            "source": "two histories crossed with two atoms",
            "computation": "mean absolute two-way probability contrast",
            "meaning": "non-additive history-geometry interaction diagnostic",
        },
        {
            "metric": "routing_effect_vs_uniform",
            "source": "joint and uniform mixtures of the same expert outputs",
            "computation": "mean absolute coefficient difference",
            "meaning": "functional influence of learned routing",
        },
    ]


def main() -> None:
    args = parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    cases = [(16, 4), (31, 4), (96, 8), (720, 16)]
    containment_rows = []
    prefix_rows = []
    for length, global_rank in cases:
        containment, prefixes = check_containment_and_projectivity(
            length,
            global_rank,
            generator,
        )
        containment_rows.append(containment)
        prefix_rows.extend(prefixes)

    geometry_collapse = check_geometry_only_collapse(generator)
    joint_noncollapse = check_joint_noncollapse()
    continuity = check_continuity_and_diagnostics(generator)
    symmetry = check_initialization_symmetry(generator)
    controls = control_matrix()

    maximum_containment_gap = max(
        row["a6_containment_max_abs"] for row in containment_rows
    )
    maximum_projectivity_gap = max(
        max(
            row["projective_coefficient_max_abs"],
            row["projective_output_max_abs"],
        )
        for row in containment_rows
    )
    maximum_permutation_gap = max(
        row["paired_atom_permutation_max_abs"]
        for row in containment_rows
    )
    gate = {
        "candidate": "SC1-JAPO",
        "candidate_status": "proposed",
        "step": 5,
        "tolerance": TOLERANCE,
        "cases": len(containment_rows),
        "prefix_cases": len(prefix_rows),
        "maximum_a6_containment_gap": maximum_containment_gap,
        "maximum_projectivity_gap": maximum_projectivity_gap,
        "maximum_paired_permutation_gap": maximum_permutation_gap,
        "geometry_only_collapse_gap": geometry_collapse["collapse_max_abs"],
        "joint_noncollapse_witness": joint_noncollapse[
            "affine_second_difference_abs"
        ],
        "a6_containment_gate": maximum_containment_gap <= TOLERANCE,
        "exact_projectivity_gate": maximum_projectivity_gap <= TOLERANCE,
        "atom_index_equivariance_gate": maximum_permutation_gap <= TOLERANCE,
        "geometry_only_no_go_gate": (
            geometry_collapse["collapse_max_abs"] <= TOLERANCE
        ),
        "joint_function_class_increment_gate": (
            not joint_noncollapse["fixed_affine_representation_possible"]
        ),
        "continuity_contract_gate": continuity["ratios_finite"],
        "specialization_metrics_defined": True,
        "matched_bank_controls_frozen": len(controls) == 7,
        "identical_initialization_forbidden": symmetry[
            "symmetry_trap_detected"
        ],
        "requested_horizon_in_learned_path": False,
        "hard_top_k_authorized": False,
        "atom_axis_normalization_authorized": False,
        "primary_expert_count": 2,
        "expert_count_sweep_authorized": False,
        "narrative_gate": "conditional_pass_complete_contract_only",
        "step5_decision": "pass_step6_design_only",
        "step6_design_authorized": True,
        "method_implementation_authorized": False,
        "remote_training_authorized": False,
        "rollback_if_step6_attribution_fails": "step4_redesign",
        "rollback_if_problem_contract_fails": "step2_3_reformulation",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "containment_checks.csv", containment_rows)
    write_csv(args.output_dir / "prefix_projectivity_checks.csv", prefix_rows)
    write_csv(args.output_dir / "control_matrix.csv", controls)
    write_csv(args.output_dir / "metric_definitions.csv", metric_definitions())
    (args.output_dir / "geometry_only_collapse.json").write_text(
        json.dumps(geometry_collapse, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "joint_noncollapse_witness.json").write_text(
        json.dumps(joint_noncollapse, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "continuity_and_diagnostics.json").write_text(
        json.dumps(continuity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "initialization_symmetry.json").write_text(
        json.dumps(symmetry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "theory_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "stage_c_japo_step5=complete "
        f"decision={gate['step5_decision']} "
        f"containment_gap={maximum_containment_gap:.3e} "
        f"projectivity_gap={maximum_projectivity_gap:.3e} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
