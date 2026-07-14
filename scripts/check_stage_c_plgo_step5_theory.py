#!/usr/bin/env python3
"""Audit the SC1-PLGO Step 5 algebra and function-class boundaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_OUTPUT_DIR = Path(
    "analysis/stage_c_sc1_plgo_step5_theory_20260714"
)
TOLERANCE = 1e-10


@dataclass(frozen=True)
class Atom:
    """Metadata for one synthesis atom."""

    kind: str
    depth: int
    start: int
    end: int


@dataclass
class Node:
    """One interval in the restricted-global nested construction."""

    start: int
    end: int
    depth: int
    scaling: torch.Tensor
    detail: torch.Tensor
    inclusion_gap: float
    left: Node | None = None
    right: Node | None = None


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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def dct_prototypes(length: int, rank: int) -> torch.Tensor:
    """Return the first ``rank`` orthonormal DCT-II columns."""
    if length <= 0:
        raise ValueError("length must be positive")
    if rank <= 0 or rank > length:
        raise ValueError("rank must lie in [1, length]")
    time = torch.arange(length, dtype=torch.float64).unsqueeze(1)
    frequency = torch.arange(rank, dtype=torch.float64).unsqueeze(0)
    basis = torch.cos(math.pi * (time + 0.5) * frequency / length)
    basis[:, 0] *= math.sqrt(1.0 / length)
    if rank > 1:
        basis[:, 1:] *= math.sqrt(2.0 / length)
    return basis


def restricted_dct_coordinates(
    total_length: int,
    start: int,
    end: int,
    global_rank: int,
) -> torch.Tensor:
    """Represent a restricted DCT span in stable local Chebyshev coordinates."""
    size = end - start
    dimension = min(size, global_rank)
    if size <= global_rank:
        return torch.eye(size, dtype=torch.float64)

    time = torch.arange(start, end, dtype=torch.float64)
    cosine_coordinate = torch.cos(math.pi * (time + 0.5) / total_length)
    lower = cosine_coordinate.min()
    upper = cosine_coordinate.max()
    normalized = 2.0 * (cosine_coordinate - lower) / (upper - lower) - 1.0
    columns = [torch.ones_like(normalized)]
    if dimension > 1:
        columns.append(normalized)
    for _degree in range(2, dimension):
        columns.append(2.0 * normalized * columns[-1] - columns[-2])
    return torch.stack(columns, dim=1)


def stable_restricted_dct_span(
    total_length: int,
    start: int,
    end: int,
    global_rank: int,
) -> torch.Tensor:
    """Orthonormalize the exact restricted DCT span in stable coordinates."""
    coordinates = restricted_dct_coordinates(
        total_length,
        start,
        end,
        global_rank,
    )
    basis, _upper = torch.linalg.qr(coordinates, mode="reduced")
    return basis


def block_diagonal(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    rows = left.shape[0] + right.shape[0]
    columns = left.shape[1] + right.shape[1]
    result = torch.zeros((rows, columns), dtype=left.dtype)
    result[: left.shape[0], : left.shape[1]] = left
    result[left.shape[0] :, left.shape[1] :] = right
    return result


def build_node(
    prototypes: torch.Tensor,
    start: int,
    end: int,
    depth: int,
) -> Node:
    """Build one restricted-global scaling space and its local complement."""
    scaling = stable_restricted_dct_span(
        prototypes.shape[0],
        start,
        end,
        prototypes.shape[1],
    )
    size = end - start
    if size == 1:
        return Node(
            start=start,
            end=end,
            depth=depth,
            scaling=scaling,
            detail=torch.empty((1, 0), dtype=prototypes.dtype),
            inclusion_gap=0.0,
        )

    middle = start + size // 2
    left = build_node(prototypes, start, middle, depth + 1)
    right = build_node(prototypes, middle, end, depth + 1)
    child_scaling = block_diagonal(left.scaling, right.scaling)
    coordinates = child_scaling.T @ scaling
    inclusion_gap = float(
        (child_scaling @ coordinates - scaling).abs().max().item()
    )

    _u, _singular_values, vh = torch.linalg.svd(
        coordinates.T,
        full_matrices=True,
    )
    parent_dimension = scaling.shape[1]
    complement_coordinates = vh[parent_dimension:].T
    detail = child_scaling @ complement_coordinates
    return Node(
        start=start,
        end=end,
        depth=depth,
        scaling=scaling,
        detail=detail,
        inclusion_gap=inclusion_gap,
        left=left,
        right=right,
    )


def collect_details(
    node: Node,
    total_length: int,
    columns: list[torch.Tensor],
    atoms: list[Atom],
    inclusion_gaps: list[float],
) -> None:
    inclusion_gaps.append(node.inclusion_gap)
    for index in range(node.detail.shape[1]):
        column = torch.zeros(total_length, dtype=node.detail.dtype)
        column[node.start : node.end] = node.detail[:, index]
        columns.append(column)
        atoms.append(
            Atom(
                kind="detail",
                depth=node.depth,
                start=node.start,
                end=node.end,
            )
        )
    if node.left is not None and node.right is not None:
        collect_details(node.left, total_length, columns, atoms, inclusion_gaps)
        collect_details(node.right, total_length, columns, atoms, inclusion_gaps)


def restricted_global_nested_basis(
    length: int,
    global_rank: int,
) -> tuple[torch.Tensor, list[Atom], torch.Tensor, float]:
    """Construct the square Restricted-Global Nested Basis (RGNB)."""
    prototypes = dct_prototypes(length, global_rank)
    root = build_node(prototypes, 0, length, 0)
    columns = [root.scaling[:, index] for index in range(root.scaling.shape[1])]
    atoms = [
        Atom(kind="global", depth=-1, start=0, end=length)
        for _index in range(root.scaling.shape[1])
    ]
    inclusion_gaps: list[float] = []
    collect_details(root, length, columns, atoms, inclusion_gaps)
    synthesis = torch.stack(columns, dim=1)
    if synthesis.shape != (length, length):
        raise AssertionError(f"unexpected RGNB shape: {synthesis.shape}")
    return synthesis, atoms, prototypes, max(inclusion_gaps, default=0.0)


def active_indices(atoms: list[Atom], horizon: int) -> torch.Tensor:
    length = atoms[0].end
    if horizon <= 0 or horizon > length:
        raise ValueError("horizon must lie in [1, T]")
    return torch.tensor(
        [index for index, atom in enumerate(atoms) if atom.start < horizon],
        dtype=torch.long,
    )


def horizon_set(length: int) -> list[int]:
    if length <= 16:
        return list(range(1, length + 1))
    candidates = {1, length, length // 8, length // 4, length // 2}
    if length in {96, 720, 721}:
        candidates.update({5, 48, 96, 144, 192, 336, 512, 720})
    return sorted(value for value in candidates if 1 <= value <= length)


def group_sizes(atoms: list[Atom]) -> dict[str, int]:
    groups: dict[str, int] = {}
    for atom in atoms:
        key = "global_root" if atom.kind == "global" else f"detail_depth_{atom.depth}"
        groups[key] = groups.get(key, 0) + 1
    return groups


def conditioning_audit(length: int, global_rank: int) -> dict[str, Any]:
    """Compare raw restricted DCT coordinates with the stable local chart."""
    prototypes = dct_prototypes(length, global_rank)
    frontier = [(0, length)]
    raw_conditions = []
    stable_conditions = []
    while frontier:
        start, end = frontier.pop()
        raw_singular_values = torch.linalg.svdvals(prototypes[start:end])
        raw_conditions.append(
            float((raw_singular_values.max() / raw_singular_values.min()).item())
        )
        stable_singular_values = torch.linalg.svdvals(
            restricted_dct_coordinates(length, start, end, global_rank)
        )
        stable_conditions.append(
            float(
                (
                    stable_singular_values.max()
                    / stable_singular_values.min()
                ).item()
            )
        )
        if end - start > 1:
            middle = start + (end - start) // 2
            frontier.extend([(start, middle), (middle, end)])
    return {
        "length": length,
        "global_rank": global_rank,
        "interval_nodes": len(raw_conditions),
        "raw_restriction_max_condition": max(raw_conditions),
        "stable_coordinate_max_condition": max(stable_conditions),
        "raw_nodes_above_1e12": sum(value > 1e12 for value in raw_conditions),
        "stable_nodes_above_1e12": sum(
            value > 1e12 for value in stable_conditions
        ),
    }


def check_basis_case(
    length: int,
    global_rank: int,
    generator: torch.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    synthesis, atoms, prototypes, inclusion_gap = restricted_global_nested_basis(
        length,
        global_rank,
    )
    identity = torch.eye(length, dtype=synthesis.dtype)
    orthogonality_gap = float(
        (synthesis.T @ synthesis - identity).abs().max().item()
    )
    root = synthesis[:, :global_rank]
    global_projector_gap = float(
        (root @ root.T - prototypes @ prototypes.T).abs().max().item()
    )
    detail = synthesis[:, global_rank:]
    prototype_moment_gap = (
        float((detail.T @ prototypes).abs().max().item())
        if detail.numel()
        else 0.0
    )
    support_gap = 0.0
    for index, atom in enumerate(atoms):
        if atom.kind == "global":
            continue
        outside = torch.cat(
            [synthesis[: atom.start, index], synthesis[atom.end :, index]]
        )
        if outside.numel():
            support_gap = max(support_gap, float(outside.abs().max().item()))

    hidden_dim = 17
    coefficient_rank = min(7, length)
    hidden = torch.randn(hidden_dim, generator=generator, dtype=torch.float64)
    coefficient_map = torch.randn(
        coefficient_rank,
        hidden_dim,
        generator=generator,
        dtype=torch.float64,
    )
    coefficient_bias = torch.randn(
        coefficient_rank,
        generator=generator,
        dtype=torch.float64,
    )
    output_basis = torch.randn(
        length,
        coefficient_rank,
        generator=generator,
        dtype=torch.float64,
    )
    temporal_bias = torch.randn(length, generator=generator, dtype=torch.float64)
    coefficient = coefficient_map @ hidden + coefficient_bias
    reference = output_basis @ coefficient + temporal_bias
    transformed_basis = synthesis.T @ output_basis
    transformed_bias = synthesis.T @ temporal_bias
    transformed_coefficients = transformed_basis @ coefficient + transformed_bias
    reconstructed = synthesis @ transformed_coefficients
    morphism_gap = float((reconstructed - reference).abs().max().item())

    prefix_rows = []
    for horizon in horizon_set(length):
        active = active_indices(atoms, horizon)
        prefix = synthesis[:horizon, active] @ transformed_coefficients[active]
        active_count = int(active.numel())
        conservative_bound = min(
            length,
            horizon + global_rank * (math.ceil(math.log2(length)) + 1),
        )
        prefix_rows.append(
            {
                "length": length,
                "global_rank": global_rank,
                "horizon": horizon,
                "active_atoms": active_count,
                "inactive_atoms": length - active_count,
                "active_to_horizon_ratio": active_count / horizon,
                "conservative_active_bound": conservative_bound,
                "bound_pass": active_count <= conservative_bound,
                "prefix_max_abs": float(
                    (prefix - reference[:horizon]).abs().max().item()
                ),
            }
        )

    return (
        {
            "length": length,
            "global_rank": global_rank,
            "atoms": len(atoms),
            "global_atoms": sum(atom.kind == "global" for atom in atoms),
            "detail_atoms": sum(atom.kind == "detail" for atom in atoms),
            "depth_groups": len(group_sizes(atoms)),
            "orthogonality_max_abs": orthogonality_gap,
            "nested_inclusion_max_abs": inclusion_gap,
            "global_projector_max_abs": global_projector_gap,
            "detail_prototype_moment_max_abs": prototype_moment_gap,
            "support_max_abs": support_gap,
            "a6_morphism_max_abs": morphism_gap,
            "prefix_cases": len(prefix_rows),
            "prefix_max_abs": max(row["prefix_max_abs"] for row in prefix_rows),
        },
        prefix_rows,
    )


def check_all_prefix_bounds(length: int, global_rank: int) -> tuple[int, bool, int]:
    _synthesis, atoms, _prototypes, _gap = restricted_global_nested_basis(
        length,
        global_rank,
    )
    largest_excess = 0
    for horizon in range(1, length + 1):
        active_count = int(active_indices(atoms, horizon).numel())
        bound = min(
            length,
            horizon + global_rank * (math.ceil(math.log2(length)) + 1),
        )
        largest_excess = max(largest_excess, active_count - bound)
    return length, largest_excess <= 0, largest_excess


def frame_control(length: int, global_rank: int) -> dict[str, Any]:
    """Audit the naive global-plus-local overcomplete union."""
    global_basis = dct_prototypes(length, global_rank)
    local_basis, _atoms, _prototypes, _gap = restricted_global_nested_basis(
        length,
        1,
    )
    frame = torch.cat([global_basis, local_basis], dim=1)
    frame_operator = frame @ frame.T
    expected_operator = (
        torch.eye(length, dtype=frame.dtype) + global_basis @ global_basis.T
    )
    operator_gap = float((frame_operator - expected_operator).abs().max().item())
    eigenvalues = torch.linalg.eigvalsh(frame_operator)

    signal = torch.linspace(-1.0, 1.0, length, dtype=torch.float64)
    canonical = frame.T @ torch.linalg.solve(frame_operator, signal)
    canonical_gap = float((frame @ canonical - signal).abs().max().item())
    kernel = torch.cat(
        [
            torch.eye(global_rank, dtype=frame.dtype),
            -(local_basis.T @ global_basis),
        ],
        dim=0,
    )
    kernel_gap = float((frame @ kernel).abs().max().item())
    alternative_gap = float(
        (frame @ (canonical + kernel[:, 0]) - signal).abs().max().item()
    )
    coherence = float((global_basis.T @ local_basis).abs().max().item())
    return {
        "length": length,
        "global_rank": global_rank,
        "frame_atoms": frame.shape[1],
        "redundancy": frame.shape[1] - length,
        "constructive_kernel_dimension": global_rank,
        "frame_lower_bound": float(eigenvalues.min().item()),
        "frame_upper_bound": float(eigenvalues.max().item()),
        "frame_operator_max_abs": operator_gap,
        "global_local_coherence": coherence,
        "canonical_reconstruction_max_abs": canonical_gap,
        "kernel_max_abs": kernel_gap,
        "alternative_reconstruction_max_abs": alternative_gap,
        "coefficient_identifiable": False,
        "candidate_status": "control_only_overcomplete_nonidentifiable",
    }


def function_class_budget(
    length: int = 720,
    global_rank: int = 16,
    hidden_dim: int = 768,
    a6_rank: int = 256,
) -> dict[str, Any]:
    _basis, atoms, _prototypes, _gap = restricted_global_nested_basis(
        length,
        global_rank,
    )
    groups = group_sizes(atoms)
    sizes = list(groups.values())
    rank_caps = [min(size, a6_rank) for size in sizes]
    a6_parameters = (
        a6_rank * hidden_dim
        + a6_rank
        + length * a6_rank
        + length
    )
    direct_atom_parameters = length * hidden_dim + length
    independent_group_parameters = sum(
        size * cap + cap * hidden_dim + size
        for size, cap in zip(sizes, rank_caps, strict=True)
    )
    all_groups_full_row_rank = all(size <= a6_rank for size in sizes)
    return {
        "length": length,
        "global_rank": global_rank,
        "hidden_dim": hidden_dim,
        "a6_rank": a6_rank,
        "group_names": list(groups),
        "group_sizes": sizes,
        "group_rank_caps_for_exact_a6_containment": rank_caps,
        "sum_group_rank_caps": sum(rank_caps),
        "same_total_rank_budget_can_contain_all_a6_blocks": (
            sum(rank_caps) <= a6_rank
        ),
        "all_groups_full_row_rank": all_groups_full_row_rank,
        "independent_group_class_equals_full_affine": all_groups_full_row_rank,
        "a6_readout_parameters": a6_parameters,
        "direct_atom_affine_parameters": direct_atom_parameters,
        "independent_group_factor_parameters": independent_group_parameters,
        "direct_atom_to_a6_ratio": direct_atom_parameters / a6_parameters,
        "independent_group_to_a6_ratio": independent_group_parameters
        / a6_parameters,
        "orthonormal_m0_function_class": "exactly_equal_to_a6",
        "square_invertible_transform_adds_function_novelty": False,
    }


def candidate_matrix() -> list[dict[str, Any]]:
    return [
        {
            "candidate": "PLGO-ONB-M0",
            "stable_synthesis": True,
            "coefficient_identifiable": True,
            "exact_prefix_restriction": True,
            "exact_a6_containment": True,
            "new_function_class": False,
            "selective_speedup_proven": False,
            "status": "control_only_isometric_reparameterization",
        },
        {
            "candidate": "PLGO-FRAME",
            "stable_synthesis": True,
            "coefficient_identifiable": False,
            "exact_prefix_restriction": True,
            "exact_a6_containment": True,
            "new_function_class": True,
            "selective_speedup_proven": False,
            "status": "control_only_overcomplete_capacity_confound",
        },
        {
            "candidate": "PLGO-INDEPENDENT-GROUP",
            "stable_synthesis": True,
            "coefficient_identifiable": True,
            "exact_prefix_restriction": True,
            "exact_a6_containment": True,
            "new_function_class": True,
            "selective_speedup_proven": False,
            "status": "rejected_as_core_full_affine_capacity_confound",
        },
        {
            "candidate": "PLGO-ATOM-CONDITIONED-GENERATOR",
            "stable_synthesis": True,
            "coefficient_identifiable": True,
            "exact_prefix_restriction": True,
            "exact_a6_containment": "not_established",
            "new_function_class": "not_established",
            "selective_speedup_proven": False,
            "status": "provisional_step6_design_question",
        },
    ]


def main() -> None:
    args = parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    cases = [
        (1, 1),
        (2, 2),
        (3, 3),
        (5, 4),
        (7, 4),
        (16, 4),
        (96, 8),
        (720, 1),
        (720, 4),
        (720, 8),
        (720, 16),
        (721, 16),
    ]
    basis_rows = []
    prefix_rows = []
    for length, global_rank in cases:
        basis_row, case_prefix_rows = check_basis_case(
            length,
            global_rank,
            generator,
        )
        basis_rows.append(basis_row)
        prefix_rows.extend(case_prefix_rows)

    bound_rows = []
    for length, global_rank in cases:
        checked_length, passed, largest_excess = check_all_prefix_bounds(
            length,
            global_rank,
        )
        bound_rows.append(
            {
                "length": checked_length,
                "global_rank": global_rank,
                "horizon_cases": length,
                "bound_pass": passed,
                "largest_bound_excess": largest_excess,
            }
        )

    frame_rows = [frame_control(96, 8), frame_control(720, 16)]
    conditioning_rows = [
        conditioning_audit(96, 8),
        conditioning_audit(720, 4),
        conditioning_audit(720, 8),
        conditioning_audit(720, 16),
        conditioning_audit(721, 16),
    ]
    budget = function_class_budget()
    candidates = candidate_matrix()
    algebra_keys = [
        "orthogonality_max_abs",
        "nested_inclusion_max_abs",
        "global_projector_max_abs",
        "detail_prototype_moment_max_abs",
        "support_max_abs",
        "a6_morphism_max_abs",
        "prefix_max_abs",
    ]
    max_algebraic_gap = max(
        float(row[key]) for row in basis_rows for key in algebra_keys
    )
    max_frame_gap = max(
        max(
            float(row["frame_operator_max_abs"]),
            float(row["canonical_reconstruction_max_abs"]),
            float(row["kernel_max_abs"]),
            float(row["alternative_reconstruction_max_abs"]),
        )
        for row in frame_rows
    )
    theory_gate = {
        "candidate": "SC1-PLGO",
        "construction": "restricted_global_nested_basis",
        "step": 5,
        "basis_cases": len(basis_rows),
        "selected_prefix_cases": len(prefix_rows),
        "all_horizon_bound_cases": sum(row["horizon_cases"] for row in bound_rows),
        "max_algebraic_gap": max_algebraic_gap,
        "max_frame_control_gap": max_frame_gap,
        "raw_restriction_max_condition": max(
            row["raw_restriction_max_condition"] for row in conditioning_rows
        ),
        "stable_coordinate_max_condition": max(
            row["stable_coordinate_max_condition"] for row in conditioning_rows
        ),
        "tolerance": TOLERANCE,
        "stable_reconstruction_gate": max_algebraic_gap <= TOLERANCE,
        "global_subspace_gate": all(
            row["global_projector_max_abs"] <= TOLERANCE for row in basis_rows
        ),
        "local_support_gate": all(
            row["support_max_abs"] <= TOLERANCE for row in basis_rows
        ),
        "prefix_restriction_gate": all(
            row["prefix_max_abs"] <= TOLERANCE for row in prefix_rows
        ),
        "active_bound_gate": all(row["bound_pass"] for row in bound_rows),
        "a6_morphism_gate": all(
            row["a6_morphism_max_abs"] <= TOLERANCE for row in basis_rows
        ),
        "overcomplete_frame_stability_gate": max_frame_gap <= TOLERANCE,
        "overcomplete_frame_identifiability_gate": False,
        "orthonormal_m0_function_novelty_gate": False,
        "independent_group_attribution_gate": False,
        "cross_horizon_coefficient_projectivity_claim": "not_established_and_not_claimed",
        "selective_efficiency_claim": "withdrawn_until_generator_level_flops_are_measured",
        "step5_decision": "partial_pass_step6_design_only",
        "method_implementation_authorized": False,
        "next_design_question": (
            "shared atom-conditioned coefficient generation versus matched dense "
            "and random-descriptor controls"
        ),
        "rollback_if_step6_fails": "step4_redesign",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "basis_checks.csv", basis_rows)
    write_csv(args.output_dir / "prefix_checks.csv", prefix_rows)
    write_csv(args.output_dir / "active_bound_checks.csv", bound_rows)
    write_csv(args.output_dir / "frame_control_checks.csv", frame_rows)
    write_csv(args.output_dir / "conditioning_checks.csv", conditioning_rows)
    write_csv(args.output_dir / "candidate_matrix.csv", candidates)
    (args.output_dir / "function_class_budget.json").write_text(
        json.dumps(budget, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "theory_gate.json").write_text(
        json.dumps(theory_gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "stage_c_plgo_step5=complete "
        f"max_gap={max_algebraic_gap:.3e} "
        f"prefix_cases={len(prefix_rows)} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
