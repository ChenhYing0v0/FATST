#!/usr/bin/env python3
"""Verify the StageC FPMO Step 5 algebra on arbitrary interval lengths."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_OUTPUT_DIR = Path("analysis/stage_c_step5_fpmo_theory_20260713")
SMALL_LENGTHS = (1, 2, 3, 5, 7, 16)
LARGE_LENGTHS = (96, 720, 721)
TOLERANCE = 1e-10


@dataclass(frozen=True)
class Atom:
    """One row of the unbalanced Haar analysis matrix."""

    kind: str
    depth: int
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def unbalanced_haar(length: int) -> tuple[torch.Tensor, list[Atom]]:
    """Build a complete orthonormal interval basis for any positive length."""
    if length <= 0:
        raise ValueError("length must be positive")
    dtype = torch.float64
    rows = [torch.full((length,), 1.0 / math.sqrt(length), dtype=dtype)]
    atoms = [Atom(kind="scaling", depth=-1, start=0, end=length)]
    frontier = [(0, length, 0)]
    while frontier:
        next_frontier = []
        for start, end, depth in frontier:
            size = end - start
            if size <= 1:
                continue
            middle = start + size // 2
            left_size = middle - start
            right_size = end - middle
            row = torch.zeros(length, dtype=dtype)
            row[start:middle] = math.sqrt(right_size / (left_size * size))
            row[middle:end] = -math.sqrt(left_size / (right_size * size))
            rows.append(row)
            atoms.append(Atom(kind="detail", depth=depth, start=start, end=end))
            next_frontier.extend(
                [(start, middle, depth + 1), (middle, end, depth + 1)]
            )
        frontier = next_frontier
    matrix = torch.stack(rows)
    if matrix.shape != (length, length):
        raise AssertionError(f"unexpected transform shape: {matrix.shape}")
    return matrix, atoms


def active_indices(atoms: list[Atom], horizon: int) -> torch.Tensor:
    if horizon <= 0 or horizon > atoms[0].end:
        raise ValueError("horizon must lie in [1, T]")
    return torch.tensor(
        [index for index, atom in enumerate(atoms) if atom.start < horizon],
        dtype=torch.long,
    )


def horizon_set(length: int) -> list[int]:
    if length in SMALL_LENGTHS:
        return list(range(1, length + 1))
    candidates = {1, length, length // 8, length // 4, length // 2}
    if length == 720:
        candidates.update({48, 96, 192, 336})
    return sorted(value for value in candidates if 1 <= value <= length)


def group_indices(atoms: list[Atom]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {"scaling": [0]}
    for index, atom in enumerate(atoms[1:], start=1):
        groups.setdefault(f"detail_depth_{atom.depth}", []).append(index)
    return groups


def check_length(
    length: int,
    generator: torch.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    transform, atoms = unbalanced_haar(length)
    identity = torch.eye(length, dtype=transform.dtype)
    orthogonality_gap = float((transform @ transform.T - identity).abs().max())

    hidden_dim = 17
    basis_rank = 7
    hidden = torch.randn(hidden_dim, generator=generator, dtype=torch.float64)
    coefficient_weight = torch.randn(
        basis_rank,
        hidden_dim,
        generator=generator,
        dtype=torch.float64,
    )
    coefficient_bias = torch.randn(
        basis_rank,
        generator=generator,
        dtype=torch.float64,
    )
    basis = torch.randn(
        length,
        basis_rank,
        generator=generator,
        dtype=torch.float64,
    )
    temporal_bias = torch.randn(length, generator=generator, dtype=torch.float64)
    coefficient = coefficient_weight @ hidden + coefficient_bias
    reference = basis @ coefficient + temporal_bias

    morphed_basis = transform @ basis
    morphed_bias = transform @ temporal_bias
    transformed_coefficients = morphed_basis @ coefficient + morphed_bias
    reconstructed = transform.T @ transformed_coefficients
    embedding_gap = float((reference - reconstructed).abs().max())

    effective_operator = basis @ coefficient_weight
    effective_bias = basis @ coefficient_bias + temporal_bias
    transformed_operator = transform @ effective_operator
    transformed_effective_bias = transform @ effective_bias
    factorization_gap = 0.0
    scale_rows = []
    for group, indices in group_indices(atoms).items():
        block = transformed_operator[indices]
        u, singular_values, vh = torch.linalg.svd(block, full_matrices=False)
        numerical_rank = int((singular_values > 1e-10).sum())
        if numerical_rank:
            left = u[:, :numerical_rank] * singular_values[:numerical_rank]
            right = vh[:numerical_rank]
            recovered = left @ right
        else:
            recovered = torch.zeros_like(block)
        factorization_gap = max(
            factorization_gap,
            float((block - recovered).abs().max()),
        )
        group_size = len(indices)
        scale_rows.append(
            {
                "length": length,
                "group": group,
                "atom_count": group_size,
                "numerical_block_rank": numerical_rank,
                "containment_rank_cap": min(group_size, basis_rank),
                "rank_cap_sufficient": numerical_rank <= min(group_size, basis_rank),
            }
        )

    prefix_rows = []
    for horizon in horizon_set(length):
        active = active_indices(atoms, horizon)
        prefix = transform.T[:horizon, active] @ transformed_coefficients[active]
        operator_prefix = (
            transform.T[:horizon, active]
            @ (
                transformed_operator[active] @ hidden
                + transformed_effective_bias[active]
            )
        )
        prefix_rows.append(
            {
                "length": length,
                "horizon": horizon,
                "active_atoms": int(active.numel()),
                "inactive_atoms": length - int(active.numel()),
                "active_to_horizon_ratio": int(active.numel()) / horizon,
                "active_atom_upper_bound": (
                    2 * horizon + math.ceil(math.log2(length))
                ),
                "prefix_max_abs": float((prefix - reference[:horizon]).abs().max()),
                "operator_prefix_max_abs": float(
                    (operator_prefix - reference[:horizon]).abs().max()
                ),
            }
        )

    transform_row = {
        "length": length,
        "atoms": len(atoms),
        "depth_groups": len(group_indices(atoms)),
        "orthogonality_max_abs": orthogonality_gap,
        "embedding_max_abs": embedding_gap,
        "scale_factorization_max_abs": factorization_gap,
        "prefix_cases": len(prefix_rows),
        "prefix_max_abs": max(row["prefix_max_abs"] for row in prefix_rows),
        "operator_prefix_max_abs": max(
            row["operator_prefix_max_abs"] for row in prefix_rows
        ),
    }
    return transform_row, prefix_rows, scale_rows


def parameter_budget(length: int = 720, hidden_dim: int = 768, rank: int = 256) -> dict[str, Any]:
    _transform, atoms = unbalanced_haar(length)
    groups = group_indices(atoms)
    group_sizes = [len(indices) for indices in groups.values()]
    rank_caps = [min(size, rank) for size in group_sizes]
    a6_parameters = rank * hidden_dim + rank + length * rank + length
    direct_atom_parameters = length * hidden_dim + length
    independent_scale_factor_parameters = sum(
        size * cap + cap * hidden_dim + size
        for size, cap in zip(group_sizes, rank_caps, strict=True)
    )
    return {
        "length": length,
        "hidden_dim": hidden_dim,
        "a6_rank": rank,
        "group_sizes": group_sizes,
        "group_rank_caps": rank_caps,
        "sum_group_rank_caps": sum(rank_caps),
        "a6_readout_parameters": a6_parameters,
        "direct_atom_affine_parameters": direct_atom_parameters,
        "independent_scale_factor_parameters": independent_scale_factor_parameters,
        "direct_atom_to_a6_ratio": direct_atom_parameters / a6_parameters,
        "independent_scale_to_a6_ratio": independent_scale_factor_parameters
        / a6_parameters,
        "all_groups_full_row_rank": all(size <= rank for size in group_sizes),
        "independent_scale_class_equals_full_affine_at_t720": all(
            size <= rank for size in group_sizes
        ),
        "same_total_rank_budget_can_contain_all_a6_blocks": sum(rank_caps) <= rank,
    }


def main() -> None:
    args = parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    transform_rows = []
    prefix_rows = []
    scale_rows = []
    for length in (*SMALL_LENGTHS, *LARGE_LENGTHS):
        transform_row, length_prefix_rows, length_scale_rows = check_length(
            length,
            generator,
        )
        transform_rows.append(transform_row)
        prefix_rows.extend(length_prefix_rows)
        scale_rows.extend(length_scale_rows)

    max_gap = max(
        max(
            row["orthogonality_max_abs"],
            row["embedding_max_abs"],
            row["scale_factorization_max_abs"],
            row["prefix_max_abs"],
            row["operator_prefix_max_abs"],
        )
        for row in transform_rows
    )
    bound_cases = 0
    bound_pass = True
    for length in (*SMALL_LENGTHS, *LARGE_LENGTHS):
        _transform, atoms = unbalanced_haar(length)
        for horizon in range(1, length + 1):
            bound_cases += 1
            bound = 2 * horizon + math.ceil(math.log2(length))
            if int(active_indices(atoms, horizon).numel()) > bound:
                bound_pass = False
    budget = parameter_budget()
    summary = {
        "candidate": "SC1-FPMO",
        "step": 5,
        "lengths": [*SMALL_LENGTHS, *LARGE_LENGTHS],
        "transform_cases": len(transform_rows),
        "prefix_cases": len(prefix_rows),
        "max_algebraic_gap": max_gap,
        "tolerance": TOLERANCE,
        "algebra_gate": max_gap <= TOLERANCE,
        "active_atom_bound_gate": bound_pass,
        "active_atom_bound_cases": bound_cases,
        "shared_latent_morph_role": "control_only",
        "scale_native_extension_status": "capacity_confounded_theory_feasible",
        "step5_decision": "partial_pass_step6_design_only",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "transform_checks.csv", transform_rows)
    write_csv(args.output_dir / "prefix_checks.csv", prefix_rows)
    write_csv(args.output_dir / "scale_group_checks.csv", scale_rows)
    (args.output_dir / "parameter_budget.json").write_text(
        json.dumps(budget, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "theory_gate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "stage_c_fpmo_step5=complete "
        f"max_gap={max_gap:.3e} prefix_cases={len(prefix_rows)} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
