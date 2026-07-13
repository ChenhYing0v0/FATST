#!/usr/bin/env python3
"""Audit the StageC PMFO/PIR algebra before method implementation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch


SERIES_LENGTH = 720
BLOCK_SIZES = (90, 30, 10, 5, 1)
RADICES = (3, 3, 2, 5)
AUDIT_HORIZONS = (1, 48, 96, 192, 336, 720)
MEASURE_NAMES = ("delta_720", "uniform_h", "log_uniform_h", "benchmark_h")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit mixed-radix PMFO and measure-induced PIR geometry."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/stage_c_step46_pmfo_pir_theory_gate_20260713"),
    )
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def helmert_detail(radix: int, dtype: torch.dtype) -> torch.Tensor:
    """Return an orthonormal complement of the normalized all-ones vector."""
    detail = torch.zeros(radix, radix - 1, dtype=dtype)
    for column in range(radix - 1):
        count = column + 1
        scale = math.sqrt(float(count * (count + 1)))
        detail[:count, column] = 1.0 / scale
        detail[count, column] = -float(count) / scale
    return detail


def coefficient_group_sizes() -> tuple[int, ...]:
    groups = [SERIES_LENGTH // BLOCK_SIZES[0]]
    parent_count = groups[0]
    for radix in RADICES:
        groups.append(parent_count * (radix - 1))
        parent_count *= radix
    if sum(groups) != SERIES_LENGTH:
        raise RuntimeError("mixed-radix coefficient groups do not span the domain")
    return tuple(groups)


def synthesize(coefficients: torch.Tensor) -> torch.Tensor:
    """Map ordered scaling/detail coefficients to unit-resolution leaves."""
    groups = coefficient_group_sizes()
    cursor = groups[0]
    values = coefficients[:, :cursor]
    for radix, group_size in zip(RADICES, groups[1:], strict=True):
        parent_count = values.shape[1]
        detail = coefficients[:, cursor : cursor + group_size].reshape(
            coefficients.shape[0], parent_count, radix - 1
        )
        scaling = torch.full(
            (radix,),
            1.0 / math.sqrt(float(radix)),
            dtype=coefficients.dtype,
            device=coefficients.device,
        )
        contrast = helmert_detail(radix, coefficients.dtype).to(coefficients.device)
        children = values.unsqueeze(-1) * scaling
        children = children + torch.einsum("bnd,rd->bnr", detail, contrast)
        values = children.reshape(coefficients.shape[0], parent_count * radix)
        cursor += group_size
    return values


def synthesize_prefix(coefficients: torch.Tensor, horizon: int) -> torch.Tensor:
    """Evaluate only interval-tree nodes intersecting the requested prefix."""
    if horizon <= 0 or horizon > SERIES_LENGTH:
        raise ValueError("horizon must be within the canonical future domain")
    groups = coefficient_group_sizes()
    coarse_count = math.ceil(horizon / BLOCK_SIZES[0])
    values = coefficients[:, :coarse_count]
    cursor = groups[0]
    for level, (radix, group_size) in enumerate(
        zip(RADICES, groups[1:], strict=True)
    ):
        parent_support = BLOCK_SIZES[level]
        child_support = BLOCK_SIZES[level + 1]
        active_parents = math.ceil(horizon / parent_support)
        active_children = math.ceil(horizon / child_support)
        detail = coefficients[
            :,
            cursor : cursor + active_parents * (radix - 1),
        ].reshape(coefficients.shape[0], active_parents, radix - 1)
        scaling = torch.full(
            (radix,),
            1.0 / math.sqrt(float(radix)),
            dtype=coefficients.dtype,
            device=coefficients.device,
        )
        contrast = helmert_detail(radix, coefficients.dtype).to(coefficients.device)
        children = values.unsqueeze(-1) * scaling
        children = children + torch.einsum("bnd,rd->bnr", detail, contrast)
        values = children.reshape(coefficients.shape[0], -1)[:, :active_children]
        cursor += group_size
    return values[:, :horizon]


def block_projection(length: int, block_size: int, dtype: torch.dtype) -> torch.Tensor:
    if length % block_size != 0:
        raise ValueError("block size must divide the temporal length")
    local = torch.full(
        (block_size, block_size), 1.0 / float(block_size), dtype=dtype
    )
    return torch.kron(torch.eye(length // block_size, dtype=dtype), local)


def nested_projectors(dtype: torch.dtype) -> list[torch.Tensor]:
    projections = [
        block_projection(SERIES_LENGTH, block_size, dtype)
        for block_size in BLOCK_SIZES
    ]
    increments = [projections[0]]
    increments.extend(
        current - previous
        for previous, current in zip(projections[:-1], projections[1:], strict=True)
    )
    return increments


def measure_weights(dtype: torch.dtype) -> dict[str, torch.Tensor]:
    horizons = torch.arange(1, SERIES_LENGTH + 1, dtype=dtype)
    distributions: dict[str, torch.Tensor] = {}
    delta = torch.zeros(SERIES_LENGTH, dtype=dtype)
    delta[-1] = 1.0
    distributions["delta_720"] = delta
    distributions["uniform_h"] = torch.full(
        (SERIES_LENGTH,), 1.0 / float(SERIES_LENGTH), dtype=dtype
    )
    log_uniform = horizons.reciprocal()
    distributions["log_uniform_h"] = log_uniform / log_uniform.sum()
    benchmark = torch.zeros(SERIES_LENGTH, dtype=dtype)
    for horizon in (96, 192, 336, 720):
        benchmark[horizon - 1] = 0.25
    distributions["benchmark_h"] = benchmark

    weights: dict[str, torch.Tensor] = {}
    for name, probabilities in distributions.items():
        per_horizon = probabilities / horizons
        weights[name] = torch.flip(
            torch.cumsum(torch.flip(per_horizon, dims=[0]), dim=0), dims=[0]
        )
    return weights


def invariant_metrics(seed: int) -> tuple[dict[str, float], list[torch.Tensor]]:
    dtype = torch.float64
    groups = coefficient_group_sizes()
    identity = torch.eye(SERIES_LENGTH, dtype=dtype)
    synthesis_rows = synthesize(identity)
    synthesis_basis = synthesis_rows.transpose(0, 1)
    orthogonality_error = float(
        (synthesis_rows @ synthesis_rows.transpose(0, 1) - identity).abs().max()
    )

    projectors = nested_projectors(dtype)
    projector_sum_error = float((sum(projectors) - identity).abs().max())
    projector_errors = []
    cross_errors = []
    for left_index, left in enumerate(projectors):
        projector_errors.append(float((left @ left - left).abs().max()))
        for right_index, right in enumerate(projectors):
            if left_index != right_index:
                cross_errors.append(float((left @ right).abs().max()))

    cursor = 0
    basis_projector_errors = []
    for group_size, expected in zip(groups, projectors, strict=True):
        group_basis = synthesis_rows[cursor : cursor + group_size]
        observed = group_basis.transpose(0, 1) @ group_basis
        basis_projector_errors.append(float((observed - expected).abs().max()))
        cursor += group_size

    generator = torch.Generator().manual_seed(seed)
    coefficients = torch.randn(32, SERIES_LENGTH, generator=generator, dtype=dtype)
    full = synthesize(coefficients)
    prefix_errors = [
        float((synthesize_prefix(coefficients, horizon) - full[:, :horizon]).abs().max())
        for horizon in AUDIT_HORIZONS
    ]

    refinement_errors = []
    for radix in RADICES:
        parent = torch.randn(19, 7, generator=generator, dtype=dtype)
        detail = torch.randn(19, 7, radix - 1, generator=generator, dtype=dtype)
        scaling = torch.full(
            (radix,), 1.0 / math.sqrt(float(radix)), dtype=dtype
        )
        contrast = helmert_detail(radix, dtype)
        children = parent.unsqueeze(-1) * scaling
        children = children + torch.einsum("bnd,rd->bnr", detail, contrast)
        recovered_parent = torch.einsum("bnr,r->bn", children, scaling)
        recovered_detail = torch.einsum("bnr,rd->bnd", children, contrast)
        refinement_errors.extend(
            [
                float((recovered_parent - parent).abs().max()),
                float((recovered_detail - detail).abs().max()),
            ]
        )

    return (
        {
            "orthogonality_max_abs": orthogonality_error,
            "projector_sum_max_abs": projector_sum_error,
            "projector_idempotence_max_abs": max(projector_errors),
            "projector_cross_max_abs": max(cross_errors),
            "basis_projector_match_max_abs": max(basis_projector_errors),
            "prefix_restriction_max_abs": max(prefix_errors),
            "refinement_recovery_max_abs": max(refinement_errors),
        },
        projectors,
    )


def tree_count_rows() -> list[dict[str, Any]]:
    rows = []
    full_coefficients = sum(coefficient_group_sizes())
    for horizon in AUDIT_HORIZONS:
        active_coarse = math.ceil(horizon / BLOCK_SIZES[0])
        active_details = []
        for level, radix in enumerate(RADICES):
            active_parents = math.ceil(horizon / BLOCK_SIZES[level])
            active_details.append(active_parents * (radix - 1))
        active_coefficients = active_coarse + sum(active_details)
        rows.append(
            {
                "horizon": horizon,
                "active_coarse_coefficients": active_coarse,
                "active_detail_l1": active_details[0],
                "active_detail_l2": active_details[1],
                "active_detail_l3": active_details[2],
                "active_detail_l4": active_details[3],
                "active_total_coefficients": active_coefficients,
                "boundary_overhead_vs_output": active_coefficients - horizon,
                "out_of_prefix_coefficients_avoided": (
                    full_coefficients - active_coefficients
                ),
                "active_fraction_of_full_tree": (
                    active_coefficients / float(full_coefficients)
                ),
                "a6_dense_basis_scalar_products": horizon * 256,
            }
        )
    return rows


def measure_geometry_rows(
    projectors: list[torch.Tensor], seed: int
) -> list[dict[str, Any]]:
    dtype = projectors[0].dtype
    weights_by_measure = measure_weights(dtype)
    generator = torch.Generator().manual_seed(seed + 1)
    errors = torch.randn(256, SERIES_LENGTH, generator=generator, dtype=dtype)
    rows = []
    for measure_name in MEASURE_NAMES:
        weights = weights_by_measure[measure_name]
        weight_operator = torch.diag(weights)
        projected_operator = sum(
            projector @ weight_operator @ projector for projector in projectors
        )
        offblock = weight_operator - projected_operator
        raw_risk = torch.einsum("bi,ij,bj->b", errors, weight_operator, errors)
        projected_risk = torch.einsum(
            "bi,ij,bj->b", errors, projected_operator, errors
        )
        component_risk = sum(
            torch.einsum(
                "bi,ij,bj->b",
                errors @ projector,
                weight_operator,
                errors @ projector,
            )
            for projector in projectors
        )
        relative_gap = (projected_risk - raw_risk).abs() / raw_risk.clamp_min(1e-12)
        cross_norms = [
            float((left @ weight_operator @ right).norm())
            for left_index, left in enumerate(projectors)
            for right_index, right in enumerate(projectors)
            if left_index != right_index
        ]
        rows.append(
            {
                "measure": measure_name,
                "step_weight_sum": float(weights.sum()),
                "step_weight_first": float(weights[0]),
                "step_weight_last": float(weights[-1]),
                "first_to_last_weight_ratio": float(weights[0] / weights[-1]),
                "offblock_fro_ratio": float(
                    offblock.norm() / weight_operator.norm().clamp_min(1e-12)
                ),
                "offblock_energy_fraction": float(
                    offblock.square().sum()
                    / weight_operator.square().sum().clamp_min(1e-12)
                ),
                "max_cross_scale_block_fro": max(cross_norms),
                "trace_preservation_abs": float(
                    (projected_operator.trace() - weight_operator.trace()).abs()
                ),
                "component_quadratic_identity_max_abs": float(
                    (component_risk - projected_risk).abs().max()
                ),
                "random_error_relative_risk_gap_mean": float(relative_gap.mean()),
                "random_error_relative_risk_gap_max": float(relative_gap.max()),
            }
        )
    return rows


def render_report(
    invariants: dict[str, float],
    tree_rows: list[dict[str, Any]],
    measure_rows: list[dict[str, Any]],
) -> str:
    invariant_pass = max(invariants.values()) <= 1e-10
    lines = [
        "# StageC Step 4-6 PMFO/PIR Theory Gate",
        "",
        "## Decision",
        "",
        f"- PMFO mixed-radix algebra invariant gate: `{'pass' if invariant_pass else 'fail'}`；",
        "- PIR is an exact block-diagonal quadratic construction for L2, but not the exact deployment risk when cross-scale blocks are non-zero；",
        "- 本报告只验证代数与measure geometry，不构成method effectiveness evidence。",
        "",
        "## PMFO Invariants",
        "",
        "| Invariant | Max absolute error |",
        "| --- | ---: |",
    ]
    for name, value in invariants.items():
        lines.append(f"| `{name}` | {value:.3e} |")
    lines.extend(
        [
            "",
            "## Domain-Local Tree Counts",
            "",
            "`active_total_coefficients` 只统计与 requested prefix 相交的 mixed-radix scaling/detail coefficients；不等价于完整模型 FLOPs。",
            "",
            "| H | Active coefficients | Boundary overhead | Avoided outside prefix | Active fraction | A6 basis products |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in tree_rows:
        lines.append(
            "| {horizon} | {active_total_coefficients} | "
            "{boundary_overhead_vs_output} | {out_of_prefix_coefficients_avoided} | "
            "{active_fraction_of_full_tree:.4f} | {a6_dense_basis_scalar_products} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Measure-Induced Geometry",
            "",
            "| Measure | First/last weight | Off-block energy | Mean random-risk gap | Max random-risk gap |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in measure_rows:
        lines.append(
            "| {measure} | {first_to_last_weight_ratio:.3f} | "
            "{offblock_energy_fraction:.6f} | "
            "{random_error_relative_risk_gap_mean:.6f} | "
            "{random_error_relative_risk_gap_max:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "`delta_720` 令 temporal weights 与 identity 成比例，因此 orthogonal refinement blocks 不产生 cross-scale coupling。其他 horizon measures 通常产生非零 off-block operator；PIR 删除这些 blocks，是由 decoder partition 决定的 structured surrogate，不是 raw deployment risk 的等价改写，也没有一般性的 upper/lower-bound 保证。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    invariants, projectors = invariant_metrics(args.seed)
    tree_rows = tree_count_rows()
    measure_rows = measure_geometry_rows(projectors, args.seed)
    invariant_pass = max(invariants.values()) <= 1e-10
    payload = {
        "series_length": SERIES_LENGTH,
        "block_sizes": BLOCK_SIZES,
        "radices": RADICES,
        "coefficient_group_sizes": coefficient_group_sizes(),
        "seed": args.seed,
        "pmfo_invariant_gate": "pass" if invariant_pass else "fail",
        "invariants": invariants,
        "pir_theory_status": "valid_l2_measure_induced_block_diagonal_surrogate",
        "method_effectiveness_evidence": False,
    }

    write_csv(output_dir / "pmfo_tree_counts.csv", tree_rows)
    write_csv(output_dir / "pir_measure_geometry.csv", measure_rows)
    (output_dir / "theory_gate.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "theory_gate_report.md").write_text(
        render_report(invariants, tree_rows, measure_rows), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
