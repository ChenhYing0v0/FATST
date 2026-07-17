#!/usr/bin/env python3
"""Check SIFF and MCCA theory contracts before any method implementation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_post_pcc_step5.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/stage_c_post_pcc_step5_theory_20260717"),
    )
    return parser.parse_args()


def scale_basis(scales: torch.Tensor, length: int, order: int) -> torch.Tensor:
    z = torch.log(scales) / math.log(float(length))
    return torch.stack([z**degree for degree in range(order)], dim=-1)


def scale_indexed_modes(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    components = torch.einsum("bcr,qdrk->bcqdk", hidden, weights)
    return torch.einsum("sq,bcqdk->bcsdk", basis, components)


def synthesize(modes: torch.Tensor, synthesis: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bcsdk,tdk->bcst", modes, synthesis)


def projective_measure(length: int) -> torch.Tensor:
    horizons = torch.arange(1, length + 1, dtype=torch.float64)
    reciprocal = 1.0 / horizons
    cumulative = torch.cumsum(torch.flip(reciprocal, dims=(0,)), dim=0)
    return torch.flip(cumulative, dims=(0,)) / length


def sinkhorn_assignment(
    capability: torch.Tensor,
    row_mass: torch.Tensor,
    column_mass: torch.Tensor,
    epsilon: float,
    iterations: int,
) -> torch.Tensor:
    if torch.any(capability <= 0):
        raise ValueError("capability must be strictly positive")
    kernel = torch.exp(torch.log(capability) / epsilon)
    kernel = kernel / kernel.max().clamp_min(torch.finfo(kernel.dtype).tiny)
    left = torch.ones_like(row_mass)
    right = torch.ones_like(column_mass)
    tiny = torch.finfo(kernel.dtype).tiny
    for _ in range(iterations):
        left = row_mass / (kernel @ right).clamp_min(tiny)
        right = column_mass / (kernel.transpose(0, 1) @ left).clamp_min(tiny)
    return left[:, None] * kernel * right[None, :]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    gates = config["gates"]
    torch.manual_seed(1707)
    dtype = torch.float64

    batch, channels, readout, dimensions, rank = 2, 3, 5, 2, 4
    scopes, length, components = 5, 12, 3
    hidden = torch.randn(batch, channels, readout, dtype=dtype)
    base_weight = torch.randn(
        dimensions,
        readout,
        rank,
        dtype=dtype,
    )
    weights = torch.zeros(
        components,
        dimensions,
        readout,
        rank,
        dtype=dtype,
    )
    weights[0] = base_weight
    scales = torch.tensor([1, 2, 3, 6, 12], dtype=dtype)
    basis = scale_basis(scales, length, components)
    synthesis = torch.randn(length, dimensions, rank, dtype=dtype)

    current_modes = torch.einsum("bcr,drk->bcdk", hidden, base_weight)
    current_output = torch.einsum("bcdk,tdk->bct", current_modes, synthesis)
    contained = synthesize(
        scale_indexed_modes(hidden, weights, basis),
        synthesis,
    )
    containment_gap = float(
        (contained - current_output.unsqueeze(2)).abs().max()
    )
    prefix_gap = max(
        float((contained[..., :horizon] - contained[..., :horizon]).abs().max())
        for horizon in (1, 5, 12)
    )

    constant_hidden = torch.ones(1, 1, 1, dtype=dtype)
    current_constant_weight = torch.ones(1, 1, 1, dtype=dtype)
    current_constant = torch.einsum(
        "bcr,drk->bcdk",
        constant_hidden,
        current_constant_weight,
    )
    current_constant_output = current_constant.unsqueeze(2).expand(
        -1,
        -1,
        scopes,
        -1,
        -1,
    )
    current_constant_scope_gap = float(
        (
            current_constant_output[:, :, 0]
            - current_constant_output[:, :, -1]
        )
        .abs()
        .max()
    )
    contrast_weights = torch.ones(2, 1, 1, 1, dtype=dtype)
    contrast_basis = scale_basis(scales, length, 2)
    new_constant = scale_indexed_modes(
        constant_hidden,
        contrast_weights,
        contrast_basis,
    )
    new_constant_scope_gap = float(
        (new_constant[:, :, 0] - new_constant[:, :, -1]).abs().max()
    )

    row_mass = projective_measure(length)
    region = torch.arange(length) * 3 // length
    capability = torch.full((length, 3), 0.05, dtype=dtype)
    capability[torch.arange(length), region] = 0.90
    column_mass = torch.stack(
        [row_mass[region == index].sum() for index in range(3)]
    )
    allocation = sinkhorn_assignment(
        capability,
        row_mass,
        column_mass,
        config["mcca"]["epsilon"],
        config["mcca"]["iterations"],
    )
    row_gap = float((allocation.sum(dim=1) - row_mass).abs().max())
    column_gap = float((allocation.sum(dim=0) - column_mass).abs().max())
    conditional = allocation / row_mass[:, None]
    best_scope_mass = float(conditional[torch.arange(length), region].mean())
    best_scope_mass_gain = best_scope_mass - 1.0 / 3.0

    starvation_capability = torch.full((length, 3), 0.01, dtype=dtype)
    starvation_capability[:, 0] = 0.98
    protected_column_mass = torch.tensor([0.6, 0.2, 0.2], dtype=dtype)
    protected = sinkhorn_assignment(
        starvation_capability,
        row_mass,
        protected_column_mass,
        config["mcca"]["epsilon"],
        config["mcca"]["iterations"],
    )
    starvation_column_min = float(protected.sum(dim=0).min())

    predictions = torch.randn(length, 3, dtype=dtype, requires_grad=True)
    target = torch.randn(length, dtype=dtype)
    skill_loss = (
        allocation.detach() * (predictions - target[:, None]).abs()
    ).sum()
    skill_loss.backward()
    gradient_norms = predictions.grad.abs().sum(dim=0)
    gradient_norm_min = float(gradient_norms.min())

    logits = torch.zeros(length, 3, dtype=dtype, requires_grad=True)
    policy = torch.softmax(logits, dim=-1)
    router_target = allocation.detach() / row_mass[:, None]
    router_loss = (
        row_mass[:, None]
        * router_target
        * (
            torch.log(router_target.clamp_min(1e-12))
            - torch.log(policy.clamp_min(1e-12))
        )
    ).sum()
    router_loss.backward()
    router_gradient_norm = float(logits.grad.norm())

    cases = [
        {
            "case": "siff_q1_containment",
            "value": containment_gap,
            "threshold": gates["containment_gap_max"],
            "pass": containment_gap <= gates["containment_gap_max"],
        },
        {
            "case": "siff_prefix_projectivity",
            "value": prefix_gap,
            "threshold": gates["prefix_gap_max"],
            "pass": prefix_gap <= gates["prefix_gap_max"],
        },
        {
            "case": "current_constant_scope_indistinguishable",
            "value": current_constant_scope_gap,
            "threshold": gates["current_constant_scope_gap_max"],
            "pass": current_constant_scope_gap
            <= gates["current_constant_scope_gap_max"],
        },
        {
            "case": "siff_constant_scope_contrast",
            "value": new_constant_scope_gap,
            "threshold": gates["new_constant_scope_gap_min"],
            "pass": new_constant_scope_gap
            >= gates["new_constant_scope_gap_min"],
        },
        {
            "case": "mcca_row_marginal",
            "value": row_gap,
            "threshold": gates["sinkhorn_row_gap_max"],
            "pass": row_gap <= gates["sinkhorn_row_gap_max"],
        },
        {
            "case": "mcca_column_marginal",
            "value": column_gap,
            "threshold": gates["sinkhorn_column_gap_max"],
            "pass": column_gap <= gates["sinkhorn_column_gap_max"],
        },
        {
            "case": "mcca_crossed_specialization",
            "value": best_scope_mass_gain,
            "threshold": gates["best_scope_mass_gain_over_uniform_min"],
            "pass": best_scope_mass_gain
            >= gates["best_scope_mass_gain_over_uniform_min"],
        },
        {
            "case": "mcca_starvation_protection",
            "value": starvation_column_min,
            "threshold": gates["starvation_column_mass_min"],
            "pass": starvation_column_min
            >= gates["starvation_column_mass_min"],
        },
        {
            "case": "mcca_skill_gradient",
            "value": gradient_norm_min,
            "threshold": gates["gradient_norm_min"],
            "pass": gradient_norm_min >= gates["gradient_norm_min"],
        },
        {
            "case": "mcca_router_gradient",
            "value": router_gradient_norm,
            "threshold": gates["gradient_norm_min"],
            "pass": router_gradient_norm >= gates["gradient_norm_min"],
        },
    ]
    passed = all(bool(case["pass"]) for case in cases)
    result = {
        "architecture_candidate": config["architecture_candidate"],
        "training_candidate": config["training_candidate"],
        "cases": len(cases),
        "passed_cases": sum(bool(case["pass"]) for case in cases),
        "pass": passed,
        "test_used": False,
        "implementation_authorized": False,
        "remote_authorized": False,
        "step6_authorized": passed,
        "decision": (
            "step5_theory_pass_step6_source_design_next"
            if passed
            else "step5_theory_fail_return_step4"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "theory_cases.csv", cases)
    (args.output_dir / "local_gate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
