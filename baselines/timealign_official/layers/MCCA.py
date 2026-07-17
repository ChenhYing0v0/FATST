"""Measure-constrained competitive assignment objectives."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from layers.PCC import (
    PCC_FINAL_ROUTE_WEIGHT,
    PCC_FINAL_SKILL_FLOOR,
    PCC_RAMP_FRACTION,
    PCC_SKILL_WEIGHT,
    PCCObjectiveResult,
    PCCSchedule,
    _weighted_route_kl,
    _weighted_target_loss,
    prefix_measure,
    standardized_capability,
    transport_prefix_credit,
)


MCCA_OBJECTIVE_MODES = frozenset(
    {
        "mcca_transport_full",
        "mcca_pointwise_full",
        "mcca_uniform_balanced",
    }
)
MCCA_KERNEL_FLOOR = 1e-8
MCCA_SINKHORN_ITERATIONS = 64


def _ramp(progress: float, ramp_fraction: float = PCC_RAMP_FRACTION) -> float:
    if not 0.0 < ramp_fraction <= 1.0:
        raise ValueError("ramp_fraction must lie in (0, 1]")
    bounded = min(max(float(progress), 0.0), 1.0)
    return min(bounded / ramp_fraction, 1.0)


def log_i_projection(
    reference: Tensor,
    row_marginal: Tensor,
    column_marginal: Tensor,
    *,
    iterations: int = MCCA_SINKHORN_ITERATIONS,
    kernel_floor: float = MCCA_KERNEL_FLOOR,
) -> Tensor:
    """KL-project a positive matrix onto prescribed row/column marginals."""
    if reference.ndim != 2:
        raise ValueError("reference must have shape [N,S]")
    if row_marginal.shape != reference.shape[:1]:
        raise ValueError("row_marginal must match reference rows")
    if column_marginal.shape != reference.shape[1:]:
        raise ValueError("column_marginal must match reference columns")
    if iterations <= 0 or kernel_floor <= 0.0:
        raise ValueError("iterations and kernel_floor must be positive")
    if bool((row_marginal <= 0).any()) or bool((column_marginal <= 0).any()):
        raise ValueError("MCCA marginals must be strictly positive")

    log_reference = reference.clamp_min(kernel_floor).log()
    log_row = row_marginal.log()
    log_column = column_marginal.log()
    log_u = torch.zeros_like(log_row)
    log_v = torch.zeros_like(log_column)
    for _ in range(iterations):
        log_u = log_row - torch.logsumexp(
            log_reference + log_v.unsqueeze(0), dim=1
        )
        log_v = log_column - torch.logsumexp(
            log_reference + log_u.unsqueeze(1), dim=0
        )
    return torch.exp(log_reference + log_u.unsqueeze(1) + log_v.unsqueeze(0))


def competitive_assignment(
    capability: Tensor,
    measure: Tensor,
    *,
    progress: float,
    uniform_column_marginal: bool = False,
    iterations: int = MCCA_SINKHORN_ITERATIONS,
    kernel_floor: float = MCCA_KERNEL_FLOOR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return competitive per-row credit and its transport marginals."""
    if capability.ndim != 4:
        raise ValueError("capability must have shape [B,C,T,S]")
    batch, channels, length, scopes = capability.shape
    if measure.shape != (length,):
        raise ValueError("measure must match the target length")
    alpha = _ramp(progress)
    uniform = capability.new_full(capability.shape, 1.0 / scopes)
    ramped = (1.0 - alpha) * uniform + alpha * capability

    row = (
        measure.view(1, 1, length)
        .expand(batch, channels, -1)
        .reshape(-1)
        / float(batch * channels)
    )
    flat_ramped = ramped.reshape(-1, scopes)
    reference = row.unsqueeze(-1) * flat_ramped
    capability_marginal = reference.sum(dim=0)
    if uniform_column_marginal:
        column = capability.new_full((scopes,), 1.0 / scopes)
    else:
        column = (
            (1.0 - PCC_FINAL_SKILL_FLOOR) * capability_marginal
            + PCC_FINAL_SKILL_FLOOR / scopes
        )
    allocation = log_i_projection(
        reference,
        row,
        column,
        iterations=iterations,
        kernel_floor=kernel_floor,
    )
    credit = (allocation / row.unsqueeze(-1)).reshape(
        batch, channels, length, scopes
    )
    return credit, allocation, row, column


def measure_constrained_competitive_loss(
    fused_forecast: Tensor,
    arm_forecasts: Tensor,
    policy: Tensor,
    target: Tensor,
    *,
    mode: str,
    progress: float,
    stop_gradient: bool = True,
) -> PCCObjectiveResult:
    """Compute the MCCA candidate or one of its frozen controls."""
    if mode not in MCCA_OBJECTIVE_MODES:
        raise ValueError(f"unsupported MCCA objective mode: {mode}")
    if fused_forecast.shape != target.shape or fused_forecast.ndim != 3:
        raise ValueError("fused_forecast and target must share shape [B,T,C]")
    expected = (
        target.shape[0],
        target.shape[2],
        target.shape[1],
        policy.shape[-1],
    )
    if tuple(arm_forecasts.shape) != expected or tuple(policy.shape) != expected:
        raise ValueError(
            "arm_forecasts and policy must share shape [B,C,T,S]; "
            f"expected {expected}, got {tuple(arm_forecasts.shape)} and "
            f"{tuple(policy.shape)}"
        )

    length = target.shape[1]
    scopes = policy.shape[-1]
    measure = prefix_measure(length, device=target.device, dtype=target.dtype)
    target_bct = target.permute(0, 2, 1)
    fused_error = (fused_forecast - target).abs().permute(0, 2, 1)
    arm_error = (arm_forecasts - target_bct.unsqueeze(-1)).abs()
    fused_loss = (
        fused_error * measure.view(1, 1, length)
    ).sum(dim=-1).mean()

    prefix_risk, prefix_capability = standardized_capability(
        arm_error,
        prefix_risk=True,
        stop_gradient=stop_gradient,
    )
    _point_risk, pointwise_capability = standardized_capability(
        arm_error,
        prefix_risk=False,
        stop_gradient=stop_gradient,
    )
    transported_capability = transport_prefix_credit(prefix_capability)
    capability = (
        pointwise_capability
        if mode == "mcca_pointwise_full"
        else transported_capability
    )
    credit, allocation, row, column = competitive_assignment(
        capability,
        measure,
        progress=progress,
        uniform_column_marginal=mode == "mcca_uniform_balanced",
    )
    if stop_gradient:
        credit = credit.detach()
        allocation = allocation.detach()
        row = row.detach()
        column = column.detach()

    alpha = _ramp(progress)
    route_weight = PCC_FINAL_ROUTE_WEIGHT * alpha
    skill_loss = _weighted_target_loss(arm_error, credit, measure)
    route_loss = _weighted_route_kl(credit, policy, measure)
    weighted_skill = PCC_SKILL_WEIGHT * skill_loss
    weighted_route = route_weight * route_loss
    total_loss = fused_loss + weighted_skill + weighted_route

    entropy_denominator = math.log(float(scopes))
    credit_entropy = -(
        credit * torch.log(credit.clamp_min(1e-12))
    ).sum(dim=-1) / entropy_denominator
    policy_entropy = -(
        policy * torch.log(policy.clamp_min(1e-12))
    ).sum(dim=-1) / entropy_denominator
    usage = (policy * measure.view(1, 1, length, 1)).sum(dim=-2).mean(
        dim=(0, 1)
    )
    flat_allocation = allocation.reshape(-1, scopes)
    row_gap = (flat_allocation.sum(dim=-1) - row).abs().max()
    column_gap = (flat_allocation.sum(dim=0) - column).abs().max()
    arm_measure_l1 = (
        arm_error * measure.view(1, 1, length, 1)
    ).sum(dim=-2).mean(dim=(0, 1))
    weighted_coordinate = measure.view(1, 1, length)
    diagnostics = {
        "pcc_total_loss": total_loss.detach(),
        "pcc_fused_measure_l1": fused_loss.detach(),
        "pcc_skill_loss": skill_loss.detach(),
        "pcc_route_kl": route_loss.detach(),
        "pcc_weighted_skill_loss": weighted_skill.detach(),
        "pcc_weighted_route_loss": weighted_route.detach(),
        "pcc_skill_floor": total_loss.new_tensor(1.0 - 0.8 * alpha),
        "pcc_route_weight": total_loss.new_tensor(route_weight),
        "pcc_credit_normalized_entropy": (
            credit_entropy * weighted_coordinate
        ).sum(dim=-1).mean().detach(),
        "pcc_policy_normalized_entropy": (
            policy_entropy * weighted_coordinate
        ).sum(dim=-1).mean().detach(),
        "pcc_policy_usage_max": usage.max().detach(),
        "pcc_credit_policy_kl": route_loss.detach(),
        "pcc_credit_min": credit.min().detach(),
        "pcc_credit_max": credit.max().detach(),
        "mcca_row_marginal_gap": row_gap.detach(),
        "mcca_column_marginal_gap": column_gap.detach(),
        "mcca_scope_mass_min": column.min().detach(),
        "mcca_scope_mass_max": column.max().detach(),
        "mcca_ramp": total_loss.new_tensor(alpha),
    }
    for scope_index, value in enumerate(arm_measure_l1):
        diagnostics[f"pcc_arm_s{scope_index}_measure_l1"] = value.detach()
        diagnostics[f"mcca_scope_s{scope_index}_mass"] = column[
            scope_index
        ].detach()

    return PCCObjectiveResult(
        total_loss=total_loss,
        fused_loss=fused_loss,
        skill_loss=skill_loss,
        route_loss=route_loss,
        weighted_skill_loss=weighted_skill,
        weighted_route_loss=weighted_route,
        pointwise_capability=pointwise_capability,
        transported_capability=transported_capability,
        skill_credit=credit,
        route_credit=credit,
        prefix_risk=prefix_risk,
        measure=measure,
        schedule=PCCSchedule(
            skill_floor=1.0 - 0.8 * alpha,
            route_weight=route_weight,
        ),
        diagnostics=diagnostics,
    )
