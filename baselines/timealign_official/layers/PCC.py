"""Projective Coupling Credit objectives for PCSD training.

The functions in this module only change the training objective.  They do not
add parameters or alter the PCSD inference graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


PCC_OBJECTIVE_MODES = frozenset(
    {
        "measure_only",
        "equal_skill",
        "pointwise_route_only",
        "pointwise_capability_skill_only",
        "pointwise_prior_composed",
        "pointwise_pcc_v0",
        "transport_skill_only",
        "transport_route_only",
        "pcc_transport_full",
    }
)

PCC_TEMPERATURE = 1.0
PCC_STANDARDIZATION_EPSILON = 1e-6
PCC_FINAL_SKILL_FLOOR = 0.2
PCC_RAMP_FRACTION = 0.25
PCC_SKILL_WEIGHT = 1.0
PCC_FINAL_ROUTE_WEIGHT = 0.1


@dataclass(frozen=True)
class PCCSchedule:
    """Continuous PCC coefficients at one optimizer progress value."""

    skill_floor: float
    route_weight: float


@dataclass
class PCCObjectiveResult:
    """Loss decomposition and stopped credit tensors for diagnostics."""

    total_loss: Tensor
    fused_loss: Tensor
    skill_loss: Tensor
    route_loss: Tensor
    weighted_skill_loss: Tensor
    weighted_route_loss: Tensor
    pointwise_capability: Tensor
    transported_capability: Tensor
    skill_credit: Tensor
    route_credit: Tensor
    prefix_risk: Tensor
    measure: Tensor
    schedule: PCCSchedule
    diagnostics: dict[str, Tensor]


def prefix_measure(length: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return dense-prefix incidence weights ``omega[t]`` with sum one."""
    if length <= 0:
        raise ValueError("length must be positive")
    inverse = 1.0 / torch.arange(1, length + 1, device=device, dtype=dtype)
    harmonic_tail = torch.flip(
        torch.cumsum(torch.flip(inverse, dims=(0,)), dim=0),
        dims=(0,),
    )
    return harmonic_tail / length


def reverse_cumsum(values: Tensor, *, dim: int) -> Tensor:
    """Cumulative sum from the final coordinate towards the first."""
    return torch.flip(
        torch.cumsum(torch.flip(values, dims=(dim,)), dim=dim),
        dims=(dim,),
    )


def standardized_capability(
    errors: Tensor,
    *,
    prefix_risk: bool,
    temperature: float = PCC_TEMPERATURE,
    standardization_epsilon: float = PCC_STANDARDIZATION_EPSILON,
    stop_gradient: bool = True,
) -> tuple[Tensor, Tensor]:
    """Map ``[B,C,T,S]`` arm errors to scope capability probabilities."""
    if errors.ndim != 4:
        raise ValueError(f"expected errors [B,C,T,S], got {tuple(errors.shape)}")
    if errors.shape[-1] < 2:
        raise ValueError("at least two scopes are required")
    if temperature <= 0.0 or standardization_epsilon <= 0.0:
        raise ValueError("temperature and standardization epsilon must be positive")

    risk = errors
    if prefix_risk:
        length = errors.shape[-2]
        denominator = torch.arange(
            1,
            length + 1,
            device=errors.device,
            dtype=errors.dtype,
        ).view(1, 1, length, 1)
        risk = errors.cumsum(dim=-2) / denominator

    capability_source = risk.detach() if stop_gradient else risk
    centered = capability_source - capability_source.mean(dim=-1, keepdim=True)
    scale = torch.sqrt(
        capability_source.var(dim=-1, keepdim=True, unbiased=False)
    ).clamp_min(standardization_epsilon)
    capability = torch.softmax(-centered / (scale * temperature), dim=-1)
    return risk, capability


def transport_prefix_credit(capability: Tensor) -> Tensor:
    """Transport prefix-indexed credit to natural target coordinates."""
    if capability.ndim != 4:
        raise ValueError(
            f"expected capability [B,C,T,S], got {tuple(capability.shape)}"
        )
    length = capability.shape[-2]
    inverse = 1.0 / torch.arange(
        1,
        length + 1,
        device=capability.device,
        dtype=capability.dtype,
    )
    numerator = reverse_cumsum(
        capability * inverse.view(1, 1, length, 1),
        dim=-2,
    )
    denominator = reverse_cumsum(inverse, dim=0).view(1, 1, length, 1)
    return numerator / denominator


def pcc_schedule(
    progress: float,
    *,
    ramp_fraction: float = PCC_RAMP_FRACTION,
    final_skill_floor: float = PCC_FINAL_SKILL_FLOOR,
    final_route_weight: float = PCC_FINAL_ROUTE_WEIGHT,
) -> PCCSchedule:
    """Return the frozen equal-skill-to-capability curriculum."""
    if not 0.0 < ramp_fraction <= 1.0:
        raise ValueError("ramp_fraction must lie in (0, 1]")
    if not 0.0 <= final_skill_floor <= 1.0:
        raise ValueError("final_skill_floor must lie in [0, 1]")
    if final_route_weight < 0.0:
        raise ValueError("final_route_weight must be non-negative")
    bounded_progress = min(max(float(progress), 0.0), 1.0)
    ramp = min(bounded_progress / ramp_fraction, 1.0)
    return PCCSchedule(
        skill_floor=1.0 - (1.0 - final_skill_floor) * ramp,
        route_weight=final_route_weight * ramp,
    )


def _weighted_target_loss(errors: Tensor, credit: Tensor, measure: Tensor) -> Tensor:
    per_target = (credit * errors).sum(dim=-1)
    return (per_target * measure.view(1, 1, -1)).sum(dim=-1).mean()


def _weighted_route_kl(
    credit: Tensor,
    policy: Tensor,
    measure: Tensor,
) -> Tensor:
    scopes = policy.shape[-1]
    kl = (
        credit
        * (
            torch.log(credit.clamp_min(1e-12))
            - torch.log(policy.clamp_min(1e-12))
        )
    ).sum(dim=-1)
    normalized = kl / torch.log(policy.new_tensor(float(scopes)))
    return (normalized * measure.view(1, 1, -1)).sum(dim=-1).mean()


def projective_coupling_credit_loss(
    fused_forecast: Tensor,
    arm_forecasts: Tensor,
    policy: Tensor,
    target: Tensor,
    *,
    mode: str,
    progress: float,
    stop_gradient: bool = True,
) -> PCCObjectiveResult:
    """Compute one frozen PCC-v1-TI or matched-control objective.

    Args:
        fused_forecast: Raw-scale prediction ``[B,T,C]``.
        arm_forecasts: Raw-scale scope predictions ``[B,C,T,S]``.
        policy: Scope probabilities ``[B,C,T,S]``.
        target: Raw-scale future values ``[B,T,C]``.
        mode: One of the nine frozen Phase-A objective modes.
        progress: Optimizer progress in ``[0,1]``.
        stop_gradient: Keep ``True`` for every authorized Phase-A arm.  The
            alternative exists only for the local conditional-path audit.
    """
    if mode not in PCC_OBJECTIVE_MODES:
        raise ValueError(f"unsupported PCC objective mode: {mode}")
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
    schedule = pcc_schedule(progress)

    uniform_credit = policy.new_full(policy.shape, 1.0 / scopes)
    pointwise_skill = (
        (1.0 - schedule.skill_floor) * pointwise_capability
        + schedule.skill_floor / scopes
    )
    transported_skill = transport_prefix_credit(
        (1.0 - schedule.skill_floor) * prefix_capability
        + schedule.skill_floor / scopes
    )

    skill_kind = {
        "equal_skill": "equal",
        "pointwise_prior_composed": "equal",
        "pointwise_capability_skill_only": "pointwise",
        "pointwise_pcc_v0": "pointwise",
        "transport_skill_only": "transport",
        "pcc_transport_full": "transport",
    }.get(mode, "none")
    route_kind = {
        "pointwise_route_only": "pointwise",
        "pointwise_prior_composed": "pointwise",
        "pointwise_pcc_v0": "pointwise",
        "transport_route_only": "transport",
        "pcc_transport_full": "transport",
    }.get(mode, "none")

    skill_credit = {
        "equal": uniform_credit,
        "pointwise": pointwise_skill,
        "transport": transported_skill,
        "none": uniform_credit,
    }[skill_kind]
    inactive_route_credit = (
        pointwise_capability
        if skill_kind == "pointwise"
        else transported_capability
    )
    route_credit = {
        "pointwise": pointwise_capability,
        "transport": transported_capability,
        "none": inactive_route_credit,
    }[route_kind]

    zero = fused_loss.new_zeros(())
    skill_loss = (
        _weighted_target_loss(arm_error, skill_credit, measure)
        if skill_kind != "none"
        else zero
    )
    route_loss = (
        _weighted_route_kl(route_credit, policy, measure)
        if route_kind != "none"
        else zero
    )
    weighted_skill = PCC_SKILL_WEIGHT * skill_loss
    active_route_weight = schedule.route_weight if route_kind != "none" else 0.0
    active_skill_floor = {
        "equal": 1.0,
        "pointwise": schedule.skill_floor,
        "transport": schedule.skill_floor,
        "none": 0.0,
    }[skill_kind]
    weighted_route = active_route_weight * route_loss
    total_loss = fused_loss + weighted_skill + weighted_route

    entropy_denominator = torch.log(policy.new_tensor(float(scopes)))
    credit_entropy = -(
        route_credit * torch.log(route_credit.clamp_min(1e-12))
    ).sum(dim=-1) / entropy_denominator
    policy_entropy = -(
        policy * torch.log(policy.clamp_min(1e-12))
    ).sum(dim=-1) / entropy_denominator
    weighted_coordinate = measure.view(1, 1, length)
    usage = (policy * measure.view(1, 1, length, 1)).sum(dim=-2).mean(
        dim=(0, 1)
    )
    argmax_match = (
        policy.argmax(dim=-1) == route_credit.argmax(dim=-1)
    ).to(dtype=policy.dtype)
    arm_measure_l1 = (
        arm_error * measure.view(1, 1, length, 1)
    ).sum(dim=-2).mean(dim=(0, 1))
    diagnostics = {
        "pcc_total_loss": total_loss.detach(),
        "pcc_fused_measure_l1": fused_loss.detach(),
        "pcc_skill_loss": skill_loss.detach(),
        "pcc_route_kl": route_loss.detach(),
        "pcc_weighted_skill_loss": weighted_skill.detach(),
        "pcc_weighted_route_loss": weighted_route.detach(),
        "pcc_skill_floor": total_loss.new_tensor(active_skill_floor),
        "pcc_route_weight": total_loss.new_tensor(active_route_weight),
        "pcc_credit_normalized_entropy": (
            credit_entropy * weighted_coordinate
        ).sum(dim=-1).mean().detach(),
        "pcc_policy_normalized_entropy": (
            policy_entropy * weighted_coordinate
        ).sum(dim=-1).mean().detach(),
        "pcc_policy_usage_max": usage.max().detach(),
        "pcc_credit_policy_kl": _weighted_route_kl(
            route_credit,
            policy,
            measure,
        ).detach(),
        "pcc_credit_argmax_accuracy": (
            argmax_match * weighted_coordinate
        ).sum(dim=-1).mean().detach(),
        "pcc_credit_min": route_credit.min().detach(),
        "pcc_credit_max": route_credit.max().detach(),
    }
    for scope_index, value in enumerate(arm_measure_l1):
        diagnostics[f"pcc_arm_s{scope_index}_measure_l1"] = value.detach()

    return PCCObjectiveResult(
        total_loss=total_loss,
        fused_loss=fused_loss,
        skill_loss=skill_loss,
        route_loss=route_loss,
        weighted_skill_loss=weighted_skill,
        weighted_route_loss=weighted_route,
        pointwise_capability=pointwise_capability,
        transported_capability=transported_capability,
        skill_credit=skill_credit,
        route_credit=route_credit,
        prefix_risk=prefix_risk,
        measure=measure,
        schedule=schedule,
        diagnostics=diagnostics,
    )
