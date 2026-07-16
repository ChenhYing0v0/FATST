#!/usr/bin/env python3
"""Check PCC-v1 nested-risk credit transport and frozen Step6 design."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_sc2_pcc_step6.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/stage_c_sc2_pcc_step6_design_20260716"),
    )
    return parser.parse_args()


def prefix_measure(length: int, *, dtype: torch.dtype) -> Tensor:
    inverse = 1.0 / torch.arange(1, length + 1, dtype=dtype)
    return torch.flip(
        torch.cumsum(torch.flip(inverse, dims=(0,)), dim=0),
        dims=(0,),
    ) / length


def reverse_cumsum(values: Tensor, *, dim: int) -> Tensor:
    return torch.flip(
        torch.cumsum(torch.flip(values, dims=(dim,)), dim=dim),
        dims=(dim,),
    )


def prefix_capability(
    errors: Tensor,
    *,
    temperature: float,
    standardization_epsilon: float,
) -> tuple[Tensor, Tensor]:
    length = errors.shape[-2]
    denominator = torch.arange(
        1,
        length + 1,
        dtype=errors.dtype,
        device=errors.device,
    ).view(1, 1, length, 1)
    prefix_risk = errors.cumsum(dim=-2) / denominator
    detached = prefix_risk.detach()
    centered = detached - detached.mean(dim=-1, keepdim=True)
    scale = torch.sqrt(
        detached.var(dim=-1, keepdim=True, unbiased=False)
    ).clamp_min(standardization_epsilon)
    capability = torch.softmax(-centered / (scale * temperature), dim=-1)
    return prefix_risk, capability


def transport_to_targets(capability: Tensor) -> Tensor:
    length = capability.shape[-2]
    inverse = 1.0 / torch.arange(
        1,
        length + 1,
        dtype=capability.dtype,
        device=capability.device,
    )
    weighted = capability * inverse.view(1, 1, length, 1)
    numerator = reverse_cumsum(weighted, dim=-2)
    denominator = reverse_cumsum(inverse, dim=0).view(1, 1, length, 1)
    return numerator / denominator


def schedule(
    progress: float,
    *,
    ramp_fraction: float,
    final_floor: float,
    final_route_weight: float,
) -> tuple[float, float]:
    ramp = min(max(progress / ramp_fraction, 0.0), 1.0)
    floor = 1.0 - (1.0 - final_floor) * ramp
    route_weight = final_route_weight * ramp
    return floor, route_weight


def add_case(
    rows: list[dict[str, Any]],
    name: str,
    value: float | int | str | bool,
    threshold: str,
    passed: bool,
) -> None:
    rows.append(
        {
            "case": name,
            "value": value,
            "threshold": threshold,
            "pass": passed,
        }
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    objective = config["objective"]
    contract = config["contract"]
    phase_a = config["phase_a"]
    rows: list[dict[str, Any]] = []

    torch.manual_seed(20260716)
    dtype = torch.float64
    batch, channels, length, scopes = 3, 4, 31, 5
    errors = torch.rand(
        batch,
        channels,
        length,
        scopes,
        dtype=dtype,
        requires_grad=True,
    )
    prefix_risk, capability = prefix_capability(
        errors,
        temperature=objective["temperature"],
        standardization_epsilon=1e-12,
    )
    transported = transport_to_targets(capability)
    weights = prefix_measure(length, dtype=dtype)

    add_case(
        rows,
        "prefix_risk_shape",
        str(list(prefix_risk.shape)),
        "[3,4,31,5]",
        list(prefix_risk.shape) == [3, 4, 31, 5],
    )
    cap_sum_gap = float((capability.sum(dim=-1) - 1.0).abs().max())
    add_case(rows, "prefix_capability_simplex", cap_sum_gap, "<=1e-12", cap_sum_gap <= 1e-12)
    transport_sum_gap = float((transported.sum(dim=-1) - 1.0).abs().max())
    add_case(
        rows,
        "transported_credit_simplex",
        transport_sum_gap,
        "<=1e-12",
        transport_sum_gap <= 1e-12,
    )
    add_case(
        rows,
        "transported_credit_nonnegative",
        float(transported.min()),
        ">=0",
        float(transported.min()) >= 0.0,
    )

    floor = objective["final_skill_floor"]
    skill_capability = (1.0 - floor) * capability + floor / scopes
    skill_credit = transport_to_targets(skill_capability)
    add_case(
        rows,
        "transported_skill_floor",
        float(skill_credit.min()),
        f">={floor / scopes}",
        float(skill_credit.min()) >= floor / scopes - 1e-12,
    )

    direct_prefix_skill = (skill_capability * prefix_risk).sum(dim=-1).mean(dim=-1).mean()
    transported_skill = (
        weights.view(1, 1, length)
        * (skill_credit * errors).sum(dim=-1)
    ).sum(dim=-1).mean()
    expansion_gap = float((direct_prefix_skill - transported_skill).abs())
    add_case(
        rows,
        "nested_prefix_transport_identity",
        expansion_gap,
        "<=1e-12",
        expansion_gap <= 1e-12,
    )

    constant = torch.softmax(torch.randn(1, 1, 1, scopes, dtype=dtype), dim=-1)
    constant = constant.expand(2, 3, length, scopes)
    constant_gap = float((transport_to_targets(constant) - constant).abs().max())
    add_case(rows, "constant_credit_fixed_point", constant_gap, "<=1e-12", constant_gap <= 1e-12)

    identical = torch.ones(2, 2, length, scopes, dtype=dtype)
    _, identical_capability = prefix_capability(
        identical,
        temperature=1.0,
        standardization_epsilon=1e-12,
    )
    uniform_gap = float((identical_capability - 1.0 / scopes).abs().max())
    add_case(
        rows,
        "identical_arms_uniform_capability",
        uniform_gap,
        "<=1e-12",
        uniform_gap <= 1e-12,
    )

    affine_errors = 3.7 * errors.detach() + 2.1
    _, affine_capability = prefix_capability(
        affine_errors,
        temperature=objective["temperature"],
        standardization_epsilon=1e-12,
    )
    affine_gap = float((affine_capability - capability).abs().max())
    add_case(rows, "positive_affine_error_invariance", affine_gap, "<=1e-9", affine_gap <= 1e-9)

    crossed = torch.full((1, 1, length, scopes), 1.0, dtype=dtype)
    crossed[..., : length // 2, 0] = 0.05
    crossed[..., length // 2 :, 0] = 1.8
    crossed[..., : length // 2, 1] = 1.8
    crossed[..., length // 2 :, 1] = 0.05
    _, crossed_prefix = prefix_capability(
        crossed,
        temperature=1.0,
        standardization_epsilon=1e-12,
    )
    crossed_transport = transport_to_targets(crossed_prefix)
    centered_point = crossed - crossed.mean(dim=-1, keepdim=True)
    point_scale = torch.sqrt(crossed.var(dim=-1, keepdim=True, unbiased=False) + 1e-12)
    pointwise = torch.softmax(-centered_point / point_scale, dim=-1)
    crossed_gap = float((crossed_transport - pointwise).abs().max())
    add_case(rows, "transport_not_pointwise_credit", crossed_gap, ">=1e-3", crossed_gap >= 1e-3)

    transported_skill.backward()
    add_case(
        rows,
        "stopgrad_credit_no_gradient_path",
        capability.requires_grad,
        "False",
        not capability.requires_grad,
    )
    finite_gradient = bool(torch.isfinite(errors.grad).all())
    add_case(
        rows,
        "skill_gradient_finite",
        finite_gradient,
        "True",
        finite_gradient,
    )

    start_floor, start_route = schedule(
        0.0,
        ramp_fraction=objective["ramp_fraction"],
        final_floor=floor,
        final_route_weight=objective["lambda_route"],
    )
    end_floor, end_route = schedule(
        1.0,
        ramp_fraction=objective["ramp_fraction"],
        final_floor=floor,
        final_route_weight=objective["lambda_route"],
    )
    add_case(
        rows,
        "schedule_starts_equal_skill",
        start_floor,
        "1.0",
        abs(start_floor - 1.0) <= 1e-12,
    )
    add_case(rows, "schedule_starts_route_off", start_route, "0.0", abs(start_route) <= 1e-12)
    add_case(
        rows,
        "schedule_reaches_final_floor",
        end_floor,
        str(floor),
        abs(end_floor - floor) <= 1e-12,
    )
    add_case(
        rows,
        "schedule_reaches_route_weight",
        end_route,
        str(objective["lambda_route"]),
        abs(end_route - objective["lambda_route"]) <= 1e-12,
    )

    expected_training_arms = {
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
    add_case(
        rows,
        "control_matrix_exact",
        len(phase_a["training_arms"]),
        "9 exact arms",
        set(phase_a["training_arms"]) == expected_training_arms,
    )
    expected_runs = len(config["datasets"]) * len(expected_training_arms)
    add_case(
        rows,
        "phase_a_run_count",
        phase_a["new_training_runs"],
        str(expected_runs),
        phase_a["new_training_runs"] == expected_runs,
    )
    protocol_pass = all(
        [
            contract["one_forward"],
            contract["one_stage_training"],
            contract["from_scratch_e2e"],
            not contract["requested_horizon_feature"],
            not contract["offline_teacher"],
            not contract["ema_teacher"],
            not contract["second_forward"],
            not contract["inference_graph_changed"],
            not config["remote_authorized"],
            not config["test_access_authorized"],
            objective["transport_complexity"] == "O(B*C*T*S)",
            objective["route_kl_normalization"] == "divide_by_log_scope_count",
        ]
    )
    add_case(rows, "protocol_contract", protocol_pass, "True", protocol_pass)

    overall_pass = all(bool(row["pass"]) for row in rows)
    result = {
        "candidate": config["candidate"],
        "parent_candidate": config["parent_candidate"],
        "test_informed": config["test_informed"],
        "cases": len(rows),
        "passed": sum(bool(row["pass"]) for row in rows),
        "overall_pass": overall_pass,
        "maximum_transport_identity_gap": expansion_gap,
        "pointwise_transport_max_gap": crossed_gap,
        "decision": (
            "step6_pass_step7a_local_authorized"
            if overall_pass and config["implementation_authorized"]
            else "step5b_pass_step6_narrative_decision_next"
            if overall_pass
            else "return_step5_redesign"
        ),
        "implementation_authorized": bool(
            overall_pass and config["implementation_authorized"]
        ),
        "remote_authorized": False,
        "test_access_authorized": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "design_cases.csv", rows)
    (args.output_dir / "local_gate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
