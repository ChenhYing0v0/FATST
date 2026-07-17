#!/usr/bin/env python3
"""Audit the frozen SIFF/MCCA Step 6 design before implementation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_post_pcc_step6.json"),
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("configs/stage_c_five_dataset_natural_profiles.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/stage_c_post_pcc_step6_design_20260717"),
    )
    return parser.parse_args()


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


def record(
    rows: list[dict[str, Any]],
    name: str,
    value: float | int | bool | str,
    threshold: str,
    passed: bool,
) -> None:
    rows.append(
        {
            "case": name,
            "value": value,
            "threshold": threshold,
            "pass": bool(passed),
        }
    )


def scale_basis(scales: Tensor, length: int) -> Tensor:
    raw = torch.log(scales) / math.log(float(length))
    centered = raw - raw.mean()
    normalized = centered / torch.sqrt(torch.mean(centered.square()))
    return torch.stack((torch.ones_like(raw), normalized), dim=-1)


def projective_measure(length: int, *, dtype: torch.dtype) -> Tensor:
    horizons = torch.arange(1, length + 1, dtype=dtype)
    reciprocal = 1.0 / horizons
    return torch.flip(
        torch.cumsum(torch.flip(reciprocal, dims=(0,)), dim=0),
        dims=(0,),
    ) / length


def log_sinkhorn(
    reference: Tensor,
    row_mass: Tensor,
    column_mass: Tensor,
    *,
    iterations: int,
    floor: float,
) -> Tensor:
    log_kernel = torch.log(reference.clamp_min(floor))
    log_row = torch.log(row_mass)
    log_column = torch.log(column_mass)
    log_left = torch.zeros_like(log_row)
    log_right = torch.zeros_like(log_column)
    for _ in range(iterations):
        log_left = log_row - torch.logsumexp(
            log_kernel + log_right.unsqueeze(0), dim=1
        )
        log_right = log_column - torch.logsumexp(
            log_kernel + log_left.unsqueeze(1), dim=0
        )
    return torch.exp(
        log_kernel + log_left.unsqueeze(1) + log_right.unsqueeze(0)
    )


def relative_gap(value: int, target: int) -> float:
    return abs(value - target) / target


def field_parameters(
    *,
    components: int,
    dimensions: int,
    readout: int,
    rank: int,
    length: int,
    independent_scopes: int = 0,
) -> int:
    leading = independent_scopes if independent_scopes else components
    return (
        leading * dimensions * (readout + 1) * rank
        + 2 * length * rank
        + length
    )


def nearest_matched_rank(
    target: int,
    *,
    leading: int,
    dimensions: int,
    readout: int,
    length: int,
) -> int:
    coefficient = leading * dimensions * (readout + 1) + 2 * length
    return max(1, round((target - length) / coefficient))


def parameter_accounting(
    config: dict[str, Any], profiles: dict[str, Any]
) -> list[dict[str, Any]]:
    length = int(config["contract"]["full_domain_length"])
    dimensions = int(config["siff"]["coordinate_dim"])
    rank = int(config["siff"]["mode_rank"])
    components = int(config["siff"]["components"])
    scopes = len(config["contract"]["scopes"])
    rows: list[dict[str, Any]] = []
    for dataset, profile in profiles.items():
        readout = int(profile["state_width"])
        target = field_parameters(
            components=components,
            dimensions=dimensions,
            readout=readout,
            rank=rank,
            length=length,
        )
        q1_rank = nearest_matched_rank(
            target,
            leading=1,
            dimensions=dimensions,
            readout=readout,
            length=length,
        )
        independent_rank = nearest_matched_rank(
            target,
            leading=scopes,
            dimensions=dimensions,
            readout=readout,
            length=length,
        )
        q1_parameters = field_parameters(
            components=1,
            dimensions=dimensions,
            readout=readout,
            rank=q1_rank,
            length=length,
        )
        independent_parameters = field_parameters(
            components=1,
            dimensions=dimensions,
            readout=readout,
            rank=independent_rank,
            length=length,
            independent_scopes=scopes,
        )
        rows.append(
            {
                "dataset": dataset,
                "readout_dim": readout,
                "siff_q": components,
                "siff_rank": rank,
                "siff_field_parameters": target,
                "q1_wide_rank": q1_rank,
                "q1_wide_field_parameters": q1_parameters,
                "q1_wide_relative_gap": relative_gap(q1_parameters, target),
                "independent_scope_rank": independent_rank,
                "independent_scope_field_parameters": independent_parameters,
                "independent_scope_relative_gap": relative_gap(
                    independent_parameters, target
                ),
            }
        )
    return rows


def siff_checks(config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    gates = config["gates"]
    dtype = torch.float64
    scales = torch.tensor(config["contract"]["scopes"], dtype=dtype)
    length = int(config["contract"]["full_domain_length"])
    basis = scale_basis(scales, length)
    record(
        rows,
        "scale_basis_constant",
        float((basis[:, 0] - 1.0).abs().max()),
        f"<={gates['scale_basis_constant_gap_max']}",
        float((basis[:, 0] - 1.0).abs().max())
        <= gates["scale_basis_constant_gap_max"],
    )
    linear_mean = float(basis[:, 1].mean().abs())
    linear_rms_gap = float(
        (torch.sqrt(torch.mean(basis[:, 1].square())) - 1.0).abs()
    )
    record(
        rows,
        "scale_basis_linear_zero_mean",
        linear_mean,
        f"<={gates['scale_basis_linear_mean_max']}",
        linear_mean <= gates["scale_basis_linear_mean_max"],
    )
    record(
        rows,
        "scale_basis_linear_unit_rms",
        linear_rms_gap,
        f"<={gates['scale_basis_linear_rms_gap_max']}",
        linear_rms_gap <= gates["scale_basis_linear_rms_gap_max"],
    )

    generator = torch.Generator().manual_seed(1707)
    hidden = torch.randn(2, 3, 7, dtype=dtype, generator=generator)
    base_weight = torch.randn(4, 7, 5, dtype=dtype, generator=generator)
    base_bias = torch.randn(4, 5, dtype=dtype, generator=generator)
    weights = torch.zeros(2, 4, 7, 5, dtype=dtype)
    biases = torch.zeros(2, 4, 5, dtype=dtype)
    weights[0] = base_weight
    biases[0] = base_bias
    current = torch.einsum("bcr,drk->bcdk", hidden, base_weight) + base_bias
    components = (
        torch.einsum("bcr,qdrk->bcqdk", hidden, weights)
        + biases.view(1, 1, 2, 4, 5)
    )
    siff = torch.einsum("sq,bcqdk->bcsdk", basis, components)
    containment_gap = float((siff - current.unsqueeze(2)).abs().max())
    record(
        rows,
        "siff_q1_exact_containment",
        containment_gap,
        f"<={gates['siff_containment_gap_max']}",
        containment_gap <= gates["siff_containment_gap_max"],
    )

    const_weights = torch.randn(2, 4, 7, 5, dtype=dtype, generator=generator)
    const_biases = torch.randn(2, 4, 5, dtype=dtype, generator=generator)
    const_components = (
        torch.einsum("bcr,qdrk->bcqdk", hidden, const_weights)
        + const_biases.view(1, 1, 2, 4, 5)
    )
    constant_basis = torch.ones_like(basis)
    constant_output = torch.einsum(
        "sq,bcqdk->bcsdk", constant_basis, const_components
    )
    merged = torch.einsum(
        "bcr,drk->bcdk", hidden, const_weights.sum(dim=0)
    ) + const_biases.sum(dim=0)
    constant_gap = float((constant_output - merged.unsqueeze(2)).abs().max())
    record(
        rows,
        "siff_const_same_parameters_collapses_to_q1",
        constant_gap,
        f"<={gates['siff_constant_control_gap_max']}",
        constant_gap <= gates["siff_constant_control_gap_max"],
    )
    record(
        rows,
        "permuted_scale_control_changes_semantics",
        bool(not torch.equal(basis[:, 1], basis.flip(0)[:, 1])),
        "true",
        not torch.equal(basis[:, 1], basis.flip(0)[:, 1]),
    )


def kl_divergence(allocation: Tensor, reference: Tensor) -> Tensor:
    return (
        allocation
        * (
            torch.log(allocation.clamp_min(1e-30))
            - torch.log(reference.clamp_min(1e-30))
        )
        - allocation
        + reference
    ).sum()


def mcca_case(
    *,
    dtype: torch.dtype,
    config: dict[str, Any],
) -> dict[str, float]:
    length = int(config["contract"]["full_domain_length"])
    scopes = len(config["contract"]["scopes"])
    instances = 6
    omega = projective_measure(length, dtype=dtype)
    row_mass = omega.repeat(instances) / instances
    target_index = torch.arange(length).repeat(instances)
    instance_index = torch.arange(instances).repeat_interleave(length)
    best = (3 * target_index // length + instance_index) % scopes
    logits = torch.full((instances * length, scopes), -2.0, dtype=dtype)
    logits[torch.arange(instances * length), best] = 2.0
    capability = torch.softmax(logits, dim=-1)
    ramp = 1.0
    uniform = torch.full_like(capability, 1.0 / scopes)
    ramped = (1.0 - ramp) * uniform + ramp * capability
    reference = row_mass[:, None] * ramped
    floor = float(config["mcca"]["final_global_skill_floor"])
    raw_marginal = reference.sum(dim=0)
    column_mass = (1.0 - floor) * raw_marginal + floor / scopes
    allocation = log_sinkhorn(
        reference,
        row_mass,
        column_mass,
        iterations=int(config["mcca"]["iterations"]),
        floor=float(config["mcca"]["kernel_floor"]),
    )
    pcc_credit = (1.0 - floor) * ramped + floor / scopes
    pcc_allocation = row_mass[:, None] * pcc_credit
    row_gap = float((allocation.sum(dim=1) - row_mass).abs().max())
    column_gap = float(
        (allocation.sum(dim=0) - column_mass).abs().max()
    )
    pcc_column_gap = float(
        (pcc_allocation.sum(dim=0) - column_mass).abs().max()
    )
    kl_advantage = float(
        kl_divergence(pcc_allocation, reference)
        - kl_divergence(allocation, reference)
    )
    return {
        "row_gap": row_gap,
        "column_gap": column_gap,
        "pcc_column_gap": pcc_column_gap,
        "kl_advantage": kl_advantage,
        "minimum_scope_mass": float(allocation.sum(dim=0).min()),
    }


def mcca_checks(config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    gates = config["gates"]
    for dtype, label, gap_gate in (
        (torch.float64, "float64", gates["sinkhorn_float64_gap_max"]),
        (torch.float32, "float32", gates["sinkhorn_float32_gap_max"]),
    ):
        values = mcca_case(dtype=dtype, config=config)
        maximum_gap = max(values["row_gap"], values["column_gap"])
        record(
            rows,
            f"mcca_{label}_marginals",
            maximum_gap,
            f"<={gap_gate}",
            maximum_gap <= gap_gate,
        )
        record(
            rows,
            f"pcc_mcca_{label}_same_column_mass",
            values["pcc_column_gap"],
            f"<={gates['pcc_mcca_column_mass_gap_max']}",
            values["pcc_column_gap"]
            <= gates["pcc_mcca_column_mass_gap_max"],
        )
        record(
            rows,
            f"mcca_{label}_closer_than_pcc_mix",
            values["kl_advantage"],
            f">={gates['mcca_kl_advantage_min']}",
            values["kl_advantage"] >= gates["mcca_kl_advantage_min"],
        )
        record(
            rows,
            f"mcca_{label}_global_starvation_floor",
            values["minimum_scope_mass"],
            f">={gates['minimum_final_scope_mass']}",
            values["minimum_scope_mass"]
            >= gates["minimum_final_scope_mass"] - gap_gate,
        )

    predictions = torch.randn(2, 3, 12, 5, dtype=torch.float64, requires_grad=True)
    target = torch.randn(2, 3, 12, 1, dtype=torch.float64)
    errors = (predictions - target).abs()
    omega = projective_measure(12, dtype=torch.float64)
    detached_credit = torch.softmax(-errors.detach(), dim=-1)
    skill_loss = (
        errors * detached_credit * omega.view(1, 1, 12, 1)
    ).sum(dim=(-2, -1)).mean()
    skill_loss.backward()
    gradient_norm = float(predictions.grad.norm())
    record(
        rows,
        "mcca_stopped_assignment_skill_gradient",
        gradient_norm,
        f">={gates['gradient_norm_min']}",
        math.isfinite(gradient_norm) and gradient_norm >= gates["gradient_norm_min"],
    )


def contract_checks(config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    contract = config["contract"]
    controls = config["controls"]
    record(
        rows,
        "fixed_past_no_requested_horizon_feature",
        not contract["requested_horizon_feature"],
        "true",
        not contract["requested_horizon_feature"],
    )
    record(
        rows,
        "one_stage_one_forward_no_teacher",
        bool(
            contract["one_forward"]
            and contract["one_stage_training"]
            and not contract["offline_teacher"]
            and not contract["ema_teacher"]
            and not contract["second_forward"]
        ),
        "true",
        bool(
            contract["one_forward"]
            and contract["one_stage_training"]
            and not contract["offline_teacher"]
            and not contract["ema_teacher"]
            and not contract["second_forward"]
        ),
    )
    factorial = controls["core_factorial"]
    expected = len(factorial["architectures"]) * len(factorial["training"])
    record(
        rows,
        "core_two_by_three_factorial_complete",
        len(factorial["arms"]),
        f"=={expected}",
        len(factorial["arms"]) == expected,
    )
    required_controls = {
        "SIFF_CONST_MCCA",
        "SIFF_PERMUTED_SCALE_MCCA",
        "PCSD_Q1_WIDE_MCCA",
        "INDEPENDENT_SCOPE_MATCHED_MCCA",
        "DENSE_SIFF_MATCHED",
    }
    architecture_controls = set(controls["architecture_attribution"])
    record(
        rows,
        "architecture_attribution_controls_complete",
        len(required_controls & architecture_controls),
        f"=={len(required_controls)}",
        required_controls <= architecture_controls,
    )
    required_training = {"PCSD_POINTWISE_MCCA", "PCSD_UNIFORM_BALANCED_OT"}
    training_controls = set(controls["training_attribution"])
    record(
        rows,
        "training_attribution_controls_complete",
        len(required_training & training_controls),
        f"=={len(required_training)}",
        required_training <= training_controls,
    )
    method_arms = set(factorial["arms"]) | architecture_controls | training_controls
    reusable = set(controls["reusable_unchanged_training_references"])
    new_arms = method_arms - reusable
    expected_runs = len(new_arms) * len(config["phase_a"]["datasets"])
    record(
        rows,
        "phase_a_new_run_count_consistent",
        expected_runs,
        f"=={config['phase_a']['new_candidate_runs']}",
        expected_runs == config["phase_a"]["new_candidate_runs"],
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    profile_contract = json.loads(args.profiles.read_text(encoding="utf-8"))
    profiles = profile_contract["dataset_profiles"]
    rows: list[dict[str, Any]] = []
    siff_checks(config, rows)
    mcca_checks(config, rows)
    contract_checks(config, rows)
    accounting = parameter_accounting(config, profiles)
    max_parameter_gap = max(
        max(row["q1_wide_relative_gap"], row["independent_scope_relative_gap"])
        for row in accounting
    )
    parameter_gate = float(config["gates"]["parameter_match_relative_gap_max"])
    record(
        rows,
        "integer_rank_parameter_controls_matched",
        max_parameter_gap,
        f"<={parameter_gate}",
        max_parameter_gap <= parameter_gate,
    )

    passed = all(bool(row["pass"]) for row in rows)
    result = {
        "architecture_candidate": config["architecture_candidate"],
        "training_candidate": config["training_candidate"],
        "cases": len(rows),
        "passed_cases": sum(bool(row["pass"]) for row in rows),
        "pass": passed,
        "max_parameter_control_relative_gap": max_parameter_gap,
        "test_used": False,
        "remote_authorized": False,
        "step7a_local_implementation_authorized": passed,
        "decision": (
            "step6_conditional_narrative_pass_step7a_local_next"
            if passed
            else "step6_fail_return_step4_or_step5"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "design_cases.csv", rows)
    write_csv(args.output_dir / "parameter_controls.csv", accounting)
    (args.output_dir / "local_gate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
