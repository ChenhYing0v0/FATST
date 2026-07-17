#!/usr/bin/env python3
"""Run the local code-theory gate for SIFF and MCCA Step 7A."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from layers.MCCA import (  # noqa: E402
    competitive_assignment,
    measure_constrained_competitive_loss,
)
from layers.PCC import pcc_schedule, prefix_measure  # noqa: E402
from layers.PCSD import PCSDCouplingFieldReadout  # noqa: E402
from layers.SIFF import (  # noqa: E402
    SIFFCouplingFieldReadout,
    siff_parameter_count,
)


OUTPUT_DIR = ROOT / "analysis" / "stage_c_post_pcc_step7a_local_20260717"
PROFILE_PATH = ROOT / "configs" / "stage_c_five_dataset_natural_profiles.json"
STEP6_PATH = ROOT / "configs" / "stage_c_post_pcc_step6.json"
MATCHED_RANKS = {
    "ETTh1": {"q1": 463, "independent": 109},
    "ETTh2": {"q1": 430, "independent": 116},
    "ETTm1": {"q1": 430, "independent": 116},
    "ETTm2": {"q1": 485, "independent": 106},
    "Weather": {"q1": 430, "independent": 116},
}


def record(
    rows: list[dict[str, Any]],
    category: str,
    name: str,
    passed: bool,
    value: float | int | str,
    threshold: float | int | str,
) -> None:
    rows.append(
        {
            "category": category,
            "case": name,
            "value": value,
            "threshold": threshold,
            "pass": bool(passed),
        }
    )


def copy_common_readout(
    source: SIFFCouplingFieldReadout | PCSDCouplingFieldReadout,
    target: SIFFCouplingFieldReadout | PCSDCouplingFieldReadout,
) -> None:
    with torch.no_grad():
        target.identity_synthesis.copy_(source.identity_synthesis)
        target.nonlinear_synthesis.copy_(source.nonlinear_synthesis)
        target.temporal_bias.copy_(source.temporal_bias)
        target.history_projection.load_state_dict(
            source.history_projection.state_dict()
        )
        target.policy_hidden.load_state_dict(source.policy_hidden.state_dict())
        target.policy_output.load_state_dict(source.policy_output.state_dict())


def architecture_cases(rows: list[dict[str, Any]], gates: dict[str, Any]) -> None:
    torch.manual_seed(2021)
    ordered = SIFFCouplingFieldReadout(
        readout_dim=12,
        mode_rank=8,
        scale_components=2,
        scale_basis_mode="ordered",
    ).double()
    basis = ordered.scale_basis.double()
    constant_gap = float((basis[:, 0] - 1.0).abs().max())
    linear_mean = float(basis[:, 1].mean().abs())
    linear_rms_gap = float((basis[:, 1].square().mean().sqrt() - 1.0).abs())
    record(
        rows,
        "SIFF_basis",
        "constant_component",
        constant_gap <= gates["scale_basis_constant_gap_max"],
        constant_gap,
        gates["scale_basis_constant_gap_max"],
    )
    record(
        rows,
        "SIFF_basis",
        "linear_zero_mean",
        linear_mean <= gates["scale_basis_linear_mean_max"],
        linear_mean,
        gates["scale_basis_linear_mean_max"],
    )
    record(
        rows,
        "SIFF_basis",
        "linear_unit_rms",
        linear_rms_gap <= gates["scale_basis_linear_rms_gap_max"],
        linear_rms_gap,
        gates["scale_basis_linear_rms_gap_max"],
    )

    hidden = torch.randn(2, 3, 12, dtype=torch.float64)
    arms = ordered.arm_forecasts(hidden)
    output = ordered(hidden, 192)
    shape_pass = tuple(arms.shape) == (2, 3, 5, 720) and tuple(
        output.shape
    ) == (2, 192, 3)
    record(rows, "SIFF_shape", "forward_contract", shape_pass, str(tuple(arms.shape)), "[2,3,5,720]")
    full = ordered(hidden, 720)
    prefix_gap = float((output - full[:, :192]).abs().max())
    record(rows, "SIFF_shape", "projective_prefix", prefix_gap == 0.0, prefix_gap, 0.0)

    q1 = SIFFCouplingFieldReadout(
        readout_dim=12,
        mode_rank=8,
        scale_components=1,
        scale_basis_mode="ordered",
    ).double()
    pcsd = PCSDCouplingFieldReadout(
        readout_dim=12,
        mode_rank=8,
    ).double()
    with torch.no_grad():
        q1.mode_weight[0].copy_(pcsd.mode_weight)
        q1.mode_bias[0].copy_(pcsd.mode_bias)
    copy_common_readout(pcsd, q1)
    containment_gap = float(
        (pcsd.arm_forecasts(hidden) - q1.arm_forecasts(hidden)).abs().max()
    )
    record(
        rows,
        "SIFF_containment",
        "q1_equals_pcsd",
        containment_gap <= gates["siff_containment_gap_max"],
        containment_gap,
        gates["siff_containment_gap_max"],
    )

    constant = SIFFCouplingFieldReadout(
        readout_dim=12,
        mode_rank=8,
        scale_components=2,
        scale_basis_mode="constant",
    ).double()
    collapsed = PCSDCouplingFieldReadout(
        readout_dim=12,
        mode_rank=8,
    ).double()
    with torch.no_grad():
        collapsed.mode_weight.copy_(constant.mode_weight.sum(dim=0))
        collapsed.mode_bias.copy_(constant.mode_bias.sum(dim=0))
    copy_common_readout(constant, collapsed)
    collapse_gap = float(
        (constant.arm_forecasts(hidden) - collapsed.arm_forecasts(hidden))
        .abs()
        .max()
    )
    record(
        rows,
        "SIFF_controls",
        "constant_collapses_to_single_field",
        collapse_gap <= gates["siff_constant_control_gap_max"],
        collapse_gap,
        gates["siff_constant_control_gap_max"],
    )

    permuted = SIFFCouplingFieldReadout(
        readout_dim=12,
        mode_rank=8,
        scale_components=2,
        scale_basis_mode="permuted",
    ).double()
    permutation_gap = float(
        (
            torch.sort(permuted.scale_basis[:, 1]).values
            - torch.sort(ordered.scale_basis[:, 1]).values
        )
        .abs()
        .max()
    )
    order_change = float(
        (permuted.scale_basis[:, 1] - ordered.scale_basis[:, 1]).abs().max()
    )
    record(rows, "SIFF_controls", "permuted_preserves_values", permutation_gap == 0.0, permutation_gap, 0.0)
    record(rows, "SIFF_controls", "permuted_changes_order", order_change > 0.0, order_change, ">0")

    independent = SIFFCouplingFieldReadout(
        readout_dim=12,
        mode_rank=8,
        scale_components=5,
        scale_basis_mode="independent",
    ).double()
    identity_gap = float(
        (independent.scale_basis - torch.eye(5, dtype=torch.float64))
        .abs()
        .max()
    )
    record(rows, "SIFF_controls", "independent_one_hot", identity_gap == 0.0, identity_gap, 0.0)

    coefficient_weight = torch.randn(8, 12, dtype=torch.float64)
    coefficient_bias = torch.randn(8, dtype=torch.float64)
    temporal_basis = torch.randn(720, 8, dtype=torch.float64)
    temporal_bias = torch.randn(720, dtype=torch.float64)
    witness = SIFFCouplingFieldReadout(
        readout_dim=12,
        mode_rank=8,
        scale_components=2,
        scale_basis_mode="ordered",
        policy_mode="fixed",
        fixed_scale=720,
    ).double()
    witness.map_a6_parameters_(
        coefficient_weight,
        coefficient_bias,
        temporal_basis,
        temporal_bias,
    )
    coefficients = torch.einsum("bcr,kr->bck", hidden, coefficient_weight)
    coefficients = coefficients + coefficient_bias.view(1, 1, -1)
    expected = torch.einsum("tk,bck->bct", temporal_basis, coefficients)
    expected = (expected + temporal_bias.view(1, 1, -1)).permute(0, 2, 1)
    a6_gap = float((witness(hidden, 720) - expected).abs().max())
    record(
        rows,
        "SIFF_containment",
        "a6_exact_subspace",
        a6_gap <= gates["siff_containment_gap_max"],
        a6_gap,
        gates["siff_containment_gap_max"],
    )


def parameter_cases(rows: list[dict[str, Any]], gates: dict[str, Any]) -> None:
    profiles = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))[
        "dataset_profiles"
    ]
    for dataset, profile in profiles.items():
        readout_dim = int(profile["patch_num"]) * int(profile["d_model"])
        target = siff_parameter_count(
            readout_dim,
            mode_rank=256,
            scale_components=2,
        )
        for control, components in (("q1", 1), ("independent", 5)):
            actual = siff_parameter_count(
                readout_dim,
                mode_rank=MATCHED_RANKS[dataset][control],
                scale_components=components,
            )
            gap = abs(actual - target) / target
            record(
                rows,
                "parameter_match",
                f"{dataset}_{control}",
                gap <= gates["parameter_match_relative_gap_max"],
                gap,
                gates["parameter_match_relative_gap_max"],
            )


def generalized_kl(allocation: torch.Tensor, reference: torch.Tensor) -> float:
    value = allocation * (
        allocation.clamp_min(1e-30).log()
        - reference.clamp_min(1e-30).log()
    ) - allocation + reference
    return float(value.sum())


def mcca_cases(rows: list[dict[str, Any]], gates: dict[str, Any]) -> None:
    for dtype, threshold in (
        (torch.float64, gates["sinkhorn_float64_gap_max"]),
        (torch.float32, gates["sinkhorn_float32_gap_max"]),
    ):
        torch.manual_seed(2021)
        logits = torch.randn(2, 3, 720, 5, dtype=dtype)
        capability = torch.softmax(logits, dim=-1)
        measure = prefix_measure(720, device=logits.device, dtype=dtype)
        credit, allocation, row, column = competitive_assignment(
            capability,
            measure,
            progress=1.0,
        )
        row_gap = float((allocation.sum(dim=-1) - row).abs().max())
        column_gap = float((allocation.sum(dim=0) - column).abs().max())
        max_gap = max(row_gap, column_gap)
        record(
            rows,
            "MCCA_solver",
            f"marginals_{str(dtype).split('.')[-1]}",
            max_gap <= threshold,
            max_gap,
            threshold,
        )
        credit_sum_gap = float((credit.sum(dim=-1) - 1.0).abs().max())
        record(
            rows,
            "MCCA_solver",
            f"credit_simplex_{str(dtype).split('.')[-1]}",
            credit_sum_gap <= threshold,
            credit_sum_gap,
            threshold,
        )

    torch.manual_seed(2022)
    capability = torch.softmax(
        torch.randn(2, 2, 96, 5, dtype=torch.float64), dim=-1
    )
    measure = prefix_measure(96, device=capability.device, dtype=capability.dtype)
    for progress in (0.0, 0.125, 0.25, 1.0):
        credit, allocation, row, column = competitive_assignment(
            capability,
            measure,
            progress=progress,
        )
        schedule = pcc_schedule(progress)
        pcc_credit = (
            (1.0 - schedule.skill_floor) * capability
            + schedule.skill_floor / capability.shape[-1]
        )
        pcc_allocation = row.unsqueeze(-1) * pcc_credit.reshape(-1, 5)
        mass_gap = float((pcc_allocation.sum(dim=0) - column).abs().max())
        record(
            rows,
            "MCCA_same_mass",
            f"pcc_column_mass_progress_{progress}",
            mass_gap <= gates["pcc_mcca_column_mass_gap_max"],
            mass_gap,
            gates["pcc_mcca_column_mass_gap_max"],
        )
        alpha = min(progress / 0.25, 1.0)
        uniform = torch.full_like(capability, 0.2)
        ramped = (1.0 - alpha) * uniform + alpha * capability
        reference = row.unsqueeze(-1) * ramped.reshape(-1, 5)
        advantage = generalized_kl(pcc_allocation, reference) - generalized_kl(
            allocation, reference
        )
        if progress > 0.0:
            record(
                rows,
                "MCCA_same_mass",
                f"kl_projection_advantage_{progress}",
                advantage >= gates["mcca_kl_advantage_min"],
                advantage,
                gates["mcca_kl_advantage_min"],
            )

    torch.manual_seed(2023)
    target = torch.randn(2, 32, 3)
    arm_forecasts = torch.randn(2, 3, 32, 5, requires_grad=True)
    policy_logits = torch.randn(2, 3, 32, 5, requires_grad=True)
    policy = torch.softmax(policy_logits, dim=-1)
    fused = (arm_forecasts * policy).sum(dim=-1).permute(0, 2, 1)
    result = measure_constrained_competitive_loss(
        fused,
        arm_forecasts,
        policy,
        target,
        mode="mcca_transport_full",
        progress=1.0,
    )
    result.total_loss.backward()
    arm_norm = float(arm_forecasts.grad.norm())
    policy_norm = float(policy_logits.grad.norm())
    finite = math.isfinite(float(result.total_loss))
    record(rows, "MCCA_gradient", "finite_loss", finite, float(result.total_loss), "finite")
    record(rows, "MCCA_gradient", "arm_gradient", arm_norm > gates["gradient_norm_min"], arm_norm, gates["gradient_norm_min"])
    record(rows, "MCCA_gradient", "policy_gradient", policy_norm > gates["gradient_norm_min"], policy_norm, gates["gradient_norm_min"])
    scope_mass = [
        float(result.diagnostics[f"mcca_scope_s{index}_mass"])
        for index in range(5)
    ]
    record(
        rows,
        "MCCA_gradient",
        "minimum_scope_mass",
        min(scope_mass) >= gates["minimum_final_scope_mass"],
        min(scope_mass),
        gates["minimum_final_scope_mass"],
    )


def main() -> None:
    step6 = json.loads(STEP6_PATH.read_text(encoding="utf-8"))
    gates = step6["gates"]
    rows: list[dict[str, Any]] = []
    architecture_cases(rows, gates)
    parameter_cases(rows, gates)
    mcca_cases(rows, gates)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "cases.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    failed = [row for row in rows if not row["pass"]]
    result = {
        "candidate": "SC1-SIFF-v1/SC2-MCCA-v1",
        "current_step": "Step7A local implementation gate",
        "cases_total": len(rows),
        "cases_passed": len(rows) - len(failed),
        "all_pass": not failed,
        "failed_cases": [row["case"] for row in failed],
        "test_used": False,
        "remote_used": False,
        "decision": (
            "step7a_pass_step7b_prelaunch_authorized"
            if not failed
            else "step7a_fail_return_step6"
        ),
    }
    (OUTPUT_DIR / "local_gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    if failed:
        raise RuntimeError(f"Step7A local gate failed: {failed}")


if __name__ == "__main__":
    main()
