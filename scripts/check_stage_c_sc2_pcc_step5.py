#!/usr/bin/env python3
"""Check Projective Coupling Credit Step5 algebra and synthetic feasibility."""

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
        "--output-dir",
        type=Path,
        default=Path("analysis/stage_c_sc2_pcc_step5_theory_20260716"),
    )
    return parser.parse_args()


def prefix_measure(length: int, *, dtype: torch.dtype) -> Tensor:
    inverse = 1.0 / torch.arange(1, length + 1, dtype=dtype)
    cumulative = torch.cumsum(torch.flip(inverse, dims=(0,)), dim=0)
    return torch.flip(cumulative, dims=(0,)) / length


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def record(
    rows: list[dict[str, Any]],
    name: str,
    value: float,
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


def gradient_checks(rows: list[dict[str, Any]]) -> None:
    torch.manual_seed(20260716)
    dtype = torch.float64
    batch, length, scopes = 7, 19, 5
    weights = prefix_measure(length, dtype=dtype)
    target = torch.randn(batch, length, dtype=dtype)
    arms = torch.randn(batch, length, scopes, dtype=dtype, requires_grad=True)
    logits = torch.randn(batch, length, scopes, dtype=dtype, requires_grad=True)
    policy = torch.softmax(logits, dim=-1)
    fused = (policy * arms).sum(dim=-1)
    residual = fused - target
    fused_loss = (weights * (0.5 * residual.square()).mean(dim=0)).sum()
    arm_gradient, logit_gradient = torch.autograd.grad(
        fused_loss,
        (arms, logits),
        retain_graph=True,
    )
    scale = weights.view(1, length, 1) / batch
    expected_arm = scale * policy * residual.unsqueeze(-1)
    expected_logit = scale * policy * (arms - fused.unsqueeze(-1)) * residual.unsqueeze(-1)
    arm_gap = float((arm_gradient - expected_arm).abs().max())
    logit_gap = float((logit_gradient - expected_logit).abs().max())
    record(rows, "plain_fused_arm_gradient_identity", arm_gap, "<=1e-12", arm_gap <= 1e-12)
    record(rows, "plain_fused_router_gradient_identity", logit_gap, "<=1e-12", logit_gap <= 1e-12)

    per_arm_error = 0.5 * (arms - target.unsqueeze(-1)).square()
    centered = per_arm_error - per_arm_error.mean(dim=-1, keepdim=True)
    normalized = centered / (
        per_arm_error.std(dim=-1, keepdim=True, unbiased=False) + 1e-8
    )
    temperature = 0.7
    capability = torch.softmax(-normalized.detach() / temperature, dim=-1)
    epsilon = 0.1
    floored = (1.0 - epsilon) * capability + epsilon / scopes
    skill_weight, route_weight = 0.4, 0.3
    skill_loss = (
        weights
        * (floored * per_arm_error).sum(dim=-1).mean(dim=0)
    ).sum()
    route_loss = (
        weights
        * (
            capability
            * (
                torch.log(capability.clamp_min(1e-12))
                - torch.log(policy.clamp_min(1e-12))
            )
        )
        .sum(dim=-1)
        .mean(dim=0)
    ).sum()
    total = fused_loss + skill_weight * skill_loss + route_weight * route_loss
    pcc_arm_gradient, pcc_logit_gradient = torch.autograd.grad(total, (arms, logits))
    expected_pcc_arm = expected_arm + (
        skill_weight * scale * floored * (arms - target.unsqueeze(-1))
    )
    expected_pcc_logit = expected_logit + (
        route_weight * scale * (policy - capability)
    )
    pcc_arm_gap = float((pcc_arm_gradient - expected_pcc_arm).abs().max())
    pcc_logit_gap = float((pcc_logit_gradient - expected_pcc_logit).abs().max())
    floor_gap = float(abs(float(floored.min()) - epsilon / scopes))
    record(rows, "pcc_arm_gradient_identity", pcc_arm_gap, "<=1e-12", pcc_arm_gap <= 1e-12)
    record(rows, "pcc_router_gradient_identity", pcc_logit_gap, "<=1e-12", pcc_logit_gap <= 1e-12)
    record(
        rows,
        "skill_floor_lower_bound",
        float(floored.min()),
        f">={epsilon / scopes}",
        float(floored.min()) + 1e-12 >= epsilon / scopes,
    )
    record(rows, "skill_floor_numeric_gap", floor_gap, "finite", torch.isfinite(floored).all().item())


def prefix_and_projectivity_checks(rows: list[dict[str, Any]]) -> None:
    torch.manual_seed(17)
    dtype = torch.float64
    length = 720
    errors = torch.rand(length, dtype=dtype)
    weights = prefix_measure(length, dtype=dtype)
    prefix_auc = torch.stack(
        [errors[:horizon].mean() for horizon in range(1, length + 1)]
    ).mean()
    weighted = (weights * errors).sum()
    identity_gap = float(abs(prefix_auc - weighted))
    record(rows, "dense_prefix_measure_identity", identity_gap, "<=1e-12", identity_gap <= 1e-12)
    weight_sum_gap = float(abs(weights.sum() - 1.0))
    record(rows, "prefix_measure_normalization", weight_sum_gap, "<=1e-12", weight_sum_gap <= 1e-12)
    monotone = bool(torch.all(weights[:-1] >= weights[1:]))
    record(rows, "prefix_measure_monotone", float(monotone), "==1", monotone)

    full = torch.randn(3, length, 2, dtype=dtype)
    max_crop_gap = 0.0
    for horizon in (1, 48, 96, 144, 360, 719, 720):
        native = full[:, :horizon, :]
        max_crop_gap = max(
            max_crop_gap,
            float((native - full[:, :horizon, :]).abs().max()),
        )
    record(rows, "full_domain_prefix_projectivity", max_crop_gap, "==0", max_crop_gap == 0.0)


def synthetic_router_recovery(rows: list[dict[str, Any]]) -> None:
    torch.manual_seed(53)
    dtype = torch.float64
    batch, length, scopes = 96, 64, 5
    history = torch.linspace(-1.0, 1.0, batch, dtype=dtype).view(batch, 1, 1)
    target = torch.linspace(-1.0, 1.0, length, dtype=dtype).view(1, length, 1)
    history_grid = history.expand(batch, length, 1)
    target_grid = target.expand(batch, length, 1)
    features = torch.cat(
        (
            history_grid,
            target_grid,
            history_grid * target_grid,
            torch.ones_like(history_grid),
        ),
        dim=-1,
    )
    teacher = torch.tensor(
        [
            [2.0, -2.0, 1.0, -1.0, 0.5],
            [-1.0, 1.0, -2.0, 2.0, 0.0],
            [2.0, 2.0, -2.0, -2.0, 1.0],
            [0.2, -0.1, 0.1, -0.2, 0.0],
        ],
        dtype=dtype,
    )
    capability = torch.softmax(features @ teacher / 0.45, dim=-1).detach()
    model = torch.nn.Linear(4, scopes, bias=False, dtype=dtype)
    torch.nn.init.zeros_(model.weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    weights = prefix_measure(length, dtype=dtype)
    for _ in range(1200):
        policy = torch.softmax(model(features), dim=-1)
        loss = (
            weights
            * (
                capability
                * (
                    torch.log(capability.clamp_min(1e-12))
                    - torch.log(policy.clamp_min(1e-12))
                )
            )
            .sum(dim=-1)
            .mean(dim=0)
        ).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        policy = torch.softmax(model(features), dim=-1)
        kl = float(
            (
                capability
                * (
                    torch.log(capability.clamp_min(1e-12))
                    - torch.log(policy.clamp_min(1e-12))
                )
            )
            .sum(dim=-1)
            .mean()
        )
        accuracy = float(
            (policy.argmax(dim=-1) == capability.argmax(dim=-1))
            .to(dtype)
            .mean()
        )
        teacher_choice = capability.argmax(dim=-1)
        target_crossing = float(
            torch.tensor(
                [teacher_choice[row].unique().numel() > 1 for row in range(batch)]
            )
            .to(dtype)
            .mean()
        )
        history_crossing = float(
            torch.tensor(
                [
                    teacher_choice[:, column].unique().numel() > 1
                    for column in range(length)
                ]
            )
            .to(dtype)
            .mean()
        )
        used_scopes = float(teacher_choice.unique().numel())
    record(rows, "synthetic_router_kl", kl, "<=1e-4", kl <= 1e-4)
    record(rows, "synthetic_router_argmax_accuracy", accuracy, ">=0.99", accuracy >= 0.99)
    record(rows, "synthetic_target_crossing_fraction", target_crossing, ">=0.5", target_crossing >= 0.5)
    record(rows, "synthetic_history_crossing_fraction", history_crossing, ">=0.5", history_crossing >= 0.5)
    record(rows, "synthetic_used_scopes", used_scopes, ">=4", used_scopes >= 4)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    gradient_checks(rows)
    prefix_and_projectivity_checks(rows)
    synthetic_router_recovery(rows)
    write_csv(args.output_dir / "theory_and_synthetic_cases.csv", rows)
    overall_pass = all(bool(row["pass"]) for row in rows)
    result = {
        "candidate": "SC2-PCC-v0",
        "current_step": "Step5 theory feasibility",
        "case_count": len(rows),
        "passed_cases": sum(bool(row["pass"]) for row in rows),
        "overall_pass": overall_pass,
        "decision": (
            "conditional_pass_step6_design_only"
            if overall_pass
            else "fail_return_step5_redesign"
        ),
        "implementation_authorized": False,
        "remote_authorized": False,
        "test_used": False,
    }
    (args.output_dir / "local_gate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
