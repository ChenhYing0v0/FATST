#!/usr/bin/env python3
"""Check the exact ISCF-SCC objective and shuffled binding control."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from layers.PCC import (  # noqa: E402
    SCC_SHUFFLE_SEED_OFFSET,
    projective_coupling_credit_loss,
    scope_coalition_credit,
)


def build_inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(20260722)
    arms = torch.randn(2, 3, 8, 5, requires_grad=True)
    logits = torch.randn(2, 3, 8, 5, requires_grad=True)
    policy = torch.softmax(logits, dim=-1)
    fused = (arms * policy).sum(dim=-1).permute(0, 2, 1)
    target = torch.randn(2, 8, 3)
    return fused, arms, policy, target, logits


def check_exact_credit_and_gradients() -> None:
    fused, arms, policy, target, logits = build_inputs()
    expected_credit, _signed_gain = scope_coalition_credit(
        fused,
        arms,
        policy,
        target,
    )
    result = projective_coupling_credit_loss(
        fused,
        arms,
        policy,
        target,
        mode="scope_coalition_credit",
        progress=1.0,
    )
    if not torch.allclose(result.route_credit, expected_credit, atol=1e-7):
        raise AssertionError("SCC route credit does not match exact LOO credit")
    if result.route_credit.requires_grad:
        raise AssertionError("SCC credit must be fully stop-gradient")
    if not torch.allclose(
        result.total_loss,
        result.fused_loss + result.weighted_route_loss,
        atol=1e-7,
    ):
        raise AssertionError("SCC total loss has an unexpected component")
    if float(result.skill_loss) != 0.0:
        raise AssertionError("SCC must not retain individual arm loss")

    result.route_loss.backward(retain_graph=True)
    if arms.grad is not None and float(arms.grad.abs().max()) != 0.0:
        raise AssertionError("SCC route loss leaked gradients into arms")
    if logits.grad is None or float(logits.grad.norm()) <= 0.0:
        raise AssertionError("SCC route loss did not calibrate policy")


def check_uniform_fallback() -> None:
    arms = torch.ones(1, 2, 4, 5)
    policy = torch.softmax(torch.randn(1, 2, 4, 5), dim=-1)
    fused = torch.ones(1, 4, 2)
    target = torch.ones(1, 4, 2)
    credit, signed_gain = scope_coalition_credit(
        fused,
        arms,
        policy,
        target,
    )
    if float(signed_gain.abs().max()) != 0.0:
        raise AssertionError("fallback construction should have zero gain")
    expected = torch.full_like(credit, 0.2)
    if not torch.allclose(credit, expected, atol=1e-7):
        raise AssertionError("all-nonpositive SCC fallback is not uniform")


def check_shuffled_control() -> None:
    fused, arms, policy, target, _logits = build_inputs()
    generator_a = torch.Generator(device="cpu")
    generator_b = torch.Generator(device="cpu")
    seed = 2021 + SCC_SHUFFLE_SEED_OFFSET
    generator_a.manual_seed(seed)
    generator_b.manual_seed(seed)
    result_a = projective_coupling_credit_loss(
        fused,
        arms,
        policy,
        target,
        mode="scope_coalition_credit_shuffled",
        progress=1.0,
        coalition_shuffle_generator=generator_a,
    )
    result_b = projective_coupling_credit_loss(
        fused,
        arms,
        policy,
        target,
        mode="scope_coalition_credit_shuffled",
        progress=1.0,
        coalition_shuffle_generator=generator_b,
    )
    original, _signed_gain = scope_coalition_credit(
        fused,
        arms,
        policy,
        target,
    )
    if not torch.equal(result_a.route_credit, result_b.route_credit):
        raise AssertionError("dedicated SCC shuffle is not reproducible")
    if not torch.allclose(
        result_a.route_credit.sort(dim=-1).values,
        original.sort(dim=-1).values,
        atol=1e-7,
    ):
        raise AssertionError("SCC shuffle did not preserve credit marginals")
    if torch.equal(result_a.route_credit, original):
        raise AssertionError("SCC shuffle failed to break scope binding")

    torch.manual_seed(1234)
    expected_global_draw = torch.rand(4)
    torch.manual_seed(1234)
    generator_c = torch.Generator(device="cpu")
    generator_c.manual_seed(seed)
    projective_coupling_credit_loss(
        fused,
        arms,
        policy,
        target,
        mode="scope_coalition_credit_shuffled",
        progress=1.0,
        coalition_shuffle_generator=generator_c,
    )
    observed_global_draw = torch.rand(4)
    if not torch.equal(expected_global_draw, observed_global_draw):
        raise AssertionError("dedicated SCC shuffle consumed global RNG")


def main() -> None:
    check_exact_credit_and_gradients()
    check_uniform_fallback()
    check_shuffled_control()
    print("iscf_scc_step7a=pass")


if __name__ == "__main__":
    main()
