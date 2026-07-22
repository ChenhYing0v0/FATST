#!/usr/bin/env python3
"""Check the ISCF-BSCA-v1 objective and frozen launch contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines" / "timealign_official"))

from layers.PCC import projective_coupling_credit_loss  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage_c_iscf_bsca_v1.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.manual_seed(20260722)
    shape = (2, 12, 3)
    arms = torch.randn(2, 3, 12, 5, requires_grad=True)
    target = torch.randn(*shape)
    logits = torch.randn(2, 3, 12, 5, requires_grad=True)
    policy = logits.softmax(dim=-1)
    fused = (policy * arms).sum(dim=-1).permute(0, 2, 1)
    result = projective_coupling_credit_loss(
        fused, arms, policy, target,
        mode="equal_uniform_scope_anchor", progress=1.0,
    )
    assert torch.allclose(result.route_credit, torch.full_like(policy, 0.2))
    assert torch.allclose(result.skill_credit, torch.full_like(policy, 0.2))
    assert result.schedule.route_weight == 0.1
    result.route_loss.backward(retain_graph=True)
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert arms.grad is None
    expected = result.fused_loss + result.skill_loss + 0.1 * result.route_loss
    assert torch.allclose(result.total_loss, expected)

    other_arms = torch.randn_like(arms, requires_grad=True)
    other_target = torch.randn_like(target)
    other_fused = (policy.detach() * other_arms).sum(dim=-1).permute(0, 2, 1)
    other = projective_coupling_credit_loss(
        other_fused, other_arms, policy.detach(), other_target,
        mode="equal_uniform_scope_anchor", progress=1.0,
    )
    assert torch.allclose(result.route_credit, other.route_credit)
    assert torch.allclose(result.route_loss.detach(), other.route_loss.detach())

    assert config["matrix"]["new_training_runs"] == 5
    assert config["matrix"]["new_formal_test_runs"] == 5
    assert config["authorization"]["formal_test_access_count_for_version"] == 1
    assert not config["authorization"]["confirmation_seeds_authorized"]
    dry = subprocess.run(
        ["bash", "scripts/remote/run_stage_c_iscf_bsca_v1.sh"],
        cwd=ROOT,
        env={"PATH": str(Path("/usr/bin")) + ":/bin", "DRY_RUN": "1"},
        text=True,
        capture_output=True,
        check=True,
    )
    assert "jobs=5" in dry.stdout
    print(json.dumps({
        "candidate": "ISCF-BSCA-v1",
        "objective_contract": "pass",
        "uniform_credit": 0.2,
        "route_weight_final": 0.1,
        "five_run_dry_run": "pass",
        "formal_test_guard": "5_of_5_required",
        "overall_pass": True,
    }, indent=2))


if __name__ == "__main__":
    main()
