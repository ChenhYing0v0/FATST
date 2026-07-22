#!/usr/bin/env python3
"""Check the frozen PSA-D1 control protocol and EQUAL objective contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from layers.PCC import projective_coupling_credit_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_iscf_psa_d1.json"),
    )
    return parser.parse_args()


def check_config(config: dict[str, Any]) -> None:
    if config["diagnostic_id"] != "SC-ISCF-PSA-D1":
        raise AssertionError("unexpected diagnostic id")
    if config["datasets"] != [
        "Weather",
        "ETTm1",
        "ETTh1",
        "ETTh2",
        "ETTm2",
    ]:
        raise AssertionError("D1 dataset order changed")
    if config["seeds"] != [2021]:
        raise AssertionError("D1 must remain a one-seed control")
    if len(config["arms"]) != 1:
        raise AssertionError("D1 must train exactly one arm")
    arm = config["arms"][0]
    expected = {
        "id": "iscf_equal_contemporaneous",
        "readout_mode": "siff-independent-scope-control",
        "projection_mode": "identity",
        "partition": "canonical",
        "objective_mode": "equal_skill",
        "train": True,
    }
    for key, value in expected.items():
        if arm[key] != value:
            raise AssertionError(f"unexpected D1 arm contract: {key}")
    if len(config["launch_order"]) != 5:
        raise AssertionError("D1 must launch exactly five runs")
    if config["matrix"]["expected_new_runs"] != 5:
        raise AssertionError("D1 matrix size changed")
    authorization = config["authorization"]
    if not authorization["step7a_implementation_authorized"]:
        raise AssertionError("D1 Step7A is not authorized")
    if not authorization["remote_training_authorized"]:
        raise AssertionError("D1 five-run training is not authorized")
    if authorization["formal_test_access_authorized"]:
        raise AssertionError("D1 must not authorize formal test")
    if authorization["confirmation_seeds_authorized"]:
        raise AssertionError("D1 must not authorize confirmation seeds")
    if authorization["method_promotion_authorized"]:
        raise AssertionError("D1 is a control, not a method")


def check_source_semantics() -> None:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "020eea3",
            "--",
            "baselines/timealign_official/layers/PCC.py",
            "baselines/timealign_official/train_repo.py",
            "scripts/evaluate_stage_c_pcsd_cf_checkpoint.py",
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "training/evaluation semantics changed since RSCC launch"
        )


def check_equal_objective() -> None:
    torch.manual_seed(20260722)
    arms = torch.randn(2, 3, 8, 5, requires_grad=True)
    logits = torch.randn(2, 3, 8, 5, requires_grad=True)
    policy = torch.softmax(logits, dim=-1)
    fused = (arms * policy).sum(dim=-1).permute(0, 2, 1)
    target = torch.randn(2, 8, 3)
    result = projective_coupling_credit_loss(
        fused,
        arms,
        policy,
        target,
        mode="equal_skill",
        progress=1.0,
    )
    if float(result.route_loss) != 0.0:
        raise AssertionError("EQUAL unexpectedly has a route loss")
    if float(result.weighted_route_loss) != 0.0:
        raise AssertionError("EQUAL unexpectedly weights a route loss")
    expected = result.fused_loss + result.weighted_skill_loss
    if not torch.allclose(result.total_loss, expected, atol=1e-7):
        raise AssertionError("EQUAL total loss decomposition changed")
    result.total_loss.backward()
    if arms.grad is None or not torch.isfinite(arms.grad).all():
        raise AssertionError("EQUAL arm gradients are invalid")
    if logits.grad is None or not torch.isfinite(logits.grad).all():
        raise AssertionError("EQUAL policy gradients are invalid")
    if any(float(arms.grad[..., scope].norm()) <= 0.0 for scope in range(5)):
        raise AssertionError("EQUAL does not reach all five scope arms")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    check_config(config)
    check_source_semantics()
    check_equal_objective()
    print("iscf_psa_d1_step7a=pass jobs=5 route_loss=0 test=false")


if __name__ == "__main__":
    main()
