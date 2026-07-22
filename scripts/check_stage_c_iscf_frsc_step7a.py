#!/usr/bin/env python3
"""Run local implementation contracts for SC-ISCF-FRSC-v0."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "baselines" / "timealign_official"
sys.path.insert(0, str(OFFICIAL))

from layers.SIFF import SIFFCouplingFieldReadout  # noqa: E402
from layers.SPS import FullRankScopeConditioningReadout  # noqa: E402
from models import TimeAlign  # noqa: E402
import train_repo as training_adapter  # noqa: E402


def tensor_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for parameter in module.parameters():
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_parent(seed: int) -> SIFFCouplingFieldReadout:
    torch.manual_seed(seed)
    return SIFFCouplingFieldReadout(
        readout_dim=32,
        series_length=720,
        coordinate_dim=4,
        mode_rank=16,
        scale_components=5,
        scale_basis_mode="independent",
        policy_mode="direct",
    )


def make_frsc(
    seed: int,
    projection_mode: str,
    strength: float,
    partition: str = "canonical",
) -> FullRankScopeConditioningReadout:
    torch.manual_seed(seed)
    return FullRankScopeConditioningReadout(
        readout_dim=32,
        series_length=720,
        coordinate_dim=4,
        mode_rank=16,
        projection_mode=projection_mode,
        conditioning_strength=strength,
        policy_mode="direct",
        partition=partition,
    )


def model_config() -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        encoder_mode="timealign-token-mlp",
        readout_mode="iscf-full-rank-scope-conditioning",
        e_layers=1,
        patch_num=2,
        d_model=4,
        d_ff=8,
        dropout=0.0,
        pos=1,
        layer_norm=1,
        enc_in=2,
        basis_rank=8,
        pcsd_coordinate_dim=4,
        pcsd_mode_rank=4,
        pcsd_policy_history_dim=4,
        pcsd_policy_hidden_dim=8,
        pcsd_policy_mode="direct",
        pcsd_fixed_scale=720,
        pcsd_partition="canonical",
        pcsd_partition_seed=15101,
        pcsd_group_chunk_size=64,
        pcsd_target_chunk_size=128,
        sps_projection_mode="scope",
        frsc_conditioning_strength=0.55,
    )


def cli_contract() -> dict[str, object]:
    original = sys.argv
    sys.argv = [
        "train_repo.py",
        "--dataset-root",
        "/tmp/dataset",
        "--dataset",
        "ETTh1",
        "--mode",
        "unified",
        "--pred-len",
        "720",
        "--target-horizons",
        "720",
        "--validation-horizons",
        "96,192,336,720",
        "--evaluation-horizons",
        "96,192,336,720",
        "--run-name",
        "iscf-frsc-smoke",
        "--output-dir",
        "/tmp/iscf-frsc-smoke",
        "--readout-mode",
        "iscf-full-rank-scope-conditioning",
        "--pcsd-mode-rank",
        "109",
        "--sps-projection-mode",
        "scope",
        "--frsc-conditioning-strength",
        "0.55",
        "--pcsd-policy-mode",
        "direct",
        "--pcc-objective-mode",
        "equal_skill",
        "--checkpoint-policy",
        "best-val",
        "--final-evaluation-split",
        "val",
    ]
    try:
        args = training_adapter.parse_args()
    finally:
        sys.argv = original
    return {
        "readout_mode": args.readout_mode,
        "projection_mode": args.sps_projection_mode,
        "conditioning_strength": args.frsc_conditioning_strength,
        "mode_rank": args.pcsd_mode_rank,
        "validation_horizons": args.validation_horizons,
        "final_split": args.final_evaluation_split,
    }


def main() -> None:
    seed = 20260722
    parent = make_parent(seed)
    identity = make_frsc(seed, "scope", 0.0)
    candidate = make_frsc(seed, "scope", 0.55)
    global_same = make_frsc(seed, "global", 0.55)
    global_best = make_frsc(seed, "global", 0.45)
    random_scope = make_frsc(seed, "scope", 0.55, partition="random")
    modules = {
        "parent": parent,
        "identity": identity,
        "candidate": candidate,
        "global_same": global_same,
        "global_best": global_best,
        "random": random_scope,
    }
    hashes = {name: tensor_hash(module) for name, module in modules.items()}
    if len(set(hashes.values())) != 1:
        raise AssertionError(f"paired parameter initialization differs: {hashes}")

    hidden = torch.randn(2, 3, 32)
    with torch.no_grad():
        outputs = {name: module(hidden) for name, module in modules.items()}
        prefix = candidate(hidden, target_prefix=192)
    if outputs["parent"].shape != (2, 720, 3) or prefix.shape != (2, 192, 3):
        raise AssertionError("FRSC output shape contract failed")
    if not all(torch.isfinite(output).all() for output in outputs.values()):
        raise AssertionError("FRSC output finite contract failed")
    identity_gap = float((outputs["identity"] - outputs["parent"]).abs().max())
    prefix_gap = float((outputs["candidate"][:, :192] - prefix).abs().max())
    if identity_gap > 1e-6 or prefix_gap > 1e-6:
        raise AssertionError(
            f"FRSC identity/prefix contract failed: {identity_gap}, {prefix_gap}"
        )

    control_gaps = {
        name: float((outputs["candidate"] - outputs[name]).abs().max())
        for name in ("parent", "global_same", "global_best", "random")
    }
    if any(gap <= 1e-7 for gap in control_gaps.values()):
        raise AssertionError(f"FRSC controls are not distinct: {control_gaps}")

    expected_minimum = 0.45
    if abs(candidate.minimum_operator_eigenvalue - expected_minimum) > 1e-12:
        raise AssertionError("FRSC minimum eigenvalue contract failed")
    if candidate.minimum_operator_eigenvalue <= 0.0:
        raise AssertionError("FRSC operator is not full rank")

    candidate.train()
    train_hidden = torch.randn(2, 3, 32, requires_grad=True)
    target = torch.randn(2, 720, 3)
    loss = (candidate(train_hidden) - target).square().mean()
    loss.backward()
    gradient_norms = [
        float(candidate.mode_weight.grad[index].norm())
        for index in range(len(candidate.scales))
    ]
    if not all(math.isfinite(value) and value > 0.0 for value in gradient_norms):
        raise AssertionError(f"FRSC gradients inactive: {gradient_norms}")

    torch.manual_seed(seed)
    model = TimeAlign.Model(model_config()).float().eval()
    x = torch.randn(1, 720, 2)
    y = torch.zeros_like(x)
    with torch.no_grad():
        model_full = model(x, y, is_training=False, target_prefix=720)[0]
        model_short = model(x, y, is_training=False, target_prefix=96)[0]
    model_prefix_gap = float((model_full[:, :96] - model_short).abs().max())
    if model_full.shape != (1, 720, 2) or model_short.shape != (1, 96, 2):
        raise AssertionError("production FRSC shape contract failed")
    if not torch.isfinite(model_full).all() or model_prefix_gap > 1e-6:
        raise AssertionError("production FRSC finite/prefix contract failed")

    cli = cli_contract()
    expected_cli = {
        "readout_mode": "iscf-full-rank-scope-conditioning",
        "projection_mode": "scope",
        "conditioning_strength": 0.55,
        "mode_rank": 109,
        "validation_horizons": [96, 192, 336, 720],
        "final_split": "val",
    }
    if cli != expected_cli:
        raise AssertionError(f"production FRSC CLI contract failed: {cli}")

    result = {
        "decision": "iscf_frsc_step7a_contract_pass",
        "parameter_hashes": hashes,
        "parameter_count": sum(
            parameter.numel() for parameter in candidate.parameters()
        ),
        "conditioning_strength": candidate.conditioning_strength,
        "minimum_operator_eigenvalue": candidate.minimum_operator_eigenvalue,
        "identity_parent_max_abs_gap": identity_gap,
        "prefix_max_abs_gap": prefix_gap,
        "control_max_abs_gaps": control_gaps,
        "scope_gradient_norms": gradient_norms,
        "production_model_full_shape": list(model_full.shape),
        "production_model_prefix_gap": model_prefix_gap,
        "production_cli": cli,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
