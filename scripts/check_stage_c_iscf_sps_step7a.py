#!/usr/bin/env python3
"""Local contract checks for SC-ISCF-SPS-v0."""

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
from layers.SPS import ScopeProjectedSynthesisReadout  # noqa: E402
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


def make_sps(
    seed: int,
    projection_mode: str,
    partition: str = "canonical",
) -> ScopeProjectedSynthesisReadout:
    torch.manual_seed(seed)
    return ScopeProjectedSynthesisReadout(
        readout_dim=32,
        series_length=720,
        coordinate_dim=4,
        mode_rank=16,
        projection_mode=projection_mode,
        policy_mode="direct",
        partition=partition,
    )


def model_config() -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        encoder_mode="timealign-token-mlp",
        readout_mode="iscf-scope-projected-synthesis",
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
        "iscf-sps-smoke",
        "--output-dir",
        "/tmp/iscf-sps-smoke",
        "--readout-mode",
        "iscf-scope-projected-synthesis",
        "--pcsd-mode-rank",
        "109",
        "--sps-projection-mode",
        "scope",
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
        "mode_rank": args.pcsd_mode_rank,
        "validation_horizons": args.validation_horizons,
        "final_split": args.final_evaluation_split,
    }


def main() -> None:
    seed = 20260721
    parent = make_parent(seed)
    identity = make_sps(seed, "identity")
    scope = make_sps(seed, "scope")
    global_control = make_sps(seed, "global")
    random_scope = make_sps(seed, "scope", partition="random")

    hashes = {
        "parent": tensor_hash(parent),
        "identity": tensor_hash(identity),
        "scope": tensor_hash(scope),
        "global": tensor_hash(global_control),
        "random": tensor_hash(random_scope),
    }
    if len(set(hashes.values())) != 1:
        raise AssertionError("paired SPS/parent parameter initialization differs")

    hidden = torch.randn(2, 3, 32)
    with torch.no_grad():
        parent_output = parent(hidden)
        identity_output = identity(hidden)
        scope_output = scope(hidden)
        global_output = global_control(hidden)
        random_output = random_scope(hidden)
        prefix_output = scope(hidden, target_prefix=192)

    if parent_output.shape != (2, 720, 3):
        raise AssertionError(f"unexpected parent shape: {parent_output.shape}")
    if prefix_output.shape != (2, 192, 3):
        raise AssertionError(f"unexpected prefix shape: {prefix_output.shape}")
    for name, output in {
        "identity": identity_output,
        "scope": scope_output,
        "global": global_output,
        "random": random_output,
    }.items():
        if output.shape != parent_output.shape or not torch.isfinite(output).all():
            raise AssertionError(f"{name} output contract failed")

    identity_gap = float((identity_output - parent_output).abs().max())
    prefix_gap = float((scope_output[:, :192] - prefix_output).abs().max())
    if identity_gap > 1e-6:
        raise AssertionError(f"identity path does not recover parent: {identity_gap}")
    if prefix_gap > 1e-6:
        raise AssertionError(f"full-domain prefix crop mismatch: {prefix_gap}")

    basis_errors = []
    idempotence_errors = []
    for scale_index in range(len(scope.scales)):
        basis = scope.projection_basis(scale_index).double()
        gram = basis.T @ basis
        basis_errors.append(
            float((gram - torch.eye(gram.shape[0], dtype=gram.dtype)).abs().max())
        )
        projector = basis @ basis.T
        idempotence_errors.append(
            float((projector @ projector - projector).abs().max())
        )
    if max(basis_errors) > 1e-10 or max(idempotence_errors) > 1e-10:
        raise AssertionError("scope projector is not orthonormal/idempotent")

    impulse_supports = []
    for scale_index, scale_value in enumerate(scope.scales):
        impulse = torch.zeros(1, 1, 720)
        impulse[..., 377] = 1.0
        projected_impulse = scope._project_scope_arm(impulse, scale_index)
        support = int((projected_impulse.abs() > 1e-7).sum())
        impulse_supports.append(support)
        if support != scale_value:
            raise AssertionError(
                f"scope {scale_value} impulse support is {support}"
            )

    scope.train()
    train_hidden = torch.randn(2, 3, 32, requires_grad=True)
    target = torch.randn(2, 720, 3)
    prediction = scope(train_hidden)
    loss = (prediction - target).square().mean()
    loss.backward()
    gradient_norms = [
        float(scope.mode_weight.grad[index].norm())
        for index in range(len(scope.scales))
    ]
    if not all(math.isfinite(value) and value > 0.0 for value in gradient_norms):
        raise AssertionError(f"scope gradients inactive: {gradient_norms}")

    scope_random_gap = float((scope_output - random_output).abs().max())
    scope_global_gap = float((scope_output - global_output).abs().max())
    if scope_random_gap <= 1e-7 or scope_global_gap <= 1e-7:
        raise AssertionError("candidate does not differ from required controls")

    expected_ranks = tuple(
        min(scale, max(1, round(16 * scale / 720)))
        for scale in scope.scales
    )
    if scope.projection_ranks != expected_ranks:
        raise AssertionError(
            f"projection ranks {scope.projection_ranks} != {expected_ranks}"
        )

    torch.manual_seed(seed)
    model = TimeAlign.Model(model_config()).float().eval()
    x = torch.randn(1, 720, 2)
    y = torch.zeros_like(x)
    with torch.no_grad():
        model_full = model(x, y, is_training=False, target_prefix=720)[0]
        model_short = model(x, y, is_training=False, target_prefix=96)[0]
    model_prefix_gap = float((model_full[:, :96] - model_short).abs().max())
    if model_full.shape != (1, 720, 2) or model_short.shape != (1, 96, 2):
        raise AssertionError("production model shape contract failed")
    if not torch.isfinite(model_full).all() or model_prefix_gap > 1e-6:
        raise AssertionError("production model finite/prefix contract failed")

    cli = cli_contract()
    if cli != {
        "readout_mode": "iscf-scope-projected-synthesis",
        "projection_mode": "scope",
        "mode_rank": 109,
        "validation_horizons": [96, 192, 336, 720],
        "final_split": "val",
    }:
        raise AssertionError(f"production CLI contract failed: {cli}")

    result = {
        "decision": "iscf_sps_step7a_contract_pass",
        "parameter_hash": hashes["scope"],
        "parameter_count": sum(parameter.numel() for parameter in scope.parameters()),
        "projection_ranks": list(scope.projection_ranks),
        "identity_parent_max_abs_gap": identity_gap,
        "prefix_max_abs_gap": prefix_gap,
        "max_basis_orthonormal_error": max(basis_errors),
        "max_projector_idempotence_error": max(idempotence_errors),
        "scope_gradient_norms": gradient_norms,
        "scope_impulse_supports": impulse_supports,
        "scope_random_max_abs_gap": scope_random_gap,
        "scope_global_max_abs_gap": scope_global_gap,
        "production_model_full_shape": list(model_full.shape),
        "production_model_prefix_gap": model_prefix_gap,
        "production_cli": cli,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
