"""Run the local code-theory gate for SIFF-v3 TSAF Step 7A."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "baselines" / "timealign_official"
sys.path.insert(0, str(BASELINE_ROOT))

from layers.PCSD import PCSDCouplingFieldReadout  # noqa: E402
from layers.SIFF import (  # noqa: E402
    SIFFCouplingFieldReadout,
    siff_parameter_count,
    tsaf_parameter_count,
)
from models import TimeAlign  # noqa: E402
from train_repo import initialization_contract, model_diagnostics  # noqa: E402


def record(
    rows: list[dict[str, Any]],
    category: str,
    case: str,
    passed: bool,
    value: Any,
    threshold: Any,
) -> None:
    """Append one machine-readable gate row."""
    rows.append(
        {
            "category": category,
            "case": case,
            "passed": bool(passed),
            "value": value,
            "threshold": threshold,
        }
    )


def set_nonconstant_allocator(module: SIFFCouplingFieldReadout) -> None:
    """Install the same deterministic nonconstant TSAF witness."""
    with torch.no_grad():
        target_weight = module.target_allocation_projection.weight
        scale_weight = module.scale_allocation_projection.weight
        output_weight = module.target_scale_allocation_output.weight
        target_weight.zero_()
        scale_weight.zero_()
        output_weight.zero_()
        target_weight[:4, :4] = torch.eye(4)
        scale_weight[:2, :2] = torch.eye(2)
        output_weight[0, :4] = torch.tensor([0.9, -0.6, 0.3, -0.2])
        module.target_scale_allocation_bias.zero_()
        module.target_scale_allocation_output.bias.zero_()


def run_gate() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute shape, invariance, semantic, parameter, and gradient checks."""
    torch.manual_seed(20260721)
    rows: list[dict[str, Any]] = []
    common = {
        "readout_dim": 16,
        "series_length": 720,
        "mode_rank": 8,
        "policy_history_dim": 7,
        "policy_hidden_dim": 12,
    }
    candidate = SIFFCouplingFieldReadout(
        **common,
        policy_mode="target-scale-field",
    )
    permuted = SIFFCouplingFieldReadout(
        **common,
        policy_mode="target-scale-field-permuted",
    )
    global_control = SIFFCouplingFieldReadout(
        **common,
        policy_mode="target-scale-global",
    )
    direct = SIFFCouplingFieldReadout(
        **common,
        policy_mode="direct",
    )
    hidden = torch.randn(2, 3, common["readout_dim"], requires_grad=True)

    prefix, arms, initial_weights = candidate.forward_with_diagnostics(
        hidden,
        96,
    )
    record(
        rows,
        "shape",
        "forecast",
        tuple(prefix.shape) == (2, 96, 3),
        tuple(prefix.shape),
        (2, 96, 3),
    )
    record(
        rows,
        "shape",
        "arms",
        tuple(arms.shape) == (2, 3, 5, 720),
        tuple(arms.shape),
        (2, 3, 5, 720),
    )
    record(
        rows,
        "shape",
        "weights",
        tuple(initial_weights.shape) == (2, 3, 720, 5),
        tuple(initial_weights.shape),
        (2, 3, 720, 5),
    )
    simplex_gap = float((initial_weights.sum(dim=-1) - 1.0).abs().max())
    record(rows, "numeric", "simplex", simplex_gap <= 1e-7, simplex_gap, "<=1e-7")
    uniform_gap = float((initial_weights - 0.2).abs().max())
    record(rows, "initialization", "uniform", uniform_gap == 0.0, uniform_gap, 0.0)

    hidden_alt = torch.randn_like(hidden)
    allocation_gap = float(
        (
            candidate.policy_weights(hidden.detach())
            - candidate.policy_weights(hidden_alt)
        ).abs().max()
    )
    record(
        rows,
        "invariance",
        "allocation_ignores_history",
        allocation_gap == 0.0,
        allocation_gap,
        0.0,
    )
    channel_gap = float(
        (
            initial_weights
            - initial_weights[0, 0].view(1, 1, 720, 5)
        ).abs().max()
    )
    record(rows, "invariance", "sample_channel_broadcast", channel_gap == 0.0, channel_gap, 0.0)
    record(
        rows,
        "structure",
        "no_history_policy_parameters",
        not any("history_projection" in name for name, _ in candidate.named_parameters()),
        sorted(name for name, _ in candidate.named_parameters() if "history" in name),
        [],
    )

    full = candidate(hidden, target_prefix=720)
    short = candidate(hidden, target_prefix=96)
    request_gap = float((full[:, :96] - short).abs().max())
    record(rows, "invariance", "full_domain_crop", request_gap == 0.0, request_gap, 0.0)

    actual_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    expected_parameters = tsaf_parameter_count(
        readout_dim=common["readout_dim"],
        mode_rank=common["mode_rank"],
        policy_history_dim=common["policy_history_dim"],
        policy_hidden_dim=common["policy_hidden_dim"],
    )
    direct_parameters = siff_parameter_count(
        readout_dim=common["readout_dim"],
        mode_rank=common["mode_rank"],
        policy_history_dim=common["policy_history_dim"],
        policy_hidden_dim=common["policy_hidden_dim"],
    )
    record(
        rows,
        "parameters",
        "exact_count",
        actual_parameters == expected_parameters,
        actual_parameters,
        expected_parameters,
    )
    record(
        rows,
        "parameters",
        "lower_than_direct",
        actual_parameters < direct_parameters,
        actual_parameters,
        f"<{direct_parameters}",
    )

    set_nonconstant_allocator(candidate)
    set_nonconstant_allocator(permuted)
    set_nonconstant_allocator(global_control)
    candidate_weights = candidate.policy_weights(hidden.detach())
    permuted_weights = permuted.policy_weights(hidden.detach())
    global_weights = global_control.policy_weights(hidden.detach())
    target_variation = float(
        (candidate_weights[:, :, 0] - candidate_weights[:, :, -1]).abs().max()
    )
    scale_semantic_gap = float((candidate_weights - permuted_weights).abs().max())
    global_target_gap = float(
        (global_weights[:, :, 0] - global_weights[:, :, -1]).abs().max()
    )
    record(
        rows,
        "semantics",
        "target_nonconstant",
        target_variation > 1e-4,
        target_variation,
        ">1e-4",
    )
    record(
        rows,
        "semantics",
        "permuted_scale_changes_allocation",
        scale_semantic_gap > 1e-4,
        scale_semantic_gap,
        ">1e-4",
    )
    record(
        rows,
        "control",
        "global_removes_target_variation",
        global_target_gap == 0.0,
        global_target_gap,
        0.0,
    )

    candidate.zero_grad(set_to_none=True)
    hidden.grad = None
    prediction, _arms, weights = candidate.forward_with_diagnostics(hidden, 720)
    target = torch.randn_like(prediction)
    loss = (prediction - target).square().mean()
    loss.backward()
    gradient_names = (
        "mode_weight",
        "target_allocation_projection.weight",
        "scale_allocation_projection.weight",
        "target_scale_allocation_output.weight",
    )
    for name in gradient_names:
        parameter = dict(candidate.named_parameters())[name]
        norm = float(parameter.grad.norm()) if parameter.grad is not None else 0.0
        record(
            rows,
            "gradient",
            name,
            torch.isfinite(torch.tensor(norm)).item() and norm > 0.0,
            norm,
            ">0 finite",
        )
    hidden_gradient = float(hidden.grad.norm()) if hidden.grad is not None else 0.0
    record(
        rows,
        "gradient",
        "history_via_arms",
        hidden_gradient > 0.0,
        hidden_gradient,
        ">0 finite",
    )
    allocation_variance = float(weights.var(dim=-2, unbiased=False).mean())
    record(
        rows,
        "health",
        "allocation_target_variance",
        allocation_variance > 0.0,
        allocation_variance,
        ">0",
    )

    base_rejected = False
    try:
        invalid = PCSDCouplingFieldReadout(
            **common,
            policy_mode="target-scale-field",
        )
        invalid.policy_weights(hidden.detach())
    except RuntimeError:
        base_rejected = True
    record(rows, "guard", "pcsd_rejects_tsaf_semantics", base_rejected, base_rejected, True)

    model_config = SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        encoder_mode="timealign-token-mlp",
        readout_mode="siff-coupling-field",
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
        pcsd_mode_rank=8,
        pcsd_policy_history_dim=4,
        pcsd_policy_hidden_dim=8,
        pcsd_policy_mode="target-scale-field",
        pcsd_fixed_scale=720,
        pcsd_partition="canonical",
        pcsd_partition_seed=15101,
        pcsd_group_chunk_size=64,
        pcsd_target_chunk_size=128,
    )
    production = TimeAlign.Model(model_config).float().eval()
    model_x = torch.randn(1, 720, 2)
    model_y = torch.zeros(1, 720, 2)
    with torch.no_grad():
        production_full = production(
            model_x,
            model_y,
            is_training=False,
            target_prefix=720,
        )[0]
        production_short = production(
            model_x,
            model_y,
            is_training=False,
            target_prefix=96,
        )[0]
    production_gap = float(
        (production_full[:, :96] - production_short).abs().max()
    )
    record(
        rows,
        "production",
        "timealign_output_shape",
        tuple(production_full.shape) == (1, 720, 2),
        tuple(production_full.shape),
        (1, 720, 2),
    )
    record(
        rows,
        "production",
        "timealign_request_invariance",
        production_gap == 0.0,
        production_gap,
        0.0,
    )
    record(
        rows,
        "production",
        "timealign_policy_mode",
        production.pcsd_readout.policy_mode == "target-scale-field",
        production.pcsd_readout.policy_mode,
        "target-scale-field",
    )
    diagnostics = model_diagnostics(production)
    initialization = initialization_contract(production)
    record(
        rows,
        "production",
        "policy_parameter_accounting",
        diagnostics["pcsd_policy_parameters"] > 0
        and diagnostics["tsaf_allocation_parameters"]
        == diagnostics["pcsd_policy_parameters"],
        diagnostics["pcsd_policy_parameters"],
        ">0 and equals TSAF allocation parameters",
    )
    record(
        rows,
        "production",
        "allocation_hash_recorded",
        "tsaf_allocation_scale_hash" in initialization,
        initialization.get("tsaf_allocation_scale_hash", "missing"),
        "recorded",
    )

    all_passed = all(row["passed"] for row in rows)
    summary = {
        "gate_id": "SC1-SIFF-v3-TSAF-v1-Step7A-local",
        "case_count": len(rows),
        "passed_count": sum(int(row["passed"]) for row in rows),
        "all_passed": all_passed,
        "remote_training": False,
        "official_test": False,
    }
    return rows, summary


def main() -> None:
    """Write the local gate artifacts and fail on any rejected case."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = run_gate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
