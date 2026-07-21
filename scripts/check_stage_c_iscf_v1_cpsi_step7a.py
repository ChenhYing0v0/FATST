#!/usr/bin/env python3
"""Run the local production gate for ISCF-v1-CPSI Step 7A."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from layers.CPSI import (  # noqa: E402
    CPSI_READOUT_CONFIG,
    CPSIReadout,
    cpsi_interaction_parameter_count,
)
from layers.SIFF import SIFFCouplingFieldReadout  # noqa: E402
from models import TimeAlign  # noqa: E402
import train_repo as training_adapter  # noqa: E402
from train_repo import initialization_contract, model_diagnostics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_iscf_v1_cpsi_step6.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/stage_c_post_d21_unconstrained_reset_20260720/"
            "iscf_v1_cpsi_step7a_20260721"
        ),
    )
    return parser.parse_args()


def record(
    rows: list[dict[str, Any]],
    category: str,
    case: str,
    passed: bool,
    value: Any,
    threshold: Any,
) -> None:
    rows.append(
        {
            "category": category,
            "case": case,
            "passed": bool(passed),
            "value": value,
            "threshold": threshold,
        }
    )


def tensor_grad_norm(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.norm().item())


def parent_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    excluded = {
        "common_projection",
        "private_projection",
        "interaction_output",
    }
    return {
        name: tensor
        for name, tensor in module.state_dict().items()
        if name not in excluded
    }


def max_state_gap(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> float:
    if left.keys() != right.keys():
        return math.inf
    return max(
        float((left[name] - right[name]).abs().max().item())
        for name in left
    )


def readout_kwargs() -> dict[str, Any]:
    return {
        "readout_dim": 8,
        "series_length": 720,
        "coordinate_dim": 4,
        "mode_rank": 4,
        "policy_history_dim": 4,
        "policy_hidden_dim": 8,
        "policy_mode": "direct",
        "group_chunk_size": 64,
        "target_chunk_size": 128,
    }


def build_base(seed: int = 2021) -> SIFFCouplingFieldReadout:
    torch.manual_seed(seed)
    return SIFFCouplingFieldReadout(
        **readout_kwargs(),
        scale_components=5,
        scale_basis_mode="independent",
    )


def build_cpsi(mode: str, seed: int = 2021) -> CPSIReadout:
    torch.manual_seed(seed)
    return CPSIReadout(
        **readout_kwargs(),
        interaction_rank=32,
        interaction_mode=mode,
    )


def run_readout_gate(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    base = build_base()
    hidden = torch.randn(2, 3, 8)
    with torch.no_grad():
        base_output, base_arms, base_policy = base.forward_with_diagnostics(
            hidden,
            720,
        )
    summaries: dict[str, dict[str, Any]] = {}
    expected_parent = parent_state(base)

    for readout_mode, interaction_mode in CPSI_READOUT_CONFIG.items():
        module = build_cpsi(interaction_mode)
        with torch.no_grad():
            output, arms, policy = module.forward_with_diagnostics(hidden, 720)
            details = module.interaction_diagnostics(hidden)
        parent_gap = max_state_gap(expected_parent, parent_state(module))
        morph_gap = float((output - base_output).abs().max().item())
        arm_gap = float((arms - base_arms).abs().max().item())
        policy_gap = float((policy - base_policy).abs().max().item())
        expected_width = (
            720 if interaction_mode == "post-synthesis" else 16
        )
        expected_rank = (
            round(16 * 32 / 720)
            if interaction_mode == "post-synthesis"
            else 32
        )
        expected_parameters = 3 * expected_width * expected_rank

        checks = {
            "output_shape": tuple(output.shape) == (2, 720, 3),
            "arm_shape": tuple(arms.shape) == (2, 3, 5, 720),
            "policy_shape": tuple(policy.shape) == (2, 3, 720, 5),
            "parent_pair": parent_gap == 0.0,
            "output_morph": morph_gap <= 1e-6,
            "arm_morph": arm_gap <= 1e-6,
            "policy_morph": policy_gap == 0.0,
            "parameter_count": module.interaction_parameters
            == expected_parameters,
            "zero_output_projection": float(
                module.interaction_output.abs().max().item()
            )
            == 0.0,
            "diagnostic_finite": all(
                torch.isfinite(value).all().item()
                for value in details.values()
            ),
        }
        for case, passed in checks.items():
            record(
                rows,
                "readout",
                f"{readout_mode}:{case}",
                passed,
                {
                    "parent_gap": parent_gap,
                    "morph_gap": morph_gap,
                    "arm_gap": arm_gap,
                    "policy_gap": policy_gap,
                    "parameters": module.interaction_parameters,
                },
                "production contract",
            )
        summaries[readout_mode] = {
            "interaction_mode": interaction_mode,
            "effective_rank": module.effective_interaction_rank,
            "interaction_width": module.interaction_width,
            "interaction_parameters": module.interaction_parameters,
            "parent_gap": parent_gap,
            "morph_gap": morph_gap,
            "pass": all(checks.values()),
        }
    return summaries


def run_equivariance_gate(rows: list[dict[str, Any]]) -> None:
    permutation = torch.tensor([2, 4, 0, 1, 3])
    modes = torch.randn(2, 3, 5, 4, 4)
    forecasts = torch.randn(2, 3, 5, 720)
    for interaction_mode in CPSI_READOUT_CONFIG.values():
        module = build_cpsi(interaction_mode)
        with torch.no_grad():
            module.interaction_output.normal_(mean=0.0, std=0.05)
            if interaction_mode == "post-synthesis":
                direct, _ = module.interact_forecasts(forecasts)
                permuted, _ = module.interact_forecasts(
                    forecasts[:, :, permutation]
                )
            else:
                direct, _ = module.interact_modes(modes)
                permuted, _ = module.interact_modes(modes[:, :, permutation])
        gap = float((permuted - direct[:, :, permutation]).abs().max().item())
        record(
            rows,
            "equivariance",
            interaction_mode,
            gap <= 1e-6,
            gap,
            "<=1e-6",
        )

    candidate = build_cpsi("common-private")
    with torch.no_grad():
        candidate.interaction_output.normal_(mean=0.0, std=0.05)
        identical = modes[:, :, :1].expand(-1, -1, 5, -1, -1)
        updated, _ = candidate.interact_modes(identical)
    identical_gap = float((updated - identical).abs().max().item())
    record(
        rows,
        "semantics",
        "candidate_zero_when_private_absent",
        identical_gap <= 1e-7,
        identical_gap,
        "<=1e-7",
    )


def run_gradient_gate(rows: list[dict[str, Any]]) -> None:
    for interaction_mode in CPSI_READOUT_CONFIG.values():
        module = build_cpsi(interaction_mode)
        width = module.interaction_width
        values = torch.randn(2, 3, 5, width)

        updated, _ = module._interaction_terms(values)
        first_loss = updated.square().mean()
        first_loss.backward()
        output_first = tensor_grad_norm(module.interaction_output)
        common_first = tensor_grad_norm(module.common_projection)
        private_first = tensor_grad_norm(module.private_projection)
        first_pass = (
            math.isfinite(float(first_loss.detach().item()))
            and output_first > 0.0
            and common_first == 0.0
            and private_first == 0.0
        )
        record(
            rows,
            "gradient",
            f"{interaction_mode}:first_backward",
            first_pass,
            {
                "loss": float(first_loss.detach().item()),
                "output": output_first,
                "common": common_first,
                "private": private_first,
            },
            "output>0; inputs=0 at zero-output morph",
        )

        with torch.no_grad():
            module.interaction_output.add_(
                -0.1 * module.interaction_output.grad
            )
        module.zero_grad(set_to_none=True)
        updated, details = module._interaction_terms(values)
        second_loss = updated.square().mean()
        second_loss.backward()
        common_second = tensor_grad_norm(module.common_projection)
        private_second = tensor_grad_norm(module.private_projection)
        message_rms = float(details["message"].square().mean().sqrt().item())
        second_pass = (
            common_second > 0.0
            and private_second > 0.0
            and message_rms > 0.0
            and all(
                math.isfinite(value)
                for value in (
                    float(second_loss.detach().item()),
                    common_second,
                    private_second,
                    message_rms,
                )
            )
        )
        record(
            rows,
            "gradient",
            f"{interaction_mode}:second_backward",
            second_pass,
            {
                "loss": float(second_loss.detach().item()),
                "common": common_second,
                "private": private_second,
                "message_rms": message_rms,
            },
            "input gradients and message >0 after output step",
        )


def model_config(readout_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        encoder_mode="timealign-token-mlp",
        readout_mode=readout_mode,
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
        cpsi_rank=32,
        pcsd_policy_history_dim=4,
        pcsd_policy_hidden_dim=8,
        pcsd_policy_mode="direct",
        pcsd_fixed_scale=720,
        pcsd_partition="canonical",
        pcsd_partition_seed=15101,
        pcsd_group_chunk_size=64,
        pcsd_target_chunk_size=128,
    )


def run_model_gate(rows: list[dict[str, Any]]) -> None:
    x = torch.randn(1, 720, 2)
    y = torch.zeros_like(x)
    for readout_mode, interaction_mode in CPSI_READOUT_CONFIG.items():
        torch.manual_seed(2021)
        model = TimeAlign.Model(model_config(readout_mode)).float().eval()
        contract = initialization_contract(model)
        diagnostics = model_diagnostics(model)
        with torch.no_grad():
            full = model(x, y, is_training=False, target_prefix=720)[0]
            short = model(x, y, is_training=False, target_prefix=96)[0]
        prefix_gap = float((short - full[:, :96]).abs().max().item())
        passed = (
            tuple(full.shape) == (1, 720, 2)
            and tuple(short.shape) == (1, 96, 2)
            and prefix_gap == 0.0
            and contract["cpsi_interaction_mode"] == interaction_mode
            and contract["cpsi_output_initial_max_abs"] == 0.0
            and diagnostics["cpsi_interaction_parameters"]
            == model.pcsd_readout.interaction_parameters
        )
        record(
            rows,
            "model",
            readout_mode,
            passed,
            {
                "full_shape": tuple(full.shape),
                "short_shape": tuple(short.shape),
                "prefix_gap": prefix_gap,
                "parent_hash": contract["cpsi_parent_initialization_hash"],
                "parameters": diagnostics["cpsi_interaction_parameters"],
            },
            "production model integration",
        )


def cli_argv(readout_mode: str) -> list[str]:
    return [
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
        readout_mode,
        "--output-dir",
        f"/tmp/{readout_mode}",
        "--readout-mode",
        readout_mode,
        "--pcsd-mode-rank",
        "109",
        "--cpsi-rank",
        "32",
        "--pcsd-policy-mode",
        "direct",
        "--pcc-objective-mode",
        "equal_skill",
        "--checkpoint-policy",
        "best-val",
        "--final-evaluation-split",
        "val",
    ]


def run_cli_gate(rows: list[dict[str, Any]]) -> None:
    original = sys.argv
    try:
        for readout_mode in CPSI_READOUT_CONFIG:
            sys.argv = cli_argv(readout_mode)
            args = training_adapter.parse_args()
            passed = (
                args.readout_mode == readout_mode
                and args.cpsi_rank == 32
                and args.pcsd_mode_rank == 109
                and args.pcsd_policy_mode == "direct"
                and args.pcc_objective_mode == "equal_skill"
                and args.validation_horizons == [96, 192, 336, 720]
                and args.final_evaluation_split == "val"
            )
            record(
                rows,
                "cli",
                readout_mode,
                passed,
                {
                    "rank": args.cpsi_rank,
                    "mode_rank": args.pcsd_mode_rank,
                    "objective": args.pcc_objective_mode,
                },
                "frozen CLI contract",
            )
    finally:
        sys.argv = original


def run_profile_parameter_gate(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    for dataset, profile in config["profiles"].items():
        expected = cpsi_interaction_parameter_count(
            coordinate_dim=4,
            mode_rank=int(profile["mode_rank"]),
            interaction_rank=32,
        )
        record(
            rows,
            "profile_parameters",
            dataset,
            expected == int(profile["cpsi_parameters"]),
            expected,
            int(profile["cpsi_parameters"]),
        )


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "checks.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    categories = sorted({row["category"] for row in rows})
    payload = {
        "cases": len(rows),
        "passed": sum(bool(row["passed"]) for row in rows),
        "categories": {
            category: {
                "cases": sum(row["category"] == category for row in rows),
                "passed": sum(
                    row["category"] == category and bool(row["passed"])
                    for row in rows
                ),
            }
            for category in categories
        },
        "readouts": summaries,
    }
    payload["decision"] = (
        "step7a_local_pass_step7b_prelaunch_next"
        if payload["cases"] == payload["passed"]
        else "step7a_hard_invalid_repair_before_prelaunch"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    summaries = run_readout_gate(rows)
    run_equivariance_gate(rows)
    run_gradient_gate(rows)
    run_model_gate(rows)
    run_cli_gate(rows)
    run_profile_parameter_gate(rows, config)
    write_outputs(args.output_dir, rows, summaries)
    failed = [row for row in rows if not row["passed"]]
    print(
        f"CPSI Step7A: {len(rows) - len(failed)}/{len(rows)} passed; "
        f"failed={len(failed)}"
    )
    if failed:
        for row in failed:
            print(json.dumps(row, sort_keys=True))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
