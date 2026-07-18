#!/usr/bin/env python3
"""Verify the CCSF zero-contrast gradient repair and multi-step stability."""

from __future__ import annotations

import argparse
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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from layers.CCSF import contrast_scope_calibration_loss  # noqa: E402
from models import TimeAlign  # noqa: E402
from check_stage_c_siff_ccsf_step7a import model_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/stage_c_siff_ccsf_runtime_repair_20260718/local_gate"
        ),
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def small_model() -> TimeAlign.Model:
    row = {
        "arm": "ccsf_relcal",
        "readout_mode": "ccsf-coupling-field",
        "objective_mode": "ccsf_relative_calibration",
        "dataset": "ETTh2",
        "mode_rank": 8,
        "patch_num": 2,
        "d_model": 4,
        "d_ff": 8,
    }
    return TimeAlign.Model(model_config(row, small=True)).float()


def zero_contrast_gradient_case() -> dict[str, Any]:
    torch.manual_seed(2021)
    model = small_model()
    readout = model.pcsd_readout
    arms = torch.ones(
        1,
        2,
        len(readout.scales),
        readout.series_length,
        requires_grad=True,
    )
    descriptor = readout._true_contrast_descriptor(arms)
    descriptor.sum().backward()
    gradient = arms.grad
    assert gradient is not None
    expected_floor = math.sqrt(readout.contrast_epsilon)
    group_rms = descriptor[..., 3]
    return {
        "case": "identical_arms_zero_contrast",
        "descriptor_finite": bool(torch.isfinite(descriptor).all()),
        "gradient_finite": bool(torch.isfinite(gradient).all()),
        "gradient_nan_count": int(torch.isnan(gradient).sum()),
        "group_rms_min": float(group_rms.min()),
        "expected_epsilon_floor": expected_floor,
        "pass": bool(
            torch.isfinite(descriptor).all()
            and torch.isfinite(gradient).all()
            and torch.all(group_rms >= expected_floor * 0.999)
        ),
    }


def multi_step_cases() -> list[dict[str, Any]]:
    rows = []
    for temperature in (0.05, 0.1, 0.25):
        torch.manual_seed(2021)
        model = small_model().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.zeros(2, 720, 2)
        target = torch.zeros(2, 720, 2)
        for step in range(1, 4):
            output, _recon, _align, details = model(
                x,
                target,
                is_training=True,
                target_prefix=720,
                return_pcsd_training_details=True,
            )
            arms = details["arm_forecasts"].permute(0, 1, 3, 2)
            result = contrast_scope_calibration_loss(
                output,
                arms,
                details["policy"],
                target,
                mode="ccsf_relative_calibration",
                progress=1.0,
                temperature=temperature,
            )
            optimizer.zero_grad()
            result.total_loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            gradient_finite = all(
                bool(torch.isfinite(gradient).all()) for gradient in gradients
            )
            optimizer.step()
            parameter_finite = all(
                bool(torch.isfinite(parameter).all())
                for parameter in model.parameters()
            )
            rows.append(
                {
                    "case": "degenerate_zero_input_three_step",
                    "temperature": temperature,
                    "step": step,
                    "loss": float(result.total_loss.detach()),
                    "loss_finite": math.isfinite(
                        float(result.total_loss.detach())
                    ),
                    "gradient_finite": gradient_finite,
                    "parameter_finite": parameter_finite,
                    "pass": bool(
                        math.isfinite(float(result.total_loss.detach()))
                        and gradient_finite
                        and parameter_finite
                    ),
                }
            )
    return rows


def main() -> None:
    torch.set_num_threads(1)
    args = parse_args()
    zero_case = zero_contrast_gradient_case()
    step_cases = multi_step_cases()
    categories = {
        "zero_contrast_forward_and_backward_finite": bool(zero_case["pass"]),
        "three_step_all_temperatures_finite": all(
            row["pass"] for row in step_cases
        ),
        "three_steps_executed_per_temperature": all(
            sum(row["temperature"] == temperature for row in step_cases) == 3
            for temperature in (0.05, 0.1, 0.25)
        ),
    }
    payload = {
        "repair_id": "SC1-SIFF-v2-CCSF-RUNTIME-REPAIR-v1",
        "failure_attribution": "optimization_or_numeric_pathology",
        "root_cause": "zero_contrast_group_rms_sqrt_derivative",
        "categories": categories,
        "categories_passed": sum(categories.values()),
        "categories_total": len(categories),
        "overall_pass": all(categories.values()),
        "remote_pilot_relaunch_authorized": False,
        "formal_phase_a_authorized": False,
        "formal_test_access_authorized": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "zero_contrast_gradient.csv", [zero_case])
    write_csv(args.output_dir / "multi_step_stability.csv", step_cases)
    (args.output_dir / "runtime_repair_gate.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    if not payload["overall_pass"]:
        raise RuntimeError(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
