#!/usr/bin/env python3
"""Run the PCSD-CF Step7B local prelaunch contract gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from models import TimeAlign  # noqa: E402
from train_repo import initialization_contract, model_diagnostics  # noqa: E402


DATASETS = ("Weather", "ETTm1", "ETTm2", "ETTh1", "ETTh2")
CHANNELS = {"ETTh1": 7, "ETTh2": 7, "ETTm1": 7, "ETTm2": 7, "Weather": 21}
SCALES = (1, 48, 144, 360, 720)
ARM_SPECS: tuple[tuple[str, str, str, int, str], ...] = (
    ("a6", "learned-basis-forecast-operator", "control", 720, "control"),
    ("pcsd_m0", "pcsd-coupling-field-m0", "control", 720, "control"),
    *(
        (f"pcsd_fixed_{scale}", "pcsd-coupling-field", "fixed", scale, "canonical")
        for scale in SCALES
    ),
    ("pcsd_equal", "pcsd-coupling-field", "equal", 720, "canonical"),
    (
        "pcsd_static",
        "pcsd-coupling-field",
        "static-target",
        720,
        "canonical",
    ),
    ("pcsd_direct", "pcsd-coupling-field", "direct", 720, "canonical"),
    ("pcsd_random", "pcsd-coupling-field", "direct", 720, "random"),
    (
        "dense_matched",
        "pcsd-dense-nonlinear-matched",
        "control",
        720,
        "control",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=Path("configs/stage_c_pcsd_cf_native_direct.json"),
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("configs/stage_c_five_dataset_natural_profiles.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/stage_c_pcsd_cf_step7b_prelaunch_20260716"),
    )
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def tensor_hash(parameters: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_config(
    dataset: str,
    profile: dict[str, Any],
    readout: str,
    policy: str,
    fixed_scale: int,
    partition: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        encoder_mode="timealign-token-mlp",
        readout_mode=readout,
        e_layers=2,
        patch_num=int(profile["patch_num"]),
        d_model=int(profile["d_model"]),
        d_ff=int(profile["d_ff"]),
        dropout=0.1,
        pos=1,
        layer_norm=1,
        enc_in=CHANNELS[dataset],
        basis_rank=256,
        pcsd_coordinate_dim=4,
        pcsd_mode_rank=256,
        pcsd_policy_history_dim=32,
        pcsd_policy_hidden_dim=64,
        pcsd_policy_mode="direct" if policy == "control" else policy,
        pcsd_fixed_scale=fixed_scale,
        pcsd_partition="canonical" if partition == "control" else partition,
        pcsd_partition_seed=15101,
        pcsd_group_chunk_size=64,
        pcsd_target_chunk_size=128,
    )


def active_readout_parameters(model: TimeAlign.Model) -> list[torch.Tensor]:
    if hasattr(model, "pcsd_readout"):
        return list(model.pcsd_readout.parameters())
    if hasattr(model, "pcsd_m0_readout"):
        return list(model.pcsd_m0_readout.parameters())
    if hasattr(model, "pcsd_dense_readout"):
        return list(model.pcsd_dense_readout.parameters())
    return [
        model.learned_basis_coeff.weight,
        model.learned_basis_coeff.bias,
        model.learned_temporal_basis,
        model.learned_temporal_bias,
    ]


def audit_matrix(
    design: dict[str, Any],
    profiles: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    pairing_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        profile = profiles[dataset]
        models: dict[str, TimeAlign.Model] = {}
        for arm, readout, policy, fixed_scale, partition in ARM_SPECS:
            torch.manual_seed(seed)
            model = TimeAlign.Model(
                model_config(
                    dataset,
                    profile,
                    readout,
                    policy,
                    fixed_scale,
                    partition,
                )
            ).float().eval()
            models[arm] = model
            hidden_width = int(profile["state_width"])
            hidden = torch.randn(
                1,
                1,
                hidden_width,
                generator=torch.Generator().manual_seed(seed + hidden_width),
            )
            module = (
                model.pcsd_readout
                if hasattr(model, "pcsd_readout")
                else (
                    model.pcsd_m0_readout
                    if hasattr(model, "pcsd_m0_readout")
                    else (
                        model.pcsd_dense_readout
                        if hasattr(model, "pcsd_dense_readout")
                        else None
                    )
                )
            )
            with torch.no_grad():
                if module is None:
                    full = model._learned_basis_forecast_operator(hidden, 720)
                    prefix = model._learned_basis_forecast_operator(hidden, 336)
                else:
                    full = module(hidden, 720)
                    prefix = module(hidden, 336)
            prefix_gap = float((prefix - full[:, :336]).abs().max())
            diagnostics = model_diagnostics(model)
            initialization = initialization_contract(model)
            dense_gap = diagnostics.get("pcsd_dense_parameter_relative_gap", 0.0)
            fixed_equivalence_gap: float | str = ""
            if hasattr(model, "pcsd_readout") and policy == "fixed":
                with torch.no_grad():
                    diagnostic_output, arms, _weights = (
                        model.pcsd_readout.forward_with_diagnostics(hidden, 720)
                    )
                selected = arms[:, :, SCALES.index(fixed_scale)].permute(0, 2, 1)
                fixed_equivalence_gap = float((diagnostic_output - selected).abs().max())
                fixed_equivalence_gap = max(
                    fixed_equivalence_gap,
                    float((full - selected).abs().max()),
                )
            rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "readout": readout,
                    "policy": policy,
                    "fixed_scale": fixed_scale,
                    "partition": partition,
                    "output_shape": str(tuple(full.shape)),
                    "prefix_max_abs": prefix_gap,
                    "fixed_fast_path_max_abs": fixed_equivalence_gap,
                    "decoder_parameters": sum(
                        parameter.numel() for parameter in active_readout_parameters(model)
                    ),
                    "dense_parameter_gap": dense_gap,
                    "encoder_hash": initialization["encoder_initialization_hash"],
                    "operator_hash": initialization.get(
                        "operator_initialization_hash", ""
                    ),
                    "pcsd_hash": initialization.get(
                        "pcsd_initialization_hash", ""
                    ),
                    "readout_hash": tensor_hash(active_readout_parameters(model)),
                    "pass": tuple(full.shape) == (1, 720, 1)
                    and tuple(prefix.shape) == (1, 336, 1)
                    and prefix_gap == 0.0
                    and (fixed_equivalence_gap == "" or fixed_equivalence_gap <= 2e-5)
                    and dense_gap
                    <= design["step7b_protocol"]["dense_parameter_gap_max"],
                }
            )
        a6 = models["a6"]
        m0 = models["pcsd_m0"]
        hidden_width = int(profile["state_width"])
        hidden = torch.randn(
            2,
            1,
            hidden_width,
            generator=torch.Generator().manual_seed(seed + 77 + hidden_width),
        )
        with torch.no_grad():
            a6_output = a6._learned_basis_forecast_operator(hidden, 720)
            m0_output = m0.pcsd_m0_readout(hidden, 720)
        a6_init = initialization_contract(a6)
        m0_init = initialization_contract(m0)
        pcsd_hashes = {
            initialization_contract(model).get("pcsd_initialization_hash", "")
            for arm, model in models.items()
            if arm.startswith("pcsd_")
            and arm != "pcsd_m0"
        }
        encoder_hashes = {
            initialization_contract(model)["encoder_initialization_hash"]
            for model in models.values()
        }
        mode_bound = hidden_width**-0.5
        direct = models["pcsd_direct"].pcsd_readout
        empirical_std = float(direct.mode_weight.std())
        expected_std = mode_bound / math.sqrt(3.0)
        pairing_rows.append(
            {
                "dataset": dataset,
                "a6_m0_output_max_abs": float((a6_output - m0_output).abs().max()),
                "a6_m0_operator_hash_equal": a6_init["operator_initialization_hash"]
                == m0_init["operator_initialization_hash"],
                "paired_encoder_hash_count": len(encoder_hashes),
                "paired_pcsd_hash_count": len(pcsd_hashes),
                "mode_weight_max_abs": float(direct.mode_weight.abs().max()),
                "mode_weight_bound": mode_bound,
                "mode_weight_std": empirical_std,
                "mode_weight_expected_std": expected_std,
                "pass": float((a6_output - m0_output).abs().max()) == 0.0
                and a6_init["operator_initialization_hash"]
                == m0_init["operator_initialization_hash"]
                and len(encoder_hashes) == 1
                and len(pcsd_hashes) == 1
                and float(direct.mode_weight.abs().max()) <= mode_bound
                and abs(empirical_std / expected_std - 1.0) <= 0.05,
            }
        )
        del models
    return rows, pairing_rows


def main() -> None:
    args = parse_args()
    design = json.loads(args.design.read_text(encoding="utf-8"))
    profiles = json.loads(args.profiles.read_text(encoding="utf-8"))[
        "dataset_profiles"
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_rows, pairing_rows = audit_matrix(design, profiles, args.seed)
    write_csv(args.output_dir / "arm_matrix_checks.csv", matrix_rows)
    write_csv(args.output_dir / "pairing_and_initialization_checks.csv", pairing_rows)
    protocol = design["step7b_protocol"]
    protocol_pass = bool(
        protocol["seed"] == args.seed
        and protocol["datasets"] == list(DATASETS)
        and protocol["final_evaluation_split"] == "val"
        and protocol["test_used"] is False
        and design["training"]["from_scratch"] is True
        and design["training"]["joint_encoder_decoder_training"] is True
        and design["training"]["frozen_replacement"] is False
        and design["training"]["warm_start"] is False
    )
    categories = {
        "sixty_arm_dataset_contracts": len(matrix_rows) == 60
        and all(row["pass"] for row in matrix_rows),
        "paired_initialization_and_m0": all(row["pass"] for row in pairing_rows),
        "validation_only_protocol": protocol_pass,
        "dense_capacity_match": max(
            float(row["dense_parameter_gap"])
            for row in matrix_rows
            if row["arm"] == "dense_matched"
        )
        <= protocol["dense_parameter_gap_max"],
    }
    result = {
        "candidate_id": design["candidate_id"],
        "diagnostic_id": design["diagnostic_id"],
        "current_step": "Step7B prelaunch gate",
        "expected_jobs": 60,
        "design_sha256": hashlib.sha256(args.design.read_bytes()).hexdigest(),
        "profile_sha256": hashlib.sha256(args.profiles.read_bytes()).hexdigest(),
        "categories": categories,
        "overall_pass": all(categories.values()),
        "test_used": False,
        "decision": (
            "step7b_prelaunch_pass_remote_seed2021_authorizable"
            if all(categories.values())
            else "step7b_prelaunch_fail_hold_remote"
        ),
    }
    (args.output_dir / "prelaunch_gate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = f"""# PCSD-CF Step7B Prelaunch Gate

- expected jobs: `60`
- overall pass: `{str(result['overall_pass']).lower()}`
- decision: `{result['decision']}`
- test used: `false`

本gate审计5个frozen profiles × 12 arms、A6/M0 exact paired initialization与output、Encoder/PCSD paired
initialization、按history width修正后的mode initialization、fixed-arm fast-path等价性、dense parameter matching
及validation-only protocol。它不训练数据，也不提供effectiveness evidence。
"""
    (args.output_dir / "prelaunch_gate_report.md").write_text(
        report,
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
