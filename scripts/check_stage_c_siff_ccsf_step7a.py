#!/usr/bin/env python3
"""Run the local implementation gate for SC1-SIFF-v2-CCSF Step 7A."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from layers.CCSF import (  # noqa: E402
    CCSFCouplingFieldReadout,
    CCSF_OBJECTIVE_MODES,
    ccsf_parameter_count,
    contrast_scope_calibration_loss,
)
from layers.PCC import (  # noqa: E402
    prefix_measure,
    projective_coupling_credit_loss,
)
from models import TimeAlign  # noqa: E402
import train_repo as training_adapter  # noqa: E402
from train_repo import initialization_contract  # noqa: E402


CHANNELS = {"ETTh1": 7, "ETTh2": 7, "ETTm1": 7, "ETTm2": 7, "Weather": 21}
CCSF_READOUT_CONTRACT = {
    "ccsf-coupling-field": (2, "ordered", "true"),
    "ccsf-no-contrast-control": (2, "ordered", "zero"),
    "ccsf-permuted-contrast-control": (2, "ordered", "permuted"),
    "ccsf-independent-scope-control": (5, "independent", "true"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_siff_ccsf_step7a.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/stage_c_siff_ccsf_step7a_20260718/local_gate"),
    )
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(named_tensors: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
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


def manifest_rows(
    config: dict[str, Any],
    profiles: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for dataset in config["datasets"]:
        profile = profiles[dataset]
        for arm in config["arms"]:
            rule = arm["rank_rule"]
            rank = (
                256
                if rule == "fixed_256"
                else config["matched_ranks"][dataset][rule]
            )
            rows.append(
                {
                    "job_index": len(rows) + 1,
                    "dataset": dataset,
                    "arm": arm["id"],
                    "readout_mode": arm["readout_mode"],
                    "objective_mode": arm["objective_mode"],
                    "correction_mode": arm["correction_mode"],
                    "mode_rank": rank,
                    "profile": profile["profile"],
                    "patch_num": profile["patch_num"],
                    "d_model": profile["d_model"],
                    "d_ff": profile["d_ff"],
                    "seed": config["seeds"][0],
                    "checkpoint_score": config["training"][
                        "validation_checkpoint_score"
                    ],
                    "formal_evaluation_split": "test",
                }
            )
    return rows


def training_argv(row: dict[str, Any], config: dict[str, Any]) -> list[str]:
    training = config["training"]
    return [
        "train_repo.py",
        "--dataset-root",
        "/home/yingch/dataset",
        "--dataset",
        row["dataset"],
        "--mode",
        "unified",
        "--seq-len",
        "720",
        "--pred-len",
        "720",
        "--target-horizons",
        "720",
        "--validation-horizons",
        "96,192,336,720",
        "--evaluation-horizons",
        "96,192,336,720",
        "--segment-horizons",
        "96,192,336,720",
        "--evaluation-prefix-mode",
        "full-crop",
        "--e-layers",
        "2",
        "--batch-size",
        str(training["batch_size"]),
        "--epochs",
        str(training["epochs"]),
        "--patience",
        str(training["patience"]),
        "--enable-early-stopping",
        "--seed",
        str(row["seed"]),
        "--run-name",
        f"CCSF_{row['arm']}",
        "--output-dir",
        f"/tmp/{row['arm']}_{row['dataset']}",
        "--device",
        "cuda",
        "--checkpoint-policy",
        "best-val",
        "--protocol-class",
        "method_screening",
        "--protocol-profile",
        "stage_c_siff_ccsf_v1",
        "--profile-hash",
        config["profiles"]["sha256"],
        "--legacy-patch-num",
        str(row["patch_num"]),
        "--legacy-d-model",
        str(row["d_model"]),
        "--legacy-d-ff",
        str(row["d_ff"]),
        "--legacy-dropout",
        "0.1",
        "--legacy-layer-norm",
        "1",
        "--learning-rate",
        str(training["learning_rate"]),
        "--readout-mode",
        row["readout_mode"],
        "--basis-rank",
        "256",
        "--pcsd-coordinate-dim",
        "4",
        "--pcsd-mode-rank",
        str(row["mode_rank"]),
        "--pcsd-policy-history-dim",
        "32",
        "--pcsd-policy-hidden-dim",
        "64",
        "--pcsd-policy-mode",
        "direct",
        "--pcsd-fixed-scale",
        "720",
        "--pcsd-partition",
        "canonical",
        "--pcsd-partition-seed",
        "15101",
        "--ccsf-correction-hidden-dim",
        "64",
        "--ccsf-calibration-temperature",
        str(config["objective"]["local_smoke_temperature"]),
        "--ccsf-calibration-weight",
        str(config["objective"]["calibration_weight"]),
        "--pcc-objective-mode",
        row["objective_mode"],
        "--pred-loss-mode",
        "full",
        "--final-evaluation-split",
        "val",
        "--no-save-predictions",
    ]


def cli_audit(rows: list[dict[str, Any]], config: dict[str, Any]) -> bool:
    original = sys.argv
    try:
        for row in rows:
            sys.argv = training_argv(row, config)
            parsed = training_adapter.parse_args()
            if not (
                parsed.dataset == row["dataset"]
                and parsed.readout_mode == row["readout_mode"]
                and parsed.pcc_objective_mode == row["objective_mode"]
                and parsed.pcsd_mode_rank == row["mode_rank"]
                and parsed.validation_horizons == [96, 192, 336, 720]
                and parsed.final_evaluation_split == "val"
                and parsed.evaluation_prefix_mode == "full-crop"
                and parsed.ccsf_calibration_temperature == 0.1
                and parsed.ccsf_calibration_weight == 0.1
            ):
                return False
    finally:
        sys.argv = original
    return True


def model_config(row: dict[str, Any], *, small: bool) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        encoder_mode="timealign-token-mlp",
        readout_mode=row["readout_mode"],
        e_layers=1 if small else 2,
        patch_num=2 if small else int(row["patch_num"]),
        d_model=4 if small else int(row["d_model"]),
        d_ff=8 if small else int(row["d_ff"]),
        dropout=0.0 if small else 0.1,
        pos=1,
        layer_norm=1,
        enc_in=2 if small else CHANNELS[row["dataset"]],
        basis_rank=8 if small else 256,
        pcsd_coordinate_dim=4,
        pcsd_mode_rank=8 if small else int(row["mode_rank"]),
        pcsd_policy_history_dim=4 if small else 32,
        pcsd_policy_hidden_dim=8 if small else 64,
        pcsd_policy_mode="direct",
        pcsd_fixed_scale=720,
        pcsd_partition="canonical",
        pcsd_partition_seed=15101,
        pcsd_group_chunk_size=64,
        pcsd_target_chunk_size=128,
        ccsf_correction_hidden_dim=64,
    )


def construction_audit(
    manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    representatives = {
        (row["dataset"], row["readout_mode"], row["mode_rank"]): row
        for row in manifest
    }
    rows = []
    for row in representatives.values():
        torch.manual_seed(int(row["seed"]))
        model = TimeAlign.Model(model_config(row, small=False)).float().eval()
        contract = initialization_contract(model)
        readout_pass = True
        correction_parameters = 0
        if row["readout_mode"] in CCSF_READOUT_CONTRACT:
            readout = model.pcsd_readout
            expected = CCSF_READOUT_CONTRACT[row["readout_mode"]]
            actual = (
                int(readout.scale_components),
                readout.scale_basis_mode,
                readout.correction_mode,
            )
            correction_parameters = int(readout.correction_parameters)
            readout_pass = actual == expected and correction_parameters == 2881
        rows.append(
            {
                "dataset": row["dataset"],
                "readout_mode": row["readout_mode"],
                "mode_rank": row["mode_rank"],
                "encoder_hash": contract["encoder_initialization_hash"],
                "correction_parameters": correction_parameters,
                "readout_contract_pass": readout_pass,
                "pass": readout_pass,
            }
        )
        del model
    return rows


def parameter_audit(
    config: dict[str, Any],
    profiles: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for dataset, profile in profiles.items():
        readout_dim = int(profile["patch_num"]) * int(profile["d_model"])
        ordered = ccsf_parameter_count(readout_dim, mode_rank=256)
        rank = int(
            config["matched_ranks"][dataset]["independent_dataset_matched"]
        )
        independent = ccsf_parameter_count(
            readout_dim,
            mode_rank=rank,
            scale_components=5,
        )
        relative_gap = abs(independent - ordered) / float(ordered)
        rows.append(
            {
                "dataset": dataset,
                "ordered_parameters": ordered,
                "independent_parameters": independent,
                "independent_rank": rank,
                "relative_gap": relative_gap,
                "threshold": 0.005,
                "pass": relative_gap <= 0.005,
            }
        )
    return rows


def base_parameter_hash(model: torch.nn.Module) -> str:
    tensors = [
        (name, parameter)
        for name, parameter in model.pcsd_readout.named_parameters()
        if not name.startswith(("correction_hidden.", "correction_output."))
    ]
    return tensor_hash(tensors)


def initialization_and_containment_audit() -> list[dict[str, Any]]:
    modes = [
        "siff-coupling-field",
        "ccsf-coupling-field",
        "ccsf-no-contrast-control",
        "ccsf-permuted-contrast-control",
    ]
    models = {}
    x = torch.randn(1, 720, 2)
    y = torch.zeros(1, 720, 2)
    for mode in modes:
        row = {
            "readout_mode": mode,
            "dataset": "ETTh2",
            "mode_rank": 8,
            "patch_num": 2,
            "d_model": 4,
            "d_ff": 8,
        }
        torch.manual_seed(2021)
        models[mode] = TimeAlign.Model(model_config(row, small=True)).eval()

    parent = models["siff-coupling-field"]
    with torch.no_grad():
        parent_output = parent(x, y, is_training=False, target_prefix=720)[0]
    parent_hash = base_parameter_hash(parent)
    rows = []
    for mode in modes[1:]:
        model = models[mode]
        with torch.no_grad():
            output = model(x, y, is_training=False, target_prefix=720)[0]
        gap = float((output - parent_output).abs().max())
        current_hash = base_parameter_hash(model)
        correction_zero = float(
            model.pcsd_readout.correction_output.weight.abs().max()
        ) == 0.0
        rows.append(
            {
                "readout_mode": mode,
                "base_hash_equal": current_hash == parent_hash,
                "initial_output_gap": gap,
                "correction_output_zero": correction_zero,
                "pass": bool(
                    current_hash == parent_hash
                    and gap == 0.0
                    and correction_zero
                ),
            }
        )
    return rows


def contrast_control_audit() -> list[dict[str, Any]]:
    torch.manual_seed(2021)
    true = CCSFCouplingFieldReadout(
        readout_dim=8,
        mode_rank=8,
        policy_history_dim=4,
        policy_hidden_dim=8,
        correction_mode="true",
    )
    zero = CCSFCouplingFieldReadout(
        readout_dim=8,
        mode_rank=8,
        policy_history_dim=4,
        policy_hidden_dim=8,
        correction_mode="zero",
    )
    permuted = CCSFCouplingFieldReadout(
        readout_dim=8,
        mode_rank=8,
        policy_history_dim=4,
        policy_hidden_dim=8,
        correction_mode="permuted",
    )
    true_state = true.state_dict()
    for control in (zero, permuted):
        compatible = {
            name: value
            for name, value in true_state.items()
            if name in control.state_dict()
            and control.state_dict()[name].shape == value.shape
            and name not in {
                "ccsf_contrast_permutation",
            }
        }
        control.load_state_dict(compatible, strict=False)
    with torch.no_grad():
        values = torch.linspace(
            -0.25,
            0.25,
            steps=true.correction_output.weight.numel(),
        ).view_as(true.correction_output.weight)
        for readout in (true, zero, permuted):
            readout.correction_output.weight.copy_(values)
            readout.correction_output.bias.zero_()
    hidden = torch.randn(2, 3, 8)
    arms = true.arm_forecasts(hidden)
    true_descriptor = true.contrast_descriptor(arms)
    zero_descriptor = zero.contrast_descriptor(arms)
    permuted_descriptor = permuted.contrast_descriptor(arms)
    expected_permuted = true_descriptor.index_select(
        3,
        true.ccsf_contrast_permutation,
    )
    true_correction = true.correction_logits(hidden, true_descriptor)
    zero_correction = zero.correction_logits(hidden, zero_descriptor)
    permuted_correction = permuted.correction_logits(
        hidden,
        permuted_descriptor,
    )
    differentiable_arms = arms.detach().requires_grad_(True)
    differentiable_descriptor = true.contrast_descriptor(differentiable_arms)
    true.correction_logits(hidden, differentiable_descriptor).square().mean().backward()
    contrast_arm_gradient = float(differentiable_arms.grad.norm())
    return [
        {
            "case": "descriptor_shape",
            "value": str(tuple(true_descriptor.shape)),
            "threshold": "(2,3,720,5,6)",
            "pass": tuple(true_descriptor.shape) == (2, 3, 720, 5, 6),
        },
        {
            "case": "true_contrast_nonzero",
            "value": float(true_descriptor.square().mean().sqrt()),
            "threshold": ">0",
            "pass": float(true_descriptor.square().mean()) > 0.0,
        },
        {
            "case": "zero_control_exact",
            "value": float(zero_descriptor.abs().max()),
            "threshold": 0.0,
            "pass": float(zero_descriptor.abs().max()) == 0.0,
        },
        {
            "case": "permutation_control_exact",
            "value": float((permuted_descriptor - expected_permuted).abs().max()),
            "threshold": 0.0,
            "pass": bool(
                torch.equal(
                    permuted.ccsf_contrast_permutation,
                    torch.tensor([1, 2, 3, 4, 0]),
                )
                and torch.equal(permuted_descriptor, expected_permuted)
            ),
        },
        {
            "case": "true_vs_zero_correction_contrast",
            "value": float(
                (true_correction - zero_correction).square().mean().sqrt()
            ),
            "threshold": ">0",
            "pass": bool(not torch.equal(true_correction, zero_correction)),
        },
        {
            "case": "true_vs_permuted_correction_contrast",
            "value": float(
                (true_correction - permuted_correction).square().mean().sqrt()
            ),
            "threshold": ">0",
            "pass": bool(not torch.equal(true_correction, permuted_correction)),
        },
        {
            "case": "contrast_to_arm_gradient",
            "value": contrast_arm_gradient,
            "threshold": ">0",
            "pass": contrast_arm_gradient > 0.0,
        },
    ]


def projectivity_audit() -> list[dict[str, Any]]:
    rows = []
    for mode in CCSF_READOUT_CONTRACT:
        row = {
            "readout_mode": mode,
            "dataset": "ETTh2",
            "mode_rank": 8,
            "patch_num": 2,
            "d_model": 4,
            "d_ff": 8,
        }
        torch.manual_seed(2021)
        model = TimeAlign.Model(model_config(row, small=True)).eval()
        x = torch.randn(1, 720, 2)
        y = torch.zeros(1, 720, 2)
        with torch.no_grad():
            full = model(x, y, is_training=False, target_prefix=720)[0]
            gaps = []
            for horizon in (1, 96, 192, 337, 720):
                prefix = model(
                    x,
                    y,
                    is_training=False,
                    target_prefix=horizon,
                )[0]
                gaps.append(float((prefix - full[:, :horizon]).abs().max()))
        gap = max(gaps)
        rows.append(
            {
                "readout_mode": mode,
                "max_prefix_gap": gap,
                "threshold": 1e-7,
                "pass": gap <= 1e-7,
            }
        )
    return rows


def objective_audit() -> list[dict[str, Any]]:
    torch.manual_seed(2021)
    target = torch.randn(2, 720, 3)
    arms = torch.randn(2, 3, 720, 5, requires_grad=True)
    policy_logits = torch.randn(2, 3, 720, 5, requires_grad=True)
    policy = torch.softmax(policy_logits, dim=-1)
    fused = (arms * policy).sum(dim=-1).permute(0, 2, 1)
    rows = []
    relative = contrast_scope_calibration_loss(
        fused,
        arms,
        policy,
        target,
        mode="ccsf_relative_calibration",
        progress=1.0,
        temperature=0.1,
    )
    standardized = contrast_scope_calibration_loss(
        fused,
        arms,
        policy,
        target,
        mode="ccsf_standardized_calibration",
        progress=1.0,
        temperature=0.1,
    )
    equal = projective_coupling_credit_loss(
        fused,
        arms,
        policy,
        target,
        mode="equal_skill",
        progress=1.0,
    )
    equal_gap = float(
        (
            relative.fused_loss
            + relative.skill_loss
            - equal.total_loss
        ).abs()
    )
    scaled = contrast_scope_calibration_loss(
        fused.detach() * 7.0,
        arms.detach() * 7.0,
        policy.detach(),
        target * 7.0,
        mode="ccsf_relative_calibration",
        progress=1.0,
        temperature=0.1,
    )
    scale_gap = float((relative.teacher - scaled.teacher).abs().max())
    tied_arms = target.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, 5)
    tied = contrast_scope_calibration_loss(
        target,
        tied_arms,
        policy.detach(),
        target,
        mode="ccsf_relative_calibration",
        progress=1.0,
        temperature=0.1,
    )
    relative.total_loss.backward()
    rows.extend(
        [
            {
                "case": "objective_modes_registered",
                "value": ",".join(sorted(CCSF_OBJECTIVE_MODES)),
                "threshold": 2,
                "pass": len(CCSF_OBJECTIVE_MODES) == 2,
            },
            {
                "case": "relative_finite",
                "value": float(relative.total_loss.detach()),
                "threshold": "finite",
                "pass": math.isfinite(float(relative.total_loss.detach())),
            },
            {
                "case": "standardized_finite",
                "value": float(standardized.total_loss.detach()),
                "threshold": "finite",
                "pass": math.isfinite(float(standardized.total_loss.detach())),
            },
            {
                "case": "equal_skill_equivalence",
                "value": equal_gap,
                "threshold": 1e-6,
                "pass": equal_gap <= 1e-6,
            },
            {
                "case": "relative_scale_invariance",
                "value": scale_gap,
                "threshold": 1e-6,
                "pass": scale_gap <= 1e-6,
            },
            {
                "case": "tie_confidence_zero",
                "value": float(tied.teacher_confidence.abs().max()),
                "threshold": 1e-7,
                "pass": float(tied.teacher_confidence.abs().max()) <= 1e-7,
            },
            {
                "case": "teacher_stop_gradient",
                "value": relative.teacher.requires_grad,
                "threshold": False,
                "pass": not relative.teacher.requires_grad,
            },
            {
                "case": "policy_gradient_nonzero",
                "value": float(policy_logits.grad.norm()),
                "threshold": ">0",
                "pass": float(policy_logits.grad.norm()) > 0.0,
            },
        ]
    )
    return rows


def arm_loss(
    row: dict[str, Any],
    output: torch.Tensor,
    target: torch.Tensor,
    details: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    mode = row["objective_mode"]
    if details is None:
        measure = prefix_measure(720, device=output.device, dtype=output.dtype)
        return (
            (output - target).abs() * measure.view(1, -1, 1)
        ).sum(dim=1).mean()
    arms = details["arm_forecasts"].permute(0, 1, 3, 2)
    if mode in CCSF_OBJECTIVE_MODES:
        return contrast_scope_calibration_loss(
            output,
            arms,
            details["policy"],
            target,
            mode=mode,
            progress=1.0,
            temperature=0.1,
        ).total_loss
    return projective_coupling_credit_loss(
        output,
        arms,
        details["policy"],
        target,
        mode=mode,
        progress=1.0,
    ).total_loss


def gradient_audit(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    representatives = {row["arm"]: row for row in manifest}
    rows = []
    for arm, row in representatives.items():
        torch.manual_seed(2021)
        model = TimeAlign.Model(model_config(row, small=True)).float().train()
        x = torch.randn(1, 720, 2)
        target = torch.randn(1, 720, 2)
        details = None
        if row["readout_mode"] in TimeAlign.COUPLING_READOUTS:
            output, _recon, _align, details = model(
                x,
                target,
                is_training=True,
                target_prefix=720,
                return_pcsd_training_details=True,
            )
        else:
            output = model(
                x,
                target,
                is_training=True,
                target_prefix=720,
            )[0]
        loss = arm_loss(row, output, target, details)
        loss.backward()
        encoder_norm = sum(
            float(parameter.grad.norm())
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
            and name.startswith(("patch_emb_x.", "encoder."))
        )
        readout_norm = sum(
            float(parameter.grad.norm())
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
            and name.startswith(("pcsd_readout.", "learned_"))
        )
        correction_output_norm = sum(
            float(parameter.grad.norm())
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
            and name.startswith("pcsd_readout.correction_output.")
        )
        correction_pass = (
            correction_output_norm > 0.0
            if row["readout_mode"] in CCSF_READOUT_CONTRACT
            else correction_output_norm == 0.0
        )
        rows.append(
            {
                "arm": arm,
                "loss": float(loss.detach()),
                "encoder_gradient_norm": encoder_norm,
                "readout_gradient_norm": readout_norm,
                "correction_output_gradient_norm": correction_output_norm,
                "pass": bool(
                    math.isfinite(float(loss.detach()))
                    and encoder_norm > 0.0
                    and readout_norm > 0.0
                    and correction_pass
                ),
            }
        )
        del model
    return rows


def correction_two_step_audit() -> list[dict[str, Any]]:
    rows = []
    for mode, (_components, _basis, correction_mode) in (
        CCSF_READOUT_CONTRACT.items()
    ):
        row = {
            "readout_mode": mode,
            "dataset": "ETTh2",
            "mode_rank": 8,
            "patch_num": 2,
            "d_model": 4,
            "d_ff": 8,
        }
        torch.manual_seed(2021)
        model = TimeAlign.Model(model_config(row, small=True)).float().train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        x = torch.randn(1, 720, 2)
        target = torch.randn(1, 720, 2)
        first_output, _recon, _align, first_details = model(
            x,
            target,
            is_training=True,
            target_prefix=720,
            return_pcsd_training_details=True,
        )
        first_loss = projective_coupling_credit_loss(
            first_output,
            first_details["arm_forecasts"].permute(0, 1, 3, 2),
            first_details["policy"],
            target,
            mode="equal_skill",
            progress=1.0,
        ).total_loss
        optimizer.zero_grad()
        first_loss.backward()
        first_output_grad = float(
            model.pcsd_readout.correction_output.weight.grad.norm()
        )
        optimizer.step()

        second_output, _recon, _align, second_details = model(
            x,
            target,
            is_training=True,
            target_prefix=720,
            return_pcsd_training_details=True,
        )
        second_loss = projective_coupling_credit_loss(
            second_output,
            second_details["arm_forecasts"].permute(0, 1, 3, 2),
            second_details["policy"],
            target,
            mode="equal_skill",
            progress=1.0,
        ).total_loss
        optimizer.zero_grad()
        second_loss.backward()
        hidden_gradient = model.pcsd_readout.correction_hidden.weight.grad
        hidden_grad_norm = float(hidden_gradient.norm())
        contrast_grad_norm = float(hidden_gradient[:, -6:].norm())
        correction_rms = float(
            second_details["correction_logits"].square().mean().sqrt()
        )
        contrast_gradient_pass = (
            contrast_grad_norm == 0.0
            if correction_mode == "zero"
            else contrast_grad_norm > 0.0
        )
        rows.append(
            {
                "readout_mode": mode,
                "correction_mode": correction_mode,
                "first_output_gradient_norm": first_output_grad,
                "second_hidden_gradient_norm": hidden_grad_norm,
                "second_contrast_column_gradient_norm": contrast_grad_norm,
                "second_correction_rms": correction_rms,
                "pass": bool(
                    first_output_grad > 0.0
                    and hidden_grad_norm > 0.0
                    and correction_rms > 0.0
                    and contrast_gradient_pass
                ),
            }
        )
        del model
    return rows


def diagnostic_tensor_audit() -> list[dict[str, Any]]:
    row = {
        "readout_mode": "ccsf-coupling-field",
        "dataset": "ETTh2",
        "mode_rank": 8,
        "patch_num": 2,
        "d_model": 4,
        "d_ff": 8,
    }
    torch.manual_seed(2021)
    model = TimeAlign.Model(model_config(row, small=True)).float().train()
    x = torch.randn(1, 720, 2)
    target = torch.randn(1, 720, 2)
    output, _recon, _align, details = model(
        x,
        target,
        is_training=True,
        target_prefix=720,
        return_pcsd_training_details=True,
    )
    expected = {
        "arm_forecasts": (1, 2, 5, 720),
        "policy": (1, 2, 720, 5),
        "base_logits": (1, 2, 720, 5),
        "base_policy": (1, 2, 720, 5),
        "contrast_descriptor": (1, 2, 720, 5, 6),
        "correction_logits": (1, 2, 720, 5),
    }
    rows = []
    for name, shape in expected.items():
        actual = tuple(details[name].shape)
        rows.append(
            {
                "tensor": name,
                "shape": str(actual),
                "expected": str(shape),
                "finite": bool(torch.isfinite(details[name]).all()),
                "pass": actual == shape and bool(torch.isfinite(details[name]).all()),
            }
        )
    rows.append(
        {
            "tensor": "output",
            "shape": str(tuple(output.shape)),
            "expected": "(1, 720, 2)",
            "finite": bool(torch.isfinite(output).all()),
            "pass": tuple(output.shape) == (1, 720, 2),
        }
    )
    return rows


def remote_runner_audit(config: Path) -> tuple[bool, bool]:
    environment = dict(os.environ)
    environment["CONFIG"] = str(config)
    dry = subprocess.run(
        ["bash", "scripts/remote/run_stage_c_siff_ccsf_v1.sh"],
        cwd=ROOT,
        env={**environment, "DRY_RUN": "1"},
        check=False,
        capture_output=True,
        text=True,
    )
    launch = subprocess.run(
        ["bash", "scripts/remote/run_stage_c_siff_ccsf_v1.sh"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    dry_jobs = sum(
        line.count("\t") == 3
        for line in dry.stdout.splitlines()
    )
    dry_pass = dry.returncode == 0 and dry_jobs == 50
    guard_pass = bool(
        launch.returncode == 3
        and "remote/test launch is not authorized" in launch.stderr
    )
    return dry_pass, guard_pass


def main() -> None:
    torch.set_num_threads(1)
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    design_path = Path(config["design_contract"]["path"])
    design = json.loads(design_path.read_text(encoding="utf-8"))
    profile_path = Path(config["profiles"]["path"])
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))[
        "dataset_profiles"
    ]
    manifest = manifest_rows(config, profiles)
    construction = construction_audit(manifest)
    parameters = parameter_audit(config, profiles)
    initialization = initialization_and_containment_audit()
    contrast = contrast_control_audit()
    projectivity = projectivity_audit()
    objectives = objective_audit()
    gradients = gradient_audit(manifest)
    two_step = correction_two_step_audit()
    tensors = diagnostic_tensor_audit()
    dry_run_pass, remote_guard_pass = remote_runner_audit(args.config)

    encoder_hashes: dict[str, set[str]] = {}
    for row in construction:
        encoder_hashes.setdefault(row["dataset"], set()).add(row["encoder_hash"])
    step6_arm_ids = {row["id"] for row in design["arms"]}
    step7a_arm_ids = {row["id"] for row in config["arms"]}
    categories = {
        "design_contract_hash": (
            file_hash(design_path) == config["design_contract"]["sha256"]
        ),
        "profile_hash": file_hash(profile_path) == config["profiles"]["sha256"],
        "arm_contract": step6_arm_ids == step7a_arm_ids,
        "matrix_size": len(manifest) == 50,
        "cli_contracts": cli_audit(manifest, config),
        "model_construction": all(row["pass"] for row in construction),
        "paired_encoder_initialization": all(
            len(values) == 1 for values in encoder_hashes.values()
        ),
        "parent_initialization_and_containment": all(
            row["pass"] for row in initialization
        ),
        "parameter_matching": all(row["pass"] for row in parameters),
        "contrast_control_semantics": all(row["pass"] for row in contrast),
        "prefix_projectivity": all(row["pass"] for row in projectivity),
        "objective_algebra": all(row["pass"] for row in objectives),
        "gradient_paths": all(row["pass"] for row in gradients),
        "two_step_correction_optimization": all(
            row["pass"] for row in two_step
        ),
        "diagnostic_tensor_contract": all(row["pass"] for row in tensors),
        "remote_manifest_dry_run": dry_run_pass,
        "remote_authorization_guard": remote_guard_pass,
        "authorization_boundary": bool(
            config["authorization"]["step7a_local_implementation_authorized"]
            and not config["authorization"][
                "validation_temperature_pilot_authorized"
            ]
            and not config["authorization"]["remote_training_authorized"]
            and not config["authorization"]["formal_test_access_authorized"]
        ),
    }
    overall_pass = all(categories.values())
    report = {
        "candidate_version": config["candidate_version"],
        "current_step": "Step7A local implementation gate",
        "categories": categories,
        "categories_passed": sum(categories.values()),
        "categories_total": len(categories),
        "jobs": len(manifest),
        "construction_cases": len(construction),
        "gradient_cases": len(gradients),
        "projectivity_cases": len(projectivity),
        "overall_pass": overall_pass,
        "next_step": (
            "Step7B prelaunch design and shared-temperature pilot authorization"
            if overall_pass
            else "return Step6 or repair exact Step7A implementation"
        ),
        "validation_temperature_pilot_authorized": False,
        "remote_training_authorized": False,
        "formal_test_access_authorized": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "jobs_seed2021.csv", manifest)
    write_csv(args.output_dir / "model_construction.csv", construction)
    write_csv(args.output_dir / "parameter_matching.csv", parameters)
    write_csv(args.output_dir / "parent_containment.csv", initialization)
    write_csv(args.output_dir / "contrast_control_cases.csv", contrast)
    write_csv(args.output_dir / "projectivity_cases.csv", projectivity)
    write_csv(args.output_dir / "objective_cases.csv", objectives)
    write_csv(args.output_dir / "gradient_paths.csv", gradients)
    write_csv(args.output_dir / "two_step_correction.csv", two_step)
    write_csv(args.output_dir / "diagnostic_tensors.csv", tensors)
    (args.output_dir / "local_gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not overall_pass:
        raise RuntimeError(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
