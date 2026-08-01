#!/usr/bin/env python3
"""Audit one frozen PCSD-CF checkpoint on sequential validation or test rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from data_provider.data_factory import data_provider  # noqa: E402
from models import TimeAlign  # noqa: E402
from train_repo import model_diagnostics  # noqa: E402


HORIZONS = (1, 48, 96, 192, 336, 720)
PREFIX_TOLERANCE = 2e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Optional output directory; defaults to run-dir.",
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=Path("configs/stage_c_pcsd_cf_native_direct.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-rows", type=int, default=256)
    parser.add_argument(
        "--evaluation-split",
        choices=("val", "test"),
        default="val",
    )
    parser.add_argument(
        "--test-audit-config",
        type=Path,
        default=Path("configs/stage_c_pcsd_cf_test_audit.json"),
    )
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def load_model(
    run_dir: Path,
    device: torch.device,
) -> tuple[TimeAlign.Model, dict[str, Any], SimpleNamespace]:
    config = json.loads(
        (run_dir / "effective_config.json").read_text(encoding="utf-8")
    )
    official = dict(config["official_args"])
    official["device"] = device
    official["use_gpu"] = device.type == "cuda"
    official_args = SimpleNamespace(**official)
    model = TimeAlign.Model(official_args).to(device).float()
    state = torch.load(
        run_dir / "checkpoint.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    return model, config, official_args


def sequential_loader(
    official_args: SimpleNamespace,
    split: str,
) -> DataLoader:
    evaluation_data, _loader = data_provider(official_args, split)
    return DataLoader(
        evaluation_data,
        batch_size=official_args.batch_size,
        shuffle=False,
        num_workers=official_args.num_workers,
        drop_last=False,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def expected_matrix_size(audit: dict[str, Any]) -> int:
    matrix = audit["matrix"]
    if "explicit_manifest_rows" in matrix:
        return int(matrix["explicit_manifest_rows"])
    datasets = audit.get("datasets", matrix.get("datasets", []))
    arms = audit.get(
        "arms",
        audit.get("effective_arms", matrix.get("arms", [])),
    )
    seeds = audit.get("seeds")
    if seeds is None:
        seeds = [matrix["seed"]] if "seed" in matrix else []
    return len(datasets) * len(arms) * len(seeds)


def test_audit_authorized(audit: dict[str, Any]) -> bool:
    authorization = audit["authorization"]
    if audit.get("candidate_version") == "SC1-PCSD-CF-v1":
        return bool(
            audit["status"] in {
                "authorized_prelaunch",
                "completed_test_fail_with_arm_headroom",
            }
            and audit["matrix"]["expected_runs"] == 60
            and authorization["user_authorized"] is True
            and authorization["checkpoint_retraining_allowed"] is False
            and authorization["checkpoint_selection"]
            == "historical-best-validation-h720-mse"
            and authorization["test_role"]
            == "primary-milestone-effectiveness-gate"
            and authorization["formal_test_access_count_for_version"] == 1
        )
    expected_runs = audit["matrix"].get(
        "expected_runs",
        audit["matrix"].get("phase_a_expected_runs"),
    )
    expected_checkpoint_selection = audit.get(
        "checkpoint_selection_contract",
        "best-validation-mean-mse-h96-h192-h336-h720",
    )
    accepted_test_roles = {
        "primary-mechanism-effectiveness-and-paper-benchmark",
        "primary-problem-existence-diagnostic",
        "test-tuned-hyperparameter-selection-and-paper-benchmark",
    }
    test_role = authorization.get("test_role")
    if test_role == "test-tuned-hyperparameter-selection-and-paper-benchmark":
        tuning_boundary_ok = bool(
            authorization.get(
                "per_dataset_aggregate_hyperparameter_tuning_allowed"
            )
            is True
            and authorization.get(
                "per_horizon_seed_metric_or_cell_tuning_allowed"
            )
            is False
        )
    else:
        tuning_boundary_ok = bool(
            authorization.get("per_dataset_horizon_or_cell_tuning_allowed")
            is False
        )
    return bool(
        audit.get("status") == "authorized_prelaunch"
        and expected_runs == expected_matrix_size(audit)
        and authorization.get("user_authorized") is True
        and authorization.get("authorization_date")
        and test_role in accepted_test_roles
        and authorization.get("checkpoint_selection")
        == expected_checkpoint_selection
        and authorization.get("checkpoint_retraining_allowed") is True
        and authorization.get("checkpoint_mutation_during_test_allowed") is False
        and tuning_boundary_ok
        and authorization.get("formal_test_access_count_for_version") == 1
    )


def bin_reduce(
    values: torch.Tensor,
    bins: list[dict[str, Any]],
    reduction: str,
) -> torch.Tensor:
    outputs = []
    for entry in bins:
        start, end = int(entry["start"]), int(entry["end"])
        chunk = values[..., start:end]
        if reduction == "mse":
            outputs.append(chunk.square().mean(dim=-1))
        elif reduction == "mae":
            outputs.append(chunk.abs().mean(dim=-1))
        elif reduction == "mean":
            outputs.append(chunk.mean(dim=-2))
        else:
            raise ValueError(f"unsupported reduction: {reduction}")
    return torch.stack(outputs, dim=-1 if reduction != "mean" else -2)


def prefix_audit(
    model: TimeAlign.Model,
    batch_x: torch.Tensor,
    target: torch.Tensor,
) -> tuple[list[dict[str, Any]], float]:
    rows = []
    with torch.no_grad():
        full = model(
            batch_x,
            target,
            is_training=False,
            target_prefix=720,
        )[0]
        for horizon in HORIZONS:
            prefix = model(
                batch_x,
                target,
                is_training=False,
                target_prefix=horizon,
            )[0]
            gap = float((prefix - full[:, :horizon]).abs().max())
            rows.append(
                {
                    "horizon": horizon,
                    "shape": list(prefix.shape),
                    "full_prefix_max_abs": gap,
                }
            )
    return rows, max(row["full_prefix_max_abs"] for row in rows)


def denormalized_arms(
    model: TimeAlign.Model,
    batch_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    memory = model.encode_history(batch_x)
    hidden = memory.flatten(start_dim=-2)
    _fused, arms, weights = model.pcsd_readout.forward_with_diagnostics(
        hidden,
        720,
    )
    outputs = []
    for scale_index in range(arms.shape[2]):
        normalized = arms[:, :, scale_index].permute(0, 2, 1)
        outputs.append(model.normalization_x(normalized, "denorm"))
    return torch.stack(outputs, dim=2), weights


def denormalized_ccsf_tensors(
    model: TimeAlign.Model,
    batch_x: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | None:
    """Return denormalized CCSF arms and full diagnostic tensors."""
    readout = model.pcsd_readout
    if not hasattr(readout, "forward_with_ccsf_diagnostics"):
        return None
    memory = model.encode_history(batch_x)
    hidden = memory.flatten(start_dim=-2)
    _fused, arms, _weights, tensors = (
        readout.forward_with_ccsf_diagnostics(hidden, 720)
    )
    outputs = []
    for scale_index in range(arms.shape[2]):
        normalized = arms[:, :, scale_index].permute(0, 2, 1)
        outputs.append(model.normalization_x(normalized, "denorm"))
    return torch.stack(outputs, dim=2), tensors


def denormalized_scale_component_contributions(
    model: TimeAlign.Model,
    batch_x: torch.Tensor,
) -> torch.Tensor | None:
    """Return leave-one-component-out deltas as ``[B,C,Q,T]``."""
    readout = model.pcsd_readout
    if not hasattr(readout, "component_ablation_forecasts"):
        return None
    memory = model.encode_history(batch_x)
    hidden = memory.flatten(start_dim=-2)
    full, ablated = readout.component_ablation_forecasts(hidden)
    full_denormalized = model.normalization_x(
        full.permute(0, 2, 1),
        "denorm",
    )
    ablated_outputs = []
    for component_index in range(ablated.shape[2]):
        normalized = ablated[:, :, component_index].permute(0, 2, 1)
        ablated_outputs.append(model.normalization_x(normalized, "denorm"))
    ablated_denormalized = torch.stack(ablated_outputs, dim=2)
    return (
        full_denormalized.unsqueeze(2) - ablated_denormalized
    ).permute(0, 3, 2, 1)


def implicit_frequency_tensors(
    model: TimeAlign.Model,
    batch_x: torch.Tensor,
) -> dict[str, torch.Tensor] | None:
    """Return the normalized D19 amplitude/phase tensors for attribution."""
    if not hasattr(model, "implicit_frequency_readout"):
        return None
    normalized_history = model.normalization_x(batch_x, "norm")
    memory = model._encode_normalized_history(normalized_history)
    hidden = memory.flatten(start_dim=-2)
    _forecast, tensors = model.implicit_frequency_readout.full_forecast(
        hidden,
        normalized_history,
    )
    return tensors


def compact_history_statistic_tensors(
    model: TimeAlign.Model,
    batch_x: torch.Tensor,
) -> dict[str, torch.Tensor] | None:
    """Return normalized D20 summary and coefficient contributions."""
    if not hasattr(model, "history_statistic_coeff"):
        return None
    normalized_history = model.normalization_x(batch_x, "norm")
    projection = model.history_statistic_projection.to(
        dtype=normalized_history.dtype
    )
    summary = torch.einsum(
        "btc,tq->bcq",
        normalized_history,
        projection,
    )
    coefficient = model.history_statistic_coeff(summary)
    prediction = torch.einsum(
        "tk,bck->btc",
        model.learned_temporal_basis.to(dtype=coefficient.dtype),
        coefficient,
    )
    return {
        "summary": summary,
        "coefficient": coefficient,
        "prediction_contribution": prediction,
    }


def fcmi_tensors(
    model: TimeAlign.Model,
    batch_x: torch.Tensor,
) -> dict[str, torch.Tensor] | None:
    """Return reduced FCMI health tensors without retaining full activations."""
    if not hasattr(model, "fcmi_readout"):
        return None
    normalized_history = model.normalization_x(batch_x, "norm")
    memory = model._encode_normalized_history(normalized_history)
    normalized_output, details = model.fcmi_readout(
        memory,
        target_prefix=720,
        return_details=True,
    )
    context = details["context"]
    main = details["main"]
    interaction = details["interaction"]
    attention = details["attention"]
    if attention is None:
        raise RuntimeError("FCMI diagnostic attention is missing")
    attention_safe = attention.clamp_min(1e-12)
    entropy = -(attention_safe * attention_safe.log()).sum(dim=-1)
    if attention.shape[-1] > 1:
        entropy = entropy / math.log(attention.shape[-1])
    else:
        entropy = torch.zeros_like(entropy)
    payload = {
        "context_coordinate_std": context.std(
            dim=1,
            unbiased=False,
        ).mean(dim=-1),
        "main_rms": main.square().mean(dim=(-2, -1)).sqrt(),
        "interaction_rms": interaction.square().mean(dim=(-2, -1)).sqrt(),
        "attention_entropy": entropy.mean(dim=1),
        "attention_target_dispersion": attention.std(
            dim=1,
            unbiased=False,
        ).mean(dim=-1),
        "normalized_output": normalized_output.permute(0, 2, 1).reshape(
            -1,
            720,
        ),
    }
    if model.readout_mode == "fcmi":
        main_state = model.fcmi_readout.main_projection(main)
        main_state = main_state + details["query"]
        main_output = model.fcmi_readout.output_projection(
            torch.nn.functional.gelu(main_state)
        ).squeeze(-1)
        full_output = normalized_output.permute(0, 2, 1).reshape(-1, 720)
        payload["main_only_output"] = main_output
        payload["interaction_prediction_contribution"] = (
            full_output - main_output
        )
    if model.readout_mode == "fcmi-dense-capacity-matched":
        dense_residual = details["dense_residual"]
        if dense_residual is None:
            raise RuntimeError("FCMI dense residual diagnostic is missing")
        payload["dense_residual"] = dense_residual
    return payload


def cpsi_tensors(
    model: TimeAlign.Model,
    batch_x: torch.Tensor,
) -> dict[str, torch.Tensor] | None:
    """Return reduced CPSI interaction health tensors per series row."""
    readout = getattr(model, "pcsd_readout", None)
    if readout is None or not hasattr(readout, "interaction_diagnostics"):
        return None
    memory = model.encode_history(batch_x)
    hidden = memory.flatten(start_dim=-2)
    tensors = readout.interaction_diagnostics(hidden)
    payload = {}
    for name in ("common", "private", "left", "right", "latent", "message"):
        value = tensors[name]
        payload[name] = value.square().mean(dim=(-2, -1)).sqrt()
    return payload


def sps_tensors(
    model: TimeAlign.Model,
    batch_x: torch.Tensor,
) -> dict[str, torch.Tensor] | None:
    """Return normalized raw/projected/removed SPS arm diagnostics."""
    readout = getattr(model, "pcsd_readout", None)
    if readout is None or not hasattr(readout, "projection_diagnostics"):
        return None
    memory = model.encode_history(batch_x)
    hidden = memory.flatten(start_dim=-2)
    tensors = readout.projection_diagnostics(hidden)
    return {
        name: value
        for name, value in tensors.items()
        if name in {"raw_arms", "projected_arms", "removed_arms"}
    }


def diagnostic_bins(design: dict[str, Any]) -> list[dict[str, Any]]:
    if "diagnostic_protocol" in design:
        return design["diagnostic_protocol"]["future_bins"]
    return design["step7b_protocol"]["future_bins"]


def protocol_training_contracts(
    protocol_config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if protocol_config is None:
        return [
            {
                "target_horizons": [720],
                "validation_horizons": [720],
                "pred_loss_mode": "full",
                "pcc_objective_mode": None,
                "training_final_evaluation_split": "val",
            }
        ]
    if "training_contracts" in protocol_config:
        return list(protocol_config["training_contracts"])
    training = protocol_config.get("training", {})
    return [
        {
            "target_horizons": training.get("target_horizons", [720]),
            "validation_horizons": training.get(
                "validation_horizons",
                [720],
            ),
            "pred_loss_mode": training.get("pred_loss_mode", "full"),
            "pcc_objective_mode": None,
            "training_final_evaluation_split": training.get(
                "training_final_evaluation_split",
                "val",
            ),
        }
    ]


def adapter_matches_training_contract(
    adapter: dict[str, Any],
    contract: dict[str, Any],
) -> bool:
    objective = contract.get("pcc_objective_mode")
    return bool(
        adapter["target_horizons"] == contract["target_horizons"]
        and adapter["validation_horizons"]
        == contract["validation_horizons"]
        and adapter["pred_loss_mode"] == contract["pred_loss_mode"]
        and adapter["final_evaluation_split"]
        == contract["training_final_evaluation_split"]
        and (
            objective is None
            or adapter["pcc_objective_mode"] == objective
        )
    )


def evaluate(args: argparse.Namespace) -> None:
    if args.run_dir is None:
        raise ValueError("run-dir is required outside synthetic smoke")
    if args.probe_rows <= 0:
        raise ValueError("probe_rows must be positive")
    protocol_config = None
    if args.test_audit_config.is_file():
        protocol_config = json.loads(
            args.test_audit_config.read_text(encoding="utf-8")
        )
    test_audit = None
    if args.evaluation_split == "test":
        test_audit = protocol_config
        if test_audit is None:
            raise FileNotFoundError(args.test_audit_config)
        if not test_audit_authorized(test_audit):
            raise PermissionError(
                "test audit authorization failed before test loader access"
            )
    artifact_dir = args.artifact_dir or args.run_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    design = json.loads(args.design.read_text(encoding="utf-8"))
    bins = diagnostic_bins(design)
    model, config, official_args = load_model(args.run_dir, device)
    loader = sequential_loader(official_args, args.evaluation_split)
    adapter = config["adapter"]
    training_contract = config["training_contract"]
    initialization = json.loads(
        (args.run_dir / "initialization_contract.json").read_text(
            encoding="utf-8"
        )
    )
    diagnostics = model_diagnostics(model)

    fused_bin_mse: list[np.ndarray] = []
    fused_bin_mae: list[np.ndarray] = []
    persistence_bin_mse: list[np.ndarray] = []
    arm_bin_mse: list[np.ndarray] = []
    arm_bin_mae: list[np.ndarray] = []
    policy_bin_usage: list[np.ndarray] = []
    probe_arms: list[np.ndarray] = []
    probe_fused: list[np.ndarray] = []
    probe_targets: list[np.ndarray] = []
    scale_component_contribution: list[np.ndarray] = []
    probe_policy: list[np.ndarray] = []
    probe_direct_policy: list[np.ndarray] = []
    probe_base_policy: list[np.ndarray] = []
    probe_base_logits: list[np.ndarray] = []
    probe_correction_logits: list[np.ndarray] = []
    probe_contrast_descriptor: list[np.ndarray] = []
    probe_if_amplitude: list[np.ndarray] = []
    probe_if_phase_sine: list[np.ndarray] = []
    probe_if_phase_cosine: list[np.ndarray] = []
    probe_history_summary: list[np.ndarray] = []
    probe_history_coefficient: list[np.ndarray] = []
    probe_history_prediction_contribution: list[np.ndarray] = []
    probe_fcmi_context_coordinate_std: list[np.ndarray] = []
    probe_fcmi_main_rms: list[np.ndarray] = []
    probe_fcmi_interaction_rms: list[np.ndarray] = []
    probe_fcmi_attention_entropy: list[np.ndarray] = []
    probe_fcmi_attention_target_dispersion: list[np.ndarray] = []
    probe_fcmi_main_only_output: list[np.ndarray] = []
    probe_fcmi_interaction_contribution: list[np.ndarray] = []
    probe_fcmi_dense_residual: list[np.ndarray] = []
    probe_cpsi: dict[str, list[np.ndarray]] = {
        name: []
        for name in ("common", "private", "left", "right", "latent", "message")
    }
    probe_sps: dict[str, list[np.ndarray]] = {
        name: [] for name in ("raw_arms", "projected_arms", "removed_arms")
    }
    probe_count = 0
    all_finite = True
    prefix_rows: list[dict[str, Any]] | None = None
    prefix_gap = 0.0
    step_squared_error = torch.zeros(720, dtype=torch.float64)
    step_absolute_error = torch.zeros(720, dtype=torch.float64)
    element_rows = 0

    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            target = batch_y[:, -720:, :].float().to(device)
            if prefix_rows is None:
                prefix_rows, prefix_gap = prefix_audit(model, batch_x[:1], target[:1])
            fused = model(batch_x, target, is_training=False, target_prefix=720)[0]
            probe_batch_count = 0
            if probe_count < args.probe_rows:
                fused_probe_rows = fused.permute(0, 2, 1).reshape(-1, 720)
                target_probe_rows = target.permute(0, 2, 1).reshape(-1, 720)
                probe_batch_count = min(
                    args.probe_rows - probe_count,
                    fused_probe_rows.shape[0],
                )
                probe_fused.append(
                    fused_probe_rows[:probe_batch_count].cpu().numpy()
                )
                probe_targets.append(
                    target_probe_rows[:probe_batch_count].cpu().numpy()
                )
            errors = (fused - target).detach().to(torch.float64).cpu()
            step_squared_error += errors.square().sum(dim=(0, 2))
            step_absolute_error += errors.abs().sum(dim=(0, 2))
            element_rows += int(errors.shape[0] * errors.shape[2])
            fused_rows = (fused - target).permute(0, 2, 1).reshape(-1, 720)
            persistence = batch_x[:, -1:, :].expand_as(target)
            persistence_rows = (
                persistence - target
            ).permute(0, 2, 1).reshape(-1, 720)
            fused_bin_mse.append(
                bin_reduce(fused_rows, bins, "mse").cpu().numpy()
            )
            fused_bin_mae.append(
                bin_reduce(fused_rows, bins, "mae").cpu().numpy()
            )
            persistence_bin_mse.append(
                bin_reduce(persistence_rows, bins, "mse").cpu().numpy()
            )

            if hasattr(model, "pcsd_readout"):
                ccsf_tensors = denormalized_ccsf_tensors(model, batch_x)
                if ccsf_tensors is None:
                    arms, weights = denormalized_arms(model, batch_x)
                    policy_tensors = None
                else:
                    arms, policy_tensors = ccsf_tensors
                    weights = policy_tensors["policy"]
                arm_errors = arms - target.unsqueeze(2)
                arm_rows = arm_errors.permute(0, 3, 2, 1).reshape(
                    -1,
                    arms.shape[2],
                    720,
                )
                arm_bin_mse.append(
                    torch.stack(
                        [
                            arm_rows[..., int(entry["start"]): int(entry["end"])]
                            .square()
                            .mean(dim=-1)
                            for entry in bins
                        ],
                        dim=1,
                    ).cpu().numpy()
                )
                arm_bin_mae.append(
                    torch.stack(
                        [
                            arm_rows[..., int(entry["start"]): int(entry["end"])]
                            .abs()
                            .mean(dim=-1)
                            for entry in bins
                        ],
                        dim=1,
                    ).cpu().numpy()
                )
                weight_rows = weights.reshape(-1, 720, weights.shape[-1])
                policy_bin_usage.append(
                    torch.stack(
                        [
                            weight_rows[
                                :, int(entry["start"]): int(entry["end"])
                            ].mean(dim=1)
                            for entry in bins
                        ],
                        dim=1,
                    ).cpu().numpy()
                )
                if probe_batch_count:
                    count = probe_batch_count
                    probe_arms.append(
                        arms.permute(0, 3, 2, 1)
                        .reshape(-1, arms.shape[2], 720)[:count]
                        .cpu()
                        .numpy()
                    )
                    if policy_tensors is not None:
                        row_shape = (-1, 720, weights.shape[-1])
                        probe_policy.append(
                            policy_tensors["policy"]
                            .reshape(row_shape)[:count]
                            .cpu()
                            .numpy()
                        )
                        probe_base_policy.append(
                            policy_tensors["base_policy"]
                            .reshape(row_shape)[:count]
                            .cpu()
                            .numpy()
                        )
                        probe_base_logits.append(
                            policy_tensors["base_logits"]
                            .reshape(row_shape)[:count]
                            .cpu()
                            .numpy()
                        )
                        probe_correction_logits.append(
                            policy_tensors["correction_logits"]
                            .reshape(row_shape)[:count]
                            .cpu()
                            .numpy()
                        )
                        probe_contrast_descriptor.append(
                            policy_tensors["contrast_descriptor"].reshape(
                                -1,
                                720,
                                weights.shape[-1],
                                policy_tensors["contrast_descriptor"].shape[-1],
                            )[:count]
                            .cpu()
                            .numpy()
                        )
                    else:
                        probe_direct_policy.append(
                            weights.reshape(-1, 720, weights.shape[-1])[:count]
                            .cpu()
                            .numpy()
                        )
                    component_delta = None
                    if (
                        adapter["readout_mode"] == "siff-coupling-field"
                        and adapter["pcc_objective_mode"] == "equal_skill"
                    ):
                        component_delta = denormalized_scale_component_contributions(
                            model,
                            batch_x,
                        )
                    if component_delta is not None:
                        scale_component_contribution.append(
                            component_delta.reshape(
                                -1,
                                component_delta.shape[2],
                                720,
                            )[:count]
                            .cpu()
                            .numpy()
                        )
                        all_finite = all_finite and bool(
                            torch.isfinite(component_delta).all()
                        )
                all_finite = all_finite and bool(
                    torch.isfinite(arms).all() and torch.isfinite(weights).all()
                )
                if policy_tensors is not None:
                    all_finite = all_finite and all(
                        bool(torch.isfinite(value).all())
                        for value in policy_tensors.values()
                    )
            if probe_batch_count:
                if_tensors = implicit_frequency_tensors(model, batch_x)
                if if_tensors is not None:
                    row_shape = (-1, if_tensors["amplitude"].shape[-1])
                    probe_if_amplitude.append(
                        if_tensors["amplitude"]
                        .reshape(row_shape)[:probe_batch_count]
                        .cpu()
                        .numpy()
                    )
                    probe_if_phase_sine.append(
                        if_tensors["phase_sine"]
                        .reshape(row_shape)[:probe_batch_count]
                        .cpu()
                        .numpy()
                    )
                    probe_if_phase_cosine.append(
                        if_tensors["phase_cosine"]
                        .reshape(row_shape)[:probe_batch_count]
                        .cpu()
                        .numpy()
                    )
                    all_finite = all_finite and all(
                        bool(torch.isfinite(value).all())
                        for value in if_tensors.values()
                    )
                history_tensors = compact_history_statistic_tensors(
                    model,
                    batch_x,
                )
                if history_tensors is not None:
                    count = probe_batch_count
                    probe_history_summary.append(
                        history_tensors["summary"]
                        .reshape(-1, history_tensors["summary"].shape[-1])[
                            :count
                        ]
                        .cpu()
                        .numpy()
                    )
                    probe_history_coefficient.append(
                        history_tensors["coefficient"]
                        .reshape(
                            -1,
                            history_tensors["coefficient"].shape[-1],
                        )[:count]
                        .cpu()
                        .numpy()
                    )
                    probe_history_prediction_contribution.append(
                        history_tensors["prediction_contribution"]
                        .permute(0, 2, 1)
                        .reshape(-1, 720)[:count]
                        .cpu()
                        .numpy()
                    )
                    all_finite = all_finite and all(
                        bool(torch.isfinite(value).all())
                        for value in history_tensors.values()
                    )
                fcmi_health = fcmi_tensors(model, batch_x)
                if fcmi_health is not None:
                    count = probe_batch_count
                    probe_fcmi_context_coordinate_std.append(
                        fcmi_health["context_coordinate_std"][:count]
                        .cpu()
                        .numpy()
                    )
                    probe_fcmi_main_rms.append(
                        fcmi_health["main_rms"][:count].cpu().numpy()
                    )
                    probe_fcmi_interaction_rms.append(
                        fcmi_health["interaction_rms"][:count].cpu().numpy()
                    )
                    probe_fcmi_attention_entropy.append(
                        fcmi_health["attention_entropy"][:count].cpu().numpy()
                    )
                    probe_fcmi_attention_target_dispersion.append(
                        fcmi_health["attention_target_dispersion"][:count]
                        .cpu()
                        .numpy()
                    )
                    if "main_only_output" in fcmi_health:
                        probe_fcmi_main_only_output.append(
                            fcmi_health["main_only_output"][:count]
                            .cpu()
                            .numpy()
                        )
                        probe_fcmi_interaction_contribution.append(
                            fcmi_health[
                                "interaction_prediction_contribution"
                            ][:count]
                            .cpu()
                            .numpy()
                        )
                    if "dense_residual" in fcmi_health:
                        probe_fcmi_dense_residual.append(
                            fcmi_health["dense_residual"][:count]
                            .cpu()
                            .numpy()
                        )
                    all_finite = all_finite and all(
                        bool(torch.isfinite(value).all())
                        for value in fcmi_health.values()
                    )
                cpsi_health = cpsi_tensors(model, batch_x)
                if cpsi_health is not None:
                    count = probe_batch_count
                    for name, value in cpsi_health.items():
                        probe_cpsi[name].append(
                            value.reshape(-1)[:count].cpu().numpy()
                        )
                    all_finite = all_finite and all(
                        bool(torch.isfinite(value).all())
                        for value in cpsi_health.values()
                    )
                sps_health = sps_tensors(model, batch_x)
                if sps_health is not None:
                    count = probe_batch_count
                    for name, value in sps_health.items():
                        probe_sps[name].append(
                            value.permute(0, 1, 2, 3)
                            .reshape(-1, value.shape[2], value.shape[3])[:count]
                            .cpu()
                            .numpy()
                        )
                    all_finite = all_finite and all(
                        bool(torch.isfinite(value).all())
                        for value in sps_health.values()
                    )
            all_finite = all_finite and bool(
                torch.isfinite(fused).all() and torch.isfinite(target).all()
            )
            probe_count += probe_batch_count

    if not fused_bin_mse or prefix_rows is None or element_rows <= 0:
        raise RuntimeError(
            f"{args.evaluation_split} evaluation produced no rows"
        )
    payload: dict[str, np.ndarray] = {
        "fused_row_bin_mse": np.concatenate(fused_bin_mse).astype(np.float32),
        "fused_row_bin_mae": np.concatenate(fused_bin_mae).astype(np.float32),
        "persistence_row_bin_mse": np.concatenate(persistence_bin_mse).astype(
            np.float32
        ),
        "bin_names": np.asarray([entry["name"] for entry in bins]),
        "scales": np.asarray(
            design.get("coupling_scales", []),
            dtype=np.int64,
        ),
    }
    if probe_fused:
        payload.update(
            {
                "probe_fused": np.concatenate(probe_fused).astype(np.float32),
                "probe_targets": np.concatenate(probe_targets).astype(
                    np.float32
                ),
            }
        )
    if arm_bin_mse:
        payload.update(
            {
                "arm_row_bin_mse": np.concatenate(arm_bin_mse).astype(
                    np.float32
                ),
                "arm_row_bin_mae": np.concatenate(arm_bin_mae).astype(
                    np.float32
                ),
                "policy_row_bin_usage": np.concatenate(policy_bin_usage).astype(
                    np.float32
                ),
                "probe_arms": np.concatenate(probe_arms).astype(np.float32),
            }
        )
        if scale_component_contribution:
            payload["scale_component_contribution"] = np.concatenate(
                scale_component_contribution
            ).astype(np.float32)
        if probe_policy:
            payload.update(
                {
                    "probe_policy": np.concatenate(probe_policy).astype(
                        np.float32
                    ),
                    "probe_base_policy": np.concatenate(
                        probe_base_policy
                    ).astype(np.float32),
                    "probe_base_logits": np.concatenate(
                        probe_base_logits
                    ).astype(np.float32),
                    "probe_correction_logits": np.concatenate(
                        probe_correction_logits
                    ).astype(np.float32),
                    "probe_contrast_descriptor": np.concatenate(
                        probe_contrast_descriptor
                    ).astype(np.float32),
                }
            )
        if probe_direct_policy:
            payload["probe_direct_policy"] = np.concatenate(
                probe_direct_policy
            ).astype(np.float32)
    if probe_if_amplitude:
        payload.update(
            {
                "probe_if_amplitude": np.concatenate(
                    probe_if_amplitude
                ).astype(np.float32),
                "probe_if_phase_sine": np.concatenate(
                    probe_if_phase_sine
                ).astype(np.float32),
                "probe_if_phase_cosine": np.concatenate(
                    probe_if_phase_cosine
                ).astype(np.float32),
            }
        )
    if probe_history_summary:
        payload.update(
            {
                "probe_history_summary": np.concatenate(
                    probe_history_summary
                ).astype(np.float32),
                "probe_history_coefficient": np.concatenate(
                    probe_history_coefficient
                ).astype(np.float32),
                "probe_history_prediction_contribution": np.concatenate(
                    probe_history_prediction_contribution
                ).astype(np.float32),
            }
        )
    if probe_fcmi_context_coordinate_std:
        payload.update(
            {
                "probe_fcmi_context_coordinate_std": np.concatenate(
                    probe_fcmi_context_coordinate_std
                ).astype(np.float32),
                "probe_fcmi_main_rms": np.concatenate(
                    probe_fcmi_main_rms
                ).astype(np.float32),
                "probe_fcmi_interaction_rms": np.concatenate(
                    probe_fcmi_interaction_rms
                ).astype(np.float32),
                "probe_fcmi_attention_entropy": np.concatenate(
                    probe_fcmi_attention_entropy
                ).astype(np.float32),
                "probe_fcmi_attention_target_dispersion": np.concatenate(
                    probe_fcmi_attention_target_dispersion
                ).astype(np.float32),
            }
        )
        if probe_fcmi_main_only_output:
            payload["probe_fcmi_main_only_output"] = np.concatenate(
                probe_fcmi_main_only_output
            ).astype(np.float32)
            payload[
                "probe_fcmi_interaction_prediction_contribution"
            ] = np.concatenate(
                probe_fcmi_interaction_contribution
            ).astype(np.float32)
        if probe_fcmi_dense_residual:
            payload["probe_fcmi_dense_residual"] = np.concatenate(
                probe_fcmi_dense_residual
            ).astype(np.float32)
    if probe_cpsi["common"]:
        for name, values in probe_cpsi.items():
            payload[f"probe_cpsi_{name}_rms"] = np.concatenate(values).astype(
                np.float32
            )
    if probe_sps["raw_arms"]:
        for name, values in probe_sps.items():
            payload[f"probe_sps_{name}"] = np.concatenate(values).astype(
                np.float32
            )
    artifact_prefix = (
        "validation" if args.evaluation_split == "val" else "test_audit"
    )
    np.savez_compressed(
        artifact_dir / f"pcsd_{artifact_prefix}_diagnostics.npz",
        **payload,
    )

    if args.evaluation_split == "test":
        cumulative_squared = torch.cumsum(step_squared_error, dim=0)
        cumulative_absolute = torch.cumsum(step_absolute_error, dim=0)
        metric_rows = []
        for horizon in range(1, 721):
            denominator = float(element_rows * horizon)
            metric_rows.append(
                {
                    "target_horizon": horizon,
                    "mse": float(cumulative_squared[horizon - 1] / denominator),
                    "mae": float(cumulative_absolute[horizon - 1] / denominator),
                    "num_rows_channels": element_rows,
                    "evaluation_split": "test",
                    "checkpoint_policy": adapter["checkpoint_policy"],
                    "candidate_version": test_audit["candidate_version"],
                    "hyperparameter_trial_id": adapter.get("hpo_trial_id"),
                    "hyperparameter_profile_id": adapter.get("hpo_profile_id"),
                    "seed": adapter["seed"],
                }
            )
        write_csv(
            artifact_dir / "test_audit_metrics_by_target_horizon.csv",
            metric_rows,
        )

    test_authorized = (
        args.evaluation_split == "val"
        or test_audit is not None
        and test_audit_authorized(test_audit)
    )
    training_contracts = protocol_training_contracts(protocol_config)
    matched_training_contract = any(
        adapter_matches_training_contract(adapter, contract)
        for contract in training_contracts
    )

    protocol_pass = bool(
        adapter["mode"] == "unified"
        and int(adapter["pred_len"]) == 720
        and adapter["checkpoint_policy"] == "best-val"
        and adapter["protocol_class"] == "method_screening"
        and adapter["evaluation_prefix_mode"] == "full-crop"
        and matched_training_contract
        and training_contract["initialization"] == "from_scratch"
        and training_contract["checkpoint_input"] is None
        and diagnostics["frozen_parameter_tensors"] == 0
        and test_authorized
    )
    readout_contract_pass = True
    if (
        hasattr(model, "pcsd_readout")
        and hasattr(model.pcsd_readout, "interaction_diagnostics")
    ):
        readout_contract_pass = bool(
            initialization.get("cpsi_parent_initialization_hash")
            and initialization.get("cpsi_input_initialization_hash")
            and diagnostics.get("cpsi_interaction_rank") == 32
            and diagnostics.get("cpsi_interaction_parameters", 0) > 0
            and all(
                f"probe_cpsi_{name}_rms" in payload
                for name in (
                    "common",
                    "private",
                    "left",
                    "right",
                    "latent",
                    "message",
                )
            )
        )
    elif hasattr(model, "pcsd_readout"):
        readout_contract_pass = bool(
            initialization.get("pcsd_initialization_hash")
            and initialization.get("pcsd_coordinate_hash")
            and initialization.get("pcsd_partition_hash")
            and diagnostics.get("pcsd_scales") == design["coupling_scales"]
            and diagnostics.get("pcsd_policy_mode")
            == adapter["pcsd_policy_mode"]
            and diagnostics.get("pcsd_partition") == adapter["pcsd_partition"]
        )
        if hasattr(model.pcsd_readout, "projection_diagnostics"):
            readout_contract_pass = bool(
                readout_contract_pass
                and diagnostics.get("sps_projection_mode")
                == adapter.get("sps_projection_mode")
                and diagnostics.get("sps_projection_ranks")
                and all(
                    f"probe_sps_{name}" in payload
                    for name in ("raw_arms", "projected_arms", "removed_arms")
                )
                and "probe_direct_policy" in payload
            )
            if hasattr(model.pcsd_readout, "conditioning_strength"):
                strength = float(adapter.get("frsc_conditioning_strength", math.nan))
                minimum_eigenvalue = float(
                    diagnostics.get(
                        "frsc_minimum_operator_eigenvalue",
                        math.nan,
                    )
                )
                readout_contract_pass = bool(
                    readout_contract_pass
                    and math.isclose(
                        float(model.pcsd_readout.conditioning_strength),
                        strength,
                        abs_tol=1e-12,
                    )
                    and math.isclose(
                        minimum_eigenvalue,
                        1.0 - strength,
                        abs_tol=1e-12,
                    )
                    and minimum_eigenvalue > 0.0
                    and diagnostics.get("frsc_full_rank") is True
                )
    elif hasattr(model, "pcsd_m0_readout"):
        readout_contract_pass = bool(
            initialization.get("pcsd_m0_initialization_hash")
            and initialization.get("operator_initialization_hash")
            and diagnostics.get("pcsd_m0_mode_rank") == 256
        )
    elif hasattr(model, "pcsd_dense_readout"):
        dense_gap_limit = design.get("gates", {}).get(
            "dense_parameter_gap_max",
            design.get("step7b_protocol", {}).get(
                "dense_parameter_gap_max",
                0.005,
            ),
        )
        readout_contract_pass = bool(
            initialization.get("pcsd_dense_initialization_hash")
            and diagnostics.get("pcsd_dense_parameter_relative_gap", 1.0)
            <= dense_gap_limit
        )
    elif hasattr(model, "implicit_frequency_readout"):
        implicit = design["implicit_forecaster"]
        expect_input_spectrum = (
            adapter["readout_mode"] == "implicit-frequency-readout"
        )
        readout_contract_pass = bool(
            initialization.get("implicit_frequency_initialization_hash")
            and initialization.get("implicit_frequency_use_input_spectrum")
            is expect_input_spectrum
            and diagnostics.get("implicit_decoder_parameters", 0) > 0
            and diagnostics.get("implicit_hidden_width")
            == implicit["hidden_width"]
            and diagnostics.get("implicit_history_spectrum_bins")
            == implicit["history_spectrum_bins"]
            and diagnostics.get("implicit_spectrum_bins")
            == implicit["spectrum_bins"]
            and diagnostics.get("implicit_fourier_norm")
            == implicit["fourier_norm"]
            and "probe_if_amplitude" in payload
        )
    elif hasattr(model, "implicit_direct_readout"):
        implicit = design["implicit_forecaster"]
        readout_contract_pass = bool(
            initialization.get("implicit_direct_initialization_hash")
            and diagnostics.get("implicit_direct_decoder_parameters", 0) > 0
            and diagnostics.get("implicit_direct_history_spectrum_bins")
            == implicit["history_spectrum_bins"]
            and diagnostics.get("implicit_direct_fourier_norm")
            == implicit["fourier_norm"]
        )
    elif hasattr(model, "history_statistic_coeff"):
        summary_contract = design["summary_contract"]
        expected_mode = adapter["history_statistic_mode"]
        projection_tolerance = design["step7b_protocol"][
            "production_projection_orthogonality_max_abs_max"
        ]
        readout_contract_pass = bool(
            initialization.get("history_statistic_initialization_hash")
            and initialization.get("history_statistic_projection_hash")
            and initialization.get("history_statistic_mode")
            == expected_mode
            and initialization.get("history_statistic_dim")
            == summary_contract["dimension"]
            and initialization.get("history_statistic_initial_weight_norm")
            == 0.0
            and diagnostics.get("history_statistic_mode") == expected_mode
            and diagnostics.get("history_statistic_dim")
            == summary_contract["dimension"]
            and diagnostics.get(
                "history_statistic_projection_orthogonality_max_abs",
                math.inf,
            )
            <= projection_tolerance
            and diagnostics.get("history_statistic_decoder_parameters")
            == summary_contract["dimension"] * 256
            and "probe_history_summary" in payload
            and "probe_history_coefficient" in payload
            and "probe_history_prediction_contribution" in payload
        )
    elif hasattr(model, "fcmi_readout"):
        fcmi_contract = design["fcmi_contract"]
        expected_dense_rank = (
            fcmi_contract["dense_ranks"].get(adapter["dataset"], 0)
            if adapter["readout_mode"]
            == "fcmi-dense-capacity-matched"
            else 0
        )
        readout_contract_pass = bool(
            initialization.get("fcmi_common_initialization_hash")
            and initialization.get("fcmi_memory_position_hash")
            and initialization.get("fcmi_target_position_hash")
            and diagnostics.get("fcmi_n_heads")
            == fcmi_contract["n_heads"]
            and diagnostics.get("fcmi_dropout")
            == fcmi_contract["dropout"]
            and diagnostics.get("fcmi_dense_rank")
            == expected_dense_rank
            and "probe_fcmi_context_coordinate_std" in payload
            and "probe_fcmi_main_rms" in payload
            and "probe_fcmi_interaction_rms" in payload
            and "probe_fcmi_attention_entropy" in payload
            and "probe_fcmi_attention_target_dispersion" in payload
            and (
                adapter["readout_mode"] != "fcmi"
                or "probe_fcmi_interaction_prediction_contribution"
                in payload
            )
            and (
                adapter["readout_mode"]
                != "fcmi-dense-capacity-matched"
                or (
                    initialization.get("fcmi_dense_initial_output_norm")
                    == 0.0
                    and "probe_fcmi_dense_residual" in payload
                )
            )
        )

    invariant = {
        "candidate_id": design.get(
            "candidate_id",
            design.get("candidate_version", "PCSD-CF"),
        ),
        "candidate_version": (
            test_audit["candidate_version"]
            if test_audit is not None
            else design.get("candidate_version")
        ),
        "hyperparameter_trial_id": adapter.get("hpo_trial_id"),
        "hyperparameter_profile_id": adapter.get("hpo_profile_id"),
        "seed": adapter["seed"],
        "diagnostic_id": design.get(
            "diagnostic_id",
            design.get("audit_id", "PCSD-CF-checkpoint-audit"),
        ),
        "dataset": adapter["dataset"],
        "readout_mode": adapter["readout_mode"],
        "policy_mode": adapter.get("pcsd_policy_mode", "control"),
        "fixed_scale": adapter.get("pcsd_fixed_scale"),
        "partition": adapter.get("pcsd_partition", "control"),
        "prefix_rows": prefix_rows,
        "full_prefix_max_abs": prefix_gap,
        "evaluation_split": args.evaluation_split,
        "evaluation_rows": int(payload["fused_row_bin_mse"].shape[0]),
        "arm_diagnostics_present": "arm_row_bin_mse" in payload,
        "probe_rows": int(payload.get("probe_fused", np.empty((0,))).shape[0]),
        "scale_component_diagnostics_present": (
            "scale_component_contribution" in payload
        ),
        "implicit_frequency_diagnostics_present": (
            "probe_if_amplitude" in payload
        ),
        "compact_history_diagnostics_present": (
            "probe_history_summary" in payload
        ),
        "cpsi_diagnostics_present": "probe_cpsi_message_rms" in payload,
        "sps_diagnostics_present": "probe_sps_raw_arms" in payload,
        "frsc_conditioning_strength": diagnostics.get(
            "frsc_conditioning_strength"
        ),
        "frsc_minimum_operator_eigenvalue": diagnostics.get(
            "frsc_minimum_operator_eigenvalue"
        ),
        "frsc_full_rank": diagnostics.get("frsc_full_rank"),
        "all_finite": all_finite,
        "protocol_pass": protocol_pass,
        "readout_contract_pass": readout_contract_pass,
        "uses_test_split": args.evaluation_split == "test",
        "test_access_date": (
            datetime.now().astimezone().date().isoformat()
            if args.evaluation_split == "test"
            else None
        ),
        "test_access_authorized": test_authorized,
        "checkpoint_sha256": file_sha256(args.run_dir / "checkpoint.pt"),
        "checkpoint_retrained": bool(
            test_audit is not None
            and test_audit["authorization"].get(
                "checkpoint_retraining_allowed",
                False,
            )
            and adapter.get("protocol_profile")
            not in test_audit["authorization"].get(
                "reused_control_protocol_profiles",
                [],
            )
        ),
        "pass": bool(
            all_finite
            and math.isfinite(prefix_gap)
            and prefix_gap <= PREFIX_TOLERANCE
            and protocol_pass
            and readout_contract_pass
        ),
    }
    invariant_name = (
        "trained_invariants.json"
        if args.evaluation_split == "val"
        else "test_audit_invariants.json"
    )
    (artifact_dir / invariant_name).write_text(
        json.dumps(invariant, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not invariant["pass"]:
        raise RuntimeError(f"trained invariant failed: {invariant}")
    print(
        f"pcsd_checkpoint=pass dataset={invariant['dataset']} "
        f"readout={invariant['readout_mode']} split={args.evaluation_split} "
        f"rows={invariant['evaluation_rows']}"
    )


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        config = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=720,
            pred_len=720,
            encoder_mode="timealign-token-mlp",
            readout_mode="siff-coupling-field",
            e_layers=2,
            patch_num=12,
            d_model=64,
            d_ff=128,
            dropout=0.1,
            pos=1,
            layer_norm=1,
            enc_in=7,
            pcsd_coordinate_dim=4,
            pcsd_mode_rank=256,
            pcsd_policy_history_dim=32,
            pcsd_policy_hidden_dim=64,
            pcsd_policy_mode="direct",
            pcsd_fixed_scale=720,
            pcsd_partition="canonical",
            pcsd_partition_seed=15101,
            pcsd_group_chunk_size=64,
            pcsd_target_chunk_size=128,
        )
        torch.manual_seed(2021)
        model = TimeAlign.Model(config).float().eval()
        x = torch.randn(2, 720, 7)
        target = torch.randn(2, 720, 7)
        rows, gap = prefix_audit(model, x[:1], target[:1])
        with torch.no_grad():
            arms, weights = denormalized_arms(model, x)
            components = denormalized_scale_component_contributions(model, x)
        if (
            len(rows) != len(HORIZONS)
            or gap != 0.0
            or tuple(arms.shape) != (2, 720, 5, 7)
            or tuple(weights.shape) != (2, 7, 720, 5)
            or components is None
            or tuple(components.shape) != (2, 7, 2, 720)
            or not bool(torch.isfinite(arms).all())
            or not bool(torch.isfinite(components).all())
        ):
            raise RuntimeError("SIFF checkpoint evaluator synthetic smoke failed")
        print("siff_checkpoint_evaluator_synthetic_smoke=pass")
        ccsf_config = SimpleNamespace(**vars(config))
        ccsf_config.readout_mode = "ccsf-coupling-field"
        ccsf_config.ccsf_correction_hidden_dim = 64
        torch.manual_seed(2021)
        ccsf_model = TimeAlign.Model(ccsf_config).float().eval()
        with torch.no_grad():
            result = denormalized_ccsf_tensors(ccsf_model, x)
        if result is None:
            raise RuntimeError("CCSF checkpoint diagnostic hook is missing")
        ccsf_arms, tensors = result
        expected_shapes = {
            "policy": (2, 7, 720, 5),
            "base_policy": (2, 7, 720, 5),
            "base_logits": (2, 7, 720, 5),
            "correction_logits": (2, 7, 720, 5),
            "contrast_descriptor": (2, 7, 720, 5, 6),
        }
        if (
            tuple(ccsf_arms.shape) != (2, 720, 5, 7)
            or any(
                key not in tensors
                or tuple(tensors[key].shape) != expected_shape
                or not bool(torch.isfinite(tensors[key]).all())
                for key, expected_shape in expected_shapes.items()
            )
        ):
            raise RuntimeError("CCSF checkpoint evaluator synthetic smoke failed")
        print("ccsf_checkpoint_evaluator_synthetic_smoke=pass")
        if_config = SimpleNamespace(**vars(config))
        if_config.readout_mode = "implicit-frequency-readout"
        if_config.if_hidden_width = 16
        if_config.if_direct_hidden_width = 32
        if_config.if_head_dropout = 0.1
        if_config.if_fourier_norm = "ortho"
        torch.manual_seed(2021)
        if_model = TimeAlign.Model(if_config).float().eval()
        with torch.no_grad():
            if_rows, if_gap = prefix_audit(
                if_model,
                x[:1],
                target[:1],
            )
            if_tensors = implicit_frequency_tensors(if_model, x)
        if (
            len(if_rows) != len(HORIZONS)
            or if_gap != 0.0
            or if_tensors is None
            or any(
                tuple(if_tensors[key].shape) != (2, 7, 361)
                or not bool(torch.isfinite(if_tensors[key]).all())
                for key in (
                    "amplitude",
                    "phase",
                    "phase_sine",
                    "phase_cosine",
                )
            )
        ):
            raise RuntimeError("D19 IF checkpoint evaluator smoke failed")
        direct_config = SimpleNamespace(**vars(if_config))
        direct_config.readout_mode = "implicit-direct-nonlinear-matched"
        torch.manual_seed(2021)
        direct_model = TimeAlign.Model(direct_config).float().eval()
        direct_rows, direct_gap = prefix_audit(
            direct_model,
            x[:1],
            target[:1],
        )
        if len(direct_rows) != len(HORIZONS) or direct_gap != 0.0:
            raise RuntimeError("D19 direct checkpoint evaluator smoke failed")
        print("d19_checkpoint_evaluator_synthetic_smoke=pass")
        d20_config = SimpleNamespace(**vars(config))
        d20_config.readout_mode = "learned-basis-compact-history-statistic"
        d20_config.basis_rank = 256
        d20_config.history_statistic_mode = "fixed-real-fourier-low32"
        d20_config.history_statistic_dim = 64
        d20_config.history_statistic_random_seed = 20260719
        torch.manual_seed(2021)
        d20_model = TimeAlign.Model(d20_config).float().eval()
        with torch.no_grad():
            d20_rows, d20_gap = prefix_audit(
                d20_model,
                x[:1],
                target[:1],
            )
            d20_tensors = compact_history_statistic_tensors(d20_model, x)
        if (
            len(d20_rows) != len(HORIZONS)
            or d20_gap > PREFIX_TOLERANCE
            or d20_tensors is None
            or tuple(d20_tensors["summary"].shape) != (2, 7, 64)
            or tuple(d20_tensors["coefficient"].shape) != (2, 7, 256)
            or tuple(d20_tensors["prediction_contribution"].shape)
            != (2, 720, 7)
            or any(
                not bool(torch.isfinite(value).all())
                for value in d20_tensors.values()
            )
        ):
            raise RuntimeError(
                "D20 checkpoint evaluator smoke failed: "
                f"rows={len(d20_rows)} gap={d20_gap} "
                "shapes="
                f"{None if d20_tensors is None else {key: tuple(value.shape) for key, value in d20_tensors.items()}}"
            )
        print("d20_checkpoint_evaluator_synthetic_smoke=pass")
        fcmi_config = SimpleNamespace(**vars(config))
        fcmi_config.readout_mode = "fcmi"
        fcmi_config.fcmi_n_heads = 8
        fcmi_config.fcmi_dropout = 0.0
        fcmi_config.fcmi_permutation_seed = 20260720
        fcmi_config.fcmi_dense_rank = 0
        torch.manual_seed(2021)
        fcmi_model = TimeAlign.Model(fcmi_config).float().eval()
        with torch.no_grad():
            fcmi_rows, fcmi_gap = prefix_audit(
                fcmi_model,
                x[:1],
                target[:1],
            )
            fcmi_health = fcmi_tensors(fcmi_model, x)
        if (
            len(fcmi_rows) != len(HORIZONS)
            or fcmi_gap > PREFIX_TOLERANCE
            or fcmi_health is None
            or "interaction_prediction_contribution" not in fcmi_health
            or any(
                not bool(torch.isfinite(value).all())
                for value in fcmi_health.values()
            )
        ):
            raise RuntimeError("D23 FCMI evaluator smoke failed")
        dense_config = SimpleNamespace(**vars(fcmi_config))
        dense_config.readout_mode = "fcmi-dense-capacity-matched"
        dense_config.fcmi_dense_rank = 16
        torch.manual_seed(2021)
        dense_model = TimeAlign.Model(dense_config).float().eval()
        with torch.no_grad():
            dense_health = fcmi_tensors(dense_model, x)
        if (
            dense_health is None
            or tuple(dense_health["dense_residual"].shape)
            != (2 * 7, 720)
            or float(dense_health["dense_residual"].abs().max()) != 0.0
        ):
            raise RuntimeError("D23 dense FCMI evaluator smoke failed")
        print("d23_fcmi_checkpoint_evaluator_synthetic_smoke=pass")
        return
    evaluate(args)


if __name__ == "__main__":
    main()
