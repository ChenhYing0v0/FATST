from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import DATASETS, ForecastDataset
from model import (
    PatchEncoderErrorProcessDecoder,
    PatchEncoderRegimeSegmentTargetOperator,
    PatchEncoderTargetSetDecoder,
)

STEP_REGIONS = [(1, 96), (97, 192), (193, 336), (337, 720)]
HORIZON_MIXED_STRATEGIES = {"horizon_mixed", "r3_prefix_risk"}


@dataclass(frozen=True)
class SupervisionUnit:
    unit_type: str
    active_steps: int
    mask_ratio: float
    interval_start: int = 0
    interval_end: int = 0
    component_rank: int = 0
    curriculum_phase: str = "none"
    condition_type: str = "none"
    condition_top_blocks: int = 0
    condition_mean_score: float = 0.0
    auxiliary_weight: float = 0.0


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_horizons(value: str) -> list[int]:
    horizons = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not horizons:
        raise ValueError("At least one target horizon is required.")
    allowed = {96, 192, 336, 720}
    unknown = sorted(set(horizons) - allowed)
    if unknown:
        raise ValueError(f"Unsupported target horizons: {unknown}")
    return sorted(set(horizons))


def metrics_by_horizon(pred: np.ndarray, true: np.ndarray) -> list[dict[str, float]]:
    diff = pred - true
    mse = np.mean(diff * diff, axis=(0, 2))
    mae = np.mean(np.abs(diff), axis=(0, 2))
    return [{"horizon": i + 1, "mse": float(mse[i]), "mae": float(mae[i])} for i in range(len(mse))]


def metrics_by_segment(pred: np.ndarray, true: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for start, end in STEP_REGIONS:
        if pred.shape[1] < start:
            continue
        segment_pred = pred[:, start - 1 : min(end, pred.shape[1]), :]
        segment_true = true[:, start - 1 : min(end, true.shape[1]), :]
        diff = segment_pred - segment_true
        rows.append(
            {
                "segment": f"{start}-{min(end, pred.shape[1])}",
                "mse": float(np.mean(diff * diff)),
                "mae": float(np.mean(np.abs(diff))),
            }
        )
    return rows


def target_state_similarity(components: dict[str, np.ndarray]) -> list[dict[str, float]]:
    states = components["target_states"]
    if states.shape[2] < 2:
        return [{"segment_a": 0, "segment_b": 0, "cosine": 1.0}]
    flat = states.reshape(-1, states.shape[2], states.shape[3])
    norm = np.linalg.norm(flat, axis=-1, keepdims=True) + 1e-12
    normed = flat / norm
    rows = []
    for i in range(states.shape[2]):
        for j in range(i + 1, states.shape[2]):
            cosine = np.sum(normed[:, i, :] * normed[:, j, :], axis=-1)
            rows.append({"segment_a": i, "segment_b": j, "cosine": float(np.mean(cosine))})
    return rows


def target_conditioning_stats(components: dict[str, np.ndarray]) -> list[dict[str, float]]:
    states = components["target_states"]
    gamma = components["gamma"]
    beta = components["beta"]
    history = components["history_readout"]
    rows = [
        {
            "scope": "all",
            "mean_abs_gamma": float(np.mean(np.abs(gamma))),
            "mean_abs_beta": float(np.mean(np.abs(beta))),
            "mean_target_state_norm": float(np.mean(np.linalg.norm(states, axis=-1))),
            "std_target_state_norm": float(np.std(np.linalg.norm(states, axis=-1))),
            "mean_history_readout_norm": float(np.mean(np.linalg.norm(history, axis=-1))),
        }
    ]
    for segment_index in range(states.shape[2]):
        rows.append(
            {
                "scope": f"segment_{segment_index}",
                "mean_abs_gamma": float(np.mean(np.abs(gamma[:, :, segment_index, :]))),
                "mean_abs_beta": float(np.mean(np.abs(beta[:, :, segment_index, :]))),
                "mean_target_state_norm": float(
                    np.mean(np.linalg.norm(states[:, :, segment_index, :], axis=-1))
                ),
                "std_target_state_norm": float(
                    np.std(np.linalg.norm(states[:, :, segment_index, :], axis=-1))
                ),
                "mean_history_readout_norm": float(np.mean(np.linalg.norm(history, axis=-1))),
            }
        )
    return rows


def prefix_residual_stats(components: dict[str, np.ndarray]) -> list[dict[str, float]]:
    residual = components["prefix_residual_norm"]
    rows = [
        {
            "scope": "all",
            "residual_mse_norm": float(np.mean(residual * residual)),
            "residual_mae_norm": float(np.mean(np.abs(residual))),
            "max_abs_residual_norm": float(np.max(np.abs(residual))),
        }
    ]
    for start, end in STEP_REGIONS:
        if residual.shape[1] < start:
            continue
        segment = residual[:, start - 1 : min(end, residual.shape[1]), :]
        rows.append(
            {
                "scope": f"{start}-{min(end, residual.shape[1])}",
                "residual_mse_norm": float(np.mean(segment * segment)),
                "residual_mae_norm": float(np.mean(np.abs(segment))),
                "max_abs_residual_norm": float(np.max(np.abs(segment))),
            }
        )
    return rows


def residual_smoothness_torch(residual: torch.Tensor) -> torch.Tensor:
    if residual.shape[1] < 3:
        return residual.new_tensor(0.0)
    second_diff = residual[:, 2:, :] - 2.0 * residual[:, 1:-1, :] + residual[:, :-2, :]
    return torch.mean(second_diff * second_diff)


def residual_smoothness_np(residual: np.ndarray) -> float:
    if residual.shape[1] < 3:
        return 0.0
    second_diff = residual[:, 2:, :] - 2.0 * residual[:, 1:-1, :] + residual[:, :-2, :]
    return float(np.mean(second_diff * second_diff))


def regime_segment_operator_stats(components: dict[str, np.ndarray]) -> list[dict[str, float | str]]:
    scale = components["regime_operator_scale"]
    shift = components["regime_operator_shift"]
    rows: list[dict[str, float | str]] = [
        {
            "scope": "all",
            "mean_abs_scale": float(np.mean(np.abs(scale))),
            "mean_abs_shift": float(np.mean(np.abs(shift))),
            "max_abs_scale": float(np.max(np.abs(scale))),
            "max_abs_shift": float(np.max(np.abs(shift))),
        }
    ]
    for segment_index in range(scale.shape[2]):
        rows.append(
            {
                "scope": f"segment_{segment_index}",
                "mean_abs_scale": float(np.mean(np.abs(scale[:, :, segment_index, :]))),
                "mean_abs_shift": float(np.mean(np.abs(shift[:, :, segment_index, :]))),
                "max_abs_scale": float(np.max(np.abs(scale[:, :, segment_index, :]))),
                "max_abs_shift": float(np.max(np.abs(shift[:, :, segment_index, :]))),
            }
        )
    return rows


def regime_feature_stats(components: dict[str, np.ndarray]) -> list[dict[str, float | str]]:
    features = components["regime_features"]
    feature_names = [
        "history_mean",
        "history_std",
        "history_abs_mean",
        "history_last_abs",
        "history_recent_mean",
        "history_recent_std",
        "history_recent_minus_previous",
        "history_second_minus_first",
        "history_slope_abs",
        "window_index_norm",
    ]
    rows: list[dict[str, float | str]] = []
    for feature_index, feature_name in enumerate(feature_names):
        values = features[:, :, feature_index]
        rows.append(
            {
                "feature": feature_name,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "mean_abs": float(np.mean(np.abs(values))),
                "max_abs": float(np.max(np.abs(values))),
            }
        )
    return rows


def mean_adjacent_state_cosine(states: np.ndarray, start_segment: int, end_segment: int) -> float:
    selected = states[:, :, start_segment:end_segment, :]
    if selected.shape[2] < 2:
        return 1.0
    left = selected[:, :, :-1, :].reshape(-1, selected.shape[-1])
    right = selected[:, :, 1:, :].reshape(-1, selected.shape[-1])
    left = left / (np.linalg.norm(left, axis=-1, keepdims=True) + 1e-12)
    right = right / (np.linalg.norm(right, axis=-1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(left * right, axis=-1)))


def error_process_stats(
    pred: np.ndarray,
    true: np.ndarray,
    components: dict[str, np.ndarray],
    segment_len: int,
) -> list[dict[str, float]]:
    base = components["base_prediction"]
    residual = components["error_residual"]
    residual_norm = components["error_residual_norm"]
    states = components["error_process_states"]
    scopes = [
        ("all", 1, pred.shape[1]),
        ("1-96", 1, 96),
        ("97-192", 97, 192),
        ("193-336", 193, 336),
        ("337-720", 337, 720),
    ]
    rows = []
    for scope, start, end in scopes:
        if pred.shape[1] < start:
            continue
        active_end = min(end, pred.shape[1])
        slc = slice(start - 1, active_end)
        base_scope = base[:, slc, :]
        pred_scope = pred[:, slc, :]
        true_scope = true[:, slc, :]
        residual_scope = residual[:, slc, :]
        residual_norm_scope = residual_norm[:, slc, :]
        base_diff = base_scope - true_scope
        final_diff = pred_scope - true_scope
        base_mse = float(np.mean(base_diff * base_diff))
        final_mse = float(np.mean(final_diff * final_diff))
        start_segment = (start - 1) // segment_len
        end_segment = (active_end + segment_len - 1) // segment_len
        state_scope = states[:, :, start_segment:end_segment, :]
        rows.append(
            {
                "scope": scope,
                "residual_base_mae_ratio": float(
                    np.mean(np.abs(residual_scope)) / (np.mean(np.abs(base_scope)) + 1e-12)
                ),
                "residual_energy": float(np.mean(residual_norm_scope * residual_norm_scope)),
                "residual_second_diff_smoothness": residual_smoothness_np(residual_norm_scope),
                "error_process_state_norm": float(np.mean(np.linalg.norm(state_scope, axis=-1))),
                "segment_state_cosine": mean_adjacent_state_cosine(states, start_segment, end_segment),
                "base_prediction_mse": base_mse,
                "final_prediction_mse": final_mse,
                "residual_gain_mse_pct": float((final_mse / (base_mse + 1e-12) - 1.0) * 100.0),
                "prediction_decomposition_max_abs": float(
                    np.max(np.abs(base_scope + residual_scope - pred_scope))
                ),
            }
        )
    return rows


def future_alignment_stats(
    model: PatchEncoderTargetSetDecoder,
    loader: DataLoader,
    device: torch.device,
    pred_len: int,
    max_batches: int | None = None,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    model.eval()
    local_losses = []
    relation_losses = []
    reconstruction_losses = []
    raw_reconstruction_losses = []
    normalized_reconstruction_losses = []
    confidence_means = []
    confidence_mins = []
    confidence_maxes = []
    teacher_student_cosines = []
    true_leakage = []
    shuffled_leakage = []
    zero_leakage = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            x, y, window_index_norm = unpack_batch(batch)
            x = x.float().to(device)
            y = y.float().to(device)
            window_index_norm = tensor_to_device(window_index_norm, device)
            no_future = model(
                x,
                pred_len=pred_len,
                return_components=True,
                window_index_norm=window_index_norm,
            )
            true_future = model(
                x,
                pred_len=pred_len,
                future_y=y,
                return_components=True,
                window_index_norm=window_index_norm,
            )
            shuffled_future = model(
                x,
                pred_len=pred_len,
                future_y=y.flip(0),
                return_components=True,
                window_index_norm=window_index_norm,
            )
            zero_future = model(
                x,
                pred_len=pred_len,
                future_y=torch.zeros_like(y),
                return_components=True,
                window_index_norm=window_index_norm,
            )
            if not isinstance(no_future, dict) or not isinstance(true_future, dict):
                raise TypeError("Expected component dict from model.")
            if not isinstance(shuffled_future, dict) or not isinstance(zero_future, dict):
                raise TypeError("Expected component dict from model.")
            true_leakage.append(
                float(torch.max(torch.abs(no_future["prediction"] - true_future["prediction"])).cpu())
            )
            shuffled_leakage.append(
                float(torch.max(torch.abs(no_future["prediction"] - shuffled_future["prediction"])).cpu())
            )
            zero_leakage.append(
                float(torch.max(torch.abs(no_future["prediction"] - zero_future["prediction"])).cpu())
            )
            student = F.normalize(true_future["future_student_state"], dim=-1)
            teacher = F.normalize(true_future["future_teacher_state"], dim=-1)
            teacher_student_cosines.append(float((student * teacher).sum(dim=-1).mean().cpu()))
            local_losses.append(float(true_future["future_local_alignment_loss"].cpu()))
            relation_losses.append(float(true_future["future_relation_alignment_loss"].cpu()))
            reconstruction_losses.append(float(true_future["future_reconstruction_loss"].cpu()))
            raw_reconstruction_losses.append(float(true_future["future_raw_reconstruction_loss"].cpu()))
            normalized_reconstruction_losses.append(
                float(true_future["future_normalized_reconstruction_loss"].cpu())
            )
            confidence_means.append(float(true_future["future_alignment_confidence_mean"].cpu()))
            confidence_mins.append(float(true_future["future_alignment_confidence_min"].cpu()))
            confidence_maxes.append(float(true_future["future_alignment_confidence_max"].cpu()))
            if max_batches is not None and batch_index >= max_batches:
                break

    rows = [
        {
            "scope": "all",
            "future_local_alignment_loss": float(np.mean(local_losses)),
            "future_relation_alignment_loss": float(np.mean(relation_losses)),
            "future_reconstruction_loss": float(np.mean(reconstruction_losses)),
            "future_raw_reconstruction_loss": float(np.mean(raw_reconstruction_losses)),
            "future_normalized_reconstruction_loss": float(np.mean(normalized_reconstruction_losses)),
            "future_alignment_confidence_mean": float(np.mean(confidence_means)),
            "future_alignment_confidence_min": float(np.min(confidence_mins)),
            "future_alignment_confidence_max": float(np.max(confidence_maxes)),
            "teacher_student_cosine": float(np.mean(teacher_student_cosines)),
            "prediction_leakage_max_abs": float(
                max(
                    max(true_leakage, default=0.0),
                    max(shuffled_leakage, default=0.0),
                    max(zero_leakage, default=0.0),
                )
            ),
        }
    ]
    audit = {
        "true_future_prediction_max_abs": float(max(true_leakage, default=0.0)),
        "shuffled_future_prediction_max_abs": float(max(shuffled_leakage, default=0.0)),
        "zero_future_prediction_max_abs": float(max(zero_leakage, default=0.0)),
        "prediction_leakage_max_abs": rows[0]["prediction_leakage_max_abs"],
    }
    return rows, audit


def weighted_mse_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    max_pred_len: int,
    target_horizons: list[int],
    mode: str,
    alpha: float,
    precomputed_weights: torch.Tensor | None = None,
    offdiag_block_matrix: torch.Tensor | None = None,
    offdiag_block_size: int = 48,
    offdiag_quadratic_weight: float = 0.0,
) -> torch.Tensor:
    if mode == "uniform":
        return torch.mean((pred - true) ** 2)
    if mode == "offdiag_block_quadratic":
        base_loss = weighted_mse_loss(
            pred,
            true,
            max_pred_len,
            target_horizons,
            "prefix_risk",
            alpha,
        )
        if offdiag_block_matrix is None:
            raise ValueError("offdiag_block_quadratic requires a precomputed block matrix.")
        penalty = offdiag_block_quadratic_loss(
            pred,
            true,
            offdiag_block_matrix,
            offdiag_block_size,
        )
        return base_loss + offdiag_quadratic_weight * penalty
    if mode not in {"prefix_risk", "region_balanced", "step_covariance_balanced"}:
        raise ValueError(f"Unknown step loss weighting mode: {mode}")
    if alpha < 0:
        raise ValueError("step_loss_alpha must be non-negative.")

    horizon = pred.shape[1]
    if precomputed_weights is None:
        weights = step_loss_weights(
            max_pred_len,
            target_horizons,
            mode,
            alpha,
            pred.device,
            pred.dtype,
        )[:horizon]
    else:
        weights = precomputed_weights.to(device=pred.device, dtype=pred.dtype)[:horizon]
    return torch.mean((pred - true) ** 2 * weights.view(1, horizon, 1))


def masked_mse_loss(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    horizon = pred.shape[1]
    active_mask = mask.to(device=pred.device, dtype=pred.dtype)[:horizon].view(1, horizon, 1)
    denominator = torch.sum(active_mask) * pred.shape[0] * pred.shape[2]
    return torch.sum((pred - true) ** 2 * active_mask) / torch.clamp(denominator, min=1.0)


def sample_block_mask(
    horizon: int,
    block_size: int,
    mask_ratio: float,
    rng: random.Random,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, SupervisionUnit]:
    if block_size <= 0:
        raise ValueError("supervision block size must be positive.")
    if not 0 < mask_ratio <= 1:
        raise ValueError("supervision mask ratio must be in (0, 1].")
    block_count = int(np.ceil(horizon / float(block_size)))
    active_blocks = max(1, min(block_count, int(round(block_count * mask_ratio))))
    selected = set(rng.sample(range(block_count), active_blocks))
    mask = torch.zeros(horizon, device=device, dtype=dtype)
    for block_index in selected:
        start = block_index * block_size
        end = min(start + block_size, horizon)
        mask[start:end] = 1.0
    active_steps = int(torch.sum(mask).item())
    return mask, SupervisionUnit(
        unit_type="mask",
        active_steps=active_steps,
        mask_ratio=float(active_steps / float(horizon)),
    )


def sample_interval_mask(
    horizon: int,
    block_size: int,
    min_blocks: int,
    max_blocks: int,
    rng: random.Random,
    device: torch.device,
    dtype: torch.dtype,
    curriculum_phase: str = "none",
) -> tuple[torch.Tensor, SupervisionUnit]:
    if block_size <= 0:
        raise ValueError("supervision block size must be positive.")
    if min_blocks <= 0 or max_blocks < min_blocks:
        raise ValueError("invalid supervision interval block range.")
    block_count = int(np.ceil(horizon / float(block_size)))
    length_blocks = rng.randint(min_blocks, min(max_blocks, block_count))
    start_block = rng.randint(0, block_count - length_blocks)
    start = start_block * block_size
    end = min(start + length_blocks * block_size, horizon)
    mask = torch.zeros(horizon, device=device, dtype=dtype)
    mask[start:end] = 1.0
    active_steps = int(end - start)
    return mask, SupervisionUnit(
        unit_type="interval",
        active_steps=active_steps,
        mask_ratio=float(active_steps / float(horizon)),
        interval_start=start + 1,
        interval_end=end,
        curriculum_phase=curriculum_phase,
    )


def target_rows_chunk(data: np.ndarray, indices: np.ndarray, seq_len: int, pred_len: int) -> np.ndarray:
    targets = np.stack([data[index + seq_len : index + seq_len + pred_len] for index in indices], axis=0)
    return targets.transpose(0, 2, 1).reshape(-1, pred_len).astype(np.float64, copy=False)


def train_split_label_basis(
    dataset_root: str | Path,
    dataset: str,
    seq_len: int,
    pred_len: int,
    chunk_windows: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    train_set = ForecastDataset(dataset_root, dataset, "train", seq_len, pred_len)
    data = train_set.data.astype(np.float64, copy=False)
    n_windows = len(data) - seq_len - pred_len + 1
    if n_windows <= 0:
        raise ValueError("Dataset split is shorter than seq_len + pred_len.")
    step_sum = np.zeros(pred_len, dtype=np.float64)
    step_cross = np.zeros((pred_len, pred_len), dtype=np.float64)
    count = 0
    for start in range(0, n_windows, chunk_windows):
        stop = min(start + chunk_windows, n_windows)
        rows = target_rows_chunk(data, np.arange(start, stop), seq_len, pred_len)
        step_sum += rows.sum(axis=0)
        step_cross += rows.T @ rows
        count += rows.shape[0]
    mean = step_sum / float(count)
    covariance = (step_cross - float(count) * np.outer(mean, mean)) / float(max(count - 1, 1))
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    return eigenvectors.astype(np.float32), eigenvalues.astype(np.float32)


def component_supervision_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    basis: torch.Tensor,
    eigenvalues: torch.Tensor,
    rank: int,
    alpha: float,
    beta: float,
    balanced: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, SupervisionUnit]:
    if not 0 <= alpha <= 1:
        raise ValueError("supervision component alpha must be in [0, 1].")
    horizon = pred.shape[1]
    active_rank = max(1, min(rank, horizon, basis.shape[1]))
    active_basis = basis[:horizon, :active_rank].to(device=pred.device, dtype=pred.dtype)
    pred_rows = pred.permute(0, 2, 1).reshape(-1, horizon)
    true_rows = true.permute(0, 2, 1).reshape(-1, horizon)
    pred_components = pred_rows @ active_basis
    true_components = true_rows @ active_basis
    component_scale = torch.mean(true_components * true_components, dim=0).clamp_min(1e-6)
    component_error = (pred_components - true_components) ** 2 / component_scale.view(1, active_rank)
    if balanced:
        active_eigenvalues = eigenvalues[:active_rank].to(device=pred.device, dtype=pred.dtype)
        weights = torch.pow(torch.clamp(active_eigenvalues, min=1e-8), -beta)
        weights = torch.clamp(weights, min=0.1, max=10.0)
        weights = weights / torch.mean(weights)
        unit_loss = torch.mean(component_error * weights.view(1, active_rank))
        unit_type = "component_balanced"
    else:
        unit_loss = torch.mean(component_error)
        unit_type = "component_top"
    time_loss = torch.mean((pred - true) ** 2)
    total_loss = (1.0 - alpha) * time_loss + alpha * unit_loss
    unit = SupervisionUnit(
        unit_type=unit_type,
        active_steps=horizon,
        mask_ratio=1.0,
        component_rank=active_rank,
    )
    return total_loss, time_loss, unit_loss, unit


def conditioned_future_unit_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    history: torch.Tensor,
    block_size: int,
    top_ratio: float,
    aux_weight: float,
    condition: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, SupervisionUnit]:
    if block_size <= 0:
        raise ValueError("supervision block size must be positive.")
    if not 0 < top_ratio <= 1:
        raise ValueError("supervision condition top ratio must be in (0, 1].")
    if aux_weight < 0:
        raise ValueError("supervision auxiliary weight must be non-negative.")

    horizon = pred.shape[1]
    blocks: list[tuple[int, int]] = []
    scores = []
    with torch.no_grad():
        last_history = history[:, -1:, :]
        for start in range(0, horizon, block_size):
            end = min(start + block_size, horizon)
            block = true[:, start:end, :]
            if condition == "label_novelty":
                reference = last_history.expand(-1, end - start, -1)
                score = torch.mean((block - reference) ** 2)
            elif condition == "local_variation":
                if end - start <= 1:
                    score = torch.zeros((), device=true.device, dtype=true.dtype)
                else:
                    diff = block[:, 1:, :] - block[:, :-1, :]
                    score = torch.mean(diff * diff)
            elif condition == "hybrid":
                reference = last_history.expand(-1, end - start, -1)
                novelty = torch.mean((block - reference) ** 2)
                if end - start <= 1:
                    variation = torch.zeros((), device=true.device, dtype=true.dtype)
                else:
                    diff = block[:, 1:, :] - block[:, :-1, :]
                    variation = torch.mean(diff * diff)
                score = novelty + variation
            else:
                raise ValueError(f"Unknown supervision condition: {condition}")
            blocks.append((start, end))
            scores.append(score)

        score_tensor = torch.stack(scores)
        top_blocks = max(1, min(len(blocks), int(round(len(blocks) * top_ratio))))
        selected = torch.topk(score_tensor, k=top_blocks, largest=True).indices.tolist()
        mask = torch.zeros(horizon, device=pred.device, dtype=pred.dtype)
        for block_index in selected:
            start, end = blocks[int(block_index)]
            mask[start:end] = 1.0
        mean_score = float(torch.mean(score_tensor[selected]).detach().cpu())

    time_loss = torch.mean((pred - true) ** 2)
    unit_loss = masked_mse_loss(pred, true, mask)
    total_loss = time_loss + aux_weight * unit_loss
    active_steps = int(torch.sum(mask).detach().cpu().item())
    unit = SupervisionUnit(
        unit_type="conditioned_sparse",
        active_steps=active_steps,
        mask_ratio=active_steps / float(horizon),
        condition_type=condition,
        condition_top_blocks=top_blocks,
        condition_mean_score=mean_score,
        auxiliary_weight=aux_weight,
    )
    return total_loss, time_loss, unit_loss, unit


def curriculum_phase(epoch: int, epochs: int) -> str:
    progress = epoch / float(max(epochs, 1))
    if progress <= 0.3:
        return "coarse"
    if progress <= 0.7:
        return "mixed"
    return "dense"


def horizon_decoupled_supervision_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    history: torch.Tensor,
    args: argparse.Namespace,
    rng: random.Random,
    epoch: int,
    component_basis: torch.Tensor | None,
    component_eigenvalues: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, SupervisionUnit]:
    strategy = args.supervision_strategy
    horizon = pred.shape[1]
    time_loss = torch.mean((pred - true) ** 2)
    if strategy == "full_time_mse":
        unit = SupervisionUnit("full_time", horizon, 1.0)
        return time_loss, time_loss, time_loss, unit
    if strategy == "random_future_mask":
        mask, unit = sample_block_mask(
            horizon,
            args.supervision_block_size,
            args.supervision_mask_ratio,
            rng,
            pred.device,
            pred.dtype,
        )
        unit_loss = masked_mse_loss(pred, true, mask)
        return unit_loss, time_loss, unit_loss, unit
    if strategy == "interval_supervision":
        mask, unit = sample_interval_mask(
            horizon,
            args.supervision_block_size,
            args.supervision_interval_min_blocks,
            args.supervision_interval_max_blocks,
            rng,
            pred.device,
            pred.dtype,
        )
        unit_loss = masked_mse_loss(pred, true, mask)
        return unit_loss, time_loss, unit_loss, unit
    if strategy == "conditioned_future_unit_scheduling":
        return conditioned_future_unit_loss(
            pred,
            true,
            history,
            args.supervision_block_size,
            args.supervision_condition_top_ratio,
            args.supervision_aux_weight,
            args.supervision_condition,
        )
    if strategy in {"component_basis_top", "component_basis_balanced"}:
        if component_basis is None or component_eigenvalues is None:
            raise ValueError(f"{strategy} requires a train-label component basis.")
        return component_supervision_loss(
            pred,
            true,
            component_basis,
            component_eigenvalues,
            args.supervision_component_rank,
            args.supervision_component_alpha,
            args.supervision_component_beta,
            balanced=strategy == "component_basis_balanced",
        )
    if strategy == "curriculum_units":
        phase = curriculum_phase(epoch, args.epochs)
        if phase == "coarse":
            if component_basis is None or component_eigenvalues is None:
                raise ValueError("curriculum_units requires a train-label component basis.")
            total_loss, time_loss, unit_loss, unit = component_supervision_loss(
                pred,
                true,
                component_basis,
                component_eigenvalues,
                args.supervision_component_rank,
                args.supervision_component_alpha,
                args.supervision_component_beta,
                balanced=False,
            )
            unit = SupervisionUnit(
                unit.unit_type,
                unit.active_steps,
                unit.mask_ratio,
                component_rank=unit.component_rank,
                curriculum_phase=phase,
            )
            return total_loss, time_loss, unit_loss, unit
        if phase == "mixed":
            mask, unit = sample_interval_mask(
                horizon,
                args.supervision_block_size,
                args.supervision_interval_min_blocks,
                args.supervision_interval_max_blocks,
                rng,
                pred.device,
                pred.dtype,
                curriculum_phase=phase,
            )
            unit_loss = masked_mse_loss(pred, true, mask)
            return unit_loss, time_loss, unit_loss, unit
        unit = SupervisionUnit("full_time", horizon, 1.0, curriculum_phase=phase)
        return time_loss, time_loss, time_loss, unit
    raise ValueError(f"Unknown supervision strategy: {strategy}")


def offdiag_block_quadratic_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    block_matrix: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError("offdiag_block_size must be positive.")
    residual = pred - true
    horizon = residual.shape[1]
    block_means = []
    for start in range(0, horizon, block_size):
        end = min(start + block_size, horizon)
        block_means.append(torch.mean(residual[:, start:end, :], dim=1))
    if len(block_means) < 2:
        return residual.new_tensor(0.0)
    error_blocks = torch.stack(block_means, dim=-1).reshape(-1, len(block_means))
    active_matrix = block_matrix[: len(block_means), : len(block_means)].to(
        device=residual.device,
        dtype=residual.dtype,
    )
    coupled_error = error_blocks @ active_matrix.T
    return torch.mean(coupled_error * coupled_error)


def step_loss_weights(
    max_pred_len: int,
    target_horizons: list[int],
    mode: str,
    alpha: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    step = torch.arange(1, max_pred_len + 1, device=device, dtype=dtype)
    if mode == "prefix_risk":
        weights = torch.pow(step / float(max_pred_len), -alpha)
        return weights / torch.mean(weights)
    if mode == "region_balanced":
        return region_balanced_step_weights(max_pred_len, target_horizons, device, dtype)
    if mode == "step_covariance_balanced":
        raise ValueError("step_covariance_balanced requires precomputed dataset-specific weights.")
    raise ValueError(f"Unknown step loss weighting mode: {mode}")


def region_balanced_step_weights(
    max_pred_len: int,
    target_horizons: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    pressure = expected_uniform_step_pressure(max_pred_len, target_horizons)
    total_pressure = sum(pressure)
    active_regions = [(start, min(end, max_pred_len)) for start, end in STEP_REGIONS if start <= max_pred_len]
    desired_share = 1.0 / float(len(active_regions))
    multipliers: list[float] = []
    for start, end in active_regions:
        share = sum(pressure[start - 1 : end]) / total_pressure
        multipliers.append(desired_share / max(share, 1e-12))
    full_weights = torch.ones(max_pred_len, device=device, dtype=dtype)
    for (start, end), multiplier in zip(active_regions, multipliers, strict=True):
        full_weights[start - 1 : end] = float(multiplier)
    return full_weights / torch.mean(full_weights)


def expected_uniform_step_pressure(max_pred_len: int, target_horizons: list[int]) -> list[float]:
    pressure = []
    for step in range(1, max_pred_len + 1):
        coeff = 0.0
        for horizon in target_horizons:
            if step <= horizon:
                coeff += 1.0 / float(horizon)
        pressure.append(coeff / float(len(target_horizons)))
    return pressure


def region_mean_series(data: np.ndarray, seq_len: int, pred_len: int) -> dict[str, np.ndarray]:
    n_windows = len(data) - seq_len - pred_len + 1
    if n_windows <= 0:
        raise ValueError("Dataset split is shorter than seq_len + pred_len.")
    cumsum = np.vstack([np.zeros((1, data.shape[1]), dtype=np.float64), data.cumsum(axis=0, dtype=np.float64)])
    means: dict[str, np.ndarray] = {}
    window_index = np.arange(n_windows)
    target_start = window_index + seq_len
    for start, end in STEP_REGIONS:
        if start > pred_len:
            continue
        active_end = min(end, pred_len)
        start_index = target_start + start - 1
        end_index = target_start + active_end
        region_sum = cumsum[end_index] - cumsum[start_index]
        means[f"{start}-{active_end}"] = region_sum / float(active_end - start + 1)
    return means


def correlation_squared(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    left_centered = left_flat - left_flat.mean()
    right_centered = right_flat - right_flat.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0.0:
        return 0.0
    corr = float(np.dot(left_centered, right_centered) / denominator)
    return corr * corr


def train_split_region_novelty(
    dataset_root: str | Path,
    dataset: str,
    seq_len: int,
    max_pred_len: int,
    eps: float,
) -> dict[str, float]:
    train_set = ForecastDataset(dataset_root, dataset, "train", seq_len, max_pred_len)
    means = region_mean_series(train_set.data.astype(np.float64), seq_len, max_pred_len)
    labels = list(means.keys())
    novelty: dict[str, float] = {}
    for index, label in enumerate(labels):
        current = means[label]
        current_centered = current - current.mean(axis=0, keepdims=True)
        region_std = float(np.sqrt(np.mean(current_centered * current_centered)))
        prev_r2 = [correlation_squared(current, means[prev_label]) for prev_label in labels[:index]]
        max_prev_r2 = max(prev_r2) if prev_r2 else 0.0
        novelty[label] = max(region_std * (1.0 - max_prev_r2), eps)
    total = sum(novelty.values())
    return {label: value / max(total, eps) for label, value in novelty.items()}


def block_mean_series(data: np.ndarray, seq_len: int, pred_len: int, block_size: int) -> np.ndarray:
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    n_windows = len(data) - seq_len - pred_len + 1
    if n_windows <= 0:
        raise ValueError("Dataset split is shorter than seq_len + pred_len.")
    cumsum = np.vstack([np.zeros((1, data.shape[1]), dtype=np.float64), data.cumsum(axis=0, dtype=np.float64)])
    window_index = np.arange(n_windows)
    target_start = window_index + seq_len
    blocks = []
    for start in range(0, pred_len, block_size):
        end = min(start + block_size, pred_len)
        block_start = target_start + start
        block_end = target_start + end
        block_sum = cumsum[block_end] - cumsum[block_start]
        blocks.append(block_sum / float(end - start))
    return np.stack(blocks, axis=-1).reshape(-1, len(blocks))


def train_split_offdiag_block_matrix(
    dataset_root: str | Path,
    dataset: str,
    seq_len: int,
    max_pred_len: int,
    block_size: int,
    eps: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    train_set = ForecastDataset(dataset_root, dataset, "train", seq_len, max_pred_len)
    block_values = block_mean_series(train_set.data.astype(np.float64), seq_len, max_pred_len, block_size)
    centered = block_values - block_values.mean(axis=0, keepdims=True)
    scale = block_values.std(axis=0, keepdims=True)
    normalized = centered / np.maximum(scale, eps)
    denom = max(normalized.shape[0] - 1, 1)
    corr = (normalized.T @ normalized) / float(denom)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = 0.5 * (corr + corr.T)
    precision = np.linalg.inv(corr + eps * np.eye(corr.shape[0], dtype=np.float64))
    offdiag = precision - np.diag(np.diag(precision))
    spectral_norm = float(np.linalg.norm(offdiag, ord=2))
    if spectral_norm > eps:
        offdiag = offdiag / spectral_norm
    matrix = torch.tensor(offdiag, dtype=torch.float32)
    fro_sq = float(np.sum(offdiag * offdiag))
    diag_fro_sq = float(np.sum(np.diag(precision) ** 2))
    stats = {
        "offdiag_block_count": float(offdiag.shape[0]),
        "offdiag_block_size": float(block_size),
        "offdiag_precision_spectral_norm": spectral_norm,
        "offdiag_precision_fro": float(np.sqrt(fro_sq)),
        "offdiag_precision_fro_share_before_norm": float(
            fro_sq / max(fro_sq + diag_fro_sq, eps)
        ),
    }
    return matrix, stats


def step_covariance_balanced_step_weights(
    max_pred_len: int,
    target_horizons: list[int],
    novelty_share: dict[str, float],
    beta: float,
    eta: float,
    eps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    pressure = expected_uniform_step_pressure(max_pred_len, target_horizons)
    total_pressure = sum(pressure)
    active_regions = [(start, min(end, max_pred_len)) for start, end in STEP_REGIONS if start <= max_pred_len]
    multipliers: list[float] = []
    for start, end in active_regions:
        label = f"{start}-{end}"
        pressure_share = sum(pressure[start - 1 : end]) / total_pressure
        novelty = novelty_share.get(label, eps)
        multiplier = (max(pressure_share, eps) ** (-beta)) * (max(novelty, eps) ** eta)
        multipliers.append(multiplier)
    full_weights = torch.ones(max_pred_len, device=device, dtype=dtype)
    for (start, end), multiplier in zip(active_regions, multipliers, strict=True):
        full_weights[start - 1 : end] = float(multiplier)
    return full_weights / torch.mean(full_weights)


def objective_weight_stats(
    max_pred_len: int,
    target_horizons: list[int],
    mode: str,
    alpha: float,
    precomputed_weights: torch.Tensor | None = None,
    novelty_share: dict[str, float] | None = None,
) -> list[dict[str, float | str]]:
    device = torch.device("cpu")
    if mode == "uniform":
        weights = torch.ones(max_pred_len, dtype=torch.float64)
    elif precomputed_weights is not None:
        weights = precomputed_weights.detach().to(device=device, dtype=torch.float64)
    elif mode == "offdiag_block_quadratic":
        weights = step_loss_weights(max_pred_len, target_horizons, "prefix_risk", alpha, device, torch.float64)
    else:
        weights = step_loss_weights(max_pred_len, target_horizons, mode, alpha, device, torch.float64)
    uniform_pressure = expected_uniform_step_pressure(max_pred_len, target_horizons)
    weighted_pressure = [
        uniform_pressure[index] * float(weights[index].item())
        for index in range(max_pred_len)
    ]
    uniform_total = sum(uniform_pressure)
    weighted_total = sum(weighted_pressure)
    rows: list[dict[str, float | str]] = []
    for start, end in STEP_REGIONS:
        if start > max_pred_len:
            continue
        active_end = min(end, max_pred_len)
        uniform_share = sum(uniform_pressure[start - 1 : active_end]) / uniform_total
        weighted_share = sum(weighted_pressure[start - 1 : active_end]) / weighted_total
        region_weights = weights[start - 1 : active_end]
        rows.append(
            {
                "scope": f"{start}-{active_end}",
                "mode": mode,
                "mean_step_weight": float(torch.mean(region_weights).item()),
                "min_step_weight": float(torch.min(region_weights).item()),
                "max_step_weight": float(torch.max(region_weights).item()),
                "uniform_pressure_share": uniform_share,
                "weighted_pressure_share": weighted_share,
                "pressure_share_delta_pct": (weighted_share / uniform_share - 1.0) * 100.0,
                "novelty_share": float((novelty_share or {}).get(f"{start}-{active_end}", 0.0)),
            }
        )
    for horizon in target_horizons:
        horizon_weights = weights[:horizon]
        rows.append(
            {
                "scope": f"horizon_{horizon}",
                "mode": mode,
                "mean_step_weight": float(torch.mean(horizon_weights).item()),
                "min_step_weight": float(torch.min(horizon_weights).item()),
                "max_step_weight": float(torch.max(horizon_weights).item()),
                "uniform_pressure_share": 1.0,
                "weighted_pressure_share": float(torch.mean(horizon_weights).item()),
                "pressure_share_delta_pct": (float(torch.mean(horizon_weights).item()) - 1.0) * 100.0,
                "novelty_share": 0.0,
            }
        )
    return rows


def evaluate(
    model: PatchEncoderTargetSetDecoder,
    loader: DataLoader,
    device: torch.device,
    pred_len: int,
    max_batches: int | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    component_rows: dict[str, list[np.ndarray]] = {
        "target_states": [],
        "gamma": [],
        "beta": [],
        "history_readout": [],
        "prefix_residual_norm": [],
    }
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            x, y, window_index_norm = unpack_batch(batch)
            x = x.float().to(device)
            y = y.float().to(device)
            window_index_norm = tensor_to_device(window_index_norm, device)
            output = model(
                x,
                pred_len=pred_len,
                return_components=True,
                window_index_norm=window_index_norm,
            )
            if not isinstance(output, dict):
                raise TypeError("PatchEncoderTargetSetDecoder must return component dict.")
            pred = output["prediction"]
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())
            for name, value in output.items():
                if not isinstance(value, torch.Tensor) or value.ndim == 0:
                    continue
                component_rows.setdefault(name, []).append(value.cpu().numpy())
            if max_batches is not None and batch_index >= max_batches:
                break
    pred_np = np.concatenate(preds, axis=0)
    true_np = np.concatenate(trues, axis=0)
    components = {name: np.concatenate(values, axis=0) for name, values in component_rows.items() if values}
    diff = pred_np - true_np
    return {"mse": float(np.mean(diff * diff)), "mae": float(np.mean(np.abs(diff)))}, pred_np, true_np, components


def prefix_consistency(
    model: PatchEncoderTargetSetDecoder,
    loader: DataLoader,
    device: torch.device,
    short_horizons: list[int],
    long_horizon: int,
    max_batches: int | None = None,
) -> list[dict[str, float]]:
    model.eval()
    mismatch: dict[int, list[np.ndarray]] = {horizon: [] for horizon in short_horizons}
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            x, _, window_index_norm = unpack_batch(batch)
            x = x.float().to(device)
            window_index_norm = tensor_to_device(window_index_norm, device)
            long_pred = model(x, pred_len=long_horizon, window_index_norm=window_index_norm)
            if isinstance(long_pred, dict):
                raise TypeError("Expected tensor prediction.")
            for horizon in short_horizons:
                short_pred = model(x, pred_len=horizon, window_index_norm=window_index_norm)
                if isinstance(short_pred, dict):
                    raise TypeError("Expected tensor prediction.")
                diff = short_pred - long_pred[:, :horizon, :]
                mismatch[horizon].append(diff.cpu().numpy())
            if max_batches is not None and batch_index >= max_batches:
                break
    rows = []
    for horizon, values in mismatch.items():
        diff_np = np.concatenate(values, axis=0)
        rows.append(
            {
                "short_horizon": horizon,
                "long_horizon": long_horizon,
                "prefix_mismatch_mse": float(np.mean(diff_np * diff_np)),
                "prefix_mismatch_mae": float(np.mean(np.abs(diff_np))),
                "truth_alignment_mse": 0.0,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_loaders(
    dataset_root: str,
    dataset: str,
    split: str,
    seq_len: int,
    horizons: list[int],
    batch_size: int,
    drop_last: bool,
    return_index: bool = False,
) -> dict[int, DataLoader]:
    return {
        horizon: DataLoader(
            ForecastDataset(dataset_root, dataset, split, seq_len, horizon, return_index=return_index),
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=drop_last,
        )
        for horizon in horizons
    }


def unpack_batch(
    batch: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if len(batch) == 2:
        x, y = batch
        return x, y, None
    x, y, window_index_norm = batch
    return x, y, window_index_norm


def tensor_to_device(value: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    return value.float().to(device)


def next_batch(
    loaders: dict[int, DataLoader],
    iterators: dict[
        int,
        Iterator[tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ],
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    try:
        return unpack_batch(next(iterators[horizon]))
    except StopIteration:
        iterators[horizon] = iter(loaders[horizon])
        return unpack_batch(next(iterators[horizon]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PatchEncoderTargetSetDecoder Phase1-R candidate.")
    parser.add_argument("--dataset-root", default="/Users/river/PaperResearch/Project/datasets")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ETTh2")
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--target-horizons", default="96,192,336,720")
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--head-dropout", type=float, default=0.0)
    parser.add_argument("--segment-len", type=int, default=48)
    parser.add_argument("--max-pred-len", type=int, default=720)
    parser.add_argument("--target-layers", type=int, default=1)
    parser.add_argument("--target-heads", type=int, default=8)
    parser.add_argument("--target-d-ff", type=int, default=256)
    parser.add_argument("--target-interaction-layers", type=int, default=0)
    parser.add_argument("--target-interaction-heads", type=int, default=0)
    parser.add_argument("--target-interaction-d-ff", type=int, default=0)
    parser.add_argument("--readout-dim", type=int, default=256)
    parser.add_argument("--prefix-residual-segments", type=int, default=0)
    parser.add_argument("--prefix-residual-dropout", type=float, default=0.0)
    parser.add_argument("--future-teacher-layers", type=int, default=0)
    parser.add_argument("--future-teacher-heads", type=int, default=0)
    parser.add_argument("--future-teacher-d-ff", type=int, default=0)
    parser.add_argument("--future-state-dim", type=int, default=0)
    parser.add_argument("--future-align-weight", type=float, default=0.0)
    parser.add_argument("--future-relation-weight", type=float, default=0.0)
    parser.add_argument("--future-recon-weight", type=float, default=0.0)
    parser.add_argument("--future-recon-normalization", choices=["none", "target_energy"], default="none")
    parser.add_argument(
        "--future-align-weighting",
        choices=["uniform", "reconstruction_confidence"],
        default="uniform",
    )
    parser.add_argument("--future-confidence-temperature", type=float, default=1.0)
    parser.add_argument("--future-confidence-floor", type=float, default=0.0)
    parser.add_argument("--future-recon-eps", type=float, default=1e-6)
    parser.add_argument(
        "--model-variant",
        choices=["target_set", "error_process", "regime_segment_operator"],
        default="target_set",
    )
    parser.add_argument("--use-window-position", action="store_true")
    parser.add_argument("--regime-hidden-dim", type=int, default=64)
    parser.add_argument("--regime-dropout", type=float, default=0.0)
    parser.add_argument("--error-process-dim", type=int, default=64)
    parser.add_argument("--error-process-layers", type=int, default=1)
    parser.add_argument("--error-residual-gate-init", type=float, default=-4.0)
    parser.add_argument("--error-energy-weight", type=float, default=0.0)
    parser.add_argument("--error-smoothness-weight", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--step-loss-weighting",
        choices=[
            "uniform",
            "prefix_risk",
            "region_balanced",
            "step_covariance_balanced",
            "offdiag_block_quadratic",
        ],
        default="uniform",
    )
    parser.add_argument("--step-loss-alpha", type=float, default=0.5)
    parser.add_argument("--step-covariance-beta", type=float, default=0.5)
    parser.add_argument("--step-covariance-eta", type=float, default=0.5)
    parser.add_argument("--step-covariance-eps", type=float, default=1e-6)
    parser.add_argument("--offdiag-block-size", type=int, default=48)
    parser.add_argument("--offdiag-quadratic-weight", type=float, default=0.05)
    parser.add_argument("--offdiag-ridge-eps", type=float, default=1e-3)
    parser.add_argument(
        "--supervision-strategy",
        choices=[
            "horizon_mixed",
            "full_time_mse",
            "r3_prefix_risk",
            "random_future_mask",
            "interval_supervision",
            "conditioned_future_unit_scheduling",
            "component_basis_top",
            "component_basis_balanced",
            "curriculum_units",
        ],
        default="horizon_mixed",
    )
    parser.add_argument("--supervision-pred-len", type=int, default=720)
    parser.add_argument("--supervision-mask-ratio", type=float, default=0.5)
    parser.add_argument("--supervision-block-size", type=int, default=48)
    parser.add_argument("--supervision-interval-min-blocks", type=int, default=1)
    parser.add_argument("--supervision-interval-max-blocks", type=int, default=4)
    parser.add_argument(
        "--supervision-condition",
        choices=["label_novelty", "local_variation", "hybrid"],
        default="label_novelty",
    )
    parser.add_argument("--supervision-condition-top-ratio", type=float, default=0.25)
    parser.add_argument("--supervision-aux-weight", type=float, default=0.1)
    parser.add_argument("--supervision-component-rank", type=int, default=16)
    parser.add_argument("--supervision-component-beta", type=float, default=0.25)
    parser.add_argument("--supervision-component-alpha", type=float, default=0.5)
    parser.add_argument("--supervision-trace-limit", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--run-name", default="PatchEncoderTargetSetDecoder")
    parser.add_argument("--output-root", default="artifacts/runs/phase1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_horizons = parse_horizons(args.target_horizons)
    future_loss_enabled = (
        args.future_align_weight > 0
        or args.future_relation_weight > 0
        or args.future_recon_weight > 0
    )
    if future_loss_enabled and args.future_teacher_layers <= 0:
        raise ValueError("future teacher layers must be positive when future loss weights are enabled.")
    if args.future_confidence_temperature <= 0:
        raise ValueError("future confidence temperature must be positive.")
    if not 0 <= args.future_confidence_floor < 1:
        raise ValueError("future confidence floor must be in [0, 1).")
    if args.future_recon_eps <= 0:
        raise ValueError("future reconstruction eps must be positive.")
    if args.error_energy_weight < 0:
        raise ValueError("error energy weight must be non-negative.")
    if args.error_smoothness_weight < 0:
        raise ValueError("error smoothness weight must be non-negative.")
    if args.regime_hidden_dim <= 0:
        raise ValueError("regime hidden dim must be positive.")
    if args.regime_dropout < 0:
        raise ValueError("regime dropout must be non-negative.")
    if args.step_covariance_beta < 0:
        raise ValueError("step covariance beta must be non-negative.")
    if args.step_covariance_eta < 0:
        raise ValueError("step covariance eta must be non-negative.")
    if args.step_covariance_eps <= 0:
        raise ValueError("step covariance eps must be positive.")
    if args.offdiag_block_size <= 0:
        raise ValueError("offdiag block size must be positive.")
    if args.offdiag_quadratic_weight < 0:
        raise ValueError("offdiag quadratic weight must be non-negative.")
    if args.offdiag_ridge_eps <= 0:
        raise ValueError("offdiag ridge eps must be positive.")
    if args.supervision_pred_len <= 0 or args.supervision_pred_len > args.max_pred_len:
        raise ValueError("supervision pred len must be in (0, max_pred_len].")
    if not 0 < args.supervision_mask_ratio <= 1:
        raise ValueError("supervision mask ratio must be in (0, 1].")
    if args.supervision_block_size <= 0:
        raise ValueError("supervision block size must be positive.")
    if args.supervision_interval_min_blocks <= 0:
        raise ValueError("supervision interval min blocks must be positive.")
    if args.supervision_interval_max_blocks < args.supervision_interval_min_blocks:
        raise ValueError("supervision interval max blocks must be >= min blocks.")
    if not 0 < args.supervision_condition_top_ratio <= 1:
        raise ValueError("supervision condition top ratio must be in (0, 1].")
    if args.supervision_aux_weight < 0:
        raise ValueError("supervision aux weight must be non-negative.")
    if args.supervision_component_rank <= 0:
        raise ValueError("supervision component rank must be positive.")
    if args.supervision_component_beta < 0:
        raise ValueError("supervision component beta must be non-negative.")
    if not 0 <= args.supervision_component_alpha <= 1:
        raise ValueError("supervision component alpha must be in [0, 1].")
    if args.supervision_trace_limit < 0:
        raise ValueError("supervision trace limit must be non-negative.")
    if args.supervision_strategy == "r3_prefix_risk" and args.step_loss_weighting != "prefix_risk":
        raise ValueError("r3_prefix_risk supervision requires --step-loss-weighting prefix_risk.")
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)

    horizon_mixed_training = args.supervision_strategy in HORIZON_MIXED_STRATEGIES
    train_horizons = target_horizons if horizon_mixed_training else [args.supervision_pred_len]
    return_index = args.use_window_position
    train_loaders = build_loaders(
        args.dataset_root,
        args.dataset,
        "train",
        args.seq_len,
        train_horizons,
        args.batch_size,
        drop_last=True,
        return_index=return_index,
    )
    val_loaders = build_loaders(
        args.dataset_root,
        args.dataset,
        "val",
        args.seq_len,
        target_horizons,
        args.batch_size,
        drop_last=False,
        return_index=return_index,
    )
    test_loaders = build_loaders(
        args.dataset_root,
        args.dataset,
        "test",
        args.seq_len,
        target_horizons,
        args.batch_size,
        drop_last=False,
        return_index=return_index,
    )

    model_registry = {
        "target_set": PatchEncoderTargetSetDecoder,
        "error_process": PatchEncoderErrorProcessDecoder,
        "regime_segment_operator": PatchEncoderRegimeSegmentTargetOperator,
    }
    model_cls = model_registry[args.model_variant]
    model_kwargs = {}
    if args.model_variant == "error_process":
        model_kwargs = {
            "error_process_dim": args.error_process_dim,
            "error_process_layers": args.error_process_layers,
            "error_residual_gate_init": args.error_residual_gate_init,
        }
    if args.model_variant == "regime_segment_operator":
        model_kwargs = {
            "regime_hidden_dim": args.regime_hidden_dim,
            "regime_dropout": args.regime_dropout,
        }
    model = model_cls(
        args.seq_len,
        DATASETS[args.dataset].channels,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        encoder_layers=args.encoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        segment_len=args.segment_len,
        max_pred_len=args.max_pred_len,
        target_layers=args.target_layers,
        target_heads=args.target_heads,
        target_d_ff=args.target_d_ff,
        target_interaction_layers=args.target_interaction_layers,
        target_interaction_heads=args.target_interaction_heads or None,
        target_interaction_d_ff=args.target_interaction_d_ff or None,
        readout_dim=args.readout_dim,
        prefix_residual_segments=args.prefix_residual_segments,
        prefix_residual_dropout=args.prefix_residual_dropout,
        future_teacher_layers=args.future_teacher_layers,
        future_teacher_heads=args.future_teacher_heads or None,
        future_teacher_d_ff=args.future_teacher_d_ff or None,
        future_state_dim=args.future_state_dim or None,
        future_recon_normalization=args.future_recon_normalization,
        future_align_weighting=args.future_align_weighting,
        future_confidence_temperature=args.future_confidence_temperature,
        future_confidence_floor=args.future_confidence_floor,
        future_recon_eps=args.future_recon_eps,
        **model_kwargs,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    horizon_label = "mixed_" + "_".join(f"h{horizon}" for horizon in target_horizons)
    run_dir = Path(args.output_root) / args.run_name / args.dataset / horizon_label / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch is None:
        steps_per_epoch = sum(len(loader) for loader in train_loaders.values())
    if args.max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, args.max_train_batches)

    covariance_novelty_share: dict[str, float] | None = None
    covariance_step_weights: torch.Tensor | None = None
    offdiag_block_matrix: torch.Tensor | None = None
    offdiag_block_stats: dict[str, float] = {}
    if args.step_loss_weighting == "step_covariance_balanced":
        covariance_novelty_share = train_split_region_novelty(
            args.dataset_root,
            args.dataset,
            args.seq_len,
            args.max_pred_len,
            args.step_covariance_eps,
        )
        covariance_step_weights = step_covariance_balanced_step_weights(
            args.max_pred_len,
            target_horizons,
            covariance_novelty_share,
            args.step_covariance_beta,
            args.step_covariance_eta,
            args.step_covariance_eps,
            device,
            torch.float32,
        )
    if args.step_loss_weighting == "offdiag_block_quadratic":
        offdiag_block_matrix, offdiag_block_stats = train_split_offdiag_block_matrix(
            args.dataset_root,
            args.dataset,
            args.seq_len,
            args.max_pred_len,
            args.offdiag_block_size,
            args.offdiag_ridge_eps,
        )
        offdiag_block_matrix = offdiag_block_matrix.to(device=device)
    component_basis: torch.Tensor | None = None
    component_eigenvalues: torch.Tensor | None = None
    component_basis_stats: dict[str, float] = {}
    if args.supervision_strategy in {"component_basis_top", "component_basis_balanced", "curriculum_units"}:
        basis_np, eigenvalues_np = train_split_label_basis(
            args.dataset_root,
            args.dataset,
            args.seq_len,
            args.supervision_pred_len,
        )
        component_basis = torch.tensor(basis_np, device=device, dtype=torch.float32)
        component_eigenvalues = torch.tensor(eigenvalues_np, device=device, dtype=torch.float32)
        total_variance = float(np.sum(eigenvalues_np))
        rank = min(args.supervision_component_rank, len(eigenvalues_np))
        component_basis_stats = {
            "component_rank": float(rank),
            "component_top_rank_variance": float(np.sum(eigenvalues_np[:rank]) / max(total_variance, 1e-12)),
            "component_total_variance": total_variance,
        }

    best_val = float("inf")
    best_state = None
    stale_epochs = 0
    log_rows: list[dict[str, object]] = []
    supervision_trace_rows: list[dict[str, float | int | str]] = []
    global_step = 0
    rng = random.Random(args.seed)
    training_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        iterators = {horizon: iter(loader) for horizon, loader in train_loaders.items()}
        losses = []
        pred_losses = []
        future_local_losses = []
        future_relation_losses = []
        future_reconstruction_losses = []
        future_raw_reconstruction_losses = []
        future_normalized_reconstruction_losses = []
        future_confidence_means = []
        error_energy_losses = []
        error_smoothness_losses = []
        unit_losses = []
        time_losses = []
        active_step_ratios = []
        component_ranks = []
        curriculum_phases = []
        supervision_steps = 0
        horizon_counts = {horizon: 0 for horizon in target_horizons}
        for step_in_epoch in range(1, steps_per_epoch + 1):
            global_step += 1
            horizon = rng.choice(target_horizons) if horizon_mixed_training else args.supervision_pred_len
            x, y, window_index_norm = next_batch(train_loaders, iterators, horizon)
            x = x.float().to(device)
            y = y.float().to(device)
            window_index_norm = tensor_to_device(window_index_norm, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                x,
                pred_len=horizon,
                future_y=y if future_loss_enabled else None,
                return_components=future_loss_enabled or args.model_variant == "error_process",
                window_index_norm=window_index_norm,
            )
            if future_loss_enabled or args.model_variant == "error_process":
                if not isinstance(output, dict):
                    raise TypeError("Expected component dict from model.")
                pred = output["prediction"]
            else:
                if isinstance(output, dict):
                    raise TypeError("Expected tensor prediction.")
                pred = output
            if horizon_mixed_training:
                pred_loss = weighted_mse_loss(
                    pred,
                    y,
                    args.max_pred_len,
                    target_horizons,
                    args.step_loss_weighting,
                    args.step_loss_alpha,
                    covariance_step_weights,
                    offdiag_block_matrix,
                    args.offdiag_block_size,
                    args.offdiag_quadratic_weight,
                )
                time_loss = pred_loss
                unit_loss = pred_loss
                unit = SupervisionUnit("horizon_mixed", horizon, 1.0)
                horizon_counts[horizon] += 1
            else:
                pred_loss, time_loss, unit_loss, unit = horizon_decoupled_supervision_loss(
                    pred,
                    y,
                    x,
                    args,
                    rng,
                    epoch,
                    component_basis,
                    component_eigenvalues,
                )
                supervision_steps += 1
            loss = pred_loss
            if future_loss_enabled:
                loss = loss + args.future_align_weight * output["future_local_alignment_loss"]
                loss = loss + args.future_relation_weight * output["future_relation_alignment_loss"]
                loss = loss + args.future_recon_weight * output["future_reconstruction_loss"]
                future_local_losses.append(float(output["future_local_alignment_loss"].detach().cpu()))
                future_relation_losses.append(float(output["future_relation_alignment_loss"].detach().cpu()))
                future_reconstruction_losses.append(float(output["future_reconstruction_loss"].detach().cpu()))
                future_raw_reconstruction_losses.append(
                    float(output["future_raw_reconstruction_loss"].detach().cpu())
                )
                future_normalized_reconstruction_losses.append(
                    float(output["future_normalized_reconstruction_loss"].detach().cpu())
                )
                future_confidence_means.append(float(output["future_alignment_confidence_mean"].detach().cpu()))
            if args.model_variant == "error_process":
                error_energy_loss = torch.mean(output["error_residual_norm"] * output["error_residual_norm"])
                error_smoothness_loss = residual_smoothness_torch(output["error_residual_norm"])
                loss = loss + args.error_energy_weight * error_energy_loss
                loss = loss + args.error_smoothness_weight * error_smoothness_loss
                error_energy_losses.append(float(error_energy_loss.detach().cpu()))
                error_smoothness_losses.append(float(error_smoothness_loss.detach().cpu()))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            pred_losses.append(float(pred_loss.detach().cpu()))
            unit_losses.append(float(unit_loss.detach().cpu()))
            time_losses.append(float(time_loss.detach().cpu()))
            active_step_ratios.append(unit.mask_ratio)
            component_ranks.append(float(unit.component_rank))
            curriculum_phases.append(unit.curriculum_phase)
            if len(supervision_trace_rows) < args.supervision_trace_limit:
                supervision_trace_rows.append(
                    {
                        "epoch": epoch,
                        "step_in_epoch": step_in_epoch,
                        "global_step": global_step,
                        "strategy": args.supervision_strategy,
                        "supervision_pred_len": horizon,
                        "unit_type": unit.unit_type,
                        "active_steps": unit.active_steps,
                        "mask_ratio": unit.mask_ratio,
                        "interval_start": unit.interval_start,
                        "interval_end": unit.interval_end,
                        "component_rank": unit.component_rank,
                        "curriculum_phase": unit.curriculum_phase,
                        "condition_type": unit.condition_type,
                        "condition_top_blocks": unit.condition_top_blocks,
                        "condition_mean_score": unit.condition_mean_score,
                        "auxiliary_weight": unit.auxiliary_weight,
                        "loss_time": float(time_loss.detach().cpu()),
                        "loss_unit": float(unit_loss.detach().cpu()),
                        "loss_total": float(loss.detach().cpu()),
                    }
                )

        val_rows = []
        for horizon in target_horizons:
            metrics, _, _, _ = evaluate(model, val_loaders[horizon], device, horizon, args.max_eval_batches)
            val_rows.append({"target_horizon": horizon, **metrics})
        mean_val_mse = float(np.mean([row["mse"] for row in val_rows]))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_prediction_loss": float(np.mean(pred_losses)),
            "train_future_local_alignment_loss": float(np.mean(future_local_losses))
            if future_local_losses
            else 0.0,
            "train_future_relation_alignment_loss": float(np.mean(future_relation_losses))
            if future_relation_losses
            else 0.0,
            "train_future_reconstruction_loss": float(np.mean(future_reconstruction_losses))
            if future_reconstruction_losses
            else 0.0,
            "train_future_raw_reconstruction_loss": float(np.mean(future_raw_reconstruction_losses))
            if future_raw_reconstruction_losses
            else 0.0,
            "train_future_normalized_reconstruction_loss": float(
                np.mean(future_normalized_reconstruction_losses)
            )
            if future_normalized_reconstruction_losses
            else 0.0,
            "train_future_alignment_confidence_mean": float(np.mean(future_confidence_means))
            if future_confidence_means
            else 0.0,
            "train_error_energy_loss": float(np.mean(error_energy_losses)) if error_energy_losses else 0.0,
            "train_error_smoothness_loss": float(np.mean(error_smoothness_losses))
            if error_smoothness_losses
            else 0.0,
            "train_supervision_strategy": args.supervision_strategy,
            "train_supervision_pred_len": args.supervision_pred_len,
            "train_unit_loss": float(np.mean(unit_losses)) if unit_losses else 0.0,
            "train_time_loss": float(np.mean(time_losses)) if time_losses else 0.0,
            "train_active_step_ratio": float(np.mean(active_step_ratios)) if active_step_ratios else 0.0,
            "train_component_rank": float(np.mean(component_ranks)) if component_ranks else 0.0,
            "train_supervision_steps": supervision_steps,
            "train_curriculum_phase": max(set(curriculum_phases), key=curriculum_phases.count)
            if curriculum_phases
            else "none",
            "val_mean_mse": mean_val_mse,
            "epoch_elapsed_sec": time.perf_counter() - epoch_start,
            "elapsed_sec": time.perf_counter() - training_start,
        }
        for horizon in target_horizons:
            row[f"train_steps_h{horizon}"] = horizon_counts[horizon]
        for val_row in val_rows:
            horizon = int(val_row["target_horizon"])
            row[f"val_mse_h{horizon}"] = float(val_row["mse"])
            row[f"val_mae_h{horizon}"] = float(val_row["mae"])
        log_rows.append(row)
        write_csv(run_dir / "training_log.csv", log_rows)
        write_csv(run_dir / "supervision_trace.csv", supervision_trace_rows)
        mean_epoch_sec = float(row["elapsed_sec"]) / epoch
        eta_sec = max(args.epochs - epoch, 0) * mean_epoch_sec
        print(
            "epoch_progress "
            f"run_name={args.run_name} dataset={args.dataset} "
            f"epoch={epoch}/{args.epochs} val_mean_mse={mean_val_mse:.6f} "
            f"elapsed_sec={float(row['elapsed_sec']):.1f} eta_sec={eta_sec:.1f}",
            flush=True,
        )

        if mean_val_mse < best_val:
            best_val = mean_val_mse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    target_metric_rows = []
    for horizon in target_horizons:
        metrics, pred_np, true_np, components = evaluate(
            model,
            test_loaders[horizon],
            device,
            horizon,
            args.max_eval_batches,
        )
        target_metric_rows.append({"target_horizon": horizon, **metrics})
        eval_dir = run_dir / f"h{horizon}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        if args.save_predictions:
            np.savez_compressed(eval_dir / "predictions_test.npz", pred=pred_np, true=true_np)
        (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        write_csv(eval_dir / "metrics_by_horizon.csv", metrics_by_horizon(pred_np, true_np))
        write_csv(eval_dir / "metrics_by_segment.csv", metrics_by_segment(pred_np, true_np))
        write_csv(eval_dir / "target_state_similarity.csv", target_state_similarity(components))
        write_csv(eval_dir / "target_conditioning_stats.csv", target_conditioning_stats(components))
        if args.prefix_residual_segments > 0:
            write_csv(eval_dir / "prefix_residual_stats.csv", prefix_residual_stats(components))
        if args.model_variant == "error_process":
            write_csv(
                eval_dir / "error_process_stats.csv",
                error_process_stats(pred_np, true_np, components, args.segment_len),
            )
        if args.future_teacher_layers > 0:
            alignment_rows, leakage_audit = future_alignment_stats(
                model,
                test_loaders[horizon],
                device,
                horizon,
                args.max_eval_batches,
            )
            write_csv(eval_dir / "future_alignment_stats.csv", alignment_rows)
            (eval_dir / "future_leakage_audit.json").write_text(json.dumps(leakage_audit, indent=2))
        if args.model_variant == "regime_segment_operator":
            write_csv(eval_dir / "regime_segment_operator_stats.csv", regime_segment_operator_stats(components))
            write_csv(eval_dir / "regime_feature_stats.csv", regime_feature_stats(components))

    long_horizon = max(target_horizons)
    short_horizons = [horizon for horizon in target_horizons if horizon < long_horizon]
    if short_horizons:
        prefix_rows = prefix_consistency(
            model,
            test_loaders[long_horizon],
            device,
            short_horizons,
            long_horizon,
            args.max_eval_batches,
        )
        write_csv(run_dir / "prefix_consistency.csv", prefix_rows)

    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    write_csv(run_dir / "metrics_by_target_horizon.csv", target_metric_rows)
    write_csv(run_dir / "training_log.csv", log_rows)
    write_csv(run_dir / "supervision_trace.csv", supervision_trace_rows)
    write_csv(
        run_dir / "objective_weight_stats.csv",
        objective_weight_stats(
            args.max_pred_len,
            target_horizons,
            args.step_loss_weighting,
            args.step_loss_alpha,
            covariance_step_weights,
            covariance_novelty_share,
        ),
    )
    if offdiag_block_matrix is not None:
        matrix_np = offdiag_block_matrix.detach().cpu().numpy()
        matrix_rows = []
        for row_index in range(matrix_np.shape[0]):
            for col_index in range(matrix_np.shape[1]):
                matrix_rows.append(
                    {
                        "row_block": row_index,
                        "col_block": col_index,
                        "value": float(matrix_np[row_index, col_index]),
                        "is_diagonal": row_index == col_index,
                    }
                )
        write_csv(run_dir / "offdiag_block_matrix.csv", matrix_rows)
    effective_config = vars(args)
    effective_config["target_horizons"] = target_horizons
    effective_config["evaluation_target_horizons"] = target_horizons
    effective_config["train_horizons_effective"] = train_horizons
    effective_config["training_evaluation_decoupled"] = not horizon_mixed_training
    effective_config["supervision_unit_config"] = {
        "strategy": args.supervision_strategy,
        "supervision_pred_len": args.supervision_pred_len,
        "mask_ratio": args.supervision_mask_ratio,
        "block_size": args.supervision_block_size,
        "interval_min_blocks": args.supervision_interval_min_blocks,
        "interval_max_blocks": args.supervision_interval_max_blocks,
        "condition": args.supervision_condition,
        "condition_top_ratio": args.supervision_condition_top_ratio,
        "aux_weight": args.supervision_aux_weight,
        "component_rank": args.supervision_component_rank,
        "component_beta": args.supervision_component_beta,
        "component_alpha": args.supervision_component_alpha,
        "trace_limit": args.supervision_trace_limit,
    }
    effective_config["steps_per_epoch_effective"] = steps_per_epoch
    effective_config["step_covariance_novelty_share"] = covariance_novelty_share or {}
    effective_config["offdiag_block_stats"] = offdiag_block_stats
    effective_config["component_basis_stats"] = component_basis_stats
    (run_dir / "effective_config.json").write_text(json.dumps(effective_config, indent=2))
    env = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    (run_dir / "environment.json").write_text(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
