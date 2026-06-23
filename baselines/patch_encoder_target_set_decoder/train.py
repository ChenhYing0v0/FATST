from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import DATASETS, ForecastDataset
from model import PatchEncoderErrorProcessDecoder, PatchEncoderTargetSetDecoder


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
    for start, end in [(1, 96), (97, 192), (193, 336), (337, 720)]:
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
    for start, end in [(1, 96), (97, 192), (193, 336), (337, 720)]:
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
        for batch_index, (x, y) in enumerate(loader, start=1):
            x = x.float().to(device)
            y = y.float().to(device)
            no_future = model(x, pred_len=pred_len, return_components=True)
            true_future = model(x, pred_len=pred_len, future_y=y, return_components=True)
            shuffled_future = model(
                x,
                pred_len=pred_len,
                future_y=y.flip(0),
                return_components=True,
            )
            zero_future = model(
                x,
                pred_len=pred_len,
                future_y=torch.zeros_like(y),
                return_components=True,
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
    mode: str,
    alpha: float,
) -> torch.Tensor:
    if mode == "uniform":
        return torch.mean((pred - true) ** 2)
    if mode != "prefix_risk":
        raise ValueError(f"Unknown step loss weighting mode: {mode}")
    if alpha < 0:
        raise ValueError("step_loss_alpha must be non-negative.")

    horizon = pred.shape[1]
    step = torch.arange(1, horizon + 1, device=pred.device, dtype=pred.dtype)
    full_step = torch.arange(1, max_pred_len + 1, device=pred.device, dtype=pred.dtype)
    weights = torch.pow(step / float(max_pred_len), -alpha)
    full_weights = torch.pow(full_step / float(max_pred_len), -alpha)
    weights = weights / torch.mean(full_weights)
    return torch.mean((pred - true) ** 2 * weights.view(1, horizon, 1))


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
        for batch_index, (x, y) in enumerate(loader, start=1):
            x = x.float().to(device)
            y = y.float().to(device)
            output = model(x, pred_len=pred_len, return_components=True)
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
        for batch_index, (x, _) in enumerate(loader, start=1):
            x = x.float().to(device)
            long_pred = model(x, pred_len=long_horizon)
            if isinstance(long_pred, dict):
                raise TypeError("Expected tensor prediction.")
            for horizon in short_horizons:
                short_pred = model(x, pred_len=horizon)
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


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
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
) -> dict[int, DataLoader]:
    return {
        horizon: DataLoader(
            ForecastDataset(dataset_root, dataset, split, seq_len, horizon),
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=drop_last,
        )
        for horizon in horizons
    }


def next_batch(
    loaders: dict[int, DataLoader],
    iterators: dict[int, Iterator[tuple[torch.Tensor, torch.Tensor]]],
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        return next(iterators[horizon])
    except StopIteration:
        iterators[horizon] = iter(loaders[horizon])
        return next(iterators[horizon])


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
        choices=["target_set", "error_process"],
        default="target_set",
    )
    parser.add_argument("--error-process-dim", type=int, default=64)
    parser.add_argument("--error-process-layers", type=int, default=1)
    parser.add_argument("--error-residual-gate-init", type=float, default=-4.0)
    parser.add_argument("--error-energy-weight", type=float, default=0.0)
    parser.add_argument("--error-smoothness-weight", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--step-loss-weighting", choices=["uniform", "prefix_risk"], default="uniform")
    parser.add_argument("--step-loss-alpha", type=float, default=0.5)
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
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)

    train_loaders = build_loaders(
        args.dataset_root,
        args.dataset,
        "train",
        args.seq_len,
        target_horizons,
        args.batch_size,
        drop_last=True,
    )
    val_loaders = build_loaders(
        args.dataset_root,
        args.dataset,
        "val",
        args.seq_len,
        target_horizons,
        args.batch_size,
        drop_last=False,
    )
    test_loaders = build_loaders(
        args.dataset_root,
        args.dataset,
        "test",
        args.seq_len,
        target_horizons,
        args.batch_size,
        drop_last=False,
    )

    model_cls = PatchEncoderErrorProcessDecoder if args.model_variant == "error_process" else PatchEncoderTargetSetDecoder
    model_kwargs = {}
    if args.model_variant == "error_process":
        model_kwargs = {
            "error_process_dim": args.error_process_dim,
            "error_process_layers": args.error_process_layers,
            "error_residual_gate_init": args.error_residual_gate_init,
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

    best_val = float("inf")
    best_state = None
    stale_epochs = 0
    log_rows: list[dict[str, float]] = []
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
        horizon_counts = {horizon: 0 for horizon in target_horizons}
        for _ in range(steps_per_epoch):
            horizon = rng.choice(target_horizons)
            x, y = next_batch(train_loaders, iterators, horizon)
            x = x.float().to(device)
            y = y.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                x,
                pred_len=horizon,
                future_y=y if future_loss_enabled else None,
                return_components=future_loss_enabled or args.model_variant == "error_process",
            )
            if future_loss_enabled or args.model_variant == "error_process":
                if not isinstance(output, dict):
                    raise TypeError("Expected component dict from model.")
                pred = output["prediction"]
            else:
                if isinstance(output, dict):
                    raise TypeError("Expected tensor prediction.")
                pred = output
            pred_loss = weighted_mse_loss(
                pred,
                y,
                args.max_pred_len,
                args.step_loss_weighting,
                args.step_loss_alpha,
            )
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
            horizon_counts[horizon] += 1

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
    effective_config = vars(args)
    effective_config["target_horizons"] = target_horizons
    effective_config["steps_per_epoch_effective"] = steps_per_epoch
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
