#!/usr/bin/env python3
"""Run StageC D1 PMFO/PIR diagnostics without training a new forecast model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
if str(TIMEALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEALIGN_ROOT))

from data_provider.data_factory import data_provider  # noqa: E402
from models.TimeAlign import Model  # noqa: E402


SERIES_LENGTH = 720
DCT_RANK = 256
DCT_LEVELS = (8, 24, 72, 144, 256)
BLOCK_SIZES = (90, 30, 10, 5, 1)
BLOCK_RANKS = tuple(SERIES_LENGTH // size for size in BLOCK_SIZES)
MEASURE_NAMES = ("delta_720", "uniform_h", "log_uniform_h", "benchmark_h")
FEATURE_NAMES = ("full_hidden", "patch_mean", "patch_shuffled", "raw_history")
SOURCE_NAMES = ("label", "residual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-root", type=Path)
    parser.add_argument("--phase-b-root", type=Path)
    parser.add_argument("--phase-c-root", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--dataset",
        choices=["Weather", "ETTm1", "ETTh2"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-batches", type=int, default=8)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--gradient-batches", type=int, default=2)
    parser.add_argument("--ridge-lambda", type=float, default=1e-2)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if args.synthetic_smoke:
        return args
    required = {
        "phase_a_root": args.phase_a_root,
        "phase_b_root": args.phase_b_root,
        "phase_c_root": args.phase_c_root,
        "contract": args.contract,
        "output_dir": args.output_dir,
        "dataset": args.dataset,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    for name in ("train_batches", "val_batches", "gradient_batches"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.ridge_lambda <= 0.0:
        parser.error("--ridge-lambda must be positive")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def run_dir(args: argparse.Namespace, profile: dict[str, Any], seed: int) -> Path:
    name = profile["profile"]
    if seed == 2021 and name.endswith("_medium"):
        run_name = (
            f"SC0DAP_R2A_r2a_p{profile['patch_num']}_"
            f"d{profile['d_model']}_ff{profile['d_ff']}"
        )
        root = args.phase_a_root
    elif seed == 2021:
        run_name = f"SC0DAP_R2B_{name}"
        root = args.phase_b_root
    else:
        run_name = f"SC0DAP_R2C_{name}"
        root = args.phase_c_root
    return root / run_name / args.dataset / "h720_full" / f"seed{seed}"


def dct_basis(length: int, rank: int, device: torch.device) -> torch.Tensor:
    steps = torch.arange(length, dtype=torch.float32, device=device) + 0.5
    freqs = torch.arange(rank, dtype=torch.float32, device=device)
    basis = torch.cos(torch.pi * torch.outer(steps, freqs) / float(length))
    basis[:, 0] *= (1.0 / float(length)) ** 0.5
    if rank > 1:
        basis[:, 1:] *= (2.0 / float(length)) ** 0.5
    return basis


def random_basis(length: int, rank: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(20260713)
    matrix = torch.randn(length, rank, generator=generator)
    basis, _ = torch.linalg.qr(matrix, mode="reduced")
    return basis.to(device=device)


def block_projection(values: torch.Tensor, block_size: int) -> torch.Tensor:
    if values.shape[-1] % block_size != 0:
        raise ValueError("block size must divide the temporal length")
    shape = values.shape
    blocks = values.reshape(*shape[:-1], shape[-1] // block_size, block_size)
    projected = blocks.mean(dim=-1, keepdim=True).expand_as(blocks)
    return projected.reshape(shape)


def block_coefficients(values: torch.Tensor, block_size: int = 5) -> torch.Tensor:
    blocks = values.reshape(values.shape[0], SERIES_LENGTH // block_size, block_size)
    return blocks.sum(dim=-1) / math.sqrt(float(block_size))


def block_coefficients_to_series(coefficients: torch.Tensor, block_size: int = 5) -> torch.Tensor:
    normalized = coefficients / math.sqrt(float(block_size))
    return normalized.unsqueeze(-1).expand(-1, -1, block_size).reshape(-1, SERIES_LENGTH)


def measure_weights(length: int, device: torch.device) -> dict[str, torch.Tensor]:
    horizons = torch.arange(1, length + 1, dtype=torch.float32, device=device)
    distributions: dict[str, torch.Tensor] = {}
    delta = torch.zeros(length, device=device)
    delta[-1] = 1.0
    distributions["delta_720"] = delta
    distributions["uniform_h"] = torch.full((length,), 1.0 / float(length), device=device)
    log_uniform = horizons.reciprocal()
    distributions["log_uniform_h"] = log_uniform / log_uniform.sum()
    benchmark = torch.zeros(length, device=device)
    for horizon in (96, 192, 336, 720):
        benchmark[horizon - 1] = 0.25
    distributions["benchmark_h"] = benchmark

    weights: dict[str, torch.Tensor] = {}
    for name, probabilities in distributions.items():
        per_horizon = probabilities / horizons
        weights[name] = torch.flip(
            torch.cumsum(torch.flip(per_horizon, dims=[0]), dim=0),
            dims=[0],
        )
        if not torch.allclose(weights[name].sum(), torch.ones((), device=device), atol=1e-6):
            raise RuntimeError(f"step weights do not sum to one for {name}")
    return weights


def weighted_mse(error: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (error.square() * weights.view(1, -1, 1)).sum(dim=1).mean()


def projected_increment_risk(error: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    error_bct = error.permute(0, 2, 1)
    previous = torch.zeros_like(error_bct)
    risk = error.new_zeros(())
    for block_size in BLOCK_SIZES:
        current = block_projection(error_bct, block_size)
        increment = current - previous
        risk = risk + weighted_mse(increment.permute(0, 2, 1), weights)
        previous = current
    return risk


def normalized_history_and_target(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = batch_x.mean(dim=1, keepdim=True).detach()
    std = torch.sqrt(batch_x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    history = (batch_x - mean) / std
    target = (batch_y[:, -SERIES_LENGTH:, :] - mean) / std
    return history, target, std


def shuffled_patch_feature(memory: torch.Tensor, seed: int) -> torch.Tensor:
    memory_cpu = memory.detach().cpu()
    rows, patches, width = memory_cpu.shape
    generator = torch.Generator().manual_seed(seed)
    permutations = torch.rand(rows, patches, generator=generator).argsort(dim=1)
    gather_index = permutations.unsqueeze(-1).expand(rows, patches, width)
    return torch.gather(memory_cpu, dim=1, index=gather_index).flatten(start_dim=1)


def collect_split(
    model: Model,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
    shuffle_seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    feature_parts = {name: [] for name in FEATURE_NAMES}
    source_parts = {name: [] for name in SOURCE_NAMES}
    model.eval()
    with torch.no_grad():
        for batch_index, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            history, target, std = normalized_history_and_target(batch_x, batch_y)
            prediction, _recon, _alignment = model(
                batch_x,
                batch_y[:, -SERIES_LENGTH:, :],
                is_training=False,
                target_prefix=SERIES_LENGTH,
            )
            prediction_normalized = (
                prediction - batch_x.mean(dim=1, keepdim=True).detach()
            ) / std
            residual = target - prediction_normalized
            memory = model.encode_history(batch_x)
            batch, channels, patches, width = memory.shape
            memory_rows = memory.reshape(batch * channels, patches, width)
            history_rows = history.permute(0, 2, 1).reshape(batch * channels, SERIES_LENGTH)

            feature_parts["full_hidden"].append(memory_rows.flatten(start_dim=1).cpu())
            feature_parts["patch_mean"].append(memory_rows.mean(dim=1).cpu())
            feature_parts["patch_shuffled"].append(
                shuffled_patch_feature(memory_rows, shuffle_seed + batch_index)
            )
            feature_parts["raw_history"].append(history_rows.cpu())
            source_parts["label"].append(
                target.permute(0, 2, 1).reshape(batch * channels, SERIES_LENGTH).cpu()
            )
            source_parts["residual"].append(
                residual.permute(0, 2, 1).reshape(batch * channels, SERIES_LENGTH).cpu()
            )
    if not feature_parts["full_hidden"]:
        raise RuntimeError("split collection produced no batches")
    features = {name: torch.cat(parts, dim=0) for name, parts in feature_parts.items()}
    sources = {name: torch.cat(parts, dim=0) for name, parts in source_parts.items()}
    return features, sources


def source_targets(
    sources: dict[str, torch.Tensor],
    dct: torch.Tensor,
) -> tuple[torch.Tensor, dict[tuple[str, str], slice]]:
    parts: list[torch.Tensor] = []
    slices: dict[tuple[str, str], slice] = {}
    cursor = 0
    dct_cpu = dct.cpu()
    for source_name in SOURCE_NAMES:
        values = sources[source_name]
        dct_values = values @ dct_cpu
        slices[(source_name, "dct")] = slice(cursor, cursor + DCT_RANK)
        cursor += DCT_RANK
        parts.append(dct_values)
        block_values = block_coefficients(values)
        slices[(source_name, "block")] = slice(cursor, cursor + BLOCK_RANKS[-2])
        cursor += BLOCK_RANKS[-2]
        parts.append(block_values)
    return torch.cat(parts, dim=1), slices


def ridge_predict(
    train_features: torch.Tensor,
    val_features: torch.Tensor,
    train_targets: torch.Tensor,
    ridge_lambda: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_train = train_features.to(device)
    x_val = val_features.to(device)
    y_train = train_targets.to(device)
    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    y_mean = y_train.mean(dim=0, keepdim=True)
    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std
    y_centered = y_train - y_mean
    sample_count = float(x_train.shape[0])
    gram = x_train.transpose(0, 1) @ x_train / sample_count
    rhs = x_train.transpose(0, 1) @ y_centered / sample_count
    regularized = gram + ridge_lambda * torch.eye(gram.shape[0], device=device)
    weights = torch.linalg.solve(regularized, rhs)
    prediction = x_val @ weights + y_mean
    return prediction.cpu(), y_mean.cpu()


def r2_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    baseline: torch.Tensor,
) -> tuple[float, float, float, float]:
    sse = float((prediction - target).square().sum().item())
    sst = float((target - baseline).square().sum().item())
    r2 = 1.0 - sse / max(sst, 1e-12)
    nrmse = math.sqrt(sse / max(sst, 1e-12))
    return r2, nrmse, sse, sst


def probe_metrics(
    dataset: str,
    seed: int,
    train_features: dict[str, torch.Tensor],
    val_features: dict[str, torch.Tensor],
    train_sources: dict[str, torch.Tensor],
    val_sources: dict[str, torch.Tensor],
    dct: torch.Tensor,
    ridge_lambda: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    train_targets, slices = source_targets(train_sources, dct)
    val_targets, _ = source_targets(val_sources, dct)
    rows: list[dict[str, Any]] = []
    for feature_name in FEATURE_NAMES:
        prediction, train_mean = ridge_predict(
            train_features[feature_name],
            val_features[feature_name],
            train_targets,
            ridge_lambda,
            device,
        )
        for source_name in SOURCE_NAMES:
            dct_slice = slices[(source_name, "dct")]
            source_prediction = prediction[:, dct_slice]
            source_target = val_targets[:, dct_slice]
            source_baseline = train_mean[:, dct_slice]
            start = 0
            for level_index, end in enumerate(DCT_LEVELS):
                r2, nrmse, sse, sst = r2_metrics(
                    source_prediction[:, start:end],
                    source_target[:, start:end],
                    source_baseline[:, start:end],
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "feature": feature_name,
                        "source": source_name,
                        "family": "dct",
                        "level_index": level_index,
                        "level_start_rank": start,
                        "level_end_rank": end,
                        "r2": r2,
                        "nrmse": nrmse,
                        "sse": sse,
                        "sst": sst,
                        "train_rows": train_features[feature_name].shape[0],
                        "val_rows": val_features[feature_name].shape[0],
                    }
                )
                start = end

            block_slice = slices[(source_name, "block")]
            prediction_series = block_coefficients_to_series(prediction[:, block_slice])
            target_series = block_coefficients_to_series(val_targets[:, block_slice])
            baseline_series = block_coefficients_to_series(train_mean[:, block_slice])
            previous_prediction = torch.zeros_like(prediction_series)
            previous_target = torch.zeros_like(target_series)
            previous_baseline = torch.zeros_like(baseline_series)
            for level_index, block_size in enumerate(BLOCK_SIZES[:-1]):
                current_prediction = block_projection(prediction_series, block_size)
                current_target = block_projection(target_series, block_size)
                current_baseline = block_projection(baseline_series, block_size)
                r2, nrmse, sse, sst = r2_metrics(
                    current_prediction - previous_prediction,
                    current_target - previous_target,
                    current_baseline - previous_baseline,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "feature": feature_name,
                        "source": source_name,
                        "family": "block",
                        "level_index": level_index,
                        "level_start_rank": 0 if level_index == 0 else BLOCK_RANKS[level_index - 1],
                        "level_end_rank": BLOCK_RANKS[level_index],
                        "r2": r2,
                        "nrmse": nrmse,
                        "sse": sse,
                        "sst": sst,
                        "train_rows": train_features[feature_name].shape[0],
                        "val_rows": val_features[feature_name].shape[0],
                    }
                )
                previous_prediction = current_prediction
                previous_target = current_target
                previous_baseline = current_baseline
    return rows


def orthonormalized_learned_basis(model: Model) -> tuple[torch.Tensor, torch.Tensor]:
    basis = model.learned_temporal_basis.detach().float().cpu()
    singular_values = torch.linalg.svdvals(basis)
    q_basis, _ = torch.linalg.qr(basis, mode="reduced")
    return q_basis, singular_values


def energy_structure_rows(
    dataset: str,
    seed: int,
    sources: dict[str, torch.Tensor],
    dct: torch.Tensor,
    random_q: torch.Tensor,
    learned_q: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bases = {"dct": dct.cpu(), "random": random_q.cpu()}
    for source_name, values in sources.items():
        total_energy = float(values.square().sum().item())
        for family, basis in bases.items():
            coefficients = values @ basis
            start = 0
            cumulative_energy = 0.0
            for level_index, end in enumerate(DCT_LEVELS):
                increment_energy = float(coefficients[:, start:end].square().sum().item())
                cumulative_energy += increment_energy
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "source": source_name,
                        "family": family,
                        "level_index": level_index,
                        "level_name": f"rank_{start}_{end}",
                        "cumulative_rank": end,
                        "increment_energy_share": increment_energy / max(total_energy, 1e-12),
                        "cumulative_energy_share": cumulative_energy / max(total_energy, 1e-12),
                        "reconstruction_error_share": 1.0 - cumulative_energy / max(total_energy, 1e-12),
                    }
                )
                start = end
            complement = max(total_energy - cumulative_energy, 0.0)
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "source": source_name,
                    "family": family,
                    "level_index": len(DCT_LEVELS),
                    "level_name": "rank_256_720_complement",
                    "cumulative_rank": SERIES_LENGTH,
                    "increment_energy_share": complement / max(total_energy, 1e-12),
                    "cumulative_energy_share": 1.0,
                    "reconstruction_error_share": 0.0,
                }
            )

        previous = torch.zeros_like(values)
        cumulative_energy = 0.0
        for level_index, (block_size, rank) in enumerate(zip(BLOCK_SIZES, BLOCK_RANKS, strict=True)):
            current = block_projection(values, block_size)
            increment = current - previous
            increment_energy = float(increment.square().sum().item())
            cumulative_energy += increment_energy
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "source": source_name,
                    "family": "block",
                    "level_index": level_index,
                    "level_name": f"block_{block_size}",
                    "cumulative_rank": rank,
                    "increment_energy_share": increment_energy / max(total_energy, 1e-12),
                    "cumulative_energy_share": cumulative_energy / max(total_energy, 1e-12),
                    "reconstruction_error_share": max(1.0 - cumulative_energy / max(total_energy, 1e-12), 0.0),
                }
            )
            previous = current

        learned_energy = float((values @ learned_q).square().sum().item())
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "seed": seed,
                    "source": source_name,
                    "family": "learned_basis_subspace",
                    "level_index": 0,
                    "level_name": "captured_rank_256",
                    "cumulative_rank": DCT_RANK,
                    "increment_energy_share": learned_energy / max(total_energy, 1e-12),
                    "cumulative_energy_share": learned_energy / max(total_energy, 1e-12),
                    "reconstruction_error_share": 1.0 - learned_energy / max(total_energy, 1e-12),
                },
                {
                    "dataset": dataset,
                    "seed": seed,
                    "source": source_name,
                    "family": "learned_basis_subspace",
                    "level_index": 1,
                    "level_name": "orthogonal_complement",
                    "cumulative_rank": SERIES_LENGTH,
                    "increment_energy_share": 1.0 - learned_energy / max(total_energy, 1e-12),
                    "cumulative_energy_share": 1.0,
                    "reconstruction_error_share": 0.0,
                },
            ]
        )
    return rows


def basis_geometry_rows(
    dataset: str,
    seed: int,
    model: Model,
    dct: torch.Tensor,
    learned_q: torch.Tensor,
    singular_values: torch.Tensor,
) -> list[dict[str, Any]]:
    basis = model.learned_temporal_basis.detach().float().cpu()
    squared = singular_values.square()
    probabilities = squared / squared.sum().clamp_min(1e-12)
    effective_rank = float(torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item())
    stable_rank = float((squared.sum() / squared.max()).item())
    condition = float((singular_values.max() / singular_values.min().clamp_min(1e-12)).item())
    column_energy = basis.square()
    column_prob = column_energy / column_energy.sum(dim=0, keepdim=True).clamp_min(1e-12)
    entropy = -(column_prob * column_prob.clamp_min(1e-12).log()).sum(dim=0) / math.log(SERIES_LENGTH)
    sorted_energy = torch.sort(column_prob, dim=0, descending=True).values
    support_90 = (sorted_energy.cumsum(dim=0) < 0.9).sum(dim=0).float().add(1.0) / SERIES_LENGTH
    smoothness = float(
        (basis[1:] - basis[:-1]).square().mean().div(basis.square().mean().clamp_min(1e-12)).item()
    )
    rows = [
        {"dataset": dataset, "seed": seed, "metric": "effective_rank", "scale": "all", "value": effective_rank},
        {"dataset": dataset, "seed": seed, "metric": "stable_rank", "scale": "all", "value": stable_rank},
        {"dataset": dataset, "seed": seed, "metric": "condition_number", "scale": "all", "value": condition},
        {
            "dataset": dataset,
            "seed": seed,
            "metric": "column_entropy_mean",
            "scale": "all",
            "value": float(entropy.mean().item()),
        },
        {
            "dataset": dataset,
            "seed": seed,
            "metric": "support_90_fraction_mean",
            "scale": "all",
            "value": float(support_90.mean().item()),
        },
        {
            "dataset": dataset,
            "seed": seed,
            "metric": "normalized_first_difference_energy",
            "scale": "all",
            "value": smoothness,
        },
    ]
    dct_cpu = dct.cpu()
    for rank in DCT_LEVELS:
        overlap = learned_q.transpose(0, 1) @ dct_cpu[:, :rank]
        normalized_overlap = float(overlap.square().sum().item() / float(rank))
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "metric": "principal_overlap_with_dct",
                "scale": f"rank_{rank}",
                "value": normalized_overlap,
            }
        )
    block_indicator = torch.zeros(SERIES_LENGTH, BLOCK_RANKS[-2])
    for index in range(BLOCK_RANKS[-2]):
        block_indicator[index * 5 : (index + 1) * 5, index] = 1.0 / math.sqrt(5.0)
    overlap = learned_q.transpose(0, 1) @ block_indicator
    rows.append(
        {
            "dataset": dataset,
            "seed": seed,
            "metric": "principal_overlap_with_block",
            "scale": "rank_144",
            "value": float(overlap.square().sum().item() / float(BLOCK_RANKS[-2])),
        }
    )
    return rows


def parameter_groups(model: Model) -> tuple[list[tuple[str, torch.nn.Parameter]], dict[str, list[int]]]:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    groups = {"encoder": [], "coeff": [], "basis": [], "all": list(range(len(named)))}
    for index, (name, _parameter) in enumerate(named):
        if name.startswith("patch_emb_x") or name.startswith("encoder."):
            groups["encoder"].append(index)
        if name.startswith("learned_basis_coeff"):
            groups["coeff"].append(index)
        if name.startswith("learned_temporal_basis") or name.startswith("learned_temporal_bias"):
            groups["basis"].append(index)
    for name in ("encoder", "coeff", "basis"):
        if not groups[name]:
            raise RuntimeError(f"empty gradient parameter group: {name}")
    return named, groups


def flatten_gradient_group(
    gradients: tuple[torch.Tensor | None, ...],
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    indices: Iterable[int],
) -> torch.Tensor:
    parts = []
    for index in indices:
        gradient = gradients[index]
        parameter = named_parameters[index][1]
        parts.append(torch.zeros_like(parameter).flatten() if gradient is None else gradient.flatten())
    return torch.cat(parts).detach().cpu()


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.norm() == 0 or second.norm() == 0:
        return float("nan")
    return float(F.cosine_similarity(first, second, dim=0).item())


def gradient_metrics(
    dataset: str,
    seed: int,
    model: Model,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
) -> list[dict[str, Any]]:
    weights = measure_weights(SERIES_LENGTH, device)
    named_parameters, groups = parameter_groups(model)
    parameters = [parameter for _name, parameter in named_parameters]
    gradient_sums: dict[tuple[str, str, str], torch.Tensor] = {}
    loss_sums: dict[tuple[str, str], float] = {}
    observed_batches = 0
    model.eval()
    for batch_index, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(loader):
        if batch_index >= max_batches:
            break
        observed_batches += 1
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        _history, target, std = normalized_history_and_target(batch_x, batch_y)
        for form in ("raw", "projected"):
            for measure_name in MEASURE_NAMES:
                prediction, _recon, _alignment = model(
                    batch_x,
                    batch_y[:, -SERIES_LENGTH:, :],
                    is_training=False,
                    target_prefix=SERIES_LENGTH,
                )
                prediction_normalized = (
                    prediction - batch_x.mean(dim=1, keepdim=True).detach()
                ) / std
                error = prediction_normalized - target
                if form == "raw":
                    loss = weighted_mse(error, weights[measure_name])
                else:
                    loss = projected_increment_risk(error, weights[measure_name])
                gradients = torch.autograd.grad(
                    loss,
                    parameters,
                    allow_unused=True,
                )
                loss_sums[(form, measure_name)] = loss_sums.get((form, measure_name), 0.0) + float(loss.item())
                for module_name, indices in groups.items():
                    key = (form, measure_name, module_name)
                    vector = flatten_gradient_group(gradients, named_parameters, indices)
                    gradient_sums[key] = gradient_sums.get(key, torch.zeros_like(vector)) + vector
    if observed_batches == 0:
        raise RuntimeError("gradient diagnostic produced no batches")
    gradient_means = {key: value / observed_batches for key, value in gradient_sums.items()}
    loss_means = {key: value / observed_batches for key, value in loss_sums.items()}
    rows: list[dict[str, Any]] = []
    for form in ("raw", "projected"):
        for measure_name in MEASURE_NAMES:
            step_weights = weights[measure_name].detach().cpu()
            entropy = float(
                (-(step_weights * step_weights.clamp_min(1e-12).log()).sum() / math.log(SERIES_LENGTH)).item()
            )
            early_to_late = float(
                (step_weights[:96].mean() / step_weights[336:].mean().clamp_min(1e-12)).item()
            )
            raw_loss = loss_means[("raw", measure_name)]
            current_loss = loss_means[(form, measure_name)]
            for module_name in groups:
                current = gradient_means[(form, measure_name, module_name)]
                raw_delta = gradient_means[("raw", "delta_720", module_name)]
                same_measure_raw = gradient_means[("raw", measure_name, module_name)]
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "form": form,
                        "measure": measure_name,
                        "module": module_name,
                        "loss": current_loss,
                        "gradient_norm": float(current.norm().item()),
                        "cosine_to_raw_delta720": cosine(current, raw_delta),
                        "cosine_to_same_measure_raw": cosine(current, same_measure_raw),
                        "loss_relative_gap_to_same_measure_raw": (
                            abs(current_loss - raw_loss) / max(abs(raw_loss), 1e-12)
                        ),
                        "step_weight_entropy": entropy,
                        "early_to_late_weight_ratio": early_to_late,
                        "gradient_batches": observed_batches,
                    }
                )
    return rows


def load_model_and_loaders(
    args: argparse.Namespace,
    contract: dict[str, Any],
    seed: int,
) -> tuple[Model, Any, Any, Any, SimpleNamespace, Path]:
    profile = contract["dataset_profiles"][args.dataset]
    directory = run_dir(args, profile, seed)
    effective = json.loads((directory / "effective_config.json").read_text(encoding="utf-8"))
    adapter = effective["adapter"]
    payload = effective["official_args"]
    expected = {
        "patch_num": int(profile["patch_num"]),
        "d_model": int(profile["d_model"]),
        "d_ff": int(profile["d_ff"]),
    }
    observed = {key: int(payload[key]) for key in expected}
    if observed != expected or int(adapter["seed"]) != seed:
        raise ValueError(f"frozen contract mismatch at {directory}: {observed} != {expected}")
    payload["device"] = torch.device(args.device)
    payload["use_gpu"] = args.device.startswith("cuda")
    official_args = SimpleNamespace(**payload)
    set_seed(seed)
    _train_data, train_loader = data_provider(official_args, "train")
    _val_data, val_loader = data_provider(official_args, "val")
    set_seed(seed + 17)
    _gradient_data, gradient_loader = data_provider(official_args, "train")
    model = Model(official_args).float().to(official_args.device)
    model.load_state_dict(load_state(directory / "checkpoint.pt"), strict=True)
    return model, train_loader, val_loader, gradient_loader, official_args, directory


def run_seed(
    args: argparse.Namespace,
    contract: dict[str, Any],
    contract_hash: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model, train_loader, val_loader, gradient_loader, official_args, directory = load_model_and_loaders(
        args, contract, seed
    )
    device = official_args.device
    dct = dct_basis(SERIES_LENGTH, DCT_RANK, device)
    random_q = random_basis(SERIES_LENGTH, DCT_RANK, device)
    train_features, train_sources = collect_split(
        model,
        train_loader,
        device,
        args.train_batches,
        seed * 1000,
    )
    val_features, val_sources = collect_split(
        model,
        val_loader,
        device,
        args.val_batches,
        seed * 1000 + 500,
    )
    learned_q, singular_values = orthonormalized_learned_basis(model)
    structure = energy_structure_rows(
        args.dataset,
        seed,
        train_sources,
        dct,
        random_q,
        learned_q,
    )
    probes = probe_metrics(
        args.dataset,
        seed,
        train_features,
        val_features,
        train_sources,
        val_sources,
        dct,
        args.ridge_lambda,
        device,
    )
    geometry = basis_geometry_rows(
        args.dataset,
        seed,
        model,
        dct,
        learned_q,
        singular_values,
    )
    gradients = gradient_metrics(
        args.dataset,
        seed,
        model,
        gradient_loader,
        device,
        args.gradient_batches,
    )
    metadata = {
        "dataset": args.dataset,
        "seed": seed,
        "contract_hash": contract_hash,
        "run_dir": str(directory),
        "profile": contract["dataset_profiles"][args.dataset]["profile"],
        "memory_shape": [
            "B",
            official_args.enc_in,
            official_args.patch_num,
            official_args.d_model,
        ],
        "state_width": official_args.patch_num * official_args.d_model,
        "basis_shape": list(model.learned_temporal_basis.shape),
        "train_rows": int(train_features["full_hidden"].shape[0]),
        "val_rows": int(val_features["full_hidden"].shape[0]),
        "train_batches": args.train_batches,
        "val_batches": args.val_batches,
        "gradient_batches": args.gradient_batches,
        "ridge_lambda": args.ridge_lambda,
        "uses_test_split": False,
        "trains_forecast_model": False,
    }
    return structure, probes, geometry, gradients, metadata


def synthetic_smoke() -> None:
    device = torch.device("cpu")
    dct = dct_basis(SERIES_LENGTH, DCT_RANK, device)
    identity = dct.transpose(0, 1) @ dct
    if not torch.allclose(identity, torch.eye(DCT_RANK), atol=1e-5):
        raise RuntimeError("DCT basis is not orthonormal")
    weights = measure_weights(SERIES_LENGTH, device)
    error = torch.randn(2, SERIES_LENGTH, 3)
    raw = weighted_mse(error, weights["delta_720"])
    projected = projected_increment_risk(error, weights["delta_720"])
    if not torch.allclose(raw, projected, rtol=1e-5, atol=1e-6):
        raise RuntimeError("block increment Parseval invariant failed")
    for patches, width in ((12, 64), (24, 32)):
        memory = torch.randn(2, 3, patches, width)
        if memory.flatten(start_dim=-2).shape[-1] != 768:
            raise RuntimeError("natural profile state width invariant failed")
    train_x = torch.randn(128, 16)
    true_weights = torch.randn(16, 4)
    train_y = train_x @ true_weights
    prediction, _mean = ridge_predict(train_x, train_x, train_y, 1e-4, device)
    if float((prediction - train_y).square().mean().item()) > 1e-3:
        raise RuntimeError("ridge probe recovery invariant failed")
    print("stage_c_d1_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    contract_hash = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    structure_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for seed in contract["global_fields"]["default_seeds"]:
        print(f"d1_seed_start dataset={args.dataset} seed={seed}", flush=True)
        structure, probes, geometry, gradients, metadata = run_seed(
            args,
            contract,
            contract_hash,
            int(seed),
        )
        structure_rows.extend(structure)
        probe_rows.extend(probes)
        geometry_rows.extend(geometry)
        gradient_rows.extend(gradients)
        metadata_rows.append(metadata)
        print(f"d1_seed_done dataset={args.dataset} seed={seed}", flush=True)
    output_dir = args.output_dir / args.dataset
    write_csv(output_dir / "d1_structure_metrics.csv", structure_rows)
    write_csv(output_dir / "d1_probe_metrics.csv", probe_rows)
    write_csv(output_dir / "d1_basis_geometry.csv", geometry_rows)
    write_csv(output_dir / "d1_gradient_metrics.csv", gradient_rows)
    (output_dir / "d1_metadata.json").write_text(
        json.dumps(metadata_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"stage_c_d1_done dataset={args.dataset} seeds={len(metadata_rows)} "
        f"output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
