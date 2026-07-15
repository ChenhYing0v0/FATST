#!/usr/bin/env python3
"""Run D12-A purged forward cross-fitting for predictable future covariance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
for import_root in (str(REPO_ROOT / "scripts"), str(TIMEALIGN_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from data_provider.data_factory import data_provider  # noqa: E402
from models.TimeAlign import Model  # noqa: E402
from run_stage_c_d1_offline_diagnostic import set_seed  # noqa: E402
from run_stage_c_sc1_d2_diagnostic import checkpoint_run_dir  # noqa: E402
from utils.tools import adjust_learning_rate  # noqa: E402


SERIES_LENGTH = 720


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-root", type=Path)
    parser.add_argument("--phase-b-root", type=Path)
    parser.add_argument("--phase-c-root", type=Path)
    parser.add_argument("--five-phase-a-root", type=Path)
    parser.add_argument("--five-phase-b-root", type=Path)
    parser.add_argument("--five-phase-c-root", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--pilot-cache-root",
        type=Path,
        help="Optional prior diagnostic root containing fold pilot checkpoints.",
    )
    parser.add_argument(
        "--dataset",
        choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if args.synthetic_smoke:
        return args
    required = (
        "phase_a_root",
        "phase_b_root",
        "phase_c_root",
        "contract",
        "design",
        "output_dir",
        "dataset",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    return args


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_effective_args(
    args: argparse.Namespace,
    contract: dict[str, Any],
) -> tuple[SimpleNamespace, Path]:
    profile = contract["dataset_profiles"][args.dataset]
    directory = checkpoint_run_dir(args, profile, int(contract["global_fields"]["default_seeds"][0]))
    effective = json.loads((directory / "effective_config.json").read_text(encoding="utf-8"))
    payload = effective["official_args"]
    expected = {
        "patch_num": int(profile["patch_num"]),
        "d_model": int(profile["d_model"]),
        "d_ff": int(profile["d_ff"]),
    }
    observed = {key: int(payload[key]) for key in expected}
    if observed != expected:
        raise ValueError(f"frozen profile mismatch: {observed} != {expected}")
    payload["device"] = torch.device(args.device)
    payload["use_gpu"] = args.device.startswith("cuda")
    payload["num_workers"] = 0
    payload["batch_size"] = int(contract["global_fields"]["effective_batch_size"])
    return SimpleNamespace(**payload), directory


def evenly_spaced_indices(start: int, end: int, count: int) -> list[int]:
    if end <= start:
        raise ValueError(f"empty index interval [{start}, {end})")
    count = min(count, end - start)
    values = np.linspace(start, end - 1, num=count)
    indices = np.rint(values).astype(np.int64)
    indices = np.unique(indices)
    if indices.size != count:
        raise RuntimeError("deterministic index selection produced duplicates")
    return indices.tolist()


def fold_ranges(length: int, design: dict[str, Any]) -> list[dict[str, int]]:
    starts = design["oof_start_fractions"]
    ends = design["oof_end_fractions"]
    purge = int(design["purge_windows"])
    if len(starts) != int(design["fold_count"]) or len(ends) != len(starts):
        raise ValueError("fold fraction contract mismatch")
    folds = []
    for fold, (start_fraction, end_fraction) in enumerate(zip(starts, ends, strict=True)):
        oof_start = int(math.floor(float(start_fraction) * length))
        oof_end = length if float(end_fraction) == 1.0 else int(
            math.floor(float(end_fraction) * length)
        )
        train_end = oof_start - purge
        if train_end <= 0 or oof_end <= oof_start:
            raise ValueError(
                f"invalid fold {fold}: train_end={train_end}, "
                f"oof=[{oof_start},{oof_end})"
            )
        train_last = train_end - 1
        if train_last + SERIES_LENGTH * 2 > oof_start:
            raise RuntimeError("raw train/OOF intervals overlap")
        folds.append(
            {
                "fold": fold,
                "train_start": 0,
                "train_end": train_end,
                "oof_start": oof_start,
                "oof_end": oof_end,
                "raw_gap": oof_start - train_last - 1,
            }
        )
    return folds


def normalized_rows(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = batch_x.mean(dim=1, keepdim=True).detach()
    std = torch.sqrt(batch_x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    history = (batch_x - mean) / std
    future = (batch_y[:, -SERIES_LENGTH:, :] - mean) / std
    history_rows = history.permute(0, 2, 1).reshape(-1, SERIES_LENGTH)
    future_rows = future.permute(0, 2, 1).reshape(-1, SERIES_LENGTH)
    return history, future, history_rows, future_rows


def normalized_a6_prediction(model: Model, normalized_history: torch.Tensor) -> torch.Tensor:
    memory = model._encode_normalized_history(normalized_history)
    hidden = memory.flatten(start_dim=-2)
    coeff = model.learned_basis_coeff(hidden)
    basis = model.learned_temporal_basis.to(dtype=hidden.dtype)
    bias = model.learned_temporal_bias.to(dtype=hidden.dtype)
    return torch.einsum("tk,bck->btc", basis, coeff) + bias.view(1, -1, 1)


def train_a6_pilot(
    official_args: SimpleNamespace,
    dataset: Any,
    indices: list[int],
    design: dict[str, Any],
    checkpoint: Path,
    history_path: Path,
) -> tuple[Model, list[dict[str, Any]]]:
    seed = int(design["pilot_seed"])
    set_seed(seed)
    model = Model(official_args).float().to(official_args.device)
    if checkpoint.exists() and history_path.exists():
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
        rows = list(csv.DictReader(history_path.open(encoding="utf-8")))
        model.eval()
        return model, rows

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(design["a6_batch_size"]),
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(official_args.learning_rate))
    criterion = nn.L1Loss()
    rows: list[dict[str, Any]] = []
    for epoch in range(int(design["a6_epochs"])):
        model.train()
        epoch_start = time.time()
        losses = []
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
            batch_x = batch_x.float().to(official_args.device)
            batch_y = batch_y.float().to(official_args.device)
            target = batch_y[:, -SERIES_LENGTH:, :]
            optimizer.zero_grad()
            prediction, _recon, _alignment = model(
                batch_x,
                target,
                is_training=True,
                target_prefix=SERIES_LENGTH,
            )
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        rows.append(
            {
                "epoch": epoch + 1,
                "train_windows": len(indices),
                "train_steps": len(loader),
                "train_l1": float(np.mean(losses)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": time.time() - epoch_start,
            }
        )
        print(
            f"pilot_epoch={epoch + 1} windows={len(indices)} "
            f"train_l1={rows[-1]['train_l1']:.7f}",
            flush=True,
        )
        adjust_learning_rate(optimizer, epoch + 1, official_args)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    write_csv(history_path, rows)
    model.eval()
    return model, rows


def dct_basis(rank: int, device: torch.device) -> torch.Tensor:
    steps = torch.arange(SERIES_LENGTH, dtype=torch.float64, device=device) + 0.5
    frequencies = torch.arange(rank, dtype=torch.float64, device=device)
    basis = torch.cos(torch.pi * torch.outer(steps, frequencies) / SERIES_LENGTH)
    basis[:, 0] *= math.sqrt(1.0 / SERIES_LENGTH)
    if rank > 1:
        basis[:, 1:] *= math.sqrt(2.0 / SERIES_LENGTH)
    return basis


def fit_ridge_pilot(
    dataset: Any,
    indices: list[int],
    channel_count: int,
    design: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor | float | int]:
    max_rows = int(design["ridge_max_rows"])
    window_count = max(1, max_rows // channel_count)
    selected = evenly_spaced_indices(indices[0], indices[-1] + 1, window_count)
    loader = DataLoader(
        Subset(dataset, selected),
        batch_size=int(design["a6_batch_size"]),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    basis = dct_basis(int(design["ridge_dct_rank"]), device)
    feature_parts = []
    target_parts = []
    for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
        batch_x = batch_x.double().to(device)
        batch_y = batch_y.double().to(device)
        _history, _future, history_rows, future_rows = normalized_rows(batch_x, batch_y)
        feature_parts.append(history_rows @ basis)
        target_parts.append(future_rows)
    features = torch.cat(feature_parts)
    targets = torch.cat(target_parts)
    feature_mean = features.mean(dim=0, keepdim=True)
    target_mean = targets.mean(dim=0, keepdim=True)
    centered_features = features - feature_mean
    centered_targets = targets - target_mean
    gram = centered_features.T @ centered_features
    relative = float(design["ridge_relative_lambda"])
    alpha = relative * float(torch.trace(gram).item()) / gram.shape[0]
    regularized = gram + alpha * torch.eye(
        gram.shape[0], dtype=gram.dtype, device=device
    )
    coefficients = torch.linalg.solve(
        regularized,
        centered_features.T @ centered_targets,
    )
    return {
        "basis": basis,
        "feature_mean": feature_mean,
        "target_mean": target_mean,
        "coefficients": coefficients,
        "alpha": alpha,
        "row_count": int(features.shape[0]),
    }


class MomentAccumulator:
    """Accumulate risk-weighted covariance sufficient statistics."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.count = 0
        self.weight_sum = 0.0
        self.weight_square_sum = 0.0
        self.weight_max = 0.0
        self.sums = {
            name: torch.zeros(SERIES_LENGTH, dtype=torch.float64, device=device)
            for name in ("label", "a6", "ridge", "a6_residual", "ridge_residual")
        }
        self.outers = {
            name: torch.zeros(
                SERIES_LENGTH,
                SERIES_LENGTH,
                dtype=torch.float64,
                device=device,
            )
            for name in self.sums
        }
        self.a6_sse = 0.0
        self.ridge_sse = 0.0
        self.std_min = float("inf")

    def update(
        self,
        label: torch.Tensor,
        a6: torch.Tensor,
        ridge: torch.Tensor,
        weights: torch.Tensor,
        std_min: float,
    ) -> None:
        values = {
            "label": label.double(),
            "a6": a6.double(),
            "ridge": ridge.double(),
            "a6_residual": (a6 - label).double(),
            "ridge_residual": (ridge - label).double(),
        }
        rows = int(label.shape[0])
        weights = weights.double().reshape(-1)
        if weights.shape[0] != rows or bool((weights <= 0.0).any()):
            raise ValueError("risk weights must be positive and match row count")
        self.count += rows
        self.weight_sum += float(weights.sum().item())
        self.weight_square_sum += float(weights.square().sum().item())
        self.weight_max = max(self.weight_max, float(weights.max().item()))
        self.std_min = min(self.std_min, std_min)
        for name, tensor in values.items():
            weighted = tensor * weights.unsqueeze(1)
            self.sums[name] += weighted.sum(dim=0)
            self.outers[name] += tensor.T @ weighted
        self.a6_sse += float(
            ((a6 - label).double().square() * weights.unsqueeze(1)).sum().item()
        )
        self.ridge_sse += float(
            ((ridge - label).double().square() * weights.unsqueeze(1)).sum().item()
        )

    def covariance(self, name: str) -> torch.Tensor:
        if self.count < 2 or self.weight_sum <= 0.0:
            raise RuntimeError("insufficient covariance rows")
        mean_outer = (
            torch.outer(self.sums[name], self.sums[name]) / self.weight_sum
        )
        return (self.outers[name] - mean_outer) / self.weight_sum

    def export(self) -> dict[str, np.ndarray]:
        payload: dict[str, np.ndarray] = {
            "count": np.asarray(self.count, dtype=np.int64),
            "weight_sum": np.asarray(self.weight_sum, dtype=np.float64),
            "weight_square_sum": np.asarray(
                self.weight_square_sum, dtype=np.float64
            ),
            "weight_max": np.asarray(self.weight_max, dtype=np.float64),
            "a6_sse": np.asarray(self.a6_sse, dtype=np.float64),
            "ridge_sse": np.asarray(self.ridge_sse, dtype=np.float64),
            "std_min": np.asarray(self.std_min, dtype=np.float64),
        }
        for name in self.sums:
            payload[f"{name}_sum"] = self.sums[name].detach().cpu().numpy()
            payload[f"{name}_outer"] = self.outers[name].detach().cpu().numpy()
            payload[f"{name}_cov"] = self.covariance(name).detach().cpu().numpy()
        return payload


def ridge_predict(
    history_rows: torch.Tensor,
    ridge: dict[str, torch.Tensor | float | int],
) -> torch.Tensor:
    basis = ridge["basis"]
    feature_mean = ridge["feature_mean"]
    coefficients = ridge["coefficients"]
    target_mean = ridge["target_mean"]
    if not all(isinstance(value, torch.Tensor) for value in (
        basis,
        feature_mean,
        coefficients,
        target_mean,
    )):
        raise TypeError("ridge tensor contract violated")
    features = history_rows.double() @ basis
    return (features - feature_mean) @ coefficients + target_mean


def evaluate_fold(
    model: Model,
    ridge: dict[str, torch.Tensor | float | int],
    dataset: Any,
    indices: list[int],
    design: dict[str, Any],
    official_args: SimpleNamespace,
) -> tuple[MomentAccumulator, float]:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(design["a6_batch_size"]),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    accumulator = MomentAccumulator(official_args.device)
    forward_gap = 0.0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
            batch_x = batch_x.float().to(official_args.device)
            batch_y = batch_y.float().to(official_args.device)
            mean = batch_x.mean(dim=1, keepdim=True)
            std = torch.sqrt(
                batch_x.var(dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            normalized_history, _future, history_rows, future_rows = normalized_rows(
                batch_x,
                batch_y,
            )
            a6_btc = normalized_a6_prediction(model, normalized_history)
            a6_rows = a6_btc.permute(0, 2, 1).reshape(-1, SERIES_LENGTH)
            ridge_rows = ridge_predict(history_rows, ridge).float()
            official, _recon, _alignment = model(
                batch_x,
                batch_y[:, -SERIES_LENGTH:, :],
                is_training=False,
                target_prefix=SERIES_LENGTH,
            )
            reconstructed = a6_btc * std + mean
            forward_gap = max(
                forward_gap,
                float((official - reconstructed).abs().max().item()),
            )
            accumulator.update(
                future_rows,
                a6_rows,
                ridge_rows,
                risk_weights(std, design),
                float(std.min().item()),
            )
    return accumulator, forward_gap


def risk_weights(std: torch.Tensor, design: dict[str, Any]) -> torch.Tensor:
    """Map RevIN scale [B,1,C] to row weights matching [B*C,720]."""
    row_std = std.squeeze(1).reshape(-1)
    mode = design.get("risk_weight_mode", "uniform")
    if mode == "uniform":
        return torch.ones_like(row_std)
    if mode == "history_std":
        return row_std
    if mode == "history_std_squared":
        return row_std.square()
    raise ValueError(f"unsupported risk_weight_mode: {mode}")


def eigensystem(covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    symmetric = 0.5 * (covariance + covariance.T)
    values, vectors = torch.linalg.eigh(symmetric)
    order = torch.argsort(values, descending=True)
    return values[order], vectors[:, order]


def effective_rank(eigenvalues: torch.Tensor) -> float:
    values = eigenvalues.clamp_min(0.0)
    probabilities = values / values.sum().clamp_min(1e-30)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return float(entropy.exp().item())


def covariance_metrics(covariance: torch.Tensor) -> dict[str, float]:
    symmetry = float((covariance - covariance.T).abs().max().item())
    eigenvalues, _vectors = eigensystem(covariance)
    trace = float(torch.trace(covariance).item())
    return {
        "trace": trace,
        "effective_rank": effective_rank(eigenvalues),
        "minimum_eigenvalue": float(eigenvalues.min().item()),
        "symmetry_max_abs": symmetry,
    }


def subspace_overlap(first: torch.Tensor, second: torch.Tensor, rank: int) -> float:
    return float(
        (first[:, :rank].T @ second[:, :rank]).square().sum().item() / rank
    )


def capture(covariance: torch.Tensor, basis: torch.Tensor, rank: int) -> float:
    columns = basis[:, :rank]
    numerator = torch.trace(columns.T @ covariance @ columns)
    return float(numerator.item() / max(float(torch.trace(covariance).item()), 1e-30))


def analyze_moments(
    fold_accumulators: list[MomentAccumulator],
    design: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fold_rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    fold_covariances: list[dict[str, torch.Tensor]] = []
    for fold, accumulator in enumerate(fold_accumulators):
        covariances = {
            name: accumulator.covariance(name).detach().cpu()
            for name in accumulator.sums
        }
        fold_covariances.append(covariances)
        label_sst = float(
            (
                accumulator.outers["label"].trace()
                - accumulator.sums["label"].square().sum()
                / accumulator.weight_sum
            ).item()
        )
        denominator = accumulator.weight_sum * SERIES_LENGTH
        a6_mse = accumulator.a6_sse / denominator
        ridge_mse = accumulator.ridge_sse / denominator
        zero_mse = label_sst / denominator
        row: dict[str, Any] = {
            "fold": fold,
            "row_count": accumulator.count,
            "weight_sum": accumulator.weight_sum,
            "weight_effective_sample_fraction": (
                accumulator.weight_sum**2
                / accumulator.weight_square_sum
                / accumulator.count
            ),
            "weight_max_share": accumulator.weight_max / accumulator.weight_sum,
            "a6_oof_mse": a6_mse,
            "ridge_oof_mse": ridge_mse,
            "zero_mse": zero_mse,
            "a6_oof_r2": 1.0 - a6_mse / zero_mse,
            "ridge_oof_r2": 1.0 - ridge_mse / zero_mse,
            "normalization_std_min": accumulator.std_min,
        }
        eigens = {}
        for name, covariance in covariances.items():
            metrics = covariance_metrics(covariance)
            for key, value in metrics.items():
                row[f"{name}_{key}"] = value
            eigens[name] = eigensystem(covariance)
        row["a6_predictable_trace_fraction"] = (
            row["a6_trace"] / max(row["label_trace"], 1e-30)
        )
        row["ridge_predictable_trace_fraction"] = (
            row["ridge_trace"] / max(row["label_trace"], 1e-30)
        )
        fold_rows.append(row)
        for pilot in ("a6", "ridge"):
            pilot_values, pilot_vectors = eigens[pilot]
            _label_values, label_vectors = eigens["label"]
            for rank in design["diagnostic_ranks"]:
                optimal_capture = float(
                    pilot_values[:rank].clamp_min(0.0).sum().item()
                    / max(pilot_values.clamp_min(0.0).sum().item(), 1e-30)
                )
                raw_capture = capture(covariances[pilot], label_vectors, int(rank))
                subspace_rows.append(
                    {
                        "scope": "fold",
                        "fold": fold,
                        "pilot": pilot,
                        "rank": int(rank),
                        "optimal_predictable_capture": optimal_capture,
                        "raw_label_basis_predictable_capture": raw_capture,
                        "raw_relative_capture_gap": (
                            optimal_capture - raw_capture
                        )
                        / max(optimal_capture, 1e-30),
                        "label_pilot_subspace_overlap": subspace_overlap(
                            label_vectors,
                            pilot_vectors,
                            int(rank),
                        ),
                    }
                )

    if len(fold_accumulators) != 2:
        raise ValueError("D12-A analyzer currently expects exactly two folds")
    for pilot in ("a6", "ridge"):
        first_vectors = eigensystem(fold_covariances[0][pilot])[1]
        second_vectors = eigensystem(fold_covariances[1][pilot])[1]
        for rank in design["diagnostic_ranks"]:
            subspace_rows.append(
                {
                    "scope": "cross_fold",
                    "fold": "0_vs_1",
                    "pilot": pilot,
                    "rank": int(rank),
                    "subspace_overlap": subspace_overlap(
                        first_vectors,
                        second_vectors,
                        int(rank),
                    ),
                }
            )

    aggregate = merge_accumulators(fold_accumulators)
    covariances = {
        name: aggregate.covariance(name).detach().cpu()
        for name in aggregate.sums
    }
    eigens = {name: eigensystem(value) for name, value in covariances.items()}
    label_sst = float(
        (
            aggregate.outers["label"].trace()
            - aggregate.sums["label"].square().sum() / aggregate.weight_sum
        ).item()
    )
    denominator = aggregate.weight_sum * SERIES_LENGTH
    zero_mse = label_sst / denominator
    summary: dict[str, Any] = {
        "row_count": aggregate.count,
        "weight_sum": aggregate.weight_sum,
        "weight_effective_sample_fraction": (
            aggregate.weight_sum**2
            / aggregate.weight_square_sum
            / aggregate.count
        ),
        "weight_max_share": aggregate.weight_max / aggregate.weight_sum,
        "a6_mean_oof_mse": aggregate.a6_sse / denominator,
        "ridge_mean_oof_mse": aggregate.ridge_sse / denominator,
        "zero_mse": zero_mse,
        "normalization_std_min": aggregate.std_min,
    }
    summary["a6_mean_oof_r2"] = 1.0 - summary["a6_mean_oof_mse"] / zero_mse
    summary["ridge_mean_oof_r2"] = 1.0 - summary["ridge_mean_oof_mse"] / zero_mse
    for name, covariance in covariances.items():
        for key, value in covariance_metrics(covariance).items():
            summary[f"{name}_{key}"] = value
    summary["a6_predictable_trace_fraction"] = (
        summary["a6_trace"] / max(summary["label_trace"], 1e-30)
    )
    summary["ridge_predictable_trace_fraction"] = (
        summary["ridge_trace"] / max(summary["label_trace"], 1e-30)
    )
    for pilot in ("a6", "ridge"):
        values, vectors = eigens[pilot]
        label_vectors = eigens["label"][1]
        for rank in design["diagnostic_ranks"]:
            optimal_capture = float(
                values[:rank].clamp_min(0.0).sum().item()
                / max(values.clamp_min(0.0).sum().item(), 1e-30)
            )
            raw_capture = capture(covariances[pilot], label_vectors, int(rank))
            relative_gap = (optimal_capture - raw_capture) / max(
                optimal_capture, 1e-30
            )
            summary[f"{pilot}_rank{rank}_optimal_capture"] = optimal_capture
            summary[f"{pilot}_rank{rank}_raw_capture"] = raw_capture
            summary[f"{pilot}_rank{rank}_raw_relative_gap"] = relative_gap
            subspace_rows.append(
                {
                    "scope": "aggregate",
                    "fold": "all",
                    "pilot": pilot,
                    "rank": int(rank),
                    "optimal_predictable_capture": optimal_capture,
                    "raw_label_basis_predictable_capture": raw_capture,
                    "raw_relative_capture_gap": relative_gap,
                    "label_pilot_subspace_overlap": subspace_overlap(
                        label_vectors,
                        vectors,
                        int(rank),
                    ),
                }
            )
    for rank in design["diagnostic_ranks"]:
        summary[f"a6_ridge_rank{rank}_overlap"] = subspace_overlap(
            eigens["a6"][1],
            eigens["ridge"][1],
            int(rank),
        )
    summary["a6_fold_top32_overlap"] = next(
        row["subspace_overlap"]
        for row in subspace_rows
        if row["scope"] == "cross_fold"
        and row["pilot"] == "a6"
        and row["rank"] == int(design["stability_rank"])
    )
    summary["ridge_fold_top32_overlap"] = next(
        row["subspace_overlap"]
        for row in subspace_rows
        if row["scope"] == "cross_fold"
        and row["pilot"] == "ridge"
        and row["rank"] == int(design["stability_rank"])
    )
    summary["a6_min_fold_oof_r2"] = min(row["a6_oof_r2"] for row in fold_rows)
    return fold_rows, subspace_rows, summary


def merge_accumulators(accumulators: list[MomentAccumulator]) -> MomentAccumulator:
    merged = MomentAccumulator(accumulators[0].device)
    merged.count = sum(value.count for value in accumulators)
    merged.weight_sum = sum(value.weight_sum for value in accumulators)
    merged.weight_square_sum = sum(
        value.weight_square_sum for value in accumulators
    )
    merged.weight_max = max(value.weight_max for value in accumulators)
    merged.a6_sse = sum(value.a6_sse for value in accumulators)
    merged.ridge_sse = sum(value.ridge_sse for value in accumulators)
    merged.std_min = min(value.std_min for value in accumulators)
    for name in merged.sums:
        merged.sums[name] = sum(
            (value.sums[name] for value in accumulators),
            start=torch.zeros_like(merged.sums[name]),
        )
        merged.outers[name] = sum(
            (value.outers[name] for value in accumulators),
            start=torch.zeros_like(merged.outers[name]),
        )
    return merged


def dataset_gate(
    summary: dict[str, Any],
    fold_rows: list[dict[str, Any]],
    design: dict[str, Any],
) -> dict[str, Any]:
    gates = design["gates"]
    invariant_pass = (
        max(
            summary[f"{name}_symmetry_max_abs"]
            for name in ("label", "a6", "ridge", "a6_residual", "ridge_residual")
        )
        <= float(gates["covariance_symmetry_max_abs"])
        and min(
            summary[f"{name}_minimum_eigenvalue"]
            for name in ("label", "a6", "ridge", "a6_residual", "ridge_residual")
        )
        >= float(gates["covariance_psd_min_eigenvalue"])
        and summary["normalization_std_min"]
        >= float(gates["normalization_std_min"])
        and summary["weight_effective_sample_fraction"]
        >= float(gates.get("weight_effective_sample_fraction_min", 0.0))
        and summary["weight_max_share"]
        <= float(gates.get("weight_max_share_max", 1.0))
    )
    checks = {
        "invariants_pass": invariant_pass,
        "a6_predictability_pass": (
            summary["a6_mean_oof_r2"] > float(gates["a6_mean_oof_r2_min"])
            and summary["a6_min_fold_oof_r2"]
            >= float(gates["a6_min_fold_oof_r2_min"])
        ),
        "a6_trace_pass": summary["a6_predictable_trace_fraction"]
        >= float(gates["a6_predictable_trace_fraction_min"]),
        "a6_fold_stability_pass": summary["a6_fold_top32_overlap"]
        >= float(gates["a6_fold_top32_overlap_min"]),
        "rank256_headroom_pass": summary["a6_rank256_raw_relative_gap"]
        >= float(gates["rank256_raw_basis_predictable_capture_gap_min"]),
        "pilot_robustness_pass": (
            summary["a6_ridge_rank32_overlap"]
            >= float(gates["pilot_top32_overlap_min"])
            and summary["ridge_mean_oof_r2"]
            > float(gates["ridge_mean_oof_r2_min"])
        ),
    }
    support = all(checks.values())
    if not invariant_pass:
        decision = "diagnostic_invalid"
    elif not checks["a6_predictability_pass"] or not checks["a6_trace_pass"]:
        decision = "predictable_signal_not_established"
    elif not checks["a6_fold_stability_pass"]:
        decision = "nonstationary_or_estimator_unstable"
    elif not checks["rank256_headroom_pass"]:
        decision = "raw_label_subspace_already_sufficient_cape_closed"
    elif not checks["pilot_robustness_pass"]:
        decision = "pilot_specific_predictable_subspace"
    else:
        decision = "cape_problem_supported"
    return {
        **checks,
        "dataset_support": support,
        "decision": decision,
        "fold_count": len(fold_rows),
    }


def synthetic_smoke() -> None:
    torch.manual_seed(17)
    device = torch.device("cpu")
    latent = torch.randn(2048, 8, dtype=torch.float64)
    basis, _upper = torch.linalg.qr(
        torch.randn(SERIES_LENGTH, 8, dtype=torch.float64),
        mode="reduced",
    )
    label = latent @ basis.T + 0.1 * torch.randn(
        2048,
        SERIES_LENGTH,
        dtype=torch.float64,
    )
    a6 = latent @ basis.T
    ridge = 0.9 * a6
    accumulator = MomentAccumulator(device)
    accumulator.update(
        label,
        a6,
        ridge,
        torch.ones(label.shape[0], dtype=torch.float64),
        1.0,
    )
    covariance = accumulator.covariance("label")
    metrics = covariance_metrics(covariance)
    if metrics["symmetry_max_abs"] > 1e-10:
        raise RuntimeError("covariance symmetry smoke failed")
    if metrics["minimum_eigenvalue"] < -1e-9:
        raise RuntimeError("covariance PSD smoke failed")
    values, vectors = eigensystem(accumulator.covariance("a6"))
    if capture(accumulator.covariance("a6"), vectors, 8) < 0.999999:
        raise RuntimeError("predictable capture smoke failed")
    if values[7] <= 0.0:
        raise RuntimeError("predictable spectrum smoke failed")
    weighted = MomentAccumulator(device)
    weights = torch.linspace(0.1, 2.0, label.shape[0], dtype=torch.float64)
    weighted.update(label, a6, ridge, weights, 1.0)
    mean = (label * weights.unsqueeze(1)).sum(dim=0) / weights.sum()
    expected = (
        (label - mean).T @ ((label - mean) * weights.unsqueeze(1))
    ) / weights.sum()
    if not torch.allclose(weighted.covariance("label"), expected, atol=1e-10):
        raise RuntimeError("weighted covariance smoke failed")
    shaped_std = torch.tensor([[[2.0, 3.0]], [[4.0, 5.0]]])
    shaped_weights = risk_weights(
        shaped_std,
        {"risk_weight_mode": "history_std_squared"},
    )
    if not torch.equal(shaped_weights, torch.tensor([4.0, 9.0, 16.0, 25.0])):
        raise RuntimeError("risk weight row-order smoke failed")
    folds = fold_ranges(
        7201,
        {
            "fold_count": 2,
            "oof_start_fractions": [0.6, 0.8],
            "oof_end_fractions": [0.8, 1.0],
            "purge_windows": 1439,
        },
    )
    if min(row["train_end"] for row in folds) <= 0:
        raise RuntimeError("fold construction smoke failed")
    print("stage_c_d12_worker_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    design = json.loads(args.design.read_text(encoding="utf-8"))
    if args.dataset not in design["datasets"]:
        raise ValueError(f"dataset not authorized: {args.dataset}")
    if int(design["purge_windows"]) != SERIES_LENGTH * 2 - 1:
        raise ValueError("purge must cover history and future raw intervals")
    official_args, config_source = load_effective_args(args, contract)
    set_seed(int(design["pilot_seed"]))
    train_dataset, _unused_loader = data_provider(official_args, "train")
    folds = fold_ranges(len(train_dataset), design)
    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    pilot_directory = (
        args.pilot_cache_root / args.dataset
        if args.pilot_cache_root is not None
        else output_dir
    )

    accumulators: list[MomentAccumulator] = []
    fold_metadata: list[dict[str, Any]] = []
    pilot_cache_hits: list[bool] = []
    for fold_range in folds:
        fold = int(fold_range["fold"])
        train_indices = list(
            range(fold_range["train_start"], fold_range["train_end"])
        )
        oof_indices = evenly_spaced_indices(
            fold_range["oof_start"],
            fold_range["oof_end"],
            int(design["oof_windows_per_fold"]),
        )
        pilot_checkpoint = pilot_directory / f"fold{fold}_a6_pilot.pt"
        pilot_history = pilot_directory / f"fold{fold}_training_history.csv"
        pilot_cache_hit = pilot_checkpoint.exists() and pilot_history.exists()
        pilot_cache_hits.append(pilot_cache_hit)
        model, training_rows = train_a6_pilot(
            official_args,
            train_dataset,
            train_indices,
            design,
            pilot_checkpoint,
            pilot_history,
        )
        ridge = fit_ridge_pilot(
            train_dataset,
            train_indices,
            int(official_args.enc_in),
            design,
            official_args.device,
        )
        accumulator, forward_gap = evaluate_fold(
            model,
            ridge,
            train_dataset,
            oof_indices,
            design,
            official_args,
        )
        accumulators.append(accumulator)
        np.savez_compressed(
            output_dir / f"fold{fold}_moments.npz",
            **accumulator.export(),
        )
        fold_metadata.append(
            {
                **fold_range,
                "oof_sampled_windows": len(oof_indices),
                "oof_first_index": oof_indices[0],
                "oof_last_index": oof_indices[-1],
                "a6_epoch_count": len(training_rows),
                "ridge_row_count": int(ridge["row_count"]),
                "ridge_alpha": float(ridge["alpha"]),
                "forward_reconstruction_max_abs": forward_gap,
                "pilot_cache_directory": str(pilot_directory),
                "pilot_cache_hit": pilot_cache_hit,
            }
        )
        del model, ridge
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_rows, subspace_rows, summary = analyze_moments(accumulators, design)
    gate = dataset_gate(summary, fold_rows, design)
    summary.update(
        {
            "dataset": args.dataset,
            "profile": contract["dataset_profiles"][args.dataset]["profile"],
            **gate,
        }
    )
    write_csv(
        output_dir / "fold_metrics.csv",
        [{"dataset": args.dataset, **row} for row in fold_rows],
    )
    write_csv(
        output_dir / "subspace_metrics.csv",
        [{"dataset": args.dataset, **row} for row in subspace_rows],
    )
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "dataset": args.dataset,
        "profile": contract["dataset_profiles"][args.dataset]["profile"],
        "contract_hash": file_hash(args.contract),
        "design_hash": file_hash(args.design),
        "config_source": str(config_source),
        "folds": fold_metadata,
        "uses_train_split": True,
        "uses_validation_split": False,
        "uses_test_split": False,
        "reuses_cached_diagnostic_pilots": all(pilot_cache_hits),
        "trains_diagnostic_pilots": not all(pilot_cache_hits),
        "risk_weight_mode": design.get("risk_weight_mode", "uniform"),
        "pilot_cache_root": (
            str(args.pilot_cache_root)
            if args.pilot_cache_root is not None
            else None
        ),
        "updates_paper_forecast_model": False,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(official_args.device),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"stage_c_d12_worker_done dataset={args.dataset} "
        f"support={summary['dataset_support']} decision={summary['decision']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
