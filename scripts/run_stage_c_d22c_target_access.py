#!/usr/bin/env python3
"""Train one dataset of the D22-C neutral raw-history target-access diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
for import_root in (str(REPO_ROOT / "scripts"), str(TIMEALIGN_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from run_stage_c_sc1_d10_raw_scale_identifiability import (  # noqa: E402
    build_dataset,
    even_indices,
)


ARMS = (
    "GLOBAL_COMPRESSED",
    "POOLED_MEMORY",
    "ORDERED_TARGET_ACCESS",
    "ORDER_SHUFFLED",
    "TARGET_SHUFFLED_QUERY",
    "GENERIC_MATCHED",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--dataset",
        choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"],
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_smoke:
        required = ("dataset_root", "dataset", "config", "output_dir")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"missing required arguments: {', '.join(missing)}")
    return args


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class IndexedSelection(Dataset[Any]):
    """Expose stable source-window ids for deterministic negative controls."""

    def __init__(self, dataset: Dataset[Any], indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[Any, ...]:
        source_index = self.indices[position]
        item = self.dataset[source_index]
        return (source_index, *item)


def sinusoidal_positions(length: int, width: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    values = torch.zeros(length, width, dtype=torch.float32)
    values[:, 0::2] = torch.sin(positions * frequencies)
    if width > 1:
        values[:, 1::2] = torch.cos(
            positions * frequencies[: values[:, 1::2].shape[1]]
        )
    return values


def permutation_bank(
    bank_size: int,
    length: int,
    seed: int,
) -> torch.Tensor:
    permutations = []
    for bank_index in range(bank_size):
        generator = torch.Generator().manual_seed(seed + bank_index * 10_007)
        permutations.append(torch.randperm(length, generator=generator))
    return torch.stack(permutations)


class NeutralTargetAccess(nn.Module):
    """One parameterization shared exactly by all six diagnostic arms."""

    def __init__(
        self,
        series_length: int,
        prediction_length: int,
        patch_count: int,
        d_model: int,
        n_heads: int,
        dropout: float,
        bank_size: int,
        permutation_seed: int,
    ) -> None:
        super().__init__()
        if series_length % patch_count:
            raise ValueError("series_length must be divisible by patch_count")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.series_length = series_length
        self.prediction_length = prediction_length
        self.patch_count = patch_count
        self.patch_length = series_length // patch_count
        self.d_model = d_model

        self.patch_encoder = nn.Sequential(
            nn.Linear(self.patch_length, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cross_attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.output_projection = nn.Linear(d_model, 1)
        self.register_buffer(
            "patch_positions",
            sinusoidal_positions(patch_count, d_model),
            persistent=True,
        )
        self.register_buffer(
            "target_positions",
            sinusoidal_positions(prediction_length, d_model),
            persistent=True,
        )
        self.register_buffer(
            "memory_permutations",
            permutation_bank(bank_size, patch_count, permutation_seed),
            persistent=True,
        )
        self.register_buffer(
            "target_permutations",
            permutation_bank(bank_size, prediction_length, permutation_seed + 1),
            persistent=True,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        history: torch.Tensor,
        row_ids: torch.Tensor,
        arm: str,
        return_health: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        if history.ndim != 2 or history.shape[1] != self.series_length:
            raise ValueError(f"unexpected history shape: {tuple(history.shape)}")
        batch = history.shape[0]
        patches = history.reshape(batch, self.patch_count, self.patch_length)
        patch_latent = self.patch_encoder(patches)
        patch_position = self.patch_positions.to(patch_latent.dtype).unsqueeze(0)
        canonical_query = self.query_encoder(
            self.target_positions.to(patch_latent.dtype)
        ).unsqueeze(0).expand(batch, -1, -1)

        memory = patch_latent + patch_position
        coordinate_query = canonical_query
        access_query = canonical_query

        bank_indices = torch.remainder(
            row_ids,
            self.memory_permutations.shape[0],
        )
        if arm == "GLOBAL_COMPRESSED":
            global_patch = patches.mean(dim=1, keepdim=True)
            memory = self.patch_encoder(global_patch) + patch_position.mean(
                dim=1,
                keepdim=True,
            )
        elif arm == "POOLED_MEMORY":
            memory = memory.mean(dim=1, keepdim=True)
        elif arm == "ORDER_SHUFFLED":
            order = self.memory_permutations.index_select(0, bank_indices)
            shuffled = torch.gather(
                patch_latent,
                1,
                order.unsqueeze(-1).expand(-1, -1, self.d_model),
            )
            memory = shuffled + patch_position
        elif arm == "TARGET_SHUFFLED_QUERY":
            order = self.target_permutations.index_select(0, bank_indices)
            coordinate_query = torch.gather(
                canonical_query,
                1,
                order.unsqueeze(-1).expand(-1, -1, self.d_model),
            )
            access_query = coordinate_query
        elif arm == "GENERIC_MATCHED":
            access_query = canonical_query.mean(dim=1, keepdim=True).expand(
                -1,
                self.prediction_length,
                -1,
            )

        context, weights = self.cross_attention(
            access_query,
            memory,
            memory,
            need_weights=return_health,
            average_attn_weights=True,
        )
        hidden = self.fusion(torch.cat((coordinate_query, context), dim=-1))
        prediction = self.output_projection(hidden).squeeze(-1)

        health: dict[str, torch.Tensor] = {}
        if return_health:
            if weights is None:
                raise RuntimeError("attention weights were not returned")
            memory_width = weights.shape[-1]
            entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(
                dim=-1
            )
            if memory_width > 1:
                entropy = entropy / math.log(memory_width)
            else:
                entropy = torch.zeros_like(entropy)
            health = {
                "attention_entropy": entropy.mean(),
                "attention_target_dispersion": weights.std(
                    dim=1,
                    unbiased=False,
                ).mean(),
                "prediction_coordinate_dispersion": prediction.std(
                    dim=1,
                    unbiased=False,
                ).mean(),
            }
        return prediction, health


def selected_indices(length: int, maximum: int) -> list[int]:
    if maximum <= 0 or maximum >= length:
        return list(range(length))
    return even_indices(0, length, maximum)


def make_loader(
    dataset: Dataset[Any],
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        IndexedSelection(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        generator=generator,
    )


def prepare_batch(
    batch: tuple[Any, ...],
    device: torch.device,
    prediction_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    window_ids, batch_x, batch_y = batch[:3]
    history = torch.as_tensor(batch_x, dtype=torch.float32, device=device)
    future = torch.as_tensor(
        batch_y[:, -prediction_length:, :],
        dtype=torch.float32,
        device=device,
    )
    mean = history.mean(dim=1, keepdim=True).detach()
    std = torch.sqrt(
        history.var(dim=1, keepdim=True, unbiased=False) + 1e-5
    ).detach()
    normalized_history = ((history - mean) / std).permute(0, 2, 1)
    normalized_future = ((future - mean) / std).permute(0, 2, 1)
    batch_size, channels, series_length = normalized_history.shape
    history_rows = normalized_history.reshape(-1, series_length)
    future_rows = normalized_future.reshape(-1, prediction_length)
    mean_rows = mean.permute(0, 2, 1).reshape(-1, 1)
    std_rows = std.permute(0, 2, 1).reshape(-1, 1)
    source_ids = torch.as_tensor(window_ids, device=device, dtype=torch.long)
    channel_ids = torch.arange(channels, device=device).repeat(batch_size)
    row_ids = source_ids.repeat_interleave(channels) * channels + channel_ids
    return history_rows, future_rows, mean_rows, std_rows, row_ids


def metric_regions(config: dict[str, Any]) -> list[tuple[str, int, int]]:
    regions = [
        (f"prefix_{int(horizon)}", 0, int(horizon))
        for horizon in config["paper_facing_horizons"]
    ]
    regions.extend(
        (
            f"bin_{entry['name']}",
            int(entry["start"]),
            int(entry["end"]),
        )
        for entry in config["coordinate_bins"]
    )
    return regions


def evaluate(
    model: NeutralTargetAccess,
    loader: DataLoader[Any],
    arm: str,
    device: torch.device,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], float, dict[str, float]]:
    regions = metric_regions(config)
    accumulators = {
        name: {"sse": 0.0, "sae": 0.0, "count": 0}
        for name, _start, _end in regions
    }
    health_sum = {
        "attention_entropy": 0.0,
        "attention_target_dispersion": 0.0,
        "prediction_coordinate_dispersion": 0.0,
    }
    health_count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            history, target, mean, std, row_ids = prepare_batch(
                batch,
                device,
                model.prediction_length,
            )
            prediction, health = model(
                history,
                row_ids,
                arm,
                return_health=True,
            )
            prediction_scaled = prediction * std + mean
            target_scaled = target * std + mean
            error = prediction_scaled - target_scaled
            for name, start, end in regions:
                region_error = error[:, start:end]
                accumulators[name]["sse"] += float(region_error.square().sum())
                accumulators[name]["sae"] += float(region_error.abs().sum())
                accumulators[name]["count"] += region_error.numel()
            for key in health_sum:
                health_sum[key] += float(health[key]) * history.shape[0]
            health_count += history.shape[0]

    rows = []
    for name, start, end in regions:
        values = accumulators[name]
        count = int(values["count"])
        if count == 0:
            raise RuntimeError(f"empty evaluation region: {name}")
        rows.append(
            {
                "region": name,
                "region_start": start,
                "region_end": end,
                "mse": values["sse"] / count,
                "mae": values["sae"] / count,
                "element_count": count,
            }
        )
    selector_regions = {
        f"prefix_{int(horizon)}"
        for horizon in config["checkpoint_selector"]["horizons"]
    }
    selector = float(
        np.mean([row["mse"] for row in rows if row["region"] in selector_regions])
    )
    health_mean = {key: value / health_count for key, value in health_sum.items()}
    return rows, selector, health_mean


def train_arm(
    arm: str,
    base_state: dict[str, torch.Tensor],
    model_kwargs: dict[str, Any],
    train_dataset: Dataset[Any],
    train_indices: list[int],
    validation_loader: DataLoader[Any],
    test_loader: DataLoader[Any],
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    seed = int(config["seed"])
    set_seed(seed)
    model = NeutralTargetAccess(**model_kwargs).to(device)
    model.load_state_dict(base_state, strict=True)
    optimization = config["optimization"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    train_loader = make_loader(
        train_dataset,
        train_indices,
        int(optimization["batch_size"]),
        True,
        seed,
        int(optimization["num_workers"]),
    )
    best_selector = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, int(optimization["epochs"]) + 1):
        model.train()
        epoch_start = time.time()
        loss_sum = 0.0
        element_count = 0
        for batch in train_loader:
            history, target, _mean, _std, row_ids = prepare_batch(
                batch,
                device,
                model.prediction_length,
            )
            optimizer.zero_grad(set_to_none=True)
            prediction, _health = model(history, row_ids, arm)
            prediction_scaled = prediction * _std + _mean
            target_scaled = target * _std + _mean
            loss = (prediction_scaled - target_scaled).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss arm={arm} epoch={epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                float(optimization["gradient_clip_norm"]),
            )
            optimizer.step()
            loss_sum += float(loss.detach()) * target.numel()
            element_count += target.numel()

        validation_rows, selector, _validation_health = evaluate(
            model,
            validation_loader,
            arm,
            device,
            config,
        )
        improved = selector < best_selector
        if improved:
            best_selector = selector
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        history_rows.append(
            {
                "arm": arm,
                "epoch": epoch,
                "train_standardized_mse": loss_sum / element_count,
                "validation_selector_mse": selector,
                "best_so_far": int(improved),
                "epoch_seconds": time.time() - epoch_start,
            }
        )
        print(
            f"arm={arm} epoch={epoch} "
            f"train_standardized_mse={loss_sum / element_count:.7f} "
            f"val_selector={selector:.7f} best_epoch={best_epoch}",
            flush=True,
        )
        if stale_epochs >= int(optimization["patience"]):
            break

    if best_state is None:
        raise RuntimeError(f"no finite checkpoint selected for arm={arm}")
    model.load_state_dict(best_state, strict=True)
    validation_rows, validation_selector, validation_health = evaluate(
        model,
        validation_loader,
        arm,
        device,
        config,
    )
    test_rows, test_selector, test_health = evaluate(
        model,
        test_loader,
        arm,
        device,
        config,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.pt"
    torch.save(best_state, checkpoint)
    write_csv(output_dir / "training_history.csv", history_rows)
    metric_rows = []
    for split, rows in (("validation", validation_rows), ("test", test_rows)):
        metric_rows.extend({"split": split, **row} for row in rows)
    write_csv(output_dir / "metrics.csv", metric_rows)
    summary = {
        "arm": arm,
        "best_epoch": best_epoch,
        "validation_selector_mse": validation_selector,
        "test_selector_mse": test_selector,
        "parameter_count": model.parameter_count(),
        "checkpoint_sha256": file_hash(checkpoint),
        "validation_health": validation_health,
        "test_health": test_health,
        "epochs_completed": len(history_rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


class SyntheticForecastDataset(Dataset[Any]):
    """Small nonlinear target-access task used only for local execution smoke."""

    def __init__(self, windows: int, length: int, channels: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.history = torch.randn(windows, length, channels, generator=generator)
        future = []
        for target_index in range(length):
            source = (target_index * 7 + 3) % length
            source_two = (target_index * 11 + 5) % length
            value = (
                0.65 * self.history[:, source, :]
                + 0.25 * torch.tanh(self.history[:, source_two, :])
            )
            future.append(value)
        self.future = torch.stack(future, dim=1)
        self.marks = torch.zeros(windows, length, 1)

    def __len__(self) -> int:
        return self.history.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.history[index],
            self.future[index],
            self.marks[index],
            self.marks[index],
        )


def model_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    model_config = config["model"]
    return {
        "series_length": int(config["series_length"]),
        "prediction_length": int(config["prediction_length"]),
        "patch_count": int(model_config["patch_count"]),
        "d_model": int(model_config["d_model"]),
        "n_heads": int(model_config["n_heads"]),
        "dropout": float(model_config["dropout"]),
        "bank_size": int(model_config["permutation_bank_size"]),
        "permutation_seed": int(config["seed"]) + 50_000,
    }


def run_dataset(
    dataset_name: str,
    train_dataset: Dataset[Any],
    validation_dataset: Dataset[Any],
    test_dataset: Dataset[Any],
    config: dict[str, Any],
    config_path: Path | None,
    output_root: Path,
    device: torch.device,
) -> None:
    optimization = config["optimization"]
    train_indices = selected_indices(
        len(train_dataset),
        int(optimization["train_windows"]),
    )
    validation_indices = selected_indices(
        len(validation_dataset),
        int(optimization["validation_windows"]),
    )
    test_indices = list(range(len(test_dataset)))
    validation_loader = make_loader(
        validation_dataset,
        validation_indices,
        int(optimization["batch_size"]),
        False,
        int(config["seed"]),
        int(optimization["num_workers"]),
    )
    test_loader = make_loader(
        test_dataset,
        test_indices,
        int(optimization["batch_size"]),
        False,
        int(config["seed"]),
        int(optimization["num_workers"]),
    )

    kwargs = model_kwargs(config)
    set_seed(int(config["seed"]))
    base_model = NeutralTargetAccess(**kwargs)
    base_state = deepcopy(base_model.state_dict())
    parameter_count = base_model.parameter_count()
    dataset_root = output_root / dataset_name
    summaries = []
    for arm in config["arms"]:
        if arm not in ARMS:
            raise ValueError(f"config contains unsupported arm: {arm}")
        arm_dir = dataset_root / arm
        summary_path = arm_dir / "summary.json"
        if summary_path.exists() and (arm_dir / "metrics.csv").exists():
            print(f"skip_existing dataset={dataset_name} arm={arm}", flush=True)
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        summaries.append(
            train_arm(
                arm,
                base_state,
                kwargs,
                train_dataset,
                train_indices,
                validation_loader,
                test_loader,
                config,
                device,
                arm_dir,
            )
        )
    observed_parameters = {int(summary["parameter_count"]) for summary in summaries}
    if observed_parameters != {parameter_count}:
        raise RuntimeError(f"parameter mismatch: {observed_parameters}")

    metadata = {
        "diagnostic_id": config["diagnostic_id"],
        "candidate_version": config["candidate_version"],
        "dataset": dataset_name,
        "seed": int(config["seed"]),
        "git_commit": git_commit(),
        "config_path": str(config_path) if config_path else "synthetic",
        "config_sha256": file_hash(config_path) if config_path else None,
        "train_windows": len(train_indices),
        "validation_windows": len(validation_indices),
        "test_windows": len(test_indices),
        "test_access_date": "2026-07-20",
        "user_authorization": config["authorization"]["user_authorization_scope"],
        "test_role": config["roles"]["test"],
        "matrix_complete": len(summaries) == len(config["arms"]),
        "checkpoint_retrained": True,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "parameter_count": parameter_count,
        "arm_parameter_gap": 0,
    }
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"dataset_complete={dataset_name} parameters={parameter_count} "
        f"test_windows={len(test_indices)}",
        flush=True,
    )


def synthetic_smoke(output_dir: Path | None) -> None:
    config = json.loads(
        (REPO_ROOT / "configs" / "stage_c_d22c_target_access_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    config["series_length"] = 72
    config["prediction_length"] = 72
    config["model"].update(
        {
            "patch_count": 12,
            "d_model": 8,
            "n_heads": 2,
            "dropout": 0.0,
            "permutation_bank_size": 17,
        }
    )
    config["optimization"].update(
        {
            "epochs": 1,
            "patience": 1,
            "batch_size": 4,
            "train_windows": 12,
            "validation_windows": 4,
            "num_workers": 0,
        }
    )
    config["paper_facing_horizons"] = [12, 24, 48, 72]
    config["checkpoint_selector"]["horizons"] = [12, 24, 48, 72]
    config["coordinate_bins"] = [
        {"name": "early", "start": 0, "end": 24},
        {"name": "late", "start": 24, "end": 72},
    ]
    root = output_dir or Path(tempfile.mkdtemp(prefix="fatst_d22c_smoke_"))
    train = SyntheticForecastDataset(16, 72, 2, 101)
    validation = SyntheticForecastDataset(6, 72, 2, 102)
    test = SyntheticForecastDataset(6, 72, 2, 103)
    run_dataset(
        "Synthetic",
        train,
        validation,
        test,
        config,
        None,
        root,
        torch.device("cpu"),
    )
    for arm in ARMS:
        if not (root / "Synthetic" / arm / "summary.json").exists():
            raise RuntimeError(f"synthetic arm missing: {arm}")
    print(f"d22c_synthetic_smoke=pass output={root}")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke(args.output_dir)
        return
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if tuple(config["arms"]) != ARMS:
        raise ValueError("arm order or membership differs from the frozen contract")
    device = torch.device(args.device)
    train_dataset = build_dataset(args.dataset_root, args.dataset, "train")
    validation_dataset = build_dataset(args.dataset_root, args.dataset, "val")
    test_dataset = build_dataset(args.dataset_root, args.dataset, "test")
    run_dataset(
        args.dataset,
        train_dataset,
        validation_dataset,
        test_dataset,
        config,
        args.config,
        args.output_dir,
        device,
    )


if __name__ == "__main__":
    main()
