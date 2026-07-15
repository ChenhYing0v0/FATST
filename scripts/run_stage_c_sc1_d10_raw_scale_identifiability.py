#!/usr/bin/env python3
"""Run one dataset of the SC1-D10 raw scale-identifiability diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "baselines" / "timealign_official"
for import_root in (str(REPO_ROOT / "scripts"), str(BASELINE_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from check_stage_c_plgo_step5_theory import (  # noqa: E402
    restricted_global_nested_basis,
)
from data_provider.data_factory import data_dict  # noqa: E402
from train_repo import OFFICIAL_PRESETS, resolve_dataset_root  # noqa: E402


GROUP_NAMES = (
    "global_root",
    "detail_depth_0",
    "detail_depth_1",
    "detail_depth_2",
    "detail_depth_3",
    "detail_depth_4",
    "detail_depth_5",
)
FAMILIES = ("canonical", "history_perm", "future_perm")
BINARY_NAMES = ("global", "detail")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--dataset", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]
    )
    parser.add_argument("--design", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_smoke:
        required = ("dataset_root", "dataset", "design", "output_dir")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"missing required arguments: {', '.join(missing)}")
    return args


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def dct_rows(length: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float64, device=device).unsqueeze(0) + 0.5
    frequencies = torch.arange(length, dtype=torch.float64, device=device).unsqueeze(1)
    basis = torch.cos(math.pi * frequencies * positions / length)
    basis[0] *= math.sqrt(1.0 / length)
    if length > 1:
        basis[1:] *= math.sqrt(2.0 / length)
    return basis


def group_indices(group_sizes: list[int]) -> list[torch.Tensor]:
    groups = []
    start = 0
    for size in group_sizes:
        groups.append(torch.arange(start, start + size, dtype=torch.long))
        start += size
    if start != sum(group_sizes):
        raise AssertionError("group index construction failed")
    return groups


def permuted_groups(group_sizes: list[int], seed: int) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(sum(group_sizes), generator=generator)
    groups = []
    start = 0
    for size in group_sizes:
        groups.append(permutation[start : start + size])
        start += size
    return groups


def even_indices(start: int, end: int, count: int) -> list[int]:
    if end <= start or end - start < count:
        raise ValueError(f"insufficient index region [{start}, {end}) for {count} windows")
    values = torch.linspace(start, end - 1, count, dtype=torch.float64).round().long()
    unique = torch.unique_consecutive(values)
    if unique.numel() != count:
        raise RuntimeError("even index selection produced duplicates")
    return [int(value) for value in unique]


def dataset_args() -> SimpleNamespace:
    return SimpleNamespace(
        augmentation_ratio=0,
        jitter=False,
        scaling=False,
        permutation=False,
        randompermutation=False,
        magwarp=False,
        timewarp=False,
        windowslice=False,
        windowwarp=False,
        rotation=False,
        spawner=False,
        dtwwarp=False,
        shapedtwwarp=False,
        wdba=False,
        discdtw=False,
        discsdtw=False,
    )


def build_dataset(dataset_root: Path, dataset: str, flag: str) -> Any:
    preset = OFFICIAL_PRESETS[dataset][720]
    root = resolve_dataset_root(dataset_root, preset)
    dataset_class = data_dict[preset.data]
    return dataset_class(
        args=dataset_args(),
        root_path=str(root),
        data_path=preset.data_path,
        flag=flag,
        size=[720, 0, 720],
        features="M",
        target="OT",
        timeenc=1,
        freq=preset.freq,
        seasonal_patterns="Monthly",
    )


def collect_windows(
    dataset: Any,
    indices: list[int],
    device: torch.device,
    batch_size: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    histories = []
    futures = []
    for start in range(0, len(indices), batch_size):
        selected = [dataset[index] for index in indices[start : start + batch_size]]
        history = torch.as_tensor(
            torch.stack([torch.as_tensor(item[0]) for item in selected]),
            dtype=torch.float64,
            device=device,
        )
        future = torch.as_tensor(
            torch.stack([torch.as_tensor(item[1][-720:]) for item in selected]),
            dtype=torch.float64,
            device=device,
        )
        mean = history.mean(dim=1, keepdim=True)
        std = torch.sqrt(history.var(dim=1, keepdim=True, unbiased=False) + epsilon)
        normalized_history = (history - mean) / std
        normalized_future = (future - mean) / std
        histories.append(normalized_history.permute(0, 2, 1).reshape(-1, 720))
        futures.append(normalized_future.permute(0, 2, 1).reshape(-1, 720))
    return torch.cat(histories, dim=0), torch.cat(futures, dim=0)


def transform_rows(
    history: torch.Tensor,
    future: torch.Tensor,
    dct: torch.Tensor,
    synthesis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return history @ dct.transpose(0, 1), future @ synthesis


def orthogonal_sketch(
    width: int,
    sketch_width: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if width < sketch_width:
        raise ValueError("source width cannot be smaller than sketch width")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(width, sketch_width, generator=generator, dtype=torch.float64)
    q, _upper = torch.linalg.qr(matrix, mode="reduced")
    return q.to(device)


def prepare_sketch(
    fit: torch.Tensor,
    holdout: torch.Tensor,
    validation: torch.Tensor,
    indices: torch.Tensor,
    sketch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = indices.to(fit.device)
    fit_group = fit.index_select(1, indices)
    holdout_group = holdout.index_select(1, indices)
    validation_group = validation.index_select(1, indices)
    mean = fit_group.mean(dim=0, keepdim=True)
    std = fit_group.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    return (
        ((fit_group - mean) / std) @ sketch,
        ((holdout_group - mean) / std) @ sketch,
        ((validation_group - mean) / std) @ sketch,
    )


def ridge_r2(
    fit_x: torch.Tensor,
    fit_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    ridge_lambda: float,
) -> float:
    x_mean = fit_x.mean(dim=0, keepdim=True)
    x_std = fit_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    y_mean = fit_y.mean(dim=0, keepdim=True)
    y_std = fit_y.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    x_fit = (fit_x - x_mean) / x_std
    y_fit = (fit_y - y_mean) / y_std
    x_eval = (eval_x - x_mean) / x_std
    y_eval = (eval_y - y_mean) / y_std
    covariance = x_fit.transpose(0, 1) @ x_fit / x_fit.shape[0]
    cross = x_fit.transpose(0, 1) @ y_fit / x_fit.shape[0]
    identity = torch.eye(covariance.shape[0], dtype=covariance.dtype, device=covariance.device)
    weight = torch.linalg.solve(covariance + ridge_lambda * identity, cross)
    prediction = x_eval @ weight
    sse = (y_eval - prediction).square().sum()
    baseline = y_eval.square().sum().clamp_min(torch.finfo(y_eval.dtype).tiny)
    value = 1.0 - sse / baseline
    if not torch.isfinite(value):
        raise RuntimeError("non-finite ridge R2")
    return float(value)


def family_groups(
    family: str,
    canonical: list[torch.Tensor],
    history_permuted: list[torch.Tensor],
    future_permuted: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if family == "canonical":
        return canonical, canonical
    if family == "history_perm":
        return history_permuted, canonical
    if family == "future_perm":
        return canonical, future_permuted
    raise ValueError(f"unknown family: {family}")


def run_dataset(args: argparse.Namespace, design: dict[str, Any]) -> None:
    device = torch.device(args.device)
    length = int(design["series_length"])
    group_sizes = [int(value) for value in design["group_sizes"]]
    sketch_width = int(design["sketch_width"])
    if sum(group_sizes) != length or tuple(group_sizes) != (16, 16, 32, 64, 128, 256, 208):
        raise ValueError("D10 group contract changed")
    train_dataset = build_dataset(args.dataset_root, args.dataset, "train")
    validation_dataset = build_dataset(args.dataset_root, args.dataset, "val")
    train_length = len(train_dataset)
    fit_end = int(train_length * float(design["fit_region_end_fraction"]))
    holdout_start = int(train_length * float(design["holdout_region_start_fraction"]))
    fit_indices = even_indices(0, fit_end, int(design["fit_windows"]))
    holdout_indices = even_indices(
        holdout_start, train_length, int(design["holdout_windows"])
    )
    fit_holdout_index_gap = min(holdout_indices) - max(fit_indices)
    if fit_holdout_index_gap < 2 * length:
        raise RuntimeError(
            "fit/holdout windows overlap in their history+future observations"
        )
    validation_indices = even_indices(
        0, len(validation_dataset), int(design["validation_windows"])
    )
    epsilon = float(design["normalization_epsilon"])
    fit_history, fit_future = collect_windows(
        train_dataset, fit_indices, device, args.batch_size, epsilon
    )
    holdout_history, holdout_future = collect_windows(
        train_dataset, holdout_indices, device, args.batch_size, epsilon
    )
    validation_history, validation_future = collect_windows(
        validation_dataset, validation_indices, device, args.batch_size, epsilon
    )
    dct = dct_rows(length, device)
    synthesis, atoms, _prototypes, _gap = restricted_global_nested_basis(
        length, int(design["future_global_rank"])
    )
    synthesis = synthesis.to(device)
    fit_history, fit_future = transform_rows(fit_history, fit_future, dct, synthesis)
    holdout_history, holdout_future = transform_rows(
        holdout_history, holdout_future, dct, synthesis
    )
    validation_history, validation_future = transform_rows(
        validation_history, validation_future, dct, synthesis
    )
    canonical = group_indices(group_sizes)
    observed_future_sizes = []
    for group_index in range(len(GROUP_NAMES)):
        observed_future_sizes.append(
            sum(
                1
                for atom in atoms
                if (0 if atom.kind == "global" else atom.depth + 1) == group_index
            )
        )
    if observed_future_sizes != group_sizes:
        raise RuntimeError("RGNB atom groups changed")
    matrix_rows: list[dict[str, Any]] = []
    binary_rows: list[dict[str, Any]] = []
    max_sketch_gap = 0.0
    for sketch_seed in [int(value) for value in design["sketch_seeds"]]:
        history_permuted = permuted_groups(group_sizes, sketch_seed + 1000)
        future_permuted = permuted_groups(group_sizes, sketch_seed + 2000)
        sketches = {
            size: orthogonal_sketch(
                size, sketch_width, sketch_seed + size * 31, device
            )
            for size in sorted(set(group_sizes + [length - group_sizes[0]]))
        }
        for sketch in sketches.values():
            identity = torch.eye(sketch_width, dtype=sketch.dtype, device=device)
            max_sketch_gap = max(
                max_sketch_gap,
                float((sketch.transpose(0, 1) @ sketch - identity).abs().max()),
            )
        for family in FAMILIES:
            history_groups, future_groups = family_groups(
                family, canonical, history_permuted, future_permuted
            )
            history_sketches = []
            future_sketches = []
            for group_index, size in enumerate(group_sizes):
                history_sketches.append(
                    prepare_sketch(
                        fit_history,
                        holdout_history,
                        validation_history,
                        history_groups[group_index],
                        sketches[size],
                    )
                )
                future_sketches.append(
                    prepare_sketch(
                        fit_future,
                        holdout_future,
                        validation_future,
                        future_groups[group_index],
                        sketches[size],
                    )
                )
            history_binary_indices = (
                history_groups[0],
                torch.cat(history_groups[1:]),
            )
            future_binary_indices = (
                future_groups[0],
                torch.cat(future_groups[1:]),
            )
            history_binary = [
                prepare_sketch(
                    fit_history,
                    holdout_history,
                    validation_history,
                    indices,
                    sketches[int(indices.numel())],
                )
                for indices in history_binary_indices
            ]
            future_binary = [
                prepare_sketch(
                    fit_future,
                    holdout_future,
                    validation_future,
                    indices,
                    sketches[int(indices.numel())],
                )
                for indices in future_binary_indices
            ]
            for ridge_lambda in [float(value) for value in design["ridge_lambdas"]]:
                for future_index, future_group in enumerate(future_sketches):
                    for history_index, history_group in enumerate(history_sketches):
                        for split, x_values, y_values in (
                            ("holdout", history_group[1], future_group[1]),
                            ("validation", history_group[2], future_group[2]),
                        ):
                            matrix_rows.append(
                                {
                                    "dataset": args.dataset,
                                    "family": family,
                                    "sketch_seed": sketch_seed,
                                    "ridge_lambda": ridge_lambda,
                                    "split": split,
                                    "future_group_index": future_index,
                                    "future_group": GROUP_NAMES[future_index],
                                    "history_group_index": history_index,
                                    "history_group": GROUP_NAMES[history_index],
                                    "input_width": sketch_width,
                                    "output_width": sketch_width,
                                    "r2": ridge_r2(
                                        history_group[0],
                                        future_group[0],
                                        x_values,
                                        y_values,
                                        ridge_lambda,
                                    ),
                                }
                            )
                for future_index, future_group in enumerate(future_binary):
                    for history_index, history_group in enumerate(history_binary):
                        for split, x_values, y_values in (
                            ("holdout", history_group[1], future_group[1]),
                            ("validation", history_group[2], future_group[2]),
                        ):
                            binary_rows.append(
                                {
                                    "dataset": args.dataset,
                                    "family": family,
                                    "sketch_seed": sketch_seed,
                                    "ridge_lambda": ridge_lambda,
                                    "split": split,
                                    "future_binary_index": future_index,
                                    "future_binary": BINARY_NAMES[future_index],
                                    "history_binary_index": history_index,
                                    "history_binary": BINARY_NAMES[history_index],
                                    "input_width": sketch_width,
                                    "output_width": sketch_width,
                                    "r2": ridge_r2(
                                        history_group[0],
                                        future_group[0],
                                        x_values,
                                        y_values,
                                        ridge_lambda,
                                    ),
                                }
                            )
    dataset_dir = args.output_dir / args.dataset
    write_csv(dataset_dir / "matrix_cell_metrics.csv", matrix_rows)
    write_csv(dataset_dir / "binary_cell_metrics.csv", binary_rows)
    dct_identity = torch.eye(length, dtype=dct.dtype, device=device)
    synthesis_identity = torch.eye(length, dtype=synthesis.dtype, device=device)
    metadata = {
        "dataset": args.dataset,
        "train_dataset_length": train_length,
        "validation_dataset_length": len(validation_dataset),
        "fit_indices": fit_indices,
        "holdout_indices": holdout_indices,
        "validation_indices": validation_indices,
        "fit_holdout_index_gap": fit_holdout_index_gap,
        "fit_holdout_observation_overlap": False,
        "fit_rows": int(fit_history.shape[0]),
        "holdout_rows": int(holdout_history.shape[0]),
        "validation_rows": int(validation_history.shape[0]),
        "channels": int(fit_history.shape[0] / len(fit_indices)),
        "group_sizes": group_sizes,
        "matrix_rows": len(matrix_rows),
        "binary_rows": len(binary_rows),
        "dct_orthogonality_max_abs": float(
            (dct @ dct.transpose(0, 1) - dct_identity).abs().max()
        ),
        "rgnb_orthogonality_max_abs": float(
            (synthesis.transpose(0, 1) @ synthesis - synthesis_identity).abs().max()
        ),
        "sketch_orthogonality_max_abs": max_sketch_gap,
        "uses_train_split": True,
        "uses_validation_split": True,
        "uses_test_split": False,
        "forecast_model_updated": False,
        "forecast_model_trained": False,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    (dataset_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"d10_dataset_done dataset={args.dataset} matrix_rows={len(matrix_rows)} "
        f"binary_rows={len(binary_rows)}",
        flush=True,
    )


def synthetic_smoke() -> None:
    device = torch.device("cpu")
    generator = torch.Generator(device="cpu").manual_seed(20260715)
    fit_x = torch.randn(128, 16, generator=generator, dtype=torch.float64)
    weight = torch.randn(16, 16, generator=generator, dtype=torch.float64)
    fit_y = fit_x @ weight + 0.01 * torch.randn(
        128, 16, generator=generator, dtype=torch.float64
    )
    eval_x = torch.randn(64, 16, generator=generator, dtype=torch.float64)
    eval_y = eval_x @ weight
    r2 = ridge_r2(fit_x, fit_y, eval_x, eval_y, 0.01)
    sketch = orthogonal_sketch(64, 16, 20260715, device)
    gap = float(
        (
            sketch.transpose(0, 1) @ sketch
            - torch.eye(16, dtype=torch.float64)
        )
        .abs()
        .max()
    )
    if r2 < 0.99 or gap > 1e-10:
        raise RuntimeError(f"synthetic smoke failed: r2={r2}, gap={gap}")
    groups = group_indices([16, 16, 32, 64, 128, 256, 208])
    if len(groups) != 7 or int(torch.cat(groups).numel()) != 720:
        raise RuntimeError("synthetic group coverage failed")
    print("stage_c_sc1_d10_worker_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    design = json.loads(args.design.read_text(encoding="utf-8"))
    run_dataset(args, design)


if __name__ == "__main__":
    main()
