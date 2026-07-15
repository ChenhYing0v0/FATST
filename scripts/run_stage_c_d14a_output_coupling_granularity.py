#!/usr/bin/env python3
"""Run one dataset of the StageC D14-A0 output-coupling diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "baselines" / "timealign_official"
for import_root in (str(REPO_ROOT / "scripts"), str(BASELINE_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from run_stage_c_sc1_d10_raw_scale_identifiability import (  # noqa: E402
    build_dataset,
    even_indices,
)


CANONICAL_SCALES = (1, 48, 144, 360, 720)
INTERMEDIATE_SCALES = (48, 144, 360)


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


def collect_windows(
    dataset: Any,
    indices: list[int],
    device: torch.device,
    batch_size: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    histories: list[torch.Tensor] = []
    futures: list[torch.Tensor] = []
    window_ids: list[torch.Tensor] = []
    channel_ids: list[torch.Tensor] = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        selected = [dataset[index] for index in batch_indices]
        history = torch.stack([torch.as_tensor(item[0]) for item in selected]).to(
            device=device, dtype=torch.float64
        )
        future = torch.stack(
            [torch.as_tensor(item[1][-720:]) for item in selected]
        ).to(device=device, dtype=torch.float64)
        mean = history.mean(dim=1, keepdim=True)
        std = torch.sqrt(history.var(dim=1, keepdim=True, unbiased=False) + epsilon)
        histories.append(((history - mean) / std).permute(0, 2, 1).reshape(-1, 720))
        futures.append(((future - mean) / std).permute(0, 2, 1).reshape(-1, 720))
        channels = history.shape[2]
        window_ids.append(
            torch.as_tensor(batch_indices, device=device).repeat_interleave(channels)
        )
        channel_ids.append(
            torch.arange(channels, device=device).repeat(len(batch_indices))
        )
    return (
        torch.cat(histories),
        torch.cat(futures),
        torch.cat(window_ids),
        torch.cat(channel_ids),
    )


def parameter_count(width: int, length: int, scale: int, rank: int) -> int:
    return (length // scale) * rank * (width + scale)


def partition_groups(
    length: int, scale: int, family: str, seed: int
) -> list[torch.Tensor]:
    if length % scale:
        raise ValueError(f"scale {scale} does not divide length {length}")
    order = torch.arange(length, dtype=torch.long)
    if family == "shifted":
        order = torch.roll(order, shifts=-(scale // 2))
    elif family == "random":
        generator = torch.Generator(device="cpu").manual_seed(seed + scale * 101)
        order = order[torch.randperm(length, generator=generator)]
    elif family != "canonical":
        raise ValueError(f"unknown partition family: {family}")
    groups = list(order.split(scale))
    covered = torch.sort(torch.cat(groups)).values
    if not torch.equal(covered, torch.arange(length)):
        raise RuntimeError(f"{family} scale={scale} is not a disjoint cover")
    return groups


def fit_pca_carrier(
    fit_history: torch.Tensor, width: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    history_mean = fit_history.mean(dim=0, keepdim=True)
    centered = fit_history - history_mean
    covariance = centered.transpose(0, 1) @ centered / centered.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    components = eigenvectors[:, -width:]
    scores = centered @ components
    score_std = scores.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    identity = torch.eye(width, dtype=components.dtype, device=components.device)
    orthogonality_gap = float(
        (components.transpose(0, 1) @ components - identity).abs().max()
    )
    standardized = scores / score_std
    standardized_covariance = standardized.transpose(0, 1) @ standardized
    standardized_covariance /= standardized.shape[0]
    retained = torch.linalg.eigvalsh(standardized_covariance).clamp_min(
        torch.finfo(eigenvalues.dtype).tiny
    )
    condition_number = float(retained.max() / retained.min())
    return history_mean, components, score_std, orthogonality_gap, condition_number


def transform_carrier(
    history: torch.Tensor,
    history_mean: torch.Tensor,
    components: torch.Tensor,
    score_std: torch.Tensor,
) -> torch.Tensor:
    return ((history - history_mean) @ components) / score_std


def fit_full_affine(
    fit_x: torch.Tensor, fit_y: torch.Tensor, numeric_ridge: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_mean = fit_x.mean(dim=0, keepdim=True)
    y_mean = fit_y.mean(dim=0, keepdim=True)
    centered_x = fit_x - x_mean
    centered_y = fit_y - y_mean
    covariance = centered_x.transpose(0, 1) @ centered_x
    identity = torch.eye(
        covariance.shape[0], dtype=covariance.dtype, device=covariance.device
    )
    weight = torch.linalg.solve(
        covariance + numeric_ridge * identity,
        centered_x.transpose(0, 1) @ centered_y,
    )
    return weight, x_mean, y_mean, covariance


def reduced_rank_weight(
    full_weight: torch.Tensor,
    covariance: torch.Tensor,
    groups: list[torch.Tensor],
    rank: int,
) -> tuple[torch.Tensor, int]:
    width, length = full_weight.shape
    output = torch.zeros_like(full_weight)
    jitter = torch.finfo(covariance.dtype).eps * float(covariance.diag().mean())
    identity = torch.eye(width, dtype=covariance.dtype, device=covariance.device)
    root = torch.linalg.cholesky(covariance + jitter * identity).transpose(0, 1)
    max_observed_rank = 0
    for cpu_indices in groups:
        indices = cpu_indices.to(full_weight.device)
        block = full_weight.index_select(1, indices)
        if rank >= min(block.shape):
            projected = block
        else:
            _left, _singular, right_t = torch.linalg.svd(
                root @ block, full_matrices=False
            )
            right = right_t[:rank].transpose(0, 1)
            projected = block @ right @ right.transpose(0, 1)
        output[:, indices] = projected
        observed_rank = int(torch.linalg.matrix_rank(projected).item())
        max_observed_rank = max(max_observed_rank, observed_rank)
        if observed_rank > rank:
            raise RuntimeError(
                f"reduced-rank invariant failed: observed={observed_rank}, rank={rank}"
            )
    if output.shape != (width, length):
        raise AssertionError("reduced-rank output shape changed")
    return output, max_observed_rank


def predict(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_mean: torch.Tensor,
    y_mean: torch.Tensor,
) -> torch.Tensor:
    return (x - x_mean) @ weight + y_mean


def loss_rows(prediction: torch.Tensor, target: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    error = prediction - target
    return (
        error.square().detach().cpu().numpy(),
        error.abs().detach().cpu().numpy(),
    )


def arm_family_and_scale(name: str) -> tuple[str, int]:
    for family in ("canonical", "shifted", "random"):
        prefix = f"{family}_s"
        if name.startswith(prefix):
            scale_text = name.removeprefix(prefix)
            if not scale_text.isdigit():
                raise ValueError(f"malformed coupling arm: {name}")
            return family, int(scale_text)
    return "control", 0


def append_region_metrics(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    mse: np.ndarray,
    mae: np.ndarray,
    design: dict[str, Any],
) -> None:
    regions: list[tuple[str, int, int]] = [("full", 0, mse.shape[1])]
    regions.extend(
        (f"bin_{entry['name']}", int(entry["start"]), int(entry["end"]))
        for entry in design["future_bins"]
    )
    regions.extend(
        (f"prefix_{horizon}", 0, int(horizon))
        for horizon in design["dense_horizons"]
    )
    for region, start, end in regions:
        rows.append(
            {
                **base,
                "region": region,
                "region_start": start,
                "region_end": end,
                "mse": float(mse[:, start:end].mean()),
                "mae": float(mae[:, start:end].mean()),
            }
        )


def build_arm_weights(
    full_weight: torch.Tensor,
    covariance: torch.Tensor,
    design: dict[str, Any],
    partition_seed: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    length = int(design["series_length"])
    width = int(design["feature_width"])
    ranks = {int(key): int(value) for key, value in design["expected_block_ranks"].items()}
    weights: dict[str, torch.Tensor] = {}
    parameter_rows: list[dict[str, Any]] = []
    for family in ("canonical", "shifted", "random"):
        scales = CANONICAL_SCALES if family == "canonical" else INTERMEDIATE_SCALES
        for scale in scales:
            groups = partition_groups(length, scale, family, partition_seed)
            weight, observed_rank = reduced_rank_weight(
                full_weight, covariance, groups, ranks[scale]
            )
            name = f"{family}_s{scale}"
            weights[name] = weight
            parameter_rows.append(
                {
                    "arm": name,
                    "family": family,
                    "scale": scale,
                    "blocks": len(groups),
                    "rank_per_block": ranks[scale],
                    "max_observed_rank": observed_rank,
                    "factor_parameters": parameter_count(width, length, scale, ranks[scale]),
                }
            )
    point_gap = float((weights["canonical_s1"] - full_weight).abs().max())
    if point_gap > 1e-10:
        raise RuntimeError(f"point arm differs from full affine: gap={point_gap}")
    return weights, parameter_rows


def choose_indices(
    dataset_length: int, fold: dict[str, Any], count_key: str, design: dict[str, Any]
) -> list[int]:
    prefix = "fit" if count_key == "fit_windows" else "calibration"
    start = int(dataset_length * float(fold[f"{prefix}_start_fraction"]))
    end = int(dataset_length * float(fold[f"{prefix}_end_fraction"]))
    return even_indices(start, end, int(design[count_key]))


def run_dataset(args: argparse.Namespace, design: dict[str, Any]) -> None:
    device = torch.device(args.device)
    length = int(design["series_length"])
    width = int(design["feature_width"])
    if length != 720 or tuple(int(value) for value in design["coupling_scales"]) != CANONICAL_SCALES:
        raise ValueError("D14-A0 scale contract changed")
    train_dataset = build_dataset(args.dataset_root, args.dataset, "train")
    validation_dataset = build_dataset(args.dataset_root, args.dataset, "val")
    validation_indices = even_indices(
        0, len(validation_dataset), int(design["validation_windows"])
    )
    epsilon = float(design["normalization_epsilon"])
    validation_history, validation_future, validation_window_ids, validation_channel_ids = collect_windows(
        validation_dataset, validation_indices, device, args.batch_size, epsilon
    )
    dataset_dir = args.output_dir / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    all_parameter_rows: list[dict[str, Any]] = []
    fold_metadata: list[dict[str, Any]] = []
    for fold in design["folds"]:
        fold_id = int(fold["fold"])
        fit_indices = choose_indices(len(train_dataset), fold, "fit_windows", design)
        calibration_indices = choose_indices(
            len(train_dataset), fold, "calibration_windows", design
        )
        index_gap = min(calibration_indices) - max(fit_indices)
        if index_gap < 2 * length:
            raise RuntimeError(f"fold {fold_id} fit/calibration observations overlap")
        fit_history, fit_future, _fit_windows, _fit_channels = collect_windows(
            train_dataset, fit_indices, device, args.batch_size, epsilon
        )
        calibration_history, calibration_future, _cal_windows, _cal_channels = collect_windows(
            train_dataset, calibration_indices, device, args.batch_size, epsilon
        )
        history_mean, components, score_std, orth_gap, condition_number = fit_pca_carrier(
            fit_history, width
        )
        fit_x = transform_carrier(fit_history, history_mean, components, score_std)
        calibration_x = transform_carrier(
            calibration_history, history_mean, components, score_std
        )
        validation_x = transform_carrier(
            validation_history, history_mean, components, score_std
        )
        full_weight, x_mean, y_mean, covariance = fit_full_affine(
            fit_x, fit_future, float(design["numeric_ridge"])
        )
        weights, parameter_rows = build_arm_weights(
            full_weight, covariance, design, int(fold["partition_seed"])
        )
        for row in parameter_rows:
            all_parameter_rows.append({"dataset": args.dataset, "fold": fold_id, **row})
        calibration_predictions = {
            name: predict(calibration_x, weight, x_mean, y_mean)
            for name, weight in weights.items()
        }
        validation_predictions = {
            name: predict(validation_x, weight, x_mean, y_mean)
            for name, weight in weights.items()
        }
        canonical_names = [f"canonical_s{scale}" for scale in CANONICAL_SCALES]
        calibration_mse = {
            name: float((calibration_predictions[name] - calibration_future).square().mean())
            for name in canonical_names
        }
        selected_name = min(calibration_mse, key=calibration_mse.get)
        calibration_predictions["train_selected_best"] = calibration_predictions[selected_name]
        validation_predictions["train_selected_best"] = validation_predictions[selected_name]
        calibration_predictions["equal_canonical"] = torch.stack(
            [calibration_predictions[name] for name in canonical_names]
        ).mean(dim=0)
        validation_predictions["equal_canonical"] = torch.stack(
            [validation_predictions[name] for name in canonical_names]
        ).mean(dim=0)
        calibration_predictions["train_mean"] = y_mean.expand_as(calibration_future)
        validation_predictions["train_mean"] = y_mean.expand_as(validation_future)
        calibration_predictions["persistence"] = calibration_history[:, -1:].expand_as(
            calibration_future
        )
        validation_predictions["persistence"] = validation_history[:, -1:].expand_as(
            validation_future
        )
        arm_names = list(weights) + [
            "train_selected_best",
            "equal_canonical",
            "train_mean",
            "persistence",
        ]
        validation_bin_mse: list[np.ndarray] = []
        validation_bin_mae: list[np.ndarray] = []
        bins = [(int(item["start"]), int(item["end"])) for item in design["future_bins"]]
        for split, predictions, target in (
            ("calibration", calibration_predictions, calibration_future),
            ("validation", validation_predictions, validation_future),
        ):
            for name in arm_names:
                mse, mae = loss_rows(predictions[name], target)
                family, scale = arm_family_and_scale(name)
                append_region_metrics(
                    metric_rows,
                    {
                        "dataset": args.dataset,
                        "fold": fold_id,
                        "split": split,
                        "arm": name,
                        "family": family,
                        "scale": scale,
                        "selected_canonical_arm": selected_name,
                    },
                    mse,
                    mae,
                    design,
                )
                if split == "validation":
                    validation_bin_mse.append(
                        np.stack([mse[:, start:end].mean(axis=1) for start, end in bins], axis=1)
                    )
                    validation_bin_mae.append(
                        np.stack([mae[:, start:end].mean(axis=1) for start, end in bins], axis=1)
                    )
        np.savez_compressed(
            dataset_dir / f"validation_bin_losses_fold{fold_id}.npz",
            arms=np.asarray(arm_names),
            mse=np.stack(validation_bin_mse),
            mae=np.stack(validation_bin_mae),
            bin_names=np.asarray([item["name"] for item in design["future_bins"]]),
            window_ids=validation_window_ids.detach().cpu().numpy(),
            channel_ids=validation_channel_ids.detach().cpu().numpy(),
        )
        fold_metadata.append(
            {
                "fold": fold_id,
                "fit_indices": fit_indices,
                "calibration_indices": calibration_indices,
                "fit_calibration_index_gap": index_gap,
                "fit_calibration_observation_overlap": False,
                "fit_rows": int(fit_history.shape[0]),
                "calibration_rows": int(calibration_history.shape[0]),
                "validation_rows": int(validation_history.shape[0]),
                "pca_orthogonality_max_abs": orth_gap,
                "feature_condition_number": condition_number,
                "selected_canonical_arm": selected_name,
                "calibration_canonical_mse": calibration_mse,
                "all_finite": all(
                    bool(torch.isfinite(value).all())
                    for value in validation_predictions.values()
                ),
            }
        )
        del fit_history, fit_future, calibration_history, calibration_future
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(dataset_dir / "fold_metrics.csv", metric_rows)
    write_csv(dataset_dir / "parameter_budget.csv", all_parameter_rows)
    metadata = {
        "dataset": args.dataset,
        "diagnostic_id": design["diagnostic_id"],
        "train_dataset_length": len(train_dataset),
        "validation_dataset_length": len(validation_dataset),
        "validation_indices": validation_indices,
        "channels": int(validation_history.shape[0] / len(validation_indices)),
        "folds": fold_metadata,
        "uses_train_split": True,
        "uses_validation_split": True,
        "uses_test_split": False,
        "forecast_model_updated": False,
        "forecast_model_trained": False,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    (dataset_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"d14a_dataset_done dataset={args.dataset} folds={len(fold_metadata)} "
        f"metric_rows={len(metric_rows)}",
        flush=True,
    )


def synthetic_smoke() -> None:
    device = torch.device("cpu")
    generator = torch.Generator(device="cpu").manual_seed(20260715)
    fit_x = torch.randn(256, 16, generator=generator, dtype=torch.float64)
    latent = torch.randn(16, 4, generator=generator, dtype=torch.float64)
    decoder = torch.randn(4, 48, generator=generator, dtype=torch.float64)
    fit_y = fit_x @ latent @ decoder
    full, _x_mean, _y_mean, covariance = fit_full_affine(fit_x, fit_y, 1e-8)
    groups = partition_groups(48, 12, "random", 3101)
    reduced, observed_rank = reduced_rank_weight(full, covariance, groups, 4)
    cover = torch.sort(torch.cat(groups)).values
    if observed_rank > 4 or not torch.equal(cover, torch.arange(48)):
        raise RuntimeError("synthetic reduced-rank or partition invariant failed")
    point_groups = partition_groups(48, 1, "canonical", 3101)
    point, _rank = reduced_rank_weight(full, covariance, point_groups, 1)
    if float((point - full).abs().max()) > 1e-10:
        raise RuntimeError("synthetic point/full-affine equivalence failed")
    budgets = [
        parameter_count(64, 720, scale, rank)
        for scale, rank in zip(CANONICAL_SCALES, (1, 28, 45, 55, 60))
    ]
    if (max(budgets) - min(budgets)) / budgets[0] > 0.01:
        raise RuntimeError("synthetic parameter matching failed")
    if not torch.isfinite(reduced).all():
        raise RuntimeError("synthetic finite invariant failed")
    if arm_family_and_scale("canonical_s144") != ("canonical", 144):
        raise RuntimeError("synthetic coupling arm parsing failed")
    if arm_family_and_scale("train_selected_best") != ("control", 0):
        raise RuntimeError("synthetic control arm parsing failed")
    print("stage_c_d14a_worker_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    design = json.loads(args.design.read_text(encoding="utf-8"))
    run_dataset(args, design)


if __name__ == "__main__":
    main()
