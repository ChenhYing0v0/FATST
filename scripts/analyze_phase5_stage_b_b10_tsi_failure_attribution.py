from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
if str(TIMEALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEALIGN_ROOT))

from data_provider.data_factory import data_provider  # noqa: E402
from models.TimeAlign import Model  # noqa: E402
from train_repo import OFFICIAL_PRESETS, build_official_args  # noqa: E402


DATASETS = ("ETTh2", "ETTm1", "Weather")
HORIZONS = (96, 192, 336, 720)
SEGMENTS = (
    ("early_0_96", 0, 96),
    ("mid_96_192", 96, 192),
    ("late_192_336", 192, 336),
    ("tail_336_720", 336, 720),
)
DEFAULT_ANALYSIS_ROOT = Path("analysis/phase5_stage_b_b10_tsi_failure_attribution_20260708")
DEFAULT_CHECKPOINT_ROOT = (
    Path("analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707")
    / "raw"
    / "official-last"
    / "TimeAlignOfficialUnified720_a6_clean_official-last"
)
DEFAULT_DATASET_ROOT = Path("/Users/river/PaperResearch/Project/datasets")


@dataclass(frozen=True)
class SplitArrays:
    coeff: np.ndarray
    memory_pool: np.ndarray
    residual: np.ndarray


@dataclass(frozen=True)
class SegmentSpace:
    u: np.ndarray
    singular_values: np.ndarray


@dataclass(frozen=True)
class FitChoice:
    alpha: float
    val_mse: float
    weights: Any


@dataclass(frozen=True)
class ShrinkChoice:
    alpha: float
    beta: float
    val_mse: float
    shared_weights: np.ndarray
    target_weights: dict[str, np.ndarray]


@dataclass(frozen=True)
class DatasetResult:
    dataset: str
    summary_rows: list[dict[str, Any]]
    segment_rows: list[dict[str, Any]]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_path(root: Path, dataset: str) -> Path:
    return root / dataset / "mixed_h96_h192_h336_h720" / "seed2021" / "checkpoint.pt"


def build_args(args: argparse.Namespace, dataset: str) -> argparse.Namespace:
    preset = OFFICIAL_PRESETS[dataset][720]
    adapter_args = SimpleNamespace(
        dataset_root=args.dataset_root,
        output_dir=args.analysis_root / "_tmp_official_args",
        dataset=dataset,
        seq_len=720,
        label_len=48,
        pred_len=720,
        e_layers=2,
        num_workers=0,
        epochs=10,
        batch_size=args.batch_size,
        patience=3,
        use_amp=False,
        seed=args.seed,
        device=args.device,
        readout_mode="learned-basis-forecast-operator",
        w_align=None,
        w_recon=0.0,
        target_horizons=list(HORIZONS),
        basis_rank=256,
        stage_token_dim=32,
        stage_field_rank=32,
        stage_gate_init=-5.0,
    )
    official_args = build_official_args(adapter_args, preset)
    official_args.batch_size = args.batch_size
    official_args.num_workers = 0
    return official_args


def load_model(official_args: argparse.Namespace, checkpoint: Path) -> Model:
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = Model(official_args)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def forward_arrays(model: Model, batch_x: torch.Tensor, batch_y: torch.Tensor) -> SplitArrays:
    batch, seq_len, channels = batch_x.shape
    x = model.normalization_x(batch_x, "norm")
    target = batch_y[:, -720:, :]
    target_norm = (target - model.normalization_x.mean) / model.normalization_x.stdev

    x = model.patch_emb_x(x.permute(0, 2, 1).reshape(-1, channels * seq_len))
    for layer_idx in range(model.e_layers):
        x = x + model.encoder[layer_idx](x)
        if model.layer_norm:
            x = model.norm_x[layer_idx](x)

    memory = x.reshape(batch, channels, model.patch_num, model.d_model)
    hidden = memory.flatten(start_dim=-2)
    coeff = model.learned_basis_coeff(hidden)
    basis = model.learned_temporal_basis.to(dtype=hidden.dtype)
    bias = model.learned_temporal_bias.to(dtype=hidden.dtype)
    pred_norm = torch.einsum("hk,bck->bch", basis, coeff) + bias.view(1, 1, -1)
    pred_norm = pred_norm.permute(0, 2, 1)
    residual = target_norm - pred_norm

    memory_std = torch.std(memory, dim=2, unbiased=False)
    memory_pool = torch.cat([memory.mean(dim=2), memory[:, :, -1, :], memory_std], dim=-1)
    return SplitArrays(
        coeff=coeff.reshape(-1, coeff.shape[-1]).detach().cpu().numpy().astype(np.float64),
        memory_pool=memory_pool.reshape(-1, memory_pool.shape[-1]).detach().cpu().numpy().astype(np.float64),
        residual=residual.permute(0, 2, 1).reshape(-1, residual.shape[1]).detach().cpu().numpy().astype(np.float64),
    )


def collect_split(
    args: argparse.Namespace,
    model: Model,
    official_args: argparse.Namespace,
    split: str,
    max_rows: int,
) -> SplitArrays:
    _data, loader = data_provider(official_args, split)
    coeff_parts: list[np.ndarray] = []
    memory_parts: list[np.ndarray] = []
    residual_parts: list[np.ndarray] = []
    total = 0
    device = torch.device(args.device)
    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
            if total >= max_rows:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            arrays = forward_arrays(model, batch_x, batch_y)
            remaining = max_rows - total
            take = min(remaining, arrays.coeff.shape[0])
            coeff_parts.append(arrays.coeff[:take])
            memory_parts.append(arrays.memory_pool[:take])
            residual_parts.append(arrays.residual[:take])
            total += take
    if not coeff_parts:
        raise RuntimeError(f"No rows collected for split={split}")
    return SplitArrays(
        coeff=np.concatenate(coeff_parts, axis=0),
        memory_pool=np.concatenate(memory_parts, axis=0),
        residual=np.concatenate(residual_parts, axis=0),
    )


def parse_float_list(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one float")
    return parsed


def feature_stats(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def apply_feature_stats(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (values - mean) / std
    return np.concatenate([z, np.ones((z.shape[0], 1), dtype=z.dtype)], axis=1)


def ridge_fit(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = train_x.T @ train_x
    reg = np.eye(xtx.shape[0], dtype=train_x.dtype) * alpha
    reg[-1, -1] = 0.0
    rhs = train_x.T @ train_y
    return np.linalg.solve(xtx + reg, rhs)


def build_segment_spaces(basis: np.ndarray, rank: int) -> dict[str, SegmentSpace]:
    spaces: dict[str, SegmentSpace] = {}
    kept_rank = rank
    svd_parts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, start, end in SEGMENTS:
        u, singular_values, _vh = np.linalg.svd(basis[start:end], full_matrices=False)
        keep = min(rank, int(np.sum(singular_values > 1e-10)))
        kept_rank = min(kept_rank, keep)
        svd_parts[name] = (u, singular_values)
    if kept_rank <= 0:
        raise RuntimeError("No usable basis row-space rank")
    for name, (u, singular_values) in svd_parts.items():
        spaces[name] = SegmentSpace(u=u[:, :kept_rank], singular_values=singular_values[:kept_rank])
    return spaces


def rowspace_targets(residual: np.ndarray, spaces: dict[str, SegmentSpace]) -> dict[str, np.ndarray]:
    targets: dict[str, np.ndarray] = {}
    for name, start, end in SEGMENTS:
        space = spaces[name]
        targets[name] = residual[:, start:end] @ space.u
    return targets


def correction_from_coords(coords: np.ndarray, space: SegmentSpace) -> np.ndarray:
    return coords @ space.u.T


def fit_shared_control(train_x: np.ndarray, train_targets: dict[str, np.ndarray], alpha: float) -> np.ndarray:
    pooled_x = np.concatenate([train_x for _name, _start, _end in SEGMENTS], axis=0)
    pooled_y = np.concatenate([train_targets[name] for name, _start, _end in SEGMENTS], axis=0)
    return ridge_fit(pooled_x, pooled_y, alpha)


def fit_pooled_multihead_control(
    train_x: np.ndarray,
    train_targets: dict[str, np.ndarray],
    alpha: float,
    heads: int,
    seed: int,
) -> list[np.ndarray]:
    pooled_x = np.concatenate([train_x for _name, _start, _end in SEGMENTS], axis=0)
    pooled_y = np.concatenate([train_targets[name] for name, _start, _end in SEGMENTS], axis=0)
    rng = np.random.default_rng(seed)
    sample_size = train_x.shape[0]
    weights: list[np.ndarray] = []
    for _head_idx in range(heads):
        indices = rng.choice(pooled_x.shape[0], size=sample_size, replace=False)
        weights.append(ridge_fit(pooled_x[indices], pooled_y[indices], alpha))
    return weights


def evaluate_weights(
    features: np.ndarray,
    residual_rows: np.ndarray,
    spaces: dict[str, SegmentSpace],
    shared_w: np.ndarray,
    pooled_weights: list[np.ndarray],
    target_weights: dict[str, np.ndarray],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    totals = {"base": 0.0, "shared": 0.0, "pooled": 0.0, "target": 0.0, "wrong_target": 0.0}
    counts = 0
    segment_rows: list[dict[str, Any]] = []
    names = [segment[0] for segment in SEGMENTS]
    shared_coords = features @ shared_w
    pooled_coords = np.mean([features @ weight for weight in pooled_weights], axis=0)
    target_coords = {name: features @ target_weights[name] for name in names}
    for idx, (name, start, end) in enumerate(SEGMENTS):
        space = spaces[name]
        residual = residual_rows[:, start:end]
        wrong_name = names[(idx + 1) % len(names)]
        base_mse = float(np.mean(residual * residual))
        shared_mse = segment_mse(residual, correction_from_coords(shared_coords, space))
        pooled_mse = segment_mse(residual, correction_from_coords(pooled_coords, space))
        target_mse = segment_mse(residual, correction_from_coords(target_coords[name], space))
        wrong_mse = segment_mse(residual, correction_from_coords(target_coords[wrong_name], space))
        count = int(residual.size)
        totals["base"] += base_mse * count
        totals["shared"] += shared_mse * count
        totals["pooled"] += pooled_mse * count
        totals["target"] += target_mse * count
        totals["wrong_target"] += wrong_mse * count
        counts += count
        segment_rows.append(
            {
                "segment": name,
                "base_mse": base_mse,
                "shared_control_mse": shared_mse,
                "pooled_multihead_control_mse": pooled_mse,
                "target_set_aware_mse": target_mse,
                "wrong_target_control_mse": wrong_mse,
                "target_vs_pooled_multihead_reduction_pct": pct_reduction(pooled_mse, target_mse),
                "target_vs_wrong_target_reduction_pct": pct_reduction(wrong_mse, target_mse),
            }
        )
    metrics = {
        "base_mse": totals["base"] / counts,
        "shared_control_mse": totals["shared"] / counts,
        "pooled_multihead_control_mse": totals["pooled"] / counts,
        "target_set_aware_mse": totals["target"] / counts,
        "wrong_target_control_mse": totals["wrong_target"] / counts,
    }
    return metrics, segment_rows


def segment_mse(residual: np.ndarray, correction: np.ndarray) -> float:
    remaining = residual - correction
    return float(np.mean(remaining * remaining))


def pct_reduction(base: float, value: float) -> float:
    return 100.0 * (base - value) / max(base, 1e-12)


def select_choice(
    train_x: np.ndarray,
    val_x: np.ndarray,
    train_targets: dict[str, np.ndarray],
    val_residual: np.ndarray,
    spaces: dict[str, SegmentSpace],
    args: argparse.Namespace,
    arm: str,
) -> FitChoice:
    candidates: list[FitChoice] = []
    for alpha in args.readout_ridge_alphas:
        if arm == "shared":
            weights = fit_shared_control(train_x, train_targets, alpha)
            metrics, _rows = evaluate_weights(
                val_x,
                val_residual,
                spaces,
                weights,
                [weights for _ in range(args.control_heads)],
                {name: weights for name, _start, _end in SEGMENTS},
            )
            val_mse = metrics["shared_control_mse"]
        elif arm == "pooled":
            weights = fit_pooled_multihead_control(train_x, train_targets, alpha, args.control_heads, args.seed)
            metrics, _rows = evaluate_weights(
                val_x,
                val_residual,
                spaces,
                weights[0],
                weights,
                {name: weights[0] for name, _start, _end in SEGMENTS},
            )
            val_mse = metrics["pooled_multihead_control_mse"]
        elif arm == "target":
            weights = {name: ridge_fit(train_x, train_targets[name], alpha) for name, _start, _end in SEGMENTS}
            first = next(iter(weights.values()))
            metrics, _rows = evaluate_weights(
                val_x,
                val_residual,
                spaces,
                first,
                [first for _ in range(args.control_heads)],
                weights,
            )
            val_mse = metrics["target_set_aware_mse"]
        else:
            raise ValueError(f"unknown arm: {arm}")
        candidates.append(FitChoice(alpha=alpha, val_mse=val_mse, weights=weights))
    return min(candidates, key=lambda item: item.val_mse)


def shrink_weights(
    shared_weights: np.ndarray,
    target_weights: dict[str, np.ndarray],
    beta: float,
) -> dict[str, np.ndarray]:
    return {
        name: shared_weights + beta * (weights - shared_weights)
        for name, weights in target_weights.items()
    }


def select_shrink_choice(
    train_x: np.ndarray,
    val_x: np.ndarray,
    train_targets: dict[str, np.ndarray],
    val_residual: np.ndarray,
    spaces: dict[str, SegmentSpace],
    args: argparse.Namespace,
) -> ShrinkChoice:
    candidates: list[ShrinkChoice] = []
    for alpha in args.readout_ridge_alphas:
        shared_w = fit_shared_control(train_x, train_targets, alpha)
        target_weights = {name: ridge_fit(train_x, train_targets[name], alpha) for name, _start, _end in SEGMENTS}
        for beta in args.shrink_betas:
            weights = shrink_weights(shared_w, target_weights, beta)
            metrics, _rows = evaluate_weights(
                val_x,
                val_residual,
                spaces,
                shared_w,
                [shared_w for _ in range(args.control_heads)],
                weights,
            )
            candidates.append(
                ShrinkChoice(
                    alpha=alpha,
                    beta=beta,
                    val_mse=metrics["target_set_aware_mse"],
                    shared_weights=shared_w,
                    target_weights=target_weights,
                )
            )
    return min(candidates, key=lambda item: item.val_mse)


def feature_sources(train: SplitArrays, val: SplitArrays, test: SplitArrays) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return {
        "coeff_late": (train.coeff, val.coeff, test.coeff),
        "memory_pool": (train.memory_pool, val.memory_pool, test.memory_pool),
        "memory_plus_coeff": (
            np.concatenate([train.memory_pool, train.coeff], axis=1),
            np.concatenate([val.memory_pool, val.coeff], axis=1),
            np.concatenate([test.memory_pool, test.coeff], axis=1),
        ),
    }


def evaluate_feature_source(
    dataset: str,
    feature_source: str,
    train_raw: np.ndarray,
    val_raw: np.ndarray,
    test_raw: np.ndarray,
    train: SplitArrays,
    val: SplitArrays,
    test: SplitArrays,
    spaces: dict[str, SegmentSpace],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mean, std = feature_stats(train_raw)
    train_x = apply_feature_stats(train_raw, mean, std)
    val_x = apply_feature_stats(val_raw, mean, std)
    test_x = apply_feature_stats(test_raw, mean, std)
    train_targets = rowspace_targets(train.residual, spaces)

    shared_choice = select_choice(train_x, val_x, train_targets, val.residual, spaces, args, "shared")
    pooled_choice = select_choice(train_x, val_x, train_targets, val.residual, spaces, args, "pooled")
    target_choice = select_choice(train_x, val_x, train_targets, val.residual, spaces, args, "target")
    shrink_choice = select_shrink_choice(train_x, val_x, train_targets, val.residual, spaces, args)
    stabilized_weights = shrink_weights(
        shrink_choice.shared_weights,
        shrink_choice.target_weights,
        shrink_choice.beta,
    )

    first_target = next(iter(target_choice.weights.values()))
    test_metrics, segment_rows = evaluate_weights(
        test_x,
        test.residual,
        spaces,
        shared_choice.weights,
        pooled_choice.weights,
        target_choice.weights,
    )
    shrink_metrics, shrink_segment_rows = evaluate_weights(
        test_x,
        test.residual,
        spaces,
        shared_choice.weights,
        pooled_choice.weights,
        stabilized_weights,
    )
    shrink_by_segment = {row["segment"]: row for row in shrink_segment_rows}
    for row in segment_rows:
        shrink_row = shrink_by_segment[row["segment"]]
        row["stabilized_target_set_mse"] = shrink_row["target_set_aware_mse"]
        row["stabilized_vs_pooled_multihead_reduction_pct"] = shrink_row[
            "target_vs_pooled_multihead_reduction_pct"
        ]
        row["stabilized_vs_wrong_target_reduction_pct"] = shrink_row[
            "target_vs_wrong_target_reduction_pct"
        ]
        row["dataset"] = dataset
        row["feature_source"] = feature_source

    target_mse = test_metrics["target_set_aware_mse"]
    stabilized_mse = shrink_metrics["target_set_aware_mse"]
    pooled_mse = test_metrics["pooled_multihead_control_mse"]
    base_mse = test_metrics["base_mse"]
    pathology = target_mse > 2.0 * base_mse or target_mse > 2.0 * pooled_mse
    stabilized_pathology = stabilized_mse > 2.0 * base_mse or stabilized_mse > 2.0 * pooled_mse
    summary = {
        "dataset": dataset,
        "feature_source": feature_source,
        "feature_dim": int(train_raw.shape[1]),
        "rowspace_rank": int(first_target.shape[1]),
        "train_rows": int(train_raw.shape[0]),
        "val_rows": int(val_raw.shape[0]),
        "test_rows": int(test_raw.shape[0]),
        "shared_alpha": float(shared_choice.alpha),
        "pooled_multihead_alpha": float(pooled_choice.alpha),
        "target_set_alpha": float(target_choice.alpha),
        "stabilized_alpha": float(shrink_choice.alpha),
        "stabilized_beta": float(shrink_choice.beta),
        "val_shared_mse": float(shared_choice.val_mse),
        "val_pooled_multihead_mse": float(pooled_choice.val_mse),
        "val_target_set_mse": float(target_choice.val_mse),
        "val_stabilized_target_set_mse": float(shrink_choice.val_mse),
        **test_metrics,
        "stabilized_target_set_mse": stabilized_mse,
        "target_vs_base_reduction_pct": pct_reduction(base_mse, target_mse),
        "target_vs_shared_reduction_pct": pct_reduction(test_metrics["shared_control_mse"], target_mse),
        "target_vs_pooled_multihead_reduction_pct": pct_reduction(pooled_mse, target_mse),
        "target_vs_wrong_target_reduction_pct": pct_reduction(test_metrics["wrong_target_control_mse"], target_mse),
        "stabilized_vs_pooled_multihead_reduction_pct": pct_reduction(pooled_mse, stabilized_mse),
        "pathology_flag": pathology,
        "stabilized_pathology_flag": stabilized_pathology,
    }
    return summary, segment_rows


def evaluate_dataset(
    dataset: str,
    basis: np.ndarray,
    train: SplitArrays,
    val: SplitArrays,
    test: SplitArrays,
    args: argparse.Namespace,
) -> DatasetResult:
    spaces = build_segment_spaces(basis, args.rowspace_rank)
    summary_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    for feature_source, arrays in feature_sources(train, val, test).items():
        summary, segments = evaluate_feature_source(dataset, feature_source, *arrays, train, val, test, spaces, args)
        summary_rows.append(summary)
        segment_rows.extend(segments)
    return DatasetResult(dataset=dataset, summary_rows=summary_rows, segment_rows=segment_rows)


def analyze_dataset(args: argparse.Namespace, dataset: str) -> DatasetResult:
    official_args = build_args(args, dataset)
    model = load_model(official_args, checkpoint_path(args.checkpoint_root, dataset))
    device = torch.device(args.device)
    model.to(device)
    train = collect_split(args, model, official_args, "train", args.max_train_rows)
    val = collect_split(args, model, official_args, "val", args.max_val_rows)
    test = collect_split(args, model, official_args, "test", args.max_test_rows)
    basis = model.learned_temporal_basis.detach().cpu().numpy().astype(np.float64)
    return evaluate_dataset(dataset, basis, train, val, test, args)


def fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt_value(row[field]) for field in fields) + " |")
    return "\n".join(lines)


def mean_for(rows: list[dict[str, Any]], feature_source: str, field: str) -> float:
    values = np.asarray([row[field] for row in rows if row["feature_source"] == feature_source], dtype=np.float64)
    return float(values.mean()) if values.size else float("nan")


def decision_lines(summary_rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Failure Attribution",
        "",
        "[Fact] 本诊断把 B10-TSI-C 的单一 late readout 拆成三个 feature sources：",
        "`coeff_late`、`memory_pool`、`memory_plus_coeff`。每个 source 使用相同的 rank-truncated",
        "basis row-space target、相同 alpha validation、相同 no-target controls。",
        "",
    ]
    for source in ("coeff_late", "memory_pool", "memory_plus_coeff"):
        lines.append(
            f"- `{source}` target vs pooled control mean reduction: "
            f"`{mean_for(summary_rows, source, 'target_vs_pooled_multihead_reduction_pct'):.4f}%`; "
            f"stabilized target vs pooled: "
            f"`{mean_for(summary_rows, source, 'stabilized_vs_pooled_multihead_reduction_pct'):.4f}%`; "
            f"stabilized pathology datasets: `"
            f"{sum(1 for row in summary_rows if row['feature_source'] == source and row['stabilized_pathology_flag'])}`."
        )
    memory_gain = mean_for(summary_rows, "memory_pool", "stabilized_vs_pooled_multihead_reduction_pct")
    coeff_gain = mean_for(summary_rows, "coeff_late", "stabilized_vs_pooled_multihead_reduction_pct")
    memory_pathologies = sum(
        1 for row in summary_rows if row["feature_source"] == "memory_pool" and row["stabilized_pathology_flag"]
    )
    lines.extend(["", "## Decision", ""])
    if memory_pathologies:
        lines.extend(
            [
                "[Decision] `B10-TSI-D` 仍出现 memory-level pathology，不能否定 target-set-aware 方向。",
                "",
                "[Next] 必须先修正 diagnostic 的 feature/readout 稳定性，再讨论 B10 rollback。",
            ]
        )
    elif memory_gain >= 1.0 and memory_gain > coeff_gain:
        lines.extend(
            [
                "[Decision] `B10-TSI-D` 支持继续到 Step 4-6：target-set-aware signal 在 memory-level",
                "intervention 下超过 no-target controls，且优于 late coeff intervention。",
                "",
                "[Boundary] 这仍是 frozen-A6 offline diagnostic，不是训练效果。下一步只能设计 prefix-consistent",
                "target-query memory readout method，并保留 no-target implementation control。",
            ]
        )
    elif memory_gain >= 0.0:
        lines.extend(
            [
                "[Decision] `B10-TSI-D` 显示 memory-level target-set signal 稳定但弱，尚不足以进入 method implementation。",
                "",
                "[Next] StageB 应在 Step 4-6 重新设计 target-query readout，或将 B10 降级为诊断证据。",
            ]
        )
    else:
        lines.extend(
            [
                "[Decision] `B10-TSI-D` 暂不支持进入 B10 method design：稳定 memory-level diagnostic",
                "仍未超过 no-target pooled control。",
                "",
                "[Rollback] 允许回到 StageB Step 2/3，但本结论只针对当前 memory-pooling diagnostic；若之后提出",
                "更强的 native target-query memory architecture，需要重新过 narrative gate。",
            ]
        )
    return lines


def write_report(path: Path, summary_rows: list[dict[str, Any]], segment_rows: list[dict[str, Any]]) -> None:
    summary_fields = [
        "dataset",
        "feature_source",
        "feature_dim",
        "rowspace_rank",
        "train_rows",
        "val_rows",
        "test_rows",
        "shared_alpha",
        "pooled_multihead_alpha",
        "target_set_alpha",
        "stabilized_alpha",
        "stabilized_beta",
        "base_mse",
        "pooled_multihead_control_mse",
        "target_set_aware_mse",
        "stabilized_target_set_mse",
        "wrong_target_control_mse",
        "target_vs_pooled_multihead_reduction_pct",
        "stabilized_vs_pooled_multihead_reduction_pct",
        "target_vs_wrong_target_reduction_pct",
        "pathology_flag",
        "stabilized_pathology_flag",
    ]
    segment_fields = [
        "dataset",
        "feature_source",
        "segment",
        "base_mse",
        "pooled_multihead_control_mse",
        "target_set_aware_mse",
        "stabilized_target_set_mse",
        "wrong_target_control_mse",
        "target_vs_pooled_multihead_reduction_pct",
        "stabilized_vs_pooled_multihead_reduction_pct",
        "target_vs_wrong_target_reduction_pct",
    ]
    lines = [
        "# Phase5 StageB B10-TSI-D Failure Attribution",
        "",
        "## 阶段记录",
        "",
        "| 字段 | 内容 |",
        "| --- | --- |",
        "| `candidate_id` | `B10-TCO` |",
        "| `diagnostic_id` | `B10-TSI-D` |",
        "| `current_step` | StageB Step 3：failure attribution and diagnostic redesign |",
        "| `problem` | 分离 target-set 信息价值、intervention point、readout/head 稳定性 |",
        "| `scope` | Frozen A6 encoder/basis；offline ridge diagnostic；不训练 forecasting model |",
        "",
        "## Diagnostic Design",
        "",
        "本诊断不再只测试 `frozen coeff -> Linear_s(coeff)`。它固定 A6 encoder 与 learned basis，",
        "比较三个 feature sources：",
        "",
        "- `coeff_late`: B10-TSI-C 对应的 late coefficient intervention；",
        "- `memory_pool`: encoder patch memory 的 mean / last / std pooling，代表更早的 memory-level intervention；",
        "- `memory_plus_coeff`: memory-level feature 与 A6 coeff 组合，用于检查 coeff 是否补充必要信息。",
        "",
        "每个 feature source 都拟合 full target-set-specific readout 和 shrinkage target-set readout，",
        "并和 `shared_control`、`pooled_multihead_control`、`wrong_target_control` 比较。输出 target 使用 basis segment 的",
        "rank-truncated output row-space coordinates，避免 B10-TSI-C 的 full coefficient inverse 与",
        "singular-value back-projection 病态。",
        "",
        "## Summary",
        "",
        markdown_table(summary_rows, summary_fields),
        "",
        "## Segment Detail",
        "",
        markdown_table(segment_rows, segment_fields),
        "",
    ]
    lines.extend(decision_lines(summary_rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B10 target-set failure attribution diagnostic.")
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--max-train-rows", type=int, default=12000)
    parser.add_argument("--max-val-rows", type=int, default=6000)
    parser.add_argument("--max-test-rows", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rowspace-rank", type=int, default=64)
    parser.add_argument(
        "--readout-ridge-alphas",
        type=parse_float_list,
        default=parse_float_list("10,100,1000,10000,100000,1000000"),
    )
    parser.add_argument("--shrink-betas", type=parse_float_list, default=parse_float_list("0,0.05,0.1,0.25,0.5,1.0"))
    parser.add_argument("--control-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    summary_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        result = analyze_dataset(args, dataset)
        summary_rows.extend(result.summary_rows)
        segment_rows.extend(result.segment_rows)
    write_csv(args.analysis_root / "b10_tsi_failure_attribution_summary.csv", summary_rows)
    write_csv(args.analysis_root / "b10_tsi_failure_attribution_segments.csv", segment_rows)
    write_report(args.analysis_root / "b10_tsi_failure_attribution_report.md", summary_rows, segment_rows)


if __name__ == "__main__":
    main()
