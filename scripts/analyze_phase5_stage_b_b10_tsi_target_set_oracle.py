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
DEFAULT_ANALYSIS_ROOT = Path("analysis/phase5_stage_b_b10_tsi_target_set_oracle_20260708")
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
    residual: np.ndarray


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


def forward_coeff_and_residual(model: Model, batch_x: torch.Tensor, batch_y: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    batch, seq_len, channels = batch_x.shape
    x = model.normalization_x(batch_x, "norm")
    target = batch_y[:, -720:, :]
    target_norm = (target - model.normalization_x.mean) / model.normalization_x.stdev

    x = model.patch_emb_x(x.permute(0, 2, 1).reshape(-1, channels * seq_len))
    for layer_idx in range(model.e_layers):
        x = x + model.encoder[layer_idx](x)
        if model.layer_norm:
            x = model.norm_x[layer_idx](x)
    hidden = x.reshape(batch, channels, model.patch_num, model.d_model).flatten(start_dim=-2)
    coeff = model.learned_basis_coeff(hidden)
    basis = model.learned_temporal_basis.to(dtype=hidden.dtype)
    bias = model.learned_temporal_bias.to(dtype=hidden.dtype)
    pred_norm = torch.einsum("hk,bck->bch", basis, coeff) + bias.view(1, 1, -1)
    pred_norm = pred_norm.permute(0, 2, 1)
    residual = target_norm - pred_norm
    coeff_rows = coeff.reshape(-1, coeff.shape[-1]).detach().cpu().numpy().astype(np.float64)
    residual_rows = residual.permute(0, 2, 1).reshape(-1, residual.shape[1])
    residual_rows = residual_rows.detach().cpu().numpy().astype(np.float64)
    return coeff_rows, residual_rows


def collect_split(
    args: argparse.Namespace,
    model: Model,
    official_args: argparse.Namespace,
    split: str,
    max_rows: int,
) -> SplitArrays:
    _data, loader = data_provider(official_args, split)
    coeff_parts: list[np.ndarray] = []
    residual_parts: list[np.ndarray] = []
    total = 0
    device = torch.device(args.device)
    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            coeff, residual = forward_coeff_and_residual(model, batch_x, batch_y)
            remaining = max_rows - total
            if remaining <= 0:
                break
            coeff_parts.append(coeff[:remaining])
            residual_parts.append(residual[:remaining])
            total += min(remaining, coeff.shape[0])
            if total >= max_rows:
                break
    if not coeff_parts:
        raise RuntimeError(f"No rows collected for split={split}")
    return SplitArrays(coeff=np.concatenate(coeff_parts, axis=0), residual=np.concatenate(residual_parts, axis=0))


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


def coefficient_targets(residual: np.ndarray, basis: np.ndarray, alpha: float) -> np.ndarray:
    gram = basis.T @ basis
    reg = np.eye(gram.shape[0], dtype=basis.dtype) * alpha
    return residual @ basis @ np.linalg.inv(gram + reg)


def segment_mse(residual: np.ndarray, correction: np.ndarray) -> float:
    remaining = residual - correction
    return float(np.mean(remaining * remaining))


def evaluate_weights(
    features: np.ndarray,
    residual_rows: np.ndarray,
    basis_by_segment: dict[str, np.ndarray],
    shared_w: np.ndarray,
    pooled_weights: list[np.ndarray],
    target_weights: dict[str, np.ndarray],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    totals = {"base": 0.0, "shared": 0.0, "pooled_multihead": 0.0, "target_set": 0.0}
    counts = 0
    segment_rows: list[dict[str, Any]] = []
    shared_delta = features @ shared_w
    pooled_delta = np.mean([features @ weight for weight in pooled_weights], axis=0)
    for name, start, end in SEGMENTS:
        segment_basis = basis_by_segment[name]
        residual = residual_rows[:, start:end]
        target_delta = features @ target_weights[name]
        base_mse = float(np.mean(residual * residual))
        shared_mse = segment_mse(residual, shared_delta @ segment_basis.T)
        pooled_mse = segment_mse(residual, pooled_delta @ segment_basis.T)
        target_mse = segment_mse(residual, target_delta @ segment_basis.T)
        count = int(residual.size)
        totals["base"] += base_mse * count
        totals["shared"] += shared_mse * count
        totals["pooled_multihead"] += pooled_mse * count
        totals["target_set"] += target_mse * count
        counts += count
        segment_rows.append(
            {
                "segment": name,
                "base_mse": base_mse,
                "shared_control_mse": shared_mse,
                "pooled_multihead_control_mse": pooled_mse,
                "target_set_aware_mse": target_mse,
                "target_vs_base_reduction_pct": 100.0 * (base_mse - target_mse) / base_mse,
                "target_vs_shared_reduction_pct": 100.0 * (shared_mse - target_mse) / shared_mse,
                "target_vs_pooled_multihead_reduction_pct": 100.0 * (pooled_mse - target_mse) / pooled_mse,
            }
        )
    metrics = {
        "base_mse": totals["base"] / counts,
        "shared_control_mse": totals["shared"] / counts,
        "pooled_multihead_control_mse": totals["pooled_multihead"] / counts,
        "target_set_aware_mse": totals["target_set"] / counts,
    }
    return metrics, segment_rows


def fit_shared_control(
    train_x: np.ndarray,
    train_targets: dict[str, np.ndarray],
    alpha: float,
) -> np.ndarray:
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


def evaluate_dataset(
    dataset: str,
    basis: np.ndarray,
    train: SplitArrays,
    val: SplitArrays,
    test: SplitArrays,
    args: argparse.Namespace,
) -> DatasetResult:
    mean, std = feature_stats(train.coeff)
    train_x = apply_feature_stats(train.coeff, mean, std)
    val_x = apply_feature_stats(val.coeff, mean, std)
    test_x = apply_feature_stats(test.coeff, mean, std)
    basis_by_segment = {name: basis[start:end].astype(np.float64) for name, start, end in SEGMENTS}
    train_targets = {
        name: coefficient_targets(train.residual[:, start:end], basis_by_segment[name], args.basis_ridge_alpha)
        for name, start, end in SEGMENTS
    }

    candidates: list[dict[str, Any]] = []
    for alpha in args.readout_ridge_alphas:
        shared_w = fit_shared_control(train_x, train_targets, alpha)
        pooled_weights = fit_pooled_multihead_control(train_x, train_targets, alpha, args.control_heads, args.seed)
        target_weights = {name: ridge_fit(train_x, train_targets[name], alpha) for name, _start, _end in SEGMENTS}
        val_metrics, _segment_rows = evaluate_weights(
            val_x,
            val.residual,
            basis_by_segment,
            shared_w,
            pooled_weights,
            target_weights,
        )
        candidates.append(
            {
                "alpha": alpha,
                "shared_w": shared_w,
                "pooled_weights": pooled_weights,
                "target_weights": target_weights,
                "val_shared_mse": val_metrics["shared_control_mse"],
                "val_pooled_multihead_mse": val_metrics["pooled_multihead_control_mse"],
                "val_target_set_mse": val_metrics["target_set_aware_mse"],
            }
        )

    shared_choice = min(candidates, key=lambda item: item["val_shared_mse"])
    pooled_choice = min(candidates, key=lambda item: item["val_pooled_multihead_mse"])
    target_choice = min(candidates, key=lambda item: item["val_target_set_mse"])

    test_metrics, segment_rows = evaluate_weights(
        test_x,
        test.residual,
        basis_by_segment,
        shared_choice["shared_w"],
        pooled_choice["pooled_weights"],
        target_choice["target_weights"],
    )
    for row in segment_rows:
        row["dataset"] = dataset

    summary = {
        "dataset": dataset,
        "train_rows": int(train.coeff.shape[0]),
        "val_rows": int(val.coeff.shape[0]),
        "test_rows": int(test.coeff.shape[0]),
        "shared_alpha": float(shared_choice["alpha"]),
        "pooled_multihead_alpha": float(pooled_choice["alpha"]),
        "target_set_alpha": float(target_choice["alpha"]),
        "val_shared_mse": float(shared_choice["val_shared_mse"]),
        "val_pooled_multihead_mse": float(pooled_choice["val_pooled_multihead_mse"]),
        "val_target_set_mse": float(target_choice["val_target_set_mse"]),
        "base_mse": test_metrics["base_mse"],
        "shared_control_mse": test_metrics["shared_control_mse"],
        "pooled_multihead_control_mse": test_metrics["pooled_multihead_control_mse"],
        "target_set_aware_mse": test_metrics["target_set_aware_mse"],
    }
    summary["target_vs_base_reduction_pct"] = (
        100.0 * (summary["base_mse"] - summary["target_set_aware_mse"]) / summary["base_mse"]
    )
    summary["target_vs_shared_reduction_pct"] = (
        100.0 * (summary["shared_control_mse"] - summary["target_set_aware_mse"])
        / summary["shared_control_mse"]
    )
    summary["target_vs_pooled_multihead_reduction_pct"] = (
        100.0 * (summary["pooled_multihead_control_mse"] - summary["target_set_aware_mse"])
        / summary["pooled_multihead_control_mse"]
    )
    return DatasetResult(dataset=dataset, summary_rows=[summary], segment_rows=segment_rows)


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


def decision_lines(summary_rows: list[dict[str, Any]]) -> list[str]:
    target_vs_pooled = np.asarray(
        [row["target_vs_pooled_multihead_reduction_pct"] for row in summary_rows],
        dtype=np.float64,
    )
    target_vs_shared = np.asarray([row["target_vs_shared_reduction_pct"] for row in summary_rows], dtype=np.float64)
    lines = [
        f"[Fact] Target-set-aware readout vs shared control 的平均额外 reduction 为 `{target_vs_shared.mean():.4f}%`。",
        f"[Fact] Target-set-aware readout vs pooled 4-head no-target control 的平均额外 reduction 为 `{target_vs_pooled.mean():.4f}%`。",
        "",
    ]
    if float(target_vs_pooled.mean()) >= 1.0 and int(np.sum(target_vs_pooled > 0.0)) >= 2:
        lines.extend(
            [
                "[Decision] `B10-TSI-C` 支持进入 Step 4-6：target-set-aware readout 的 headroom 不能完全由 no-target-set capacity control 解释。",
                "",
                "[Boundary] 这仍是 frozen-A6 offline oracle，不是训练结果。Step 4-6 必须设计 prefix-consistent target-set-conditioned architecture，并再次设置 no-target-set implementation control。",
            ]
        )
    else:
        lines.extend(
            [
                "[Decision] `B10-TSI-C` 未通过：target-set-aware readout 没有稳定超过 pooled 4-head no-target capacity control。",
                "",
                "[Rollback] B10 不应进入 method design；StageB 返回 Step 2/3，或将 B7 objective optimization 保留为小贡献候选。",
            ]
        )
    return lines


def write_report(path: Path, summary_rows: list[dict[str, Any]], segment_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase5 StageB B10-TSI-C Target-Set Oracle Control",
        "",
        "## 阶段记录",
        "",
        "| 字段 | 内容 |",
        "| --- | --- |",
        "| `candidate_id` | `B10-TCO` |",
        "| `diagnostic_id` | `B10-TSI-C` |",
        "| `current_step` | Step 3：target-set oracle/control diagnostic |",
        "| `problem` | 检查 target-set-specific coefficient readout 的 headroom 是否超过 no-target-set capacity control |",
        "| `scope` | Frozen A6 encoder/basis；offline ridge oracle；不训练新 forecasting model |",
        "| `decision` | 见文末；本诊断只决定能否进入 Step 4-6 method design |",
        "",
        "## Readout Definition",
        "",
        "本诊断固定 clean A6 的 encoder、`coeff` 与 `learned_temporal_basis`，只在 normalized basis-coeff",
        "interface 内拟合 coefficient delta oracle：",
        "",
        "```text",
        "A6:        y_s = basis_s @ coeff",
        "TS-aware:  y_s = basis_s @ (coeff + Linear_s(coeff))",
        "Control:   y_s = basis_s @ (coeff + Linear_shared(coeff))",
        "Pooled-4H: y_s = basis_s @ (coeff + mean_j Linear_pooled_j(coeff))",
        "```",
        "",
        "`TS-aware` 为每个 target segment 使用不同 readout；`Pooled-4H` 有 4 个 pooled heads，但不按 target set",
        "选择 head，是主要 no-target-set capacity control。",
        "",
        "## Summary",
        "",
        markdown_table(
            summary_rows,
            [
                "dataset",
                "train_rows",
                "val_rows",
                "test_rows",
                "shared_alpha",
                "pooled_multihead_alpha",
                "target_set_alpha",
                "base_mse",
                "shared_control_mse",
                "pooled_multihead_control_mse",
                "target_set_aware_mse",
                "target_vs_shared_reduction_pct",
                "target_vs_pooled_multihead_reduction_pct",
            ],
        ),
        "",
        "## Segment Detail",
        "",
        markdown_table(
            segment_rows,
            [
                "dataset",
                "segment",
                "base_mse",
                "shared_control_mse",
                "pooled_multihead_control_mse",
                "target_set_aware_mse",
                "target_vs_shared_reduction_pct",
                "target_vs_pooled_multihead_reduction_pct",
            ],
        ),
        "",
        "## Decision",
        "",
    ]
    lines.extend(decision_lines(summary_rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B10 target-set oracle/control diagnostic.")
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--max-train-rows", type=int, default=20000)
    parser.add_argument("--max-val-rows", type=int, default=10000)
    parser.add_argument("--max-test-rows", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--basis-ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--readout-ridge-alphas",
        type=parse_float_list,
        default=parse_float_list("10,100,1000,10000,100000,1000000,10000000"),
    )
    parser.add_argument("--control-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    summary_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        result = analyze_dataset(args, dataset)
        summary_rows.extend(result.summary_rows)
        segment_rows.extend(result.segment_rows)
    write_csv(args.analysis_root / "b10_tsi_target_set_oracle_summary.csv", summary_rows)
    write_csv(args.analysis_root / "b10_tsi_target_set_oracle_segments.csv", segment_rows)
    write_report(args.analysis_root / "b10_tsi_target_set_oracle_report.md", summary_rows, segment_rows)


if __name__ == "__main__":
    main()
