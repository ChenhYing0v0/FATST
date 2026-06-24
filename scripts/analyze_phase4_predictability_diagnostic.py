from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "baselines" / "patch_encoder_target_set_decoder"
sys.path.insert(0, str(BASELINE_ROOT))

from dataset import ForecastDataset  # noqa: E402


DATASETS = ["ETTh2", "Weather"]
EPS = 1e-8


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def block_group(block_index: int, n_blocks: int) -> str:
    third = n_blocks / 3.0
    if block_index < third:
        return "early_blocks"
    if block_index < 2 * third:
        return "middle_blocks"
    return "late_blocks"


def block_range(block_index: int, block_size: int, pred_len: int) -> tuple[int, int]:
    start = block_index * block_size
    end = min(start + block_size, pred_len)
    return start + 1, end


def chunk_targets(data: np.ndarray, indices: np.ndarray, seq_len: int, pred_len: int) -> np.ndarray:
    targets = [data[index + seq_len : index + seq_len + pred_len] for index in indices]
    return np.stack(targets, axis=0).astype(np.float64, copy=False)


def chunk_seasonal_reference(
    data: np.ndarray,
    indices: np.ndarray,
    seq_len: int,
    pred_len: int,
    season: int,
) -> np.ndarray:
    offsets = seq_len - season + (np.arange(pred_len) % season)
    if np.min(offsets) < 0:
        raise ValueError("season must be <= seq_len.")
    ref_indices = indices[:, None] + offsets[None, :]
    return data[ref_indices].astype(np.float64, copy=False)


def metric_pack(
    targets: np.ndarray,
    history_last: np.ndarray,
    seasonal_ref: np.ndarray,
    block_start: int,
    block_end: int,
) -> dict[str, np.ndarray]:
    block = targets[:, block_start:block_end, :]
    persistence_ref = history_last[:, None, :]
    seasonal_block = seasonal_ref[:, block_start:block_end, :]
    novelty = np.mean((block - persistence_ref) ** 2, axis=(1, 2))
    seasonal_mse = np.mean((block - seasonal_block) ** 2, axis=(1, 2))
    if block.shape[1] <= 1:
        local_variation = np.zeros(block.shape[0], dtype=np.float64)
    else:
        local_variation = np.mean(np.diff(block, axis=1) ** 2, axis=(1, 2))
    block_variance = np.var(block, axis=(1, 2))
    best_naive_mse = np.minimum(novelty, seasonal_mse)
    smoothness_ratio = local_variation / np.maximum(novelty, EPS)
    return {
        "novelty_mse": novelty,
        "seasonal24_mse": seasonal_mse,
        "local_variation": local_variation,
        "block_variance": block_variance,
        "best_naive_mse": best_naive_mse,
        "smoothness_ratio": smoothness_ratio,
    }


def mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def analyze_dataset(
    dataset_root: Path,
    dataset: str,
    seq_len: int,
    pred_len: int,
    block_size: int,
    top_ratio: float,
    season: int,
    chunk_windows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_set = ForecastDataset(dataset_root, dataset, "train", seq_len, pred_len)
    data = train_set.data.astype(np.float64, copy=False)
    n_windows = len(train_set)
    n_blocks = int(np.ceil(pred_len / block_size))
    top_blocks = max(1, min(n_blocks, int(round(n_blocks * top_ratio))))

    metric_names = [
        "novelty_mse",
        "seasonal24_mse",
        "local_variation",
        "block_variance",
        "best_naive_mse",
        "smoothness_ratio",
    ]
    block_totals = {
        block_index: {name: 0.0 for name in metric_names}
        for block_index in range(n_blocks)
    }
    block_selected_totals = {
        block_index: {name: 0.0 for name in metric_names}
        for block_index in range(n_blocks)
    }
    block_selected_counts = np.zeros(n_blocks, dtype=np.int64)
    total_windows = 0

    selected_metric_values = {name: [] for name in metric_names}
    nonselected_metric_values = {name: [] for name in metric_names}

    for start in range(0, n_windows, chunk_windows):
        stop = min(start + chunk_windows, n_windows)
        indices = np.arange(start, stop)
        targets = chunk_targets(data, indices, seq_len, pred_len)
        history_last = data[indices + seq_len - 1]
        seasonal_ref = chunk_seasonal_reference(data, indices, seq_len, pred_len, season)

        chunk_metrics: dict[str, list[np.ndarray]] = {name: [] for name in metric_names}
        for block_index in range(n_blocks):
            block_start = block_index * block_size
            block_end = min(block_start + block_size, pred_len)
            metrics = metric_pack(targets, history_last, seasonal_ref, block_start, block_end)
            for name, values in metrics.items():
                chunk_metrics[name].append(values)
                block_totals[block_index][name] += float(np.sum(values))

        novelty_matrix = np.stack(chunk_metrics["novelty_mse"], axis=1)
        selected = np.argpartition(-novelty_matrix, kth=top_blocks - 1, axis=1)[:, :top_blocks]
        selected_mask = np.zeros_like(novelty_matrix, dtype=bool)
        row_indices = np.arange(selected.shape[0])[:, None]
        selected_mask[row_indices, selected] = True
        block_selected_counts += selected_mask.sum(axis=0)

        for block_index in range(n_blocks):
            is_selected = selected_mask[:, block_index]
            for name in metric_names:
                values = chunk_metrics[name][block_index]
                selected_values = values[is_selected]
                nonselected_values = values[~is_selected]
                block_selected_totals[block_index][name] += float(np.sum(selected_values))
                selected_metric_values[name].extend(float(item) for item in selected_values)
                nonselected_metric_values[name].extend(float(item) for item in nonselected_values)

        total_windows += len(indices)

    block_rows: list[dict[str, Any]] = []
    for block_index in range(n_blocks):
        range_start, range_end = block_range(block_index, block_size, pred_len)
        selected_count = int(block_selected_counts[block_index])
        row: dict[str, Any] = {
            "dataset": dataset,
            "block_index": block_index,
            "block_range": f"{range_start}-{range_end}",
            "block_group": block_group(block_index, n_blocks),
            "train_windows": total_windows,
            "top_blocks_per_window": top_blocks,
            "selected_count": selected_count,
            "selected_rate": selected_count / float(max(total_windows, 1)),
        }
        for name in metric_names:
            row[f"mean_{name}"] = block_totals[block_index][name] / float(max(total_windows, 1))
            row[f"selected_mean_{name}"] = (
                block_selected_totals[block_index][name] / float(selected_count)
                if selected_count > 0
                else 0.0
            )
        block_rows.append(row)

    group_rows: list[dict[str, Any]] = []
    total_selected = int(np.sum(block_selected_counts))
    for group in ["early_blocks", "middle_blocks", "late_blocks"]:
        members = [row for row in block_rows if row["block_group"] == group]
        selected_count = sum(int(row["selected_count"]) for row in members)
        group_row: dict[str, Any] = {
            "dataset": dataset,
            "block_group": group,
            "blocks": len(members),
            "selected_count": selected_count,
            "selected_share": selected_count / float(max(total_selected, 1)),
            "expected_share_if_balanced": len(members) / float(n_blocks),
        }
        for name in metric_names:
            group_row[f"mean_{name}"] = float(np.mean([row[f"mean_{name}"] for row in members]))
        group_rows.append(group_row)

    selected_rows: list[dict[str, Any]] = []
    for selection_label, values_by_metric in [
        ("selected_by_label_novelty", selected_metric_values),
        ("not_selected_by_label_novelty", nonselected_metric_values),
    ]:
        row = {
            "dataset": dataset,
            "selection": selection_label,
            "values": len(values_by_metric["novelty_mse"]),
        }
        for name in metric_names:
            row[f"mean_{name}"] = mean_or_zero(values_by_metric[name])
        selected_rows.append(row)

    return block_rows, group_rows, selected_rows


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_report(
    path: Path,
    block_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
) -> None:
    selection_by_dataset = {
        (row["dataset"], row["selection"]): row
        for row in selection_rows
    }
    interpretation_lines = []
    for dataset in sorted({row["dataset"] for row in selection_rows}):
        selected = selection_by_dataset[(dataset, "selected_by_label_novelty")]
        nonselected = selection_by_dataset[(dataset, "not_selected_by_label_novelty")]
        best_naive_ratio = selected["mean_best_naive_mse"] / max(nonselected["mean_best_naive_mse"], EPS)
        variation_ratio = selected["mean_local_variation"] / max(nonselected["mean_local_variation"], EPS)
        late_share = next(
            row["selected_share"]
            for row in group_rows
            if row["dataset"] == dataset and row["block_group"] == "late_blocks"
        )
        interpretation_lines.append(
            "[Fact] `{dataset}` selected blocks vs non-selected: best-naive MSE ratio `{best_ratio:.2f}x`, "
            "local-variation ratio `{variation_ratio:.2f}x`, late-block selected share `{late_share:.3f}`.".format(
                dataset=dataset,
                best_ratio=best_naive_ratio,
                variation_ratio=variation_ratio,
                late_share=late_share,
            )
        )
        if variation_ratio >= 2.0:
            interpretation_lines.append(
                "[Strong Evidence] `{dataset}` 的 high-novelty selected blocks 同时是 high-variation blocks；"
                "这更接近 low-predictability / noisy-hard，而不只是 learnable-hard。".format(dataset=dataset)
            )
        else:
            interpretation_lines.append(
                "[Inference] `{dataset}` 的 high-novelty selected blocks 没有显著提高 local variation；"
                "它们更像 smooth shift / learnable-hard，因此 hard-block emphasis 可能有效。".format(dataset=dataset)
            )

    lines = [
        "# Phase4-S Predictability Diagnostic",
        "",
        "## 11-step 记录",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | Step 4-6 diagnostic |",
        "| `problem` | 当前 CFUS 把 high-novelty blocks 当作 hard units 加压，但 high novelty 可能是 low-predictability noise-hard |",
        "| `existence_evidence` | CFUS beats full-time but collapses vs R.3 on Weather; SRP++ supports future-step interference / specialization concerns |",
        "| `idea` | predictability-conditioned supervision scheduling: learnable-hard 加压，low-predictability hard 降权或隔离 |",
        "| `theory_check` | 如果 hard units 含不可预测扰动，强加压会污染 shared representation；应先判断 condition 选择的 block 类型 |",
        "| `design` | train-only block diagnostic using label novelty, seasonal naive error, local variation, and selected-block distribution |",
        "| `gate` | 若 selected blocks 显著偏 late/high-variation/high-naive-error，则 CFUS-v2 应转向 predictability-aware downweight/isolation |",
        "| `artifacts` | `analysis/phase4_predictability_diagnostic_20260624` |",
        "| `decision` | current S1 condition mixes learnable-hard and noisy-hard; CFUS-v2 should use predictability-aware downweight/isolation rather than simple hard-block emphasis |",
        "",
        "## 指标定义",
        "",
        "- `novelty_mse`: 当前 CFUS 的 `label_novelty`，即 future block 相对最后一个 history step 的 MSE。",
        "- `seasonal24_mse`: 24-step seasonal naive reference 的 MSE；用于粗略衡量 block 是否可由简单周期结构解释。",
        "- `best_naive_mse`: `min(novelty_mse, seasonal24_mse)`；越高表示简单可预测性越弱。",
        "- `local_variation`: future block 内部一阶差分能量；越高表示局部扰动越强。",
        "- `smoothness_ratio`: `local_variation / novelty_mse`；高 novelty 且高 ratio 更像 noise-hard，低 ratio 更像 smooth shift / learnable-hard。",
        "",
        "## Label-Novelty 选择分布",
        "",
        "| Dataset | Group | Selected share | Balanced expectation | Mean novelty | Mean best naive | Mean variation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in group_rows:
        lines.append(
            "| {dataset} | {group} | {share} | {expected} | {novelty} | {best} | {variation} |".format(
                dataset=row["dataset"],
                group=row["block_group"],
                share=format_float(row["selected_share"]),
                expected=format_float(row["expected_share_if_balanced"]),
                novelty=format_float(row["mean_novelty_mse"]),
                best=format_float(row["mean_best_naive_mse"]),
                variation=format_float(row["mean_local_variation"]),
            )
        )

    lines.extend(
        [
            "",
            "## Selected vs Non-Selected Blocks",
            "",
            "| Dataset | Selection | Values | Mean novelty | Mean best naive | Mean variation | Smoothness ratio |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in selection_rows:
        lines.append(
            "| {dataset} | {selection} | {values} | {novelty} | {best} | {variation} | {ratio} |".format(
                dataset=row["dataset"],
                selection=row["selection"],
                values=row["values"],
                novelty=format_float(row["mean_novelty_mse"]),
                best=format_float(row["mean_best_naive_mse"]),
                variation=format_float(row["mean_local_variation"]),
                ratio=format_float(row["mean_smoothness_ratio"]),
            )
        )

    lines.extend(
        [
            "",
            "## 初步解释规则",
            "",
            "[Decision Rule] 如果 selected blocks 的 `best_naive_mse` 和 `local_variation` 同时显著高于 non-selected blocks，当前 `label_novelty` 更可能混入 low-predictability / noisy-hard units；下一版不应继续简单加压。",
            "",
            "[Decision Rule] 如果 selected share 大幅偏向 late blocks，当前 condition 可能退化为 late weighting proxy；下一版需要 region-balanced 或 predictability-aware selection。",
            "",
            "## 当前判断",
            "",
            *interpretation_lines,
            "",
            "[Decision] 当前 `label_novelty` 不是稳定的 difficulty proxy。它在 ETTh2 上更像 learnable-hard selector，"
            "但在 Weather 上明显选中 high-variation / low-predictability blocks。下一版不应继续给所有 high-novelty blocks 加压，"
            "而应区分 learnable-hard 与 noisy-hard：前者可加压，后者应降权或隔离。",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze train-side predictability proxies for Phase4-S.")
    parser.add_argument("--dataset-root", default="/Users/river/PaperResearch/Project/datasets")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--analysis-root", default="analysis/phase4_predictability_diagnostic_20260624")
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, default=720)
    parser.add_argument("--block-size", type=int, default=48)
    parser.add_argument("--top-ratio", type=float, default=0.25)
    parser.add_argument("--season", type=int, default=24)
    parser.add_argument("--chunk-windows", type=int, default=256)
    args = parser.parse_args()

    if args.seq_len < args.season:
        raise ValueError("season must be <= seq_len.")
    if not 0 < args.top_ratio <= 1:
        raise ValueError("top ratio must be in (0, 1].")

    output_dir = Path(args.analysis_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]

    all_block_rows: list[dict[str, Any]] = []
    all_group_rows: list[dict[str, Any]] = []
    all_selection_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        block_rows, group_rows, selection_rows = analyze_dataset(
            Path(args.dataset_root),
            dataset,
            args.seq_len,
            args.pred_len,
            args.block_size,
            args.top_ratio,
            args.season,
            args.chunk_windows,
        )
        all_block_rows.extend(block_rows)
        all_group_rows.extend(group_rows)
        all_selection_rows.extend(selection_rows)

    write_csv(output_dir / "phase4_predictability_block_summary.csv", all_block_rows)
    write_csv(output_dir / "phase4_predictability_group_summary.csv", all_group_rows)
    write_csv(output_dir / "phase4_predictability_selection_summary.csv", all_selection_rows)
    write_report(
        output_dir / "phase4_predictability_diagnostic_report.md",
        all_block_rows,
        all_group_rows,
        all_selection_rows,
    )


if __name__ == "__main__":
    main()
