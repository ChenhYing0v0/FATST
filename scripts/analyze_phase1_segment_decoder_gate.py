from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["ETTh2", "ETTm1", "Weather"]
HORIZONS = [96, 192, 336, 720]
MODELS = ["PatchEncoderFixedHead", "PatchEncoderSegmentQueryHead"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase1-A Future-Segment Decoder Gate results.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_class(module_path: Path, class_name: str) -> type:
    spec = importlib.util.spec_from_file_location(f"{module_path.parent.name}_{module_path.stem}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def parameter_counts() -> dict[tuple[str, int], int]:
    fixed_cls = load_class(
        REPO_ROOT / "baselines" / "patch_encoder_fixed_head" / "model.py",
        "PatchEncoderFixedHead",
    )
    segment_cls = load_class(
        REPO_ROOT / "baselines" / "patch_encoder_segment_query_head" / "model.py",
        "PatchEncoderSegmentQueryHead",
    )
    counts: dict[tuple[str, int], int] = {}
    for horizon in HORIZONS:
        fixed = fixed_cls(336, horizon, 1)
        segment = segment_cls(336, horizon, 1)
        counts[("PatchEncoderFixedHead", horizon)] = sum(p.numel() for p in fixed.parameters())
        counts[("PatchEncoderSegmentQueryHead", horizon)] = sum(p.numel() for p in segment.parameters())
    return counts


def collect_metrics(raw_dir: Path, seed: int, counts: dict[tuple[str, int], int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                run_dir = raw_dir / model / dataset / f"h{horizon}" / f"seed{seed}"
                metrics_path = run_dir / "metrics.json"
                if not metrics_path.is_file():
                    raise FileNotFoundError(metrics_path)
                metrics = read_json(metrics_path)
                epochs = len(read_csv(run_dir / "training_log.csv"))
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "horizon": horizon,
                        "seed": seed,
                        "mse": float(metrics["mse"]),
                        "mae": float(metrics["mae"]),
                        "epochs": epochs,
                        "parameter_count": counts[(model, horizon)],
                    }
                )
    return rows


def compare_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    index = {(row["model"], row["dataset"], row["horizon"]): row for row in rows}
    for dataset in DATASETS:
        for horizon in HORIZONS:
            fixed = index[("PatchEncoderFixedHead", dataset, horizon)]
            segment = index[("PatchEncoderSegmentQueryHead", dataset, horizon)]
            comparisons.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "fixed_mse": fixed["mse"],
                    "segment_mse": segment["mse"],
                    "delta_mse": segment["mse"] - fixed["mse"],
                    "relative_mse_change": (segment["mse"] - fixed["mse"]) / fixed["mse"],
                    "fixed_mae": fixed["mae"],
                    "segment_mae": segment["mae"],
                    "delta_mae": segment["mae"] - fixed["mae"],
                    "relative_mae_change": (segment["mae"] - fixed["mae"]) / fixed["mae"],
                    "fixed_epochs": fixed["epochs"],
                    "segment_epochs": segment["epochs"],
                    "fixed_parameter_count": fixed["parameter_count"],
                    "segment_parameter_count": segment["parameter_count"],
                    "parameter_ratio": segment["parameter_count"] / fixed["parameter_count"],
                    "segment_passes_mse": segment["mse"] < fixed["mse"],
                }
            )
    return comparisons


def collect_segment_comparisons(raw_dir: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            fixed_path = (
                raw_dir
                / "PatchEncoderFixedHead"
                / dataset
                / f"h{horizon}"
                / f"seed{seed}"
                / "metrics_by_segment.csv"
            )
            segment_path = (
                raw_dir
                / "PatchEncoderSegmentQueryHead"
                / dataset
                / f"h{horizon}"
                / f"seed{seed}"
                / "metrics_by_segment.csv"
            )
            fixed_rows = {row["segment"]: row for row in read_csv(fixed_path)}
            segment_rows = {row["segment"]: row for row in read_csv(segment_path)}
            for segment_name in fixed_rows:
                fixed = fixed_rows[segment_name]
                segment = segment_rows[segment_name]
                fixed_mse = float(fixed["mse"])
                segment_mse = float(segment["mse"])
                fixed_mae = float(fixed["mae"])
                segment_mae = float(segment["mae"])
                rows.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "segment": segment_name,
                        "fixed_mse": fixed_mse,
                        "segment_mse": segment_mse,
                        "delta_mse": segment_mse - fixed_mse,
                        "relative_mse_change": (segment_mse - fixed_mse) / fixed_mse,
                        "fixed_mae": fixed_mae,
                        "segment_mae": segment_mae,
                        "delta_mae": segment_mae - fixed_mae,
                        "relative_mae_change": (segment_mae - fixed_mae) / fixed_mae,
                        "segment_passes_mse": segment_mse < fixed_mse,
                    }
                )
    return rows


def collect_similarity(raw_dir: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            path = (
                raw_dir
                / "PatchEncoderSegmentQueryHead"
                / dataset
                / f"h{horizon}"
                / f"seed{seed}"
                / "segment_query_similarity.csv"
            )
            values = [float(row["cosine"]) for row in read_csv(path)]
            rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "n_pairs": len(values),
                    "cosine_mean": float(np.mean(values)),
                    "cosine_std": float(np.std(values)),
                    "cosine_min": float(np.min(values)),
                    "cosine_max": float(np.max(values)),
                }
            )
    return rows


def plot_metric_heatmap(report_dir: Path, comparisons: list[dict[str, Any]]) -> Path:
    matrix = np.zeros((len(DATASETS), len(HORIZONS)), dtype=np.float64)
    for i, dataset in enumerate(DATASETS):
        for j, horizon in enumerate(HORIZONS):
            row = next(row for row in comparisons if row["dataset"] == dataset and row["horizon"] == horizon)
            matrix[i, j] = float(row["relative_mse_change"]) * 100.0

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    image = ax.imshow(matrix, cmap="Reds", aspect="auto")
    ax.set_title("Phase1-A: SegmentQueryHead relative MSE change vs FixedHead")
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([str(h) for h in HORIZONS])
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels(DATASETS)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Dataset")
    for i in range(len(DATASETS)):
        for j in range(len(HORIZONS)):
            ax.text(j, i, f"{matrix[i, j]:+.1f}%", ha="center", va="center", color="black")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Relative MSE change (%)")
    fig.tight_layout()
    path = report_dir / "phase1_segment_decoder_relative_mse_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_segment_distribution(report_dir: Path, segment_rows: list[dict[str, Any]]) -> Path:
    values = np.array([float(row["relative_mse_change"]) * 100.0 for row in segment_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(values, bins=20, color="#4c78a8", edgecolor="white")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Phase1-A: segment-level relative MSE changes")
    ax.set_xlabel("SegmentQueryHead relative MSE change vs FixedHead (%)")
    ax.set_ylabel("Segment count")
    fig.tight_layout()
    path = report_dir / "phase1_segment_decoder_segment_delta_hist.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def format_pct(value: float) -> str:
    return f"{value * 100.0:+.2f}%"


def write_report(
    report_dir: Path,
    comparisons: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    similarity_rows: list[dict[str, Any]],
    heatmap_path: Path,
    hist_path: Path,
) -> Path:
    rel_changes = np.array([float(row["relative_mse_change"]) for row in comparisons], dtype=np.float64)
    segment_rel = np.array([float(row["relative_mse_change"]) for row in segment_rows], dtype=np.float64)
    wins = sum(1 for row in comparisons if row["segment_passes_mse"])
    segment_wins = sum(1 for row in segment_rows if row["segment_passes_mse"])
    best_row = min(comparisons, key=lambda row: float(row["relative_mse_change"]))
    worst_row = max(comparisons, key=lambda row: float(row["relative_mse_change"]))

    lines = [
        "# Phase1-A Future-Segment Decoder Gate 结果报告",
        "",
        "## 实验定位",
        "",
        "[Fact] 本实验对应长研究执行模板的第 8-10 步：远程训练、评估结果、判断是否通过。",
        "",
        "[Fact] 远程 gate 使用 `PatchEncoderFixedHead` 与 `PatchEncoderSegmentQueryHead`，在",
        "`ETTh2`、`ETTm1`、`Weather` 和 horizons `{96,192,336,720}` 上进行 one-to-one",
        "training。代码版本来自远程 driver log：`b90ce0b8fbed4759f361ea30687d9e5ce1cb3188`。",
        "",
        "## 主结论",
        "",
        f"[Strong Evidence] `PatchEncoderSegmentQueryHead` 在 12/12 个 dataset-horizon 设置上",
        f"均未超过 `PatchEncoderFixedHead`。MSE relative change 范围为",
        f"`{format_pct(float(rel_changes.min()))}` 到 `{format_pct(float(rel_changes.max()))}`，",
        f"平均为 `{format_pct(float(rel_changes.mean()))}`。",
        "",
        f"[Strong Evidence] segment-level comparison 也没有形成补救证据：总计 `{len(segment_rows)}`",
        f"个 segment 评价项中，SegmentQueryHead 只赢 `{segment_wins}` 个。",
        "",
        "[Decision] Phase1-A 第一版 `PatchEncoderSegmentQueryHead` 不通过。它不能作为论文核心",
        "decoder 创新点，也不应进入 Phase1-B one-model compatibility。",
        "",
        "## Main Metric Table",
        "",
        "| Dataset | Horizon | Fixed MSE | Segment MSE | Rel MSE | Fixed MAE | Segment MAE | Rel MAE | Param ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparisons:
        lines.append(
            "| {dataset} | {horizon} | {fixed_mse:.6f} | {segment_mse:.6f} | {rel_mse} | "
            "{fixed_mae:.6f} | {segment_mae:.6f} | {rel_mae} | {param_ratio:.3f} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                fixed_mse=float(row["fixed_mse"]),
                segment_mse=float(row["segment_mse"]),
                rel_mse=format_pct(float(row["relative_mse_change"])),
                fixed_mae=float(row["fixed_mae"]),
                segment_mae=float(row["segment_mae"]),
                rel_mae=format_pct(float(row["relative_mae_change"])),
                param_ratio=float(row["parameter_ratio"]),
            )
        )

    lines.extend(
        [
            "",
            "## 图像",
            "",
            f"![Relative MSE heatmap]({heatmap_path.name})",
            "",
            f"![Segment delta histogram]({hist_path.name})",
            "",
            "## 机制诊断",
            "",
            f"[Fact] 最接近 fixed head 的设置是 `{best_row['dataset']} / H={best_row['horizon']}`，",
            f"仍然退化 `{format_pct(float(best_row['relative_mse_change']))}` MSE。",
            "",
            f"[Fact] 最差设置是 `{worst_row['dataset']} / H={worst_row['horizon']}`，退化",
            f"`{format_pct(float(worst_row['relative_mse_change']))}` MSE。",
            "",
            "[Inference] 失败原因更可能是 decoder-side capacity / readout form 不足，而不是",
            "future segment 这个问题完全不存在。当前 head 用一个 segment state 通过 shared linear",
            "head 输出 48 个点；相比 fixed flatten head 为每个 output step 保留大矩阵 row，",
            "它明显更受限。",
            "",
            "## Segment Query Similarity",
            "",
            "| Dataset | Horizon | Pairs | Mean cosine | Min | Max |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in similarity_rows:
        lines.append(
            "| {dataset} | {horizon} | {n_pairs} | {mean:.4f} | {minv:.4f} | {maxv:.4f} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                n_pairs=row["n_pairs"],
                mean=float(row["cosine_mean"]),
                minv=float(row["cosine_min"]),
                maxv=float(row["cosine_max"]),
            )
        )

    lines.extend(
        [
            "",
            "## 是否值得继续",
            "",
            "[Decision] 不应继续把当前 `SegmentQueryHead` 作为主线硬推，也不应在它上面直接叠加",
            "future-aware 或 MoE。否则后续机制的收益会被一个弱 decoder base 吞掉，论文故事也会变成",
            "“先削弱 head，再用复杂机制补回来”。",
            "",
            "[Rollback] 回退到长研究执行模板的第 5-6 步：重新评估 idea 的理论可行性，并重新设计方案。",
            "",
            "下一轮更合理的修补方向不是简单增加 cross-attention，而是先补足 readout capacity：",
            "",
            "1. `SegmentQueryDenseHead`: segment state 只负责 conditioning，每个 segment 仍保留",
            "   parameter-controlled dense readout。",
            "2. `StepQueryHead`: 从 segment-level 改成 step-level query，但需要控制参数和计算量。",
            "3. `FixedHeadAdapter`: 保留 fixed flatten head 的强 readout，在其前后加入轻量",
            "   future-segment adapter，先不要完全替换 fixed head。",
            "",
            "[Recommendation] 下一步优先做 `FixedHeadAdapter` 或 parameter-matched",
            "`SegmentQueryDenseHead`，因为它们更符合当前证据：fixed head 的 readout capacity 很强，",
            "问题是缺少 future-side interface，而不是应该被直接删除。",
            "",
        ]
    )

    path = report_dir / "phase1_segment_decoder_gate_report.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    counts = parameter_counts()
    metrics = collect_metrics(raw_dir, args.seed, counts)
    comparisons = compare_metrics(metrics)
    segment_rows = collect_segment_comparisons(raw_dir, args.seed)
    similarity_rows = collect_similarity(raw_dir, args.seed)

    write_csv(report_dir / "phase1_segment_decoder_metrics.csv", metrics)
    write_csv(report_dir / "phase1_segment_decoder_comparison.csv", comparisons)
    write_csv(report_dir / "phase1_segment_decoder_segment_comparison.csv", segment_rows)
    write_csv(report_dir / "phase1_segment_query_similarity_summary.csv", similarity_rows)

    heatmap_path = plot_metric_heatmap(report_dir, comparisons)
    hist_path = plot_segment_distribution(report_dir, segment_rows)
    report_path = write_report(report_dir, comparisons, segment_rows, similarity_rows, heatmap_path, hist_path)

    rel_changes = [float(row["relative_mse_change"]) for row in comparisons]
    segment_rel = [float(row["relative_mse_change"]) for row in segment_rows]
    summary = {
        "status": "failed",
        "decision": "PatchEncoderSegmentQueryHead does not pass Phase1-A.",
        "n_main_comparisons": len(comparisons),
        "n_main_wins": sum(1 for row in comparisons if row["segment_passes_mse"]),
        "relative_mse_change_min": min(rel_changes),
        "relative_mse_change_max": max(rel_changes),
        "relative_mse_change_mean": float(np.mean(rel_changes)),
        "n_segment_comparisons": len(segment_rows),
        "n_segment_wins": sum(1 for row in segment_rows if row["segment_passes_mse"]),
        "segment_relative_mse_change_min": min(segment_rel),
        "segment_relative_mse_change_max": max(segment_rel),
        "segment_relative_mse_change_mean": float(np.mean(segment_rel)),
        "rollback_step": "Step 5-6: theoretical feasibility and method design.",
        "report": str(report_path),
    }
    (report_dir / "phase1_segment_decoder_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
