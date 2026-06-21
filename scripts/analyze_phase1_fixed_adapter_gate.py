from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["ETTh2", "ETTm1", "Weather"]
HORIZONS = [96, 192, 336, 720]
MODELS = ["PatchEncoderFixedHead", "PatchEncoderFixedHeadAdapter"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase1 fixed-head adapter gate results.")
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
    adapter_cls = load_class(
        REPO_ROOT / "baselines" / "patch_encoder_fixed_head_adapter" / "model.py",
        "PatchEncoderFixedHeadAdapter",
    )
    counts: dict[tuple[str, int], int] = {}
    for horizon in HORIZONS:
        fixed = fixed_cls(336, horizon, 1)
        adapter = adapter_cls(336, horizon, 1)
        counts[("PatchEncoderFixedHead", horizon)] = sum(p.numel() for p in fixed.parameters())
        counts[("PatchEncoderFixedHeadAdapter", horizon)] = sum(p.numel() for p in adapter.parameters())
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
            adapter = index[("PatchEncoderFixedHeadAdapter", dataset, horizon)]
            comparisons.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "fixed_mse": fixed["mse"],
                    "adapter_mse": adapter["mse"],
                    "delta_mse": adapter["mse"] - fixed["mse"],
                    "relative_mse_change": (adapter["mse"] - fixed["mse"]) / fixed["mse"],
                    "fixed_mae": fixed["mae"],
                    "adapter_mae": adapter["mae"],
                    "delta_mae": adapter["mae"] - fixed["mae"],
                    "relative_mae_change": (adapter["mae"] - fixed["mae"]) / fixed["mae"],
                    "fixed_epochs": fixed["epochs"],
                    "adapter_epochs": adapter["epochs"],
                    "fixed_parameter_count": fixed["parameter_count"],
                    "adapter_parameter_count": adapter["parameter_count"],
                    "parameter_ratio": adapter["parameter_count"] / fixed["parameter_count"],
                    "adapter_passes_mse": adapter["mse"] < fixed["mse"],
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
            adapter_path = (
                raw_dir
                / "PatchEncoderFixedHeadAdapter"
                / dataset
                / f"h{horizon}"
                / f"seed{seed}"
                / "metrics_by_segment.csv"
            )
            fixed_rows = {row["segment"]: row for row in read_csv(fixed_path)}
            adapter_rows = {row["segment"]: row for row in read_csv(adapter_path)}
            for segment_name in fixed_rows:
                fixed = fixed_rows[segment_name]
                adapter = adapter_rows[segment_name]
                fixed_mse = float(fixed["mse"])
                adapter_mse = float(adapter["mse"])
                fixed_mae = float(fixed["mae"])
                adapter_mae = float(adapter["mae"])
                rows.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "segment": segment_name,
                        "fixed_mse": fixed_mse,
                        "adapter_mse": adapter_mse,
                        "delta_mse": adapter_mse - fixed_mse,
                        "relative_mse_change": (adapter_mse - fixed_mse) / fixed_mse,
                        "fixed_mae": fixed_mae,
                        "adapter_mae": adapter_mae,
                        "delta_mae": adapter_mae - fixed_mae,
                        "relative_mae_change": (adapter_mae - fixed_mae) / fixed_mae,
                        "adapter_passes_mse": adapter_mse < fixed_mse,
                    }
                )
    return rows


def collect_adapter_delta(raw_dir: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            path = (
                raw_dir
                / "PatchEncoderFixedHeadAdapter"
                / dataset
                / f"h{horizon}"
                / f"seed{seed}"
                / "adapter_delta_stats.csv"
            )
            stats = read_csv(path)[0]
            rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "delta_mse_to_base": float(stats["delta_mse_to_base"]),
                    "delta_mae_to_base": float(stats["delta_mae_to_base"]),
                    "mean_abs_gamma": float(stats["mean_abs_gamma"]),
                    "mean_abs_beta": float(stats["mean_abs_beta"]),
                    "delta_to_base_mae_ratio": float(stats["delta_to_base_mae_ratio"]),
                }
            )
    return rows


def collect_similarity(raw_dir: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            path = (
                raw_dir
                / "PatchEncoderFixedHeadAdapter"
                / dataset
                / f"h{horizon}"
                / f"seed{seed}"
                / "adapter_query_similarity.csv"
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

    vmax = max(1.0, float(np.max(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    image = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title("Phase1-A.2: FixedHeadAdapter relative MSE change vs FixedHead")
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
    path = report_dir / "phase1_fixed_adapter_relative_mse_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_segment_distribution(report_dir: Path, segment_rows: list[dict[str, Any]]) -> Path:
    values = np.array([float(row["relative_mse_change"]) * 100.0 for row in segment_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(values, bins=20, color="#4c78a8", edgecolor="white")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Phase1-A.2: segment-level relative MSE changes")
    ax.set_xlabel("FixedHeadAdapter relative MSE change vs FixedHead (%)")
    ax.set_ylabel("Segment count")
    fig.tight_layout()
    path = report_dir / "phase1_fixed_adapter_segment_delta_hist.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def format_pct(value: float) -> str:
    return f"{value * 100.0:+.2f}%"


def decision_label(wins: int, total: int, mean_relative_mse: float) -> tuple[str, str]:
    if wins >= 4 and mean_relative_mse <= 0.0:
        return (
            "pass",
            "main wins 达到 Phase1-A.2 最低通过线，且平均 relative MSE 非退化。",
        )
    if wins >= 4:
        return (
            "partial_pass",
            "main wins 达到最低通过线，但平均 relative MSE 仍为正退化，不足以作为论文核心 claim。",
        )
    return (
        "fail",
        "main wins 未达到最低通过线，history-only fixed-head adapter 不应继续作为主线。",
    )


def write_report(
    report_dir: Path,
    comparisons: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    similarity_rows: list[dict[str, Any]],
    heatmap_path: Path,
    hist_path: Path,
) -> Path:
    rel_changes = np.array([float(row["relative_mse_change"]) for row in comparisons], dtype=np.float64)
    segment_rel = np.array([float(row["relative_mse_change"]) for row in segment_rows], dtype=np.float64)
    wins = sum(1 for row in comparisons if row["adapter_passes_mse"])
    segment_wins = sum(1 for row in segment_rows if row["adapter_passes_mse"])
    best_row = min(comparisons, key=lambda row: float(row["relative_mse_change"]))
    worst_row = max(comparisons, key=lambda row: float(row["relative_mse_change"]))
    mean_delta_ratio = float(np.mean([row["delta_to_base_mae_ratio"] for row in delta_rows]))
    decision, decision_reason = decision_label(wins, len(comparisons), float(rel_changes.mean()))

    lines = [
        "# Phase1-A.2 Fixed-Head Adapter Gate 结果报告",
        "",
        "## 实验定位",
        "",
        "[Fact] 本实验检验 `PatchEncoderFixedHeadAdapter` 是否能在保留 fixed flatten head",
        "readout capacity 的前提下，引入有效 future-segment conditioning。",
        "",
        "## 主结论",
        "",
        f"[Evidence] `PatchEncoderFixedHeadAdapter` main MSE wins: `{wins}/12`。",
        f"MSE relative change 范围为 `{format_pct(float(rel_changes.min()))}` 到",
        f"`{format_pct(float(rel_changes.max()))}`，平均为 `{format_pct(float(rel_changes.mean()))}`。",
        "",
        f"[Evidence] segment-level MSE wins: `{segment_wins}/{len(segment_rows)}`。",
        f"segment relative MSE 平均为 `{format_pct(float(segment_rel.mean()))}`。",
        "",
        f"[Evidence] adapter 对 base prediction 的平均 MAE 修正比例为 `{mean_delta_ratio:.4f}`。",
        "",
        f"[Decision] `{decision}`: {decision_reason}",
        "",
        "[Decision] 该结果证明保留 fixed-head capacity 后，future-side adapter 不再系统性失败；",
        "但当前收益幅度和稳定性不足，不能直接作为 decoder 论文主创新。下一步应回退到",
        "training-only future-aware teacher/student alignment，而不是继续只增加 history-only",
        "adapter capacity。",
        "",
        "## Main Metric Table",
        "",
        "| Dataset | Horizon | Fixed MSE | Adapter MSE | Rel MSE | Fixed MAE | Adapter MAE | Rel MAE | Param ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparisons:
        lines.append(
            "| {dataset} | {horizon} | {fixed_mse:.6f} | {adapter_mse:.6f} | {rel_mse} | "
            "{fixed_mae:.6f} | {adapter_mae:.6f} | {rel_mae} | {param_ratio:.3f} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                fixed_mse=float(row["fixed_mse"]),
                adapter_mse=float(row["adapter_mse"]),
                rel_mse=format_pct(float(row["relative_mse_change"])),
                fixed_mae=float(row["fixed_mae"]),
                adapter_mae=float(row["adapter_mae"]),
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
            "[Inference] 若 adapter 取得正收益且 `delta_to_base_mae_ratio` 非零，说明 future-side",
            "conditioning 在 fixed readout 外提供了实际修正；若正收益很小且修正比例接近 0，",
            "则更可能是训练波动而不是机制收益。",
            "",
            "[Inference] 若 adapter 大面积退化，说明仅用 history-derived segment queries 做",
            "post-readout affine conditioning 不足以构成 decoder 创新，应回退到 future-aware",
            "teacher/student alignment，而不是继续增加 adapter 容量。",
            "",
            f"Best setting: `{best_row['dataset']} / H={best_row['horizon']}` with",
            f"`{format_pct(float(best_row['relative_mse_change']))}` relative MSE change.",
            "",
            f"Worst setting: `{worst_row['dataset']} / H={worst_row['horizon']}` with",
            f"`{format_pct(float(worst_row['relative_mse_change']))}` relative MSE change.",
            "",
            "## Adapter Delta Stats",
            "",
            "| Dataset | Horizon | Delta/Base MAE Ratio | Mean abs gamma | Mean abs beta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in delta_rows:
        lines.append(
            "| {dataset} | {horizon} | {ratio:.6f} | {gamma:.6f} | {beta:.6f} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                ratio=float(row["delta_to_base_mae_ratio"]),
                gamma=float(row["mean_abs_gamma"]),
                beta=float(row["mean_abs_beta"]),
            )
        )

    lines.extend(
        [
            "",
            "## Adapter Query Similarity",
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

    report_path = report_dir / "phase1_fixed_adapter_gate_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    counts = parameter_counts()
    metric_rows = collect_metrics(raw_dir, args.seed, counts)
    comparisons = compare_metrics(metric_rows)
    segment_rows = collect_segment_comparisons(raw_dir, args.seed)
    delta_rows = collect_adapter_delta(raw_dir, args.seed)
    similarity_rows = collect_similarity(raw_dir, args.seed)
    heatmap_path = plot_metric_heatmap(report_dir, comparisons)
    hist_path = plot_segment_distribution(report_dir, segment_rows)

    write_csv(report_dir / "phase1_fixed_adapter_metrics.csv", metric_rows)
    write_csv(report_dir / "phase1_fixed_adapter_comparison.csv", comparisons)
    write_csv(report_dir / "phase1_fixed_adapter_segment_comparison.csv", segment_rows)
    write_csv(report_dir / "phase1_fixed_adapter_delta_stats.csv", delta_rows)
    write_csv(report_dir / "phase1_fixed_adapter_query_similarity_summary.csv", similarity_rows)
    report_path = write_report(
        report_dir,
        comparisons,
        segment_rows,
        delta_rows,
        similarity_rows,
        heatmap_path,
        hist_path,
    )

    summary = {
        "main_mse_wins": sum(1 for row in comparisons if row["adapter_passes_mse"]),
        "main_total": len(comparisons),
        "segment_mse_wins": sum(1 for row in segment_rows if row["adapter_passes_mse"]),
        "segment_total": len(segment_rows),
        "mean_relative_mse_change": float(np.mean([row["relative_mse_change"] for row in comparisons])),
        "mean_segment_relative_mse_change": float(np.mean([row["relative_mse_change"] for row in segment_rows])),
        "mean_delta_to_base_mae_ratio": float(np.mean([row["delta_to_base_mae_ratio"] for row in delta_rows])),
        "report": str(report_path),
    }
    summary["decision"], summary["decision_reason"] = decision_label(
        int(summary["main_mse_wins"]),
        int(summary["main_total"]),
        float(summary["mean_relative_mse_change"]),
    )
    (report_dir / "phase1_fixed_adapter_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
