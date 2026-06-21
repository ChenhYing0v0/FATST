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
MODELS = ["PatchEncoderFixedHead", "PatchEncoderFixedHeadAdapter", "PatchEncoderFutureAwareAdapter"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase1 future-aware adapter gate results.")
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
    classes = {
        "PatchEncoderFixedHead": load_class(
            REPO_ROOT / "baselines" / "patch_encoder_fixed_head" / "model.py",
            "PatchEncoderFixedHead",
        ),
        "PatchEncoderFixedHeadAdapter": load_class(
            REPO_ROOT / "baselines" / "patch_encoder_fixed_head_adapter" / "model.py",
            "PatchEncoderFixedHeadAdapter",
        ),
        "PatchEncoderFutureAwareAdapter": load_class(
            REPO_ROOT / "baselines" / "patch_encoder_future_aware_adapter" / "model.py",
            "PatchEncoderFutureAwareAdapter",
        ),
    }
    counts: dict[tuple[str, int], int] = {}
    for horizon in HORIZONS:
        for model, cls in classes.items():
            instance = cls(336, horizon, 1)
            counts[(model, horizon)] = sum(p.numel() for p in instance.parameters())
    return counts


def collect_metrics(raw_dir: Path, seed: int, counts: dict[tuple[str, int], int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                run_dir = raw_dir / model / dataset / f"h{horizon}" / f"seed{seed}"
                metrics = read_json(run_dir / "metrics.json")
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "horizon": horizon,
                        "seed": seed,
                        "mse": float(metrics["mse"]),
                        "mae": float(metrics["mae"]),
                        "epochs": len(read_csv(run_dir / "training_log.csv")),
                        "parameter_count": counts[(model, horizon)],
                    }
                )
    return rows


def compare(rows: list[dict[str, Any]], baseline_model: str) -> list[dict[str, Any]]:
    index = {(row["model"], row["dataset"], row["horizon"]): row for row in rows}
    output = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            base = index[(baseline_model, dataset, horizon)]
            future = index[("PatchEncoderFutureAwareAdapter", dataset, horizon)]
            output.append(
                {
                    "baseline": baseline_model,
                    "dataset": dataset,
                    "horizon": horizon,
                    "baseline_mse": base["mse"],
                    "future_mse": future["mse"],
                    "delta_mse": future["mse"] - base["mse"],
                    "relative_mse_change": (future["mse"] - base["mse"]) / base["mse"],
                    "baseline_mae": base["mae"],
                    "future_mae": future["mae"],
                    "delta_mae": future["mae"] - base["mae"],
                    "relative_mae_change": (future["mae"] - base["mae"]) / base["mae"],
                    "baseline_epochs": base["epochs"],
                    "future_epochs": future["epochs"],
                    "baseline_parameter_count": base["parameter_count"],
                    "future_parameter_count": future["parameter_count"],
                    "parameter_ratio": future["parameter_count"] / base["parameter_count"],
                    "future_passes_mse": future["mse"] < base["mse"],
                }
            )
    return output


def collect_alignment(raw_dir: Path, seed: int) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            path = (
                raw_dir
                / "PatchEncoderFutureAwareAdapter"
                / dataset
                / f"h{horizon}"
                / f"seed{seed}"
                / "future_alignment_stats.csv"
            )
            row = read_csv(path)[0]
            rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "alignment_loss": float(row["alignment_loss"]),
                    "reconstruction_loss": float(row["reconstruction_loss"]),
                    "teacher_student_cosine": float(row["teacher_student_cosine"]),
                    "prediction_leakage_max_abs": float(row["prediction_leakage_max_abs"]),
                }
            )
    return rows


def collect_delta(raw_dir: Path, seed: int) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            path = (
                raw_dir
                / "PatchEncoderFutureAwareAdapter"
                / dataset
                / f"h{horizon}"
                / f"seed{seed}"
                / "adapter_delta_stats.csv"
            )
            row = read_csv(path)[0]
            rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "delta_mse_to_base": float(row["delta_mse_to_base"]),
                    "delta_mae_to_base": float(row["delta_mae_to_base"]),
                    "mean_abs_gamma": float(row["mean_abs_gamma"]),
                    "mean_abs_beta": float(row["mean_abs_beta"]),
                    "delta_to_base_mae_ratio": float(row["delta_to_base_mae_ratio"]),
                }
            )
    return rows


def plot_heatmap(report_dir: Path, comparisons: list[dict[str, Any]], baseline: str) -> Path:
    matrix = np.zeros((len(DATASETS), len(HORIZONS)), dtype=np.float64)
    for i, dataset in enumerate(DATASETS):
        for j, horizon in enumerate(HORIZONS):
            row = next(row for row in comparisons if row["dataset"] == dataset and row["horizon"] == horizon)
            matrix[i, j] = float(row["relative_mse_change"]) * 100.0
    vmax = max(1.0, float(np.max(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    image = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title(f"FutureAwareAdapter relative MSE change vs {baseline}")
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
    safe = baseline.replace("PatchEncoder", "").lower()
    path = report_dir / f"phase1_future_aware_vs_{safe}_relative_mse_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def format_pct(value: float) -> str:
    return f"{value * 100.0:+.2f}%"


def decision_label(vs_fixed: list[dict[str, Any]], alignment_rows: list[dict[str, Any]]) -> tuple[str, str]:
    wins = sum(1 for row in vs_fixed if row["future_passes_mse"])
    mean_rel = float(np.mean([row["relative_mse_change"] for row in vs_fixed]))
    leakage = max(row["prediction_leakage_max_abs"] for row in alignment_rows)
    if leakage > 1e-7:
        return "fail_leakage", f"prediction leakage audit failed: max_abs={leakage:.6g}."
    if wins >= 6 and mean_rel < 0.0:
        return "pass", "future-aware adapter improves average MSE without leakage."
    if wins >= 4:
        return "partial_pass", "future-aware adapter has some wins but does not yet provide stable average improvement."
    return "fail", "future-aware adapter does not produce enough performance evidence."


def write_report(
    report_dir: Path,
    vs_fixed: list[dict[str, Any]],
    vs_adapter: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    fixed_heatmap: Path,
    adapter_heatmap: Path,
) -> Path:
    fixed_rel = np.array([float(row["relative_mse_change"]) for row in vs_fixed], dtype=np.float64)
    adapter_rel = np.array([float(row["relative_mse_change"]) for row in vs_adapter], dtype=np.float64)
    fixed_wins = sum(1 for row in vs_fixed if row["future_passes_mse"])
    adapter_wins = sum(1 for row in vs_adapter if row["future_passes_mse"])
    leakage = max(row["prediction_leakage_max_abs"] for row in alignment_rows)
    decision, reason = decision_label(vs_fixed, alignment_rows)

    lines = [
        "# Phase1-A.3 Future-Aware Adapter Gate 结果报告",
        "",
        "## 实验定位",
        "",
        "[Fact] 本实验检验 training-only future teacher branch 是否能让 history-derived",
        "future segment adapter 获得稳定收益。推理路径不接收 ground-truth future。",
        "",
        "## 主结论",
        "",
        f"[Evidence] vs `PatchEncoderFixedHead`: main MSE wins `{fixed_wins}/12`,",
        f"mean relative MSE `{format_pct(float(fixed_rel.mean()))}`，range",
        f"`{format_pct(float(fixed_rel.min()))}` 到 `{format_pct(float(fixed_rel.max()))}`。",
        "",
        f"[Evidence] vs `PatchEncoderFixedHeadAdapter`: main MSE wins `{adapter_wins}/12`,",
        f"mean relative MSE `{format_pct(float(adapter_rel.mean()))}`。",
        "",
        f"[Evidence] leakage audit max abs prediction difference: `{leakage:.8f}`。",
        "",
        f"[Decision] `{decision}`: {reason}",
        "",
        "## 图像",
        "",
        f"![Vs FixedHead]({fixed_heatmap.name})",
        "",
        f"![Vs FixedHeadAdapter]({adapter_heatmap.name})",
        "",
        "## Vs FixedHead",
        "",
        "| Dataset | Horizon | Fixed MSE | FutureAware MSE | Rel MSE | Fixed MAE | FutureAware MAE | Rel MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in vs_fixed:
        lines.append(
            "| {dataset} | {horizon} | {base_mse:.6f} | {future_mse:.6f} | {rel_mse} | "
            "{base_mae:.6f} | {future_mae:.6f} | {rel_mae} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                base_mse=float(row["baseline_mse"]),
                future_mse=float(row["future_mse"]),
                rel_mse=format_pct(float(row["relative_mse_change"])),
                base_mae=float(row["baseline_mae"]),
                future_mae=float(row["future_mae"]),
                rel_mae=format_pct(float(row["relative_mae_change"])),
            )
        )
    lines.extend(
        [
            "",
            "## Alignment Diagnostics",
            "",
            "| Dataset | Horizon | Alignment loss | Recon loss | Teacher/student cosine | Leakage max abs | Delta/Base MAE Ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    delta_index = {(row["dataset"], row["horizon"]): row for row in delta_rows}
    for row in alignment_rows:
        delta = delta_index[(row["dataset"], row["horizon"])]
        lines.append(
            "| {dataset} | {horizon} | {align:.6f} | {recon:.6f} | {cos:.6f} | {leak:.8f} | {ratio:.6f} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                align=float(row["alignment_loss"]),
                recon=float(row["reconstruction_loss"]),
                cos=float(row["teacher_student_cosine"]),
                leak=float(row["prediction_leakage_max_abs"]),
                ratio=float(delta["delta_to_base_mae_ratio"]),
            )
        )
    report_path = report_dir / "phase1_future_aware_adapter_gate_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    counts = parameter_counts()
    metric_rows = collect_metrics(raw_dir, args.seed, counts)
    vs_fixed = compare(metric_rows, "PatchEncoderFixedHead")
    vs_adapter = compare(metric_rows, "PatchEncoderFixedHeadAdapter")
    alignment_rows = collect_alignment(raw_dir, args.seed)
    delta_rows = collect_delta(raw_dir, args.seed)
    fixed_heatmap = plot_heatmap(report_dir, vs_fixed, "PatchEncoderFixedHead")
    adapter_heatmap = plot_heatmap(report_dir, vs_adapter, "PatchEncoderFixedHeadAdapter")
    report_path = write_report(report_dir, vs_fixed, vs_adapter, alignment_rows, delta_rows, fixed_heatmap, adapter_heatmap)

    write_csv(report_dir / "phase1_future_aware_metrics.csv", metric_rows)
    write_csv(report_dir / "phase1_future_aware_vs_fixed.csv", vs_fixed)
    write_csv(report_dir / "phase1_future_aware_vs_adapter.csv", vs_adapter)
    write_csv(report_dir / "phase1_future_aware_alignment_stats.csv", alignment_rows)
    write_csv(report_dir / "phase1_future_aware_delta_stats.csv", delta_rows)
    decision, reason = decision_label(vs_fixed, alignment_rows)
    summary = {
        "decision": decision,
        "decision_reason": reason,
        "vs_fixed_main_mse_wins": sum(1 for row in vs_fixed if row["future_passes_mse"]),
        "vs_adapter_main_mse_wins": sum(1 for row in vs_adapter if row["future_passes_mse"]),
        "mean_relative_mse_vs_fixed": float(np.mean([row["relative_mse_change"] for row in vs_fixed])),
        "mean_relative_mse_vs_adapter": float(np.mean([row["relative_mse_change"] for row in vs_adapter])),
        "max_prediction_leakage_abs": float(max(row["prediction_leakage_max_abs"] for row in alignment_rows)),
        "mean_teacher_student_cosine": float(np.mean([row["teacher_student_cosine"] for row in alignment_rows])),
        "report": str(report_path),
    }
    (report_dir / "phase1_future_aware_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
