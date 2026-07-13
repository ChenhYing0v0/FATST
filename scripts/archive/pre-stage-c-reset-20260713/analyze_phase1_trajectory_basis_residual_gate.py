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
MODELS = [
    "PatchEncoderFixedHead",
    "PatchEncoderFixedHeadAdapter",
    "PatchEncoderStepSpecificStateAdapter",
    "PatchEncoderTrajectoryBasisResidual",
]
BASELINES = [
    "PatchEncoderFixedHead",
    "PatchEncoderFixedHeadAdapter",
    "PatchEncoderStepSpecificStateAdapter",
]
CANDIDATE = "PatchEncoderTrajectoryBasisResidual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase1-A.6 trajectory basis residual gate.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS, choices=HORIZONS)
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


def parameter_counts(horizons: list[int]) -> dict[tuple[str, int], int]:
    class_specs = {
        "PatchEncoderFixedHead": (
            REPO_ROOT / "baselines" / "patch_encoder_fixed_head" / "model.py",
            "PatchEncoderFixedHead",
        ),
        "PatchEncoderFixedHeadAdapter": (
            REPO_ROOT / "baselines" / "patch_encoder_fixed_head_adapter" / "model.py",
            "PatchEncoderFixedHeadAdapter",
        ),
        "PatchEncoderStepSpecificStateAdapter": (
            REPO_ROOT / "baselines" / "patch_encoder_step_specific_state_adapter" / "model.py",
            "PatchEncoderStepSpecificStateAdapter",
        ),
        CANDIDATE: (
            REPO_ROOT / "baselines" / "patch_encoder_trajectory_basis_residual" / "model.py",
            "PatchEncoderTrajectoryBasisResidual",
        ),
    }
    classes = {name: load_class(path, class_name) for name, (path, class_name) in class_specs.items()}
    counts: dict[tuple[str, int], int] = {}
    for horizon in horizons:
        for model, cls in classes.items():
            instance = cls(336, horizon, 1)
            counts[(model, horizon)] = sum(p.numel() for p in instance.parameters())
    return counts


def collect_metrics(
    raw_dir: Path,
    seed: int,
    datasets: list[str],
    horizons: list[int],
    counts: dict[tuple[str, int], int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for dataset in datasets:
            for horizon in horizons:
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


def compare_candidate(rows: list[dict[str, Any]], baseline: str, datasets: list[str], horizons: list[int]) -> list[dict[str, Any]]:
    index = {(row["model"], row["dataset"], row["horizon"]): row for row in rows}
    comparisons = []
    for dataset in datasets:
        for horizon in horizons:
            base = index[(baseline, dataset, horizon)]
            candidate = index[(CANDIDATE, dataset, horizon)]
            comparisons.append(
                {
                    "baseline": baseline,
                    "dataset": dataset,
                    "horizon": horizon,
                    "baseline_mse": base["mse"],
                    "candidate_mse": candidate["mse"],
                    "delta_mse": candidate["mse"] - base["mse"],
                    "relative_mse_change": (candidate["mse"] - base["mse"]) / base["mse"],
                    "baseline_mae": base["mae"],
                    "candidate_mae": candidate["mae"],
                    "delta_mae": candidate["mae"] - base["mae"],
                    "relative_mae_change": (candidate["mae"] - base["mae"]) / base["mae"],
                    "baseline_epochs": base["epochs"],
                    "candidate_epochs": candidate["epochs"],
                    "baseline_parameter_count": base["parameter_count"],
                    "candidate_parameter_count": candidate["parameter_count"],
                    "parameter_ratio": candidate["parameter_count"] / base["parameter_count"],
                    "candidate_passes_mse": candidate["mse"] < base["mse"],
                }
            )
    return comparisons


def collect_segment_comparisons(
    raw_dir: Path,
    seed: int,
    baseline: str,
    datasets: list[str],
    horizons: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for horizon in horizons:
            base_path = raw_dir / baseline / dataset / f"h{horizon}" / f"seed{seed}" / "metrics_by_segment.csv"
            candidate_path = raw_dir / CANDIDATE / dataset / f"h{horizon}" / f"seed{seed}" / "metrics_by_segment.csv"
            base_rows = {row["segment"]: row for row in read_csv(base_path)}
            candidate_rows = {row["segment"]: row for row in read_csv(candidate_path)}
            for segment in base_rows:
                base = base_rows[segment]
                candidate = candidate_rows[segment]
                base_mse = float(base["mse"])
                candidate_mse = float(candidate["mse"])
                rows.append(
                    {
                        "baseline": baseline,
                        "dataset": dataset,
                        "horizon": horizon,
                        "segment": segment,
                        "baseline_mse": base_mse,
                        "candidate_mse": candidate_mse,
                        "delta_mse": candidate_mse - base_mse,
                        "relative_mse_change": (candidate_mse - base_mse) / base_mse,
                        "candidate_passes_mse": candidate_mse < base_mse,
                    }
                )
    return rows


def collect_residual_stats(raw_dir: Path, seed: int, datasets: list[str], horizons: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for horizon in horizons:
            path = raw_dir / CANDIDATE / dataset / f"h{horizon}" / f"seed{seed}" / "trajectory_residual_stats.csv"
            for row in read_csv(path):
                rows.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "scope": row["scope"],
                        "residual_mse": float(row["residual_mse"]),
                        "residual_mae": float(row["residual_mae"]),
                        "raw_residual_mae": float(row["raw_residual_mae"]),
                        "residual_to_base_mae_ratio": float(row["residual_to_base_mae_ratio"]),
                        "residual_to_prediction_mae_ratio": float(row["residual_to_prediction_mae_ratio"]),
                        "mean_gate": float(row["mean_gate"]),
                        "max_gate": float(row["max_gate"]),
                        "coefficient_l2": float(row["coefficient_l2"]),
                        "residual_norm_smoothness": float(row["residual_norm_smoothness"]),
                    }
                )
    return rows


def summarize_comparison(comparisons: list[dict[str, Any]], datasets: list[str]) -> dict[str, Any]:
    rel = np.array([float(row["relative_mse_change"]) for row in comparisons], dtype=np.float64)
    return {
        "wins": sum(1 for row in comparisons if row["candidate_passes_mse"]),
        "total": len(comparisons),
        "mean_relative_mse": float(np.mean(rel)),
        "min_relative_mse": float(np.min(rel)),
        "max_relative_mse": float(np.max(rel)),
        "dataset_wins": {
            dataset: sum(
                1 for row in comparisons if row["dataset"] == dataset and bool(row["candidate_passes_mse"])
            )
            for dataset in datasets
        },
        "dataset_with_zero_wins": [
            dataset
            for dataset in datasets
            if not any(row["dataset"] == dataset and bool(row["candidate_passes_mse"]) for row in comparisons)
        ],
    }


def summarize_by_horizon(comparisons: list[dict[str, Any]], horizons: list[int]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for horizon in horizons:
        values = [float(row["relative_mse_change"]) for row in comparisons if row["horizon"] == horizon]
        wins = sum(1 for row in comparisons if row["horizon"] == horizon and bool(row["candidate_passes_mse"]))
        summary[str(horizon)] = {
            "mean_relative_mse": float(np.mean(values)),
            "wins": wins,
            "total": len(values),
        }
    return summary


def plot_heatmap(
    report_dir: Path,
    comparisons: list[dict[str, Any]],
    baseline: str,
    datasets: list[str],
    horizons: list[int],
) -> Path:
    matrix = np.zeros((len(datasets), len(horizons)), dtype=np.float64)
    for i, dataset in enumerate(datasets):
        for j, horizon in enumerate(horizons):
            row = next(row for row in comparisons if row["dataset"] == dataset and row["horizon"] == horizon)
            matrix[i, j] = float(row["relative_mse_change"]) * 100.0

    vmax = max(1.0, float(np.max(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(8.8, 3.7))
    image = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title(f"Phase1-A.6: TrajectoryBasisResidual relative MSE vs {baseline}")
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Dataset")
    for i in range(len(datasets)):
        for j in range(len(horizons)):
            ax.text(j, i, f"{matrix[i, j]:+.1f}%", ha="center", va="center", color="black")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Relative MSE change (%)")
    fig.tight_layout()
    path = report_dir / f"phase1_trajectory_basis_residual_vs_{baseline}_relative_mse_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def format_pct(value: float) -> str:
    return f"{value * 100.0:+.2f}%"


def decision(
    vs_fixed: dict[str, Any],
    vs_adapter: dict[str, Any],
    by_horizon_fixed: dict[str, dict[str, float]],
    residual_rows: list[dict[str, Any]],
) -> tuple[str, str]:
    all_residual = [row for row in residual_rows if row["scope"] == "all"]
    mean_residual_ratio = float(np.mean([row["residual_to_base_mae_ratio"] for row in all_residual]))
    short_mean = float(
        np.mean(
            [
                by_horizon_fixed[str(horizon)]["mean_relative_mse"]
                for horizon in [96, 192]
                if str(horizon) in by_horizon_fixed
            ]
        )
    )
    pass_fixed = vs_fixed["wins"] >= 7 and vs_fixed["mean_relative_mse"] < 0
    pass_adapter = vs_adapter["wins"] >= 6 and vs_adapter["mean_relative_mse"] < 0
    pass_short = short_mean <= 0.005
    pass_residual = mean_residual_ratio > 1e-5
    if pass_fixed and pass_adapter and pass_short and pass_residual:
        return "pass", "candidate meets the A.6 performance, short-horizon stability, and residual-activity criteria."
    if (vs_fixed["wins"] >= 6 or vs_adapter["wins"] >= 6) and pass_residual:
        return "partial", "candidate has some signal but does not meet the full A.6 pass criteria."
    return "failed", "candidate does not meet the A.6 gate criteria."


def write_report(
    path: Path,
    summaries: dict[str, dict[str, Any]],
    by_horizon: dict[str, dict[str, dict[str, float]]],
    residual_rows: list[dict[str, Any]],
    heatmaps: dict[str, Path],
    decision_value: str,
    reason: str,
) -> None:
    all_residual = [row for row in residual_rows if row["scope"] == "all"]
    residual_ratio = float(np.mean([row["residual_to_base_mae_ratio"] for row in all_residual]))
    mean_gate = float(np.mean([row["mean_gate"] for row in all_residual]))
    coefficient_l2 = float(np.mean([row["coefficient_l2"] for row in all_residual]))
    lines = [
        "# Phase1-A.6 Trajectory Basis Residual Gate 结果报告",
        "",
        "## 实验定位",
        "",
        "[Fact] 本 gate 检验 `PatchEncoderTrajectoryBasisResidual` 是否能在保留 fixed-head",
        "base trajectory 的前提下，通过 future-position low-rank residual 改善 output process。",
        "",
        "## 主结论",
        "",
        f"[Decision] `{decision_value}`: {reason}",
        "",
        "## Summary",
        "",
        "| Baseline | MSE wins | Mean Rel MSE | Range | Zero-win datasets |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for baseline, summary in summaries.items():
        zero = ", ".join(summary["dataset_with_zero_wins"]) or "none"
        lines.append(
            f"| {baseline} | {summary['wins']}/{summary['total']} | "
            f"{format_pct(summary['mean_relative_mse'])} | "
            f"{format_pct(summary['min_relative_mse'])} to {format_pct(summary['max_relative_mse'])} | {zero} |"
        )
    lines.extend(["", "## Horizon Diagnostics", ""])
    for baseline, horizon_summary in by_horizon.items():
        lines.append(f"### vs {baseline}")
        lines.append("")
        lines.append("| Horizon | Wins | Mean Rel MSE |")
        lines.append("| ---: | ---: | ---: |")
        for horizon, row in horizon_summary.items():
            lines.append(f"| {horizon} | {row['wins']}/{row['total']} | {format_pct(row['mean_relative_mse'])} |")
        lines.append("")
    lines.extend(
        [
            "## Residual Diagnostics",
            "",
            f"- mean residual/base MAE ratio: `{residual_ratio:.6f}`",
            f"- mean gate: `{mean_gate:.6f}`",
            f"- mean coefficient L2: `{coefficient_l2:.6f}`",
            "",
            "## Heatmaps",
            "",
        ]
    )
    for baseline, heatmap in heatmaps.items():
        lines.append(f"![TrajectoryBasisResidual vs {baseline}]({heatmap.name})")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    counts = parameter_counts(args.horizons)
    rows = collect_metrics(raw_dir, args.seed, args.datasets, args.horizons, counts)
    write_csv(report_dir / "phase1_trajectory_basis_residual_metrics.csv", rows)

    summaries: dict[str, dict[str, Any]] = {}
    horizon_summaries: dict[str, dict[str, dict[str, float]]] = {}
    heatmaps: dict[str, Path] = {}
    for baseline in BASELINES:
        comparisons = compare_candidate(rows, baseline, args.datasets, args.horizons)
        write_csv(report_dir / f"{CANDIDATE}_vs_{baseline}.csv", comparisons)
        segment_rows = collect_segment_comparisons(raw_dir, args.seed, baseline, args.datasets, args.horizons)
        write_csv(report_dir / f"{CANDIDATE}_vs_{baseline}_segments.csv", segment_rows)
        summaries[baseline] = summarize_comparison(comparisons, args.datasets)
        horizon_summaries[baseline] = summarize_by_horizon(comparisons, args.horizons)
        heatmaps[baseline] = plot_heatmap(report_dir, comparisons, baseline, args.datasets, args.horizons)

    residual_rows = collect_residual_stats(raw_dir, args.seed, args.datasets, args.horizons)
    write_csv(report_dir / "phase1_trajectory_basis_residual_stats.csv", residual_rows)

    decision_value, reason = decision(
        summaries["PatchEncoderFixedHead"],
        summaries["PatchEncoderFixedHeadAdapter"],
        horizon_summaries["PatchEncoderFixedHead"],
        residual_rows,
    )
    summary = {
        "decision": decision_value,
        "reason": reason,
        "summaries": summaries,
        "horizon_summaries": horizon_summaries,
        "mean_residual_to_base_mae_ratio": float(
            np.mean([row["residual_to_base_mae_ratio"] for row in residual_rows if row["scope"] == "all"])
        ),
        "mean_gate": float(np.mean([row["mean_gate"] for row in residual_rows if row["scope"] == "all"])),
        "mean_coefficient_l2": float(np.mean([row["coefficient_l2"] for row in residual_rows if row["scope"] == "all"])),
    }
    (report_dir / "phase1_trajectory_basis_residual_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(
        report_dir / "phase1_trajectory_basis_residual_gate_report.md",
        summaries,
        horizon_summaries,
        residual_rows,
        heatmaps,
        decision_value,
        reason,
    )
    print(f"Wrote {report_dir / 'phase1_trajectory_basis_residual_gate_report.md'}")


if __name__ == "__main__":
    main()
