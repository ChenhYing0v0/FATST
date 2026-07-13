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
]
BASELINES = ["PatchEncoderFixedHead", "PatchEncoderFixedHeadAdapter"]
CANDIDATE = "PatchEncoderStepSpecificStateAdapter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase1-A.5 step-specific state gate results.")
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
        CANDIDATE: load_class(
            REPO_ROOT / "baselines" / "patch_encoder_step_specific_state_adapter" / "model.py",
            "PatchEncoderStepSpecificStateAdapter",
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


def compare_candidate(rows: list[dict[str, Any]], baseline: str) -> list[dict[str, Any]]:
    index = {(row["model"], row["dataset"], row["horizon"]): row for row in rows}
    comparisons = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
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


def collect_segment_comparisons(raw_dir: Path, seed: int, baseline: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            base_path = raw_dir / baseline / dataset / f"h{horizon}" / f"seed{seed}" / "metrics_by_segment.csv"
            candidate_path = (
                raw_dir / CANDIDATE / dataset / f"h{horizon}" / f"seed{seed}" / "metrics_by_segment.csv"
            )
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


def collect_candidate_diagnostics(raw_dir: Path, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    modulation_rows: list[dict[str, Any]] = []
    activation_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            run_dir = raw_dir / CANDIDATE / dataset / f"h{horizon}" / f"seed{seed}"
            for row in read_csv(run_dir / "state_modulation_stats.csv"):
                modulation_rows.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "scope": row["scope"],
                        "mean_abs_gamma": float(row["mean_abs_gamma"]),
                        "mean_abs_beta": float(row["mean_abs_beta"]),
                        "mean_segment_state_norm": float(row["mean_segment_state_norm"]),
                        "std_segment_state_norm": float(row["std_segment_state_norm"]),
                    }
                )
            values = [float(row["cosine"]) for row in read_csv(run_dir / "segment_state_activation_similarity.csv")]
            activation_rows.append(
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
    return modulation_rows, activation_rows


def plot_heatmap(report_dir: Path, comparisons: list[dict[str, Any]], baseline: str) -> Path:
    matrix = np.zeros((len(DATASETS), len(HORIZONS)), dtype=np.float64)
    for i, dataset in enumerate(DATASETS):
        for j, horizon in enumerate(HORIZONS):
            row = next(row for row in comparisons if row["dataset"] == dataset and row["horizon"] == horizon)
            matrix[i, j] = float(row["relative_mse_change"]) * 100.0

    vmax = max(1.0, float(np.max(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    image = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title(f"Phase1-A.5: StepSpecificState relative MSE vs {baseline}")
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
    path = report_dir / f"phase1_step_specific_state_vs_{baseline}_relative_mse_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def format_pct(value: float) -> str:
    return f"{value * 100.0:+.2f}%"


def summarize_comparison(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    rel = np.array([float(row["relative_mse_change"]) for row in comparisons], dtype=np.float64)
    wins = sum(1 for row in comparisons if row["candidate_passes_mse"])
    dataset_wins = {
        dataset: sum(
            1
            for row in comparisons
            if row["dataset"] == dataset and bool(row["candidate_passes_mse"])
        )
        for dataset in DATASETS
    }
    return {
        "wins": wins,
        "total": len(comparisons),
        "mean_relative_mse": float(np.mean(rel)),
        "min_relative_mse": float(np.min(rel)),
        "max_relative_mse": float(np.max(rel)),
        "dataset_wins": dataset_wins,
        "dataset_with_zero_wins": [dataset for dataset, count in dataset_wins.items() if count == 0],
    }


def decision(summary_vs_fixed: dict[str, Any], modulation_rows: list[dict[str, Any]]) -> tuple[str, str]:
    all_scope = [row for row in modulation_rows if row["scope"] == "all"]
    mean_abs_gamma = float(np.mean([row["mean_abs_gamma"] for row in all_scope]))
    mean_abs_beta = float(np.mean([row["mean_abs_beta"] for row in all_scope]))
    non_degenerate = mean_abs_gamma > 1e-4 or mean_abs_beta > 1e-4
    if (
        summary_vs_fixed["wins"] >= 6
        and summary_vs_fixed["mean_relative_mse"] < 0.0
        and not summary_vs_fixed["dataset_with_zero_wins"]
        and non_degenerate
    ):
        return "pass", "candidate meets A.5 performance, stability, and non-degenerate modulation criteria."
    if summary_vs_fixed["wins"] >= 4 and non_degenerate:
        return "partial", "candidate has some signal but does not meet the full A.5 pass criteria."
    return "fail", "candidate lacks enough stable wins or degenerates in state modulation."


def write_report(
    report_dir: Path,
    summaries: dict[str, dict[str, Any]],
    modulation_rows: list[dict[str, Any]],
    activation_rows: list[dict[str, Any]],
    heatmaps: dict[str, Path],
) -> Path:
    label, reason = decision(summaries["PatchEncoderFixedHead"], modulation_rows)
    all_scope = [row for row in modulation_rows if row["scope"] == "all"]
    mean_abs_gamma = float(np.mean([row["mean_abs_gamma"] for row in all_scope]))
    mean_abs_beta = float(np.mean([row["mean_abs_beta"] for row in all_scope]))
    mean_activation_cosine = float(np.mean([row["cosine_mean"] for row in activation_rows]))

    lines = [
        "# Phase1-A.5 Step-Specific State Decoder Gate 结果报告",
        "",
        "## 实验定位",
        "",
        "[Fact] 本 gate 检验 `PatchEncoderStepSpecificStateAdapter` 是否能在保留 fixed-head",
        "readout rows 的前提下，通过 readout 前的 segment-specific latent modulation 改善预测。",
        "",
        "## 主结论",
        "",
        f"[Decision] `{label}`: {reason}",
        "",
        "## Summary",
        "",
        "| Baseline | MSE wins | Mean Rel MSE | Range | Zero-win datasets |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for baseline, summary in summaries.items():
        zero_win = ", ".join(summary["dataset_with_zero_wins"]) or "none"
        lines.append(
            "| "
            f"{baseline} | {summary['wins']}/{summary['total']} | "
            f"{format_pct(summary['mean_relative_mse'])} | "
            f"{format_pct(summary['min_relative_mse'])} to {format_pct(summary['max_relative_mse'])} | "
            f"{zero_win} |"
        )

    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"- mean_abs_gamma: `{mean_abs_gamma:.6f}`",
            f"- mean_abs_beta: `{mean_abs_beta:.6f}`",
            f"- mean segment activation cosine: `{mean_activation_cosine:.6f}`",
            "",
            "## Heatmaps",
            "",
        ]
    )
    for baseline, path in heatmaps.items():
        lines.append(f"![StepSpecificState vs {baseline}]({path.name})")
        lines.append("")

    report_path = report_dir / "phase1_step_specific_state_gate_report.md"
    report_path.write_text("\n".join(lines))
    summary_path = report_dir / "phase1_step_specific_state_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "decision": label,
                "reason": reason,
                "summaries": summaries,
                "mean_abs_gamma": mean_abs_gamma,
                "mean_abs_beta": mean_abs_beta,
                "mean_activation_cosine": mean_activation_cosine,
            },
            indent=2,
        )
    )
    return report_path


def main() -> None:
    args = parse_args()
    global DATASETS, HORIZONS
    DATASETS = args.datasets
    HORIZONS = args.horizons
    raw_dir = Path(args.raw_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    counts = parameter_counts()
    metrics = collect_metrics(raw_dir, args.seed, counts)
    write_csv(report_dir / "phase1_step_specific_state_metrics.csv", metrics)

    summaries: dict[str, dict[str, Any]] = {}
    heatmaps: dict[str, Path] = {}
    for baseline in BASELINES:
        comparisons = compare_candidate(metrics, baseline)
        segment_rows = collect_segment_comparisons(raw_dir, args.seed, baseline)
        write_csv(report_dir / f"{CANDIDATE}_vs_{baseline}.csv", comparisons)
        write_csv(report_dir / f"{CANDIDATE}_vs_{baseline}_segments.csv", segment_rows)
        summaries[baseline] = summarize_comparison(comparisons)
        heatmaps[baseline] = plot_heatmap(report_dir, comparisons, baseline)

    modulation_rows, activation_rows = collect_candidate_diagnostics(raw_dir, args.seed)
    write_csv(report_dir / "phase1_step_specific_state_modulation_stats.csv", modulation_rows)
    write_csv(report_dir / "phase1_step_specific_state_activation_similarity.csv", activation_rows)
    report_path = write_report(report_dir, summaries, modulation_rows, activation_rows, heatmaps)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
