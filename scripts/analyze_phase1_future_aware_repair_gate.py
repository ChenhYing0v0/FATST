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
BASELINES = ["PatchEncoderFixedHead", "PatchEncoderFixedHeadAdapter"]
CANDIDATES = ["PatchEncoderFutureAwareAlignOnly", "PatchEncoderFutureAwareScaleNorm"]
MODELS = BASELINES + CANDIDATES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase1 future-aware repair gate results.")
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
        "PatchEncoderFutureAwareAlignOnly": load_class(
            REPO_ROOT / "baselines" / "patch_encoder_future_aware_adapter" / "model.py",
            "PatchEncoderFutureAwareAdapter",
        ),
        "PatchEncoderFutureAwareScaleNorm": load_class(
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


def compare(rows: list[dict[str, Any]], candidate: str, baseline: str) -> list[dict[str, Any]]:
    index = {(row["model"], row["dataset"], row["horizon"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            base = index[(baseline, dataset, horizon)]
            cand = index[(candidate, dataset, horizon)]
            output.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "dataset": dataset,
                    "horizon": horizon,
                    "baseline_mse": base["mse"],
                    "candidate_mse": cand["mse"],
                    "delta_mse": cand["mse"] - base["mse"],
                    "relative_mse_change": (cand["mse"] - base["mse"]) / base["mse"],
                    "baseline_mae": base["mae"],
                    "candidate_mae": cand["mae"],
                    "delta_mae": cand["mae"] - base["mae"],
                    "relative_mae_change": (cand["mae"] - base["mae"]) / base["mae"],
                    "baseline_epochs": base["epochs"],
                    "candidate_epochs": cand["epochs"],
                    "baseline_parameter_count": base["parameter_count"],
                    "candidate_parameter_count": cand["parameter_count"],
                    "parameter_ratio": cand["parameter_count"] / base["parameter_count"],
                    "candidate_passes_mse": cand["mse"] < base["mse"],
                }
            )
    return output


def collect_alignment(raw_dir: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in CANDIDATES:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                path = raw_dir / model / dataset / f"h{horizon}" / f"seed{seed}" / "future_alignment_stats.csv"
                row = read_csv(path)[0]
                raw_recon = row.get("raw_reconstruction_loss", row["reconstruction_loss"])
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "horizon": horizon,
                        "alignment_loss": float(row["alignment_loss"]),
                        "reconstruction_loss": float(row["reconstruction_loss"]),
                        "raw_reconstruction_loss": float(raw_recon),
                        "teacher_student_cosine": float(row["teacher_student_cosine"]),
                        "prediction_leakage_max_abs": float(row["prediction_leakage_max_abs"]),
                    }
                )
    return rows


def collect_delta(raw_dir: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in CANDIDATES:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                path = raw_dir / model / dataset / f"h{horizon}" / f"seed{seed}" / "adapter_delta_stats.csv"
                row = read_csv(path)[0]
                rows.append(
                    {
                        "model": model,
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


def summarize_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rel = np.array([float(row["relative_mse_change"]) for row in rows], dtype=np.float64)
    return {
        "candidate": rows[0]["candidate"],
        "baseline": rows[0]["baseline"],
        "wins": int(sum(1 for row in rows if row["candidate_passes_mse"])),
        "mean_relative_mse": float(rel.mean()),
        "min_relative_mse": float(rel.min()),
        "max_relative_mse": float(rel.max()),
    }


def plot_heatmap(report_dir: Path, comparisons: list[dict[str, Any]], candidate: str, baseline: str) -> Path:
    matrix = np.zeros((len(DATASETS), len(HORIZONS)), dtype=np.float64)
    for i, dataset in enumerate(DATASETS):
        for j, horizon in enumerate(HORIZONS):
            row = next(row for row in comparisons if row["dataset"] == dataset and row["horizon"] == horizon)
            matrix[i, j] = float(row["relative_mse_change"]) * 100.0
    vmax = max(1.0, float(np.max(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    image = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title(f"{candidate} relative MSE change vs {baseline}")
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
    path = report_dir / f"{candidate}_vs_{baseline}_relative_mse_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def format_pct(value: float) -> str:
    return f"{value * 100.0:+.2f}%"


def choose_decision(
    fixed_summaries: list[dict[str, Any]],
    adapter_summaries: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
) -> tuple[str, str, str]:
    leakage = max(row["prediction_leakage_max_abs"] for row in alignment_rows)
    if leakage > 1e-7:
        return "fail_leakage", "none", f"prediction leakage audit failed: max_abs={leakage:.6g}."

    ranked = sorted(fixed_summaries, key=lambda row: (row["mean_relative_mse"], -row["wins"]))
    best = ranked[0]
    adapter_index = {row["candidate"]: row for row in adapter_summaries}
    best_adapter = adapter_index[best["candidate"]]
    if best["wins"] >= 6 and best["mean_relative_mse"] < 0.0:
        return "repair_pass", best["candidate"], "a repaired future-aware variant beats fixed head on average."
    if best["wins"] >= 4 or best_adapter["mean_relative_mse"] < 0.0:
        return "repair_partial", best["candidate"], "repair improves some settings but still lacks stable fixed-head gains."
    return "repair_fail", best["candidate"], "neither repair variant provides enough evidence over fixed head."


def write_report(
    report_dir: Path,
    comparisons: dict[tuple[str, str], list[dict[str, Any]]],
    alignment_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    heatmaps: dict[tuple[str, str], Path],
) -> tuple[Path, dict[str, Any]]:
    fixed_summaries = [summarize_comparison(comparisons[(candidate, "PatchEncoderFixedHead")]) for candidate in CANDIDATES]
    adapter_summaries = [
        summarize_comparison(comparisons[(candidate, "PatchEncoderFixedHeadAdapter")]) for candidate in CANDIDATES
    ]
    decision, best_candidate, reason = choose_decision(fixed_summaries, adapter_summaries, alignment_rows)
    leakage = max(row["prediction_leakage_max_abs"] for row in alignment_rows)
    summary = {
        "decision": decision,
        "best_candidate": best_candidate,
        "decision_reason": reason,
        "max_prediction_leakage_abs": float(leakage),
        "vs_fixed": fixed_summaries,
        "vs_adapter": adapter_summaries,
        "mean_teacher_student_cosine": float(np.mean([row["teacher_student_cosine"] for row in alignment_rows])),
    }

    lines = [
        "# Phase1-A.4 Future-Aware Repair Gate 结果报告",
        "",
        "## 实验定位",
        "",
        "[Fact] 本 gate 针对 Phase1-A.3 暴露的 reconstruction loss scale imbalance 做最小修补，",
        "比较 `AlignOnly` 与 `ScaleNorm` 两个候选，而不扩大模型容量。",
        "",
        "## 主结论",
        "",
        f"[Decision] `{decision}`: {reason}",
        "",
        f"[Evidence] best candidate: `{best_candidate}`；leakage max abs `{leakage:.8f}`。",
        "",
        "## Summary",
        "",
        "| Candidate | Baseline | MSE wins | Mean Rel MSE | Range |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in fixed_summaries + adapter_summaries:
        lines.append(
            "| {candidate} | {baseline} | {wins}/12 | {mean} | {minv} to {maxv} |".format(
                candidate=row["candidate"],
                baseline=row["baseline"],
                wins=row["wins"],
                mean=format_pct(float(row["mean_relative_mse"])),
                minv=format_pct(float(row["min_relative_mse"])),
                maxv=format_pct(float(row["max_relative_mse"])),
            )
        )

    lines.extend(["", "## Heatmaps", ""])
    for candidate in CANDIDATES:
        path = heatmaps[(candidate, "PatchEncoderFixedHead")]
        lines.extend([f"![{candidate} vs FixedHead]({path.name})", ""])

    lines.extend(
        [
            "## Alignment Diagnostics",
            "",
            "| Model | Dataset | Horizon | Alignment loss | Recon loss | Raw recon loss | Cosine | Leakage | Delta/Base MAE |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    delta_index = {(row["model"], row["dataset"], row["horizon"]): row for row in delta_rows}
    for row in alignment_rows:
        delta = delta_index[(row["model"], row["dataset"], row["horizon"])]
        lines.append(
            "| {model} | {dataset} | {horizon} | {align:.6f} | {recon:.6f} | {raw:.6f} | "
            "{cos:.6f} | {leak:.8f} | {ratio:.6f} |".format(
                model=row["model"],
                dataset=row["dataset"],
                horizon=row["horizon"],
                align=float(row["alignment_loss"]),
                recon=float(row["reconstruction_loss"]),
                raw=float(row["raw_reconstruction_loss"]),
                cos=float(row["teacher_student_cosine"]),
                leak=float(row["prediction_leakage_max_abs"]),
                ratio=float(delta["delta_to_base_mae_ratio"]),
            )
        )

    report_path = report_dir / "phase1_future_aware_repair_gate_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path, summary


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    counts = parameter_counts()
    metric_rows = collect_metrics(raw_dir, args.seed, counts)
    comparisons = {
        (candidate, baseline): compare(metric_rows, candidate, baseline)
        for candidate in CANDIDATES
        for baseline in BASELINES
    }
    alignment_rows = collect_alignment(raw_dir, args.seed)
    delta_rows = collect_delta(raw_dir, args.seed)
    heatmaps = {
        key: plot_heatmap(report_dir, rows, key[0], key[1])
        for key, rows in comparisons.items()
        if key[1] == "PatchEncoderFixedHead"
    }
    report_path, summary = write_report(report_dir, comparisons, alignment_rows, delta_rows, heatmaps)

    write_csv(report_dir / "phase1_future_aware_repair_metrics.csv", metric_rows)
    for (candidate, baseline), rows in comparisons.items():
        write_csv(report_dir / f"{candidate}_vs_{baseline}.csv", rows)
    write_csv(report_dir / "phase1_future_aware_repair_alignment_stats.csv", alignment_rows)
    write_csv(report_dir / "phase1_future_aware_repair_delta_stats.csv", delta_rows)
    summary["report"] = str(report_path)
    (report_dir / "phase1_future_aware_repair_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
