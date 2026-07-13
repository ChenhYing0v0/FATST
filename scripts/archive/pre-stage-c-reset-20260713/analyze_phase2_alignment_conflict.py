from __future__ import annotations

import argparse
import csv
import json
from math import sqrt
from pathlib import Path
from statistics import mean


DATASETS = ["ETTh2", "ETTm1", "Weather"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_dev = [x - x_mean for x in xs]
    y_dev = [y - y_mean for y in ys]
    x_norm = sqrt(sum(value * value for value in x_dev))
    y_norm = sqrt(sum(value * value for value in y_dev))
    if x_norm == 0.0 or y_norm == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(x_dev, y_dev, strict=True)) / (x_norm * y_norm)


def format_pct(value: float) -> str:
    return f"{value:+.2f}%"


def collect_rows(analysis_root: Path) -> list[dict[str, object]]:
    vs_r3 = read_csv(analysis_root / "phase2_future_state_alignment_vs_r3.csv")
    alignment = read_csv(analysis_root / "phase2_future_state_alignment_alignment_stats.csv")
    alignment_by_key = {
        (row["dataset"], int(row["horizon"])): row
        for row in alignment
    }
    rows: list[dict[str, object]] = []
    for row in vs_r3:
        dataset = row["dataset"]
        horizon = int(row["horizon"])
        stats = alignment_by_key[(dataset, horizon)]
        relative_mse = float(row["relative_mse_vs_r3_pct"])
        teacher_student_cosine = float(stats["teacher_student_cosine"])
        local_alignment_loss = float(stats["future_local_alignment_loss"])
        relation_alignment_loss = float(stats["future_relation_alignment_loss"])
        reconstruction_loss = float(stats["future_reconstruction_loss"])
        rows.append(
            {
                "dataset": dataset,
                "horizon": horizon,
                "relative_mse_vs_r3_pct": relative_mse,
                "future_local_alignment_loss": local_alignment_loss,
                "future_relation_alignment_loss": relation_alignment_loss,
                "future_reconstruction_loss": reconstruction_loss,
                "teacher_student_cosine": teacher_student_cosine,
                "prediction_leakage_max_abs": float(stats["prediction_leakage_max_abs"]),
                "performance_bucket": "improve" if relative_mse < 0 else "degrade",
                "geometry_bucket": "low_cosine" if teacher_student_cosine < 0.7 else "high_cosine",
            }
        )
    return rows


def summarize(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    dataset_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        subset = [row for row in rows if row["dataset"] == dataset]
        dataset_rows.append(
            {
                "dataset": dataset,
                "mean_relative_mse_vs_r3_pct": mean(float(row["relative_mse_vs_r3_pct"]) for row in subset),
                "mse_wins_vs_r3": sum(1 for row in subset if float(row["relative_mse_vs_r3_pct"]) < 0),
                "mean_teacher_student_cosine": mean(float(row["teacher_student_cosine"]) for row in subset),
                "mean_future_local_alignment_loss": mean(
                    float(row["future_local_alignment_loss"]) for row in subset
                ),
                "mean_future_relation_alignment_loss": mean(
                    float(row["future_relation_alignment_loss"]) for row in subset
                ),
                "mean_future_reconstruction_loss": mean(
                    float(row["future_reconstruction_loss"]) for row in subset
                ),
            }
        )
    correlations = {
        "relative_mse_vs_teacher_student_cosine": pearson(
            [float(row["relative_mse_vs_r3_pct"]) for row in rows],
            [float(row["teacher_student_cosine"]) for row in rows],
        ),
        "relative_mse_vs_local_alignment_loss": pearson(
            [float(row["relative_mse_vs_r3_pct"]) for row in rows],
            [float(row["future_local_alignment_loss"]) for row in rows],
        ),
        "relative_mse_vs_relation_alignment_loss": pearson(
            [float(row["relative_mse_vs_r3_pct"]) for row in rows],
            [float(row["future_relation_alignment_loss"]) for row in rows],
        ),
        "relative_mse_vs_reconstruction_loss": pearson(
            [float(row["relative_mse_vs_r3_pct"]) for row in rows],
            [float(row["future_reconstruction_loss"]) for row in rows],
        ),
    }
    summary = {
        "settings": len(rows),
        "dataset_summary": dataset_rows,
        "correlations": correlations,
        "low_cosine_degradation_count": sum(
            1
            for row in rows
            if row["geometry_bucket"] == "low_cosine" and float(row["relative_mse_vs_r3_pct"]) > 0
        ),
        "low_cosine_count": sum(1 for row in rows if row["geometry_bucket"] == "low_cosine"),
        "high_cosine_improvement_count": sum(
            1
            for row in rows
            if row["geometry_bucket"] == "high_cosine" and float(row["relative_mse_vs_r3_pct"]) < 0
        ),
        "high_cosine_count": sum(1 for row in rows if row["geometry_bucket"] == "high_cosine"),
    }
    return dataset_rows, summary


def write_report(
    path: Path,
    rows: list[dict[str, object]],
    dataset_rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    corr = summary["correlations"]
    lines = [
        "# Phase2-A Alignment Conflict Diagnosis",
        "",
        "## Purpose",
        "",
        "[Fact] This report uses the completed Phase2-A artifacts to diagnose why uniform future-state alignment improved `ETTm1` but degraded every `ETTh2` horizon.",
        "",
        "## Main Finding",
        "",
        "[Strong Evidence] The failure pattern is more consistent with target-state geometry conflict than with pure reconstruction-scale imbalance.",
        "",
        "- `ETTh2` is the only dataset with all-horizon degradation vs R.3.",
        "- `ETTh2` also has the lowest teacher/student cosine and highest local alignment loss.",
        "- `Weather` has extremely large reconstruction loss but only slight average degradation/improvement mix, so raw reconstruction scale alone does not explain the failure.",
        "",
        "## Dataset Summary",
        "",
        "| Dataset | Mean MSE vs R.3 | Wins | Mean cosine | Local loss | Relation loss | Recon loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in dataset_rows:
        lines.append(
            "| {dataset} | {mse} | {wins}/4 | {cos:.4f} | {local:.4f} | {rel:.4f} | {recon:.4f} |".format(
                dataset=row["dataset"],
                mse=format_pct(float(row["mean_relative_mse_vs_r3_pct"])),
                wins=row["mse_wins_vs_r3"],
                cos=float(row["mean_teacher_student_cosine"]),
                local=float(row["mean_future_local_alignment_loss"]),
                rel=float(row["mean_future_relation_alignment_loss"]),
                recon=float(row["mean_future_reconstruction_loss"]),
            )
        )
    lines += [
        "",
        "## Correlation Diagnostics",
        "",
        "| Quantity pair | Pearson r | Interpretation |",
        "| --- | ---: | --- |",
        f"| MSE delta vs teacher/student cosine | `{corr['relative_mse_vs_teacher_student_cosine']:.4f}` | negative means lower cosine tends to coincide with worse MSE delta |",
        f"| MSE delta vs local alignment loss | `{corr['relative_mse_vs_local_alignment_loss']:.4f}` | positive means stronger mismatch tends to coincide with worse MSE delta |",
        f"| MSE delta vs relation alignment loss | `{corr['relative_mse_vs_relation_alignment_loss']:.4f}` | relation mismatch is weaker evidence if this value is small |",
        f"| MSE delta vs reconstruction loss | `{corr['relative_mse_vs_reconstruction_loss']:.4f}` | near zero or negative weakens the pure scale-imbalance explanation |",
        "",
        "## Per-Setting Rows",
        "",
        "| Dataset | Horizon | MSE vs R.3 | Cosine | Local loss | Relation loss | Recon loss | Bucket |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {horizon} | {mse} | {cos:.4f} | {local:.4f} | {rel:.4f} | {recon:.4f} | {bucket}/{geometry} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                mse=format_pct(float(row["relative_mse_vs_r3_pct"])),
                cos=float(row["teacher_student_cosine"]),
                local=float(row["future_local_alignment_loss"]),
                rel=float(row["future_relation_alignment_loss"]),
                recon=float(row["future_reconstruction_loss"]),
                bucket=row["performance_bucket"],
                geometry=row["geometry_bucket"],
            )
        )
    lines += [
        "",
        "## Decision Impact",
        "",
        "[Inference] If Phase2-R.1 confidence weighting fixes `ETTh2`, the paper story should focus on reliability-aware future-state calibration. If it fails, the correct rollback is not larger teacher capacity or MoE, but step 2-3: redefine the decoder problem around output-process / error-process modeling.",
        "",
        "[Next] When Phase2-R.1 artifacts become available, compare its confidence statistics against this Phase2-A conflict map. The decisive question is whether down-weighting low-reliability teacher anchors reduces `ETTh2` degradation without erasing `ETTm1/Weather` gains.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Phase2-A future alignment conflict.")
    parser.add_argument(
        "--analysis-root",
        default="analysis/phase2_future_state_alignment_gate_20260622",
    )
    parser.add_argument(
        "--output-root",
        default="analysis/phase2_alignment_conflict_diagnosis_20260623",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root = Path(args.analysis_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(analysis_root)
    dataset_rows, summary = summarize(rows)
    write_csv(output_root / "phase2_alignment_conflict_rows.csv", rows)
    write_csv(output_root / "phase2_alignment_conflict_dataset_summary.csv", dataset_rows)
    (output_root / "phase2_alignment_conflict_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(
        output_root / "phase2_alignment_conflict_diagnosis_report.md",
        rows,
        dataset_rows,
        summary,
    )


if __name__ == "__main__":
    main()
