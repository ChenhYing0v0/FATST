from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = ["ETTh2", "ETTm1", "Weather"]
HORIZONS = [96, 192, 336, 720]
SEED = 2021
MIXED_LABEL = "mixed_h96_h192_h336_h720"
FIXED_LABEL = {
    96: "mixed_h96",
    192: "mixed_h192",
    336: "mixed_h336",
    720: "mixed_h720",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def clean_run_dir(clean_root: Path, dataset: str) -> Path:
    return (
        clean_root
        / "official-last"
        / "TimeAlignOfficialUnified720_A6LBF_r256_main_official-last"
        / dataset
        / MIXED_LABEL
        / f"seed{SEED}"
    )


def fixed_run_dir(official_root: Path, dataset: str, horizon: int) -> Path:
    return (
        official_root
        / "official"
        / "official-last"
        / f"TimeAlignOfficialFixedH{horizon}_official-last"
        / dataset
        / FIXED_LABEL[horizon]
        / f"seed{SEED}"
    )


def prefix_step_weight(step_1_indexed: int, horizons: list[int]) -> float:
    return sum(1.0 / horizon for horizon in horizons if step_1_indexed <= horizon) / len(horizons)


def average_prefix_weight(start_0: int, end_0: int, horizons: list[int]) -> float:
    weights = [prefix_step_weight(step, horizons) for step in range(start_0 + 1, end_0 + 1)]
    return mean(weights)


def build_weight_profile(horizons: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tail_weight = average_prefix_weight(336, 720, horizons)
    for start, end in [(0, 96), (96, 192), (192, 336), (336, 720)]:
        avg_weight = average_prefix_weight(start, end, horizons)
        rows.append(
            {
                "segment_start": start,
                "segment_end": end,
                "avg_prefix_weight": avg_weight,
                "relative_to_tail_weight": avg_weight / tail_weight,
            }
        )
    return rows


def segment_key(row: dict[str, str]) -> tuple[int, int, int]:
    return (int(row["target_horizon"]), int(row["segment_start"]), int(row["segment_end"]))


def load_segments(path: Path) -> dict[tuple[int, int, int], dict[str, float]]:
    rows: dict[tuple[int, int, int], dict[str, float]] = {}
    if not path.exists():
        return rows
    for row in read_csv(path):
        rows[segment_key(row)] = {
            "mse": float(row["mse"]),
            "mae": float(row["mae"]),
        }
    return rows


def collect_segment_gaps(clean_root: Path, official_root: Path, horizons: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tail_weight = average_prefix_weight(336, 720, horizons)
    for dataset in DATASETS:
        clean_segments = load_segments(clean_run_dir(clean_root, dataset) / "metrics_by_segment.csv")
        for horizon in horizons:
            fixed_segments = load_segments(fixed_run_dir(official_root, dataset, horizon) / "metrics_by_segment.csv")
            for key, fixed in sorted(fixed_segments.items()):
                target_horizon, start, end = key
                if target_horizon != horizon:
                    continue
                clean = clean_segments.get(key)
                if clean is None:
                    continue
                avg_weight = average_prefix_weight(start, end, horizons)
                rows.append(
                    {
                        "dataset": dataset,
                        "target_horizon": target_horizon,
                        "segment_start": start,
                        "segment_end": end,
                        "avg_prefix_weight": avg_weight,
                        "relative_to_tail_weight": avg_weight / tail_weight,
                        "clean_mse": clean["mse"],
                        "fixed_mse": fixed["mse"],
                        "relative_mse_pct": pct(clean["mse"], fixed["mse"]),
                        "clean_mae": clean["mae"],
                        "fixed_mae": fixed["mae"],
                        "relative_mae_pct": pct(clean["mae"], fixed["mae"]),
                        "clean_wins_mse": clean["mse"] < fixed["mse"],
                    }
                )
    return rows


def bucket_name(start: int, end: int) -> str:
    if start < 96:
        return "early_0_96"
    if start < 192:
        return "mid_96_192"
    if start < 336:
        return "late_192_336"
    return "tail_336_720"


def summarize_gaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for dataset in DATASETS + ["ALL"]:
        dataset_rows = rows if dataset == "ALL" else [row for row in rows if row["dataset"] == dataset]
        if not dataset_rows:
            continue
        for bucket in ["early_0_96", "mid_96_192", "late_192_336", "tail_336_720", "ALL"]:
            if bucket == "ALL":
                subset = dataset_rows
            else:
                subset = [
                    row
                    for row in dataset_rows
                    if bucket_name(int(row["segment_start"]), int(row["segment_end"])) == bucket
                ]
            if not subset:
                continue
            summary.append(
                {
                    "dataset": dataset,
                    "bucket": bucket,
                    "segments": len(subset),
                    "mse_wins": sum(1 for row in subset if row["clean_wins_mse"]),
                    "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in subset),
                    "mean_relative_mae_pct": mean(row["relative_mae_pct"] for row in subset),
                    "mean_relative_to_tail_weight": mean(row["relative_to_tail_weight"] for row in subset),
                }
            )
    return summary


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def find_summary(summary: list[dict[str, Any]], dataset: str, bucket: str) -> dict[str, Any] | None:
    return next((row for row in summary if row["dataset"] == dataset and row["bucket"] == bucket), None)


def write_report(output_dir: Path, weight_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    early_all = find_summary(summary_rows, "ALL", "early_0_96")
    tail_all = find_summary(summary_rows, "ALL", "tail_336_720")
    all_row = find_summary(summary_rows, "ALL", "ALL")
    if early_all and tail_all:
        early_vs_tail_gap = tail_all["mean_relative_mse_pct"] - early_all["mean_relative_mse_pct"]
    else:
        early_vs_tail_gap = float("nan")
    if all_row is None:
        decision = "incomplete"
    elif early_all and tail_all and early_vs_tail_gap > 1.0:
        decision = "prefix_imbalance_problem_candidate"
    else:
        decision = "prefix_imbalance_not_supported_yet"

    lines = [
        "# Phase5 StageB Unified Prefix Optimization Diagnostic",
        "",
        "## Decision",
        "",
        f"[Decision] `{decision}`.",
        "[Current Step] StageB Step 2/3 problem-existence diagnostic after clean A6 validation.",
        "[Candidate] `B7-UPO`: unified prefix optimization / nested-prefix supervision imbalance.",
        "",
        "## Why This Is Not B6-PLO",
        "",
        "B6 asked whether label/residual structure requires a learned-basis or frequency-space objective. This diagnostic asks a different question: whether the current unified multi-prefix loss over-weights short prefix steps and under-weights long-tail steps because nested horizons are averaged as tasks.",
        "",
        "## Prefix Supervision Weight",
        "",
        "For horizons `[96, 192, 336, 720]`, the current loss is the mean of per-prefix mean losses. A step `t` receives weight `mean(1/H for H >= t)`, so earlier steps are repeated in more prefix tasks.",
        "",
        *markdown_table(weight_rows, ["segment_start", "segment_end", "avg_prefix_weight", "relative_to_tail_weight"]),
        "",
        "## Segment Gap Summary vs Fixed-Horizon TimeAlign",
        "",
        *markdown_table(
            summary_rows,
            [
                "dataset",
                "bucket",
                "segments",
                "mse_wins",
                "mean_relative_mse_pct",
                "mean_relative_mae_pct",
                "mean_relative_to_tail_weight",
            ],
        ),
        "",
        "## Reading",
        "",
        f"- [Fact] Overall segment-level mean relative MSE vs fixed is `{fmt_pct(all_row['mean_relative_mse_pct']) if all_row else 'NA'}`.",
        f"- [Fact] Early bucket mean relative MSE is `{fmt_pct(early_all['mean_relative_mse_pct']) if early_all else 'NA'}`; tail bucket is `{fmt_pct(tail_all['mean_relative_mse_pct']) if tail_all else 'NA'}`.",
        f"- [Fact] Tail-minus-early relative MSE gap is `{fmt_pct(early_vs_tail_gap)}`; positive means A6 gains are weaker in the under-weighted tail.",
        "- [Inference] If the tail gap is stable by dataset, B7 is a stronger StageB route than B6 because it targets unified multi-horizon training mechanics, not generic frequency auxiliary losses.",
        "- [Rollback] If follow-up gradient/task diagnostics do not support horizon-task interference, do not implement a new loss; keep StageB paused.",
    ]
    (output_dir / "stage_b_b7_unified_prefix_optimization_report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze StageB unified prefix optimization problem evidence.")
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=Path("analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/raw"),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path("analysis/phase5_timealign_hss_a3e_ettm1_replacement_gate_20260701/raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/phase5_stage_b_unified_prefix_optimization_20260707"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weight_rows = build_weight_profile(HORIZONS)
    segment_rows = collect_segment_gaps(args.clean_root, args.official_root, HORIZONS)
    summary_rows = summarize_gaps(segment_rows)

    write_csv(args.output_dir / "stage_b_b7_prefix_weight_profile.csv", weight_rows)
    write_csv(args.output_dir / "stage_b_b7_segment_gaps_vs_fixed.csv", segment_rows)
    write_csv(args.output_dir / "stage_b_b7_segment_gap_summary.csv", summary_rows)
    write_report(args.output_dir, weight_rows, summary_rows)


if __name__ == "__main__":
    main()
