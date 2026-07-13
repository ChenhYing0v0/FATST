from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = ["ETTh2", "ETTm2", "Weather"]
HORIZONS = [96, 192, 336, 720]
SEED = 2021
FIXED_LABEL = {
    96: "mixed_h96",
    192: "mixed_h192",
    336: "mixed_h336",
    720: "mixed_h720",
}
UNIFIED_LABEL = "mixed_h96_h192_h336_h720"


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


def fixed_dir(root: Path, dataset: str, horizon: int) -> Path:
    return root / f"TimeAlignCarrierFixedH{horizon}" / dataset / FIXED_LABEL[horizon] / f"seed{SEED}"


def unified_dir(root: Path, dataset: str) -> Path:
    return root / "TimeAlignCarrierUnified720" / dataset / UNIFIED_LABEL / f"seed{SEED}"


def load_metric_row(path: Path, target_horizon: int) -> dict[str, float] | None:
    if not path.exists():
        return None
    for row in read_csv(path):
        if int(row["target_horizon"]) == target_horizon:
            return {"mse": float(row["mse"]), "mae": float(row["mae"])}
    return None


def load_main(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fixed_rows = []
    unified_rows = []
    gap_rows = []
    for dataset in DATASETS:
        unified_path = unified_dir(root, dataset) / "metrics_by_target_horizon.csv"
        for horizon in HORIZONS:
            fixed = load_metric_row(fixed_dir(root, dataset, horizon) / "metrics_by_target_horizon.csv", horizon)
            unified = load_metric_row(unified_path, horizon)
            if fixed is not None:
                fixed_rows.append(
                    {
                        "model": f"TimeAlignCarrierFixedH{horizon}",
                        "dataset": dataset,
                        "target_horizon": horizon,
                        **fixed,
                    }
                )
            if unified is not None:
                unified_rows.append(
                    {
                        "model": "TimeAlignCarrierUnified720",
                        "dataset": dataset,
                        "target_horizon": horizon,
                        **unified,
                    }
                )
            if fixed is None or unified is None:
                continue
            gap_rows.append(
                {
                    "dataset": dataset,
                    "target_horizon": horizon,
                    "fixed_mse": fixed["mse"],
                    "unified_mse": unified["mse"],
                    "relative_mse_pct": pct(unified["mse"], fixed["mse"]),
                    "fixed_mae": fixed["mae"],
                    "unified_mae": unified["mae"],
                    "relative_mae_pct": pct(unified["mae"], fixed["mae"]),
                    "unified_win": unified["mse"] < fixed["mse"],
                }
            )
    return fixed_rows, unified_rows, gap_rows


def load_training_summary(root: Path) -> list[dict[str, Any]]:
    rows = []
    runs: list[tuple[str, str, Path]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            run_name = f"TimeAlignCarrierFixedH{horizon}"
            runs.append((run_name, dataset, fixed_dir(root, dataset, horizon)))
        runs.append(("TimeAlignCarrierUnified720", dataset, unified_dir(root, dataset)))
    for run_name, dataset, run_dir in runs:
        log_path = run_dir / "training_log.csv"
        if not log_path.exists():
            continue
        logs = read_csv(log_path)
        if not logs:
            continue
        best = min(logs, key=lambda row: float(row["val_mean_mse"]))
        rows.append(
            {
                "run_name": run_name,
                "dataset": dataset,
                "epochs_ran": len(logs),
                "best_epoch": int(best["epoch"]),
                "best_val_mean_mse": float(best["val_mean_mse"]),
                "last_val_mean_mse": float(logs[-1]["val_mean_mse"]),
                "train_prediction_l1_drop_pct": pct(
                    float(logs[-1]["train_prediction_l1"]),
                    float(logs[0]["train_prediction_l1"]),
                ),
                "train_alignment_loss_first": float(logs[0]["train_alignment_loss"]),
                "train_alignment_loss_last": float(logs[-1]["train_alignment_loss"]),
            }
        )
    return rows


def summarize_gap(gap_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        subset = [row for row in gap_rows if row["dataset"] == dataset]
        if not subset:
            continue
        rows.append(
            {
                "dataset": dataset,
                "settings": len(subset),
                "unified_wins": sum(1 for row in subset if row["unified_win"]),
                "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in subset),
                "mean_relative_mae_pct": mean(row["relative_mae_pct"] for row in subset),
            }
        )
    if gap_rows:
        rows.append(
            {
                "dataset": "ALL",
                "settings": len(gap_rows),
                "unified_wins": sum(1 for row in gap_rows if row["unified_win"]),
                "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in gap_rows),
                "mean_relative_mae_pct": mean(row["relative_mae_pct"] for row in gap_rows),
            }
        )
    return rows


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


def write_report(output_dir: Path, fixed_rows: list[dict[str, Any]], gap_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    all_summary = next((row for row in summary_rows if row["dataset"] == "ALL"), None)
    if all_summary is None:
        decision = "incomplete"
    elif all_summary["mean_relative_mse_pct"] > 0.5:
        decision = "unified_gap_exists_candidate_for_hss"
    else:
        decision = "no_clear_unified_gap_do_not_build_hss_yet"
    lines = [
        "# Phase5 TimeAlign Carrier Gate Report",
        "",
        "## Decision",
        "",
        f"[Decision] `{decision}`.",
        "",
        "## Fixed-Horizon Metrics",
        "",
        *markdown_table(fixed_rows, ["model", "dataset", "target_horizon", "mse", "mae"]),
        "",
        "## Unified-vs-Fixed Gap",
        "",
        *markdown_table(
            gap_rows,
            [
                "dataset",
                "target_horizon",
                "fixed_mse",
                "unified_mse",
                "relative_mse_pct",
                "relative_mae_pct",
                "unified_win",
            ],
        ),
        "",
        "## Gap Summary",
        "",
        *markdown_table(
            summary_rows,
            ["dataset", "settings", "unified_wins", "mean_relative_mse_pct", "mean_relative_mae_pct"],
        ),
        "",
        "## Gate Reading",
        "",
        "- [Fact] This gate tests whether a TimeAlign-style carrier is viable before adding HSS.",
        "- [Fact] A positive unified gap is useful only if fixed-horizon TimeAlign is a reasonable carrier.",
        "- [Decision] If the fixed carrier is weak, do not build HSS on it; first repair or reject the carrier.",
    ]
    (output_dir / "phase5_timealign_carrier_gate_report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase5 TimeAlign carrier gate artifacts.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed_rows, unified_rows, gap_rows = load_main(args.raw_root)
    summary_rows = summarize_gap(gap_rows)
    training_rows = load_training_summary(args.raw_root)
    write_csv(args.output_dir / "phase5_timealign_fixed_metrics.csv", fixed_rows)
    write_csv(args.output_dir / "phase5_timealign_unified_metrics.csv", unified_rows)
    write_csv(args.output_dir / "phase5_timealign_unified_gap.csv", gap_rows)
    write_csv(args.output_dir / "phase5_timealign_unified_gap_summary.csv", summary_rows)
    write_csv(args.output_dir / "phase5_timealign_training_summary.csv", training_rows)
    write_report(args.output_dir, fixed_rows, gap_rows, summary_rows)


if __name__ == "__main__":
    main()
