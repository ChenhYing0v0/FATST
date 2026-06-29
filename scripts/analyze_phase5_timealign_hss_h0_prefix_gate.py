from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATASETS = ["ETTh2", "ETTm2", "Weather"]
LOSS_MODES = ["full", "multi-prefix", "balanced-step", "stochastic-prefix", "continuous-prefix"]
HORIZONS = [96, 192, 336, 720]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def relative_pct(value: float, baseline: float) -> float:
    if baseline == 0:
        return float("nan")
    return (value / baseline - 1.0) * 100.0


def fmt_pct(value: float) -> str:
    return f"{value:.2f}"


def run_dir(root: Path, checkpoint_policy: str, loss_mode: str, dataset: str, seed: int) -> Path:
    safe_loss_mode = loss_mode.replace("-", "_")
    run_name = f"TimeAlignOfficialUnified720_H0_{safe_loss_mode}_{checkpoint_policy}"
    return root / checkpoint_policy / run_name / dataset / "mixed_h96_h192_h336_h720" / f"seed{seed}"


def collect_metrics(root: Path, checkpoint_policy: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for loss_mode in LOSS_MODES:
            path = run_dir(root, checkpoint_policy, loss_mode, dataset, seed) / "metrics_by_target_horizon.csv"
            for row in read_csv(path):
                rows.append(
                    {
                        "dataset": dataset,
                        "loss_mode": loss_mode,
                        "target_horizon": int(row["target_horizon"]),
                        "mse": float(row["mse"]),
                        "mae": float(row["mae"]),
                        "source_path": str(path),
                    }
                )
    return rows


def collect_training(root: Path, checkpoint_policy: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for loss_mode in LOSS_MODES:
            path = run_dir(root, checkpoint_policy, loss_mode, dataset, seed) / "training_log.csv"
            for row in read_csv(path):
                out: dict[str, Any] = {
                    "dataset": dataset,
                    "loss_mode": loss_mode,
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "train_prediction_l1": float(row["train_prediction_l1"]),
                    "train_prediction_full_l1": float(row.get("train_prediction_full_l1", row["train_prediction_l1"])),
                    "train_reconstruction_l1": float(row["train_reconstruction_l1"]),
                    "train_alignment_loss": float(row["train_alignment_loss"]),
                    "val_mean_mse": float(row["val_mean_mse"]),
                    "lr": float(row["lr"]),
                    "epoch_seconds": float(row["epoch_seconds"]),
                }
                rows.append(out)
    return rows


def load_fixed_reference(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    rows = read_csv(path)
    ref: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        ref[(row["dataset"], int(row["target_horizon"]))] = {
            "fixed_mse": float(row["fixed_mse"]),
            "fixed_mae": float(row["fixed_mae"]),
        }
    return ref


def compare(metric_rows: list[dict[str, Any]], fixed_ref: dict[tuple[str, int], dict[str, float]]) -> list[dict[str, Any]]:
    by_key = {
        (row["dataset"], row["loss_mode"], row["target_horizon"]): row
        for row in metric_rows
    }
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            full = by_key.get((dataset, "full", horizon))
            multi = by_key.get((dataset, "multi-prefix", horizon))
            fixed = fixed_ref.get((dataset, horizon))
            if not full:
                continue
            for loss_mode in LOSS_MODES:
                current = by_key.get((dataset, loss_mode, horizon))
                if not current:
                    continue
                row: dict[str, Any] = {
                    "dataset": dataset,
                    "target_horizon": horizon,
                    "loss_mode": loss_mode,
                    "mse": current["mse"],
                    "mae": current["mae"],
                    "relative_mse_vs_full_pct": relative_pct(current["mse"], full["mse"]),
                    "relative_mae_vs_full_pct": relative_pct(current["mae"], full["mae"]),
                    "beats_full": current["mse"] < full["mse"],
                }
                if multi:
                    row["relative_mse_vs_multi_prefix_pct"] = relative_pct(current["mse"], multi["mse"])
                    row["beats_multi_prefix"] = current["mse"] < multi["mse"]
                if fixed:
                    row["fixed_mse"] = fixed["fixed_mse"]
                    row["relative_mse_vs_fixed_pct"] = relative_pct(current["mse"], fixed["fixed_mse"])
                    row["beats_fixed"] = current["mse"] < fixed["fixed_mse"]
                rows.append(row)
    return rows


def summarize(compare_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS + ["ALL"]:
        subset_ds = compare_rows if dataset == "ALL" else [row for row in compare_rows if row["dataset"] == dataset]
        for loss_mode in LOSS_MODES:
            subset = [row for row in subset_ds if row["loss_mode"] == loss_mode]
            if not subset:
                continue
            row: dict[str, Any] = {
                "dataset": dataset,
                "loss_mode": loss_mode,
                "settings": len(subset),
                "wins_vs_full": sum(1 for item in subset if item["beats_full"]),
                "mean_relative_mse_vs_full_pct": sum(item["relative_mse_vs_full_pct"] for item in subset) / len(subset),
                "mean_relative_mae_vs_full_pct": sum(item["relative_mae_vs_full_pct"] for item in subset) / len(subset),
            }
            if all("relative_mse_vs_multi_prefix_pct" in item for item in subset):
                row["wins_vs_multi_prefix"] = sum(1 for item in subset if item["beats_multi_prefix"])
                row["mean_relative_mse_vs_multi_prefix_pct"] = (
                    sum(item["relative_mse_vs_multi_prefix_pct"] for item in subset) / len(subset)
                )
            fixed_subset = [item for item in subset if "relative_mse_vs_fixed_pct" in item]
            if fixed_subset:
                row["wins_vs_fixed"] = sum(1 for item in fixed_subset if item["beats_fixed"])
                row["mean_relative_mse_vs_fixed_pct"] = (
                    sum(item["relative_mse_vs_fixed_pct"] for item in fixed_subset) / len(fixed_subset)
                )
            rows.append(row)
    return rows


def best_epoch_rows(training_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in training_rows:
        grouped.setdefault((row["dataset"], row["loss_mode"]), []).append(row)
    for (dataset, loss_mode), items in sorted(grouped.items()):
        best = min(items, key=lambda item: item["val_mean_mse"])
        last = max(items, key=lambda item: item["epoch"])
        rows.append(
            {
                "dataset": dataset,
                "loss_mode": loss_mode,
                "best_epoch": best["epoch"],
                "best_val_mean_mse": best["val_mean_mse"],
                "last_epoch": last["epoch"],
                "last_val_mean_mse": last["val_mean_mse"],
                "last_gap_to_best_pct": relative_pct(last["val_mean_mse"], best["val_mean_mse"]),
            }
        )
    return rows


def write_report(path: Path, summary_rows: list[dict[str, Any]], compare_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase5 TimeAlign-HSS H0 Prefix Gate",
        "",
        "## Summary",
        "",
        "| dataset | loss_mode | settings | wins_vs_full | mean_mse_vs_full | wins_vs_fixed | mean_mse_vs_fixed |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {loss_mode} | {settings} | {wins_full} | {mse_full} | {wins_fixed} | {mse_fixed} |".format(
                dataset=row["dataset"],
                loss_mode=row["loss_mode"],
                settings=row["settings"],
                wins_full=row["wins_vs_full"],
                mse_full=fmt_pct(row["mean_relative_mse_vs_full_pct"]),
                wins_fixed=row.get("wins_vs_fixed", ""),
                mse_fixed=fmt_pct(row["mean_relative_mse_vs_fixed_pct"])
                if "mean_relative_mse_vs_fixed_pct" in row
                else "",
            )
        )
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `balanced-step` tests whether D0 was just non-overlapping region reweighting.",
            "- `stochastic-prefix` tests whether prefix supervision works as a train-time schedule over benchmark prefixes.",
            "- `continuous-prefix` tests whether the schedule can move away from fixed benchmark horizon ids.",
            "- H0 passes as a paper-story carrier only if a schedule-like mode approaches or beats `multi-prefix` while preserving ETTm2/Weather gap reduction and ETTh2 gains.",
            "",
            "## Per-Horizon Rows",
            "",
            "| dataset | horizon | loss_mode | mse | vs_full | vs_multi_prefix | vs_fixed |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in compare_rows:
        lines.append(
            "| {dataset} | {horizon} | {loss_mode} | {mse:.6f} | {full} | {multi} | {fixed} |".format(
                dataset=row["dataset"],
                horizon=row["target_horizon"],
                loss_mode=row["loss_mode"],
                mse=row["mse"],
                full=fmt_pct(row["relative_mse_vs_full_pct"]),
                multi=fmt_pct(row["relative_mse_vs_multi_prefix_pct"])
                if "relative_mse_vs_multi_prefix_pct" in row
                else "",
                fixed=fmt_pct(row["relative_mse_vs_fixed_pct"]) if "relative_mse_vs_fixed_pct" in row else "",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase5 TimeAlign-HSS H0 prefix gate.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-policy", choices=["official-last", "best-val"], default="official-last")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--fixed-reference-csv",
        type=Path,
        default=Path("analysis/phase5_timealign_official_gate_20260626/phase5_timealign_official_unified_gap.csv"),
    )
    args = parser.parse_args()

    metric_rows = collect_metrics(args.root, args.checkpoint_policy, args.seed)
    training_rows = collect_training(args.root, args.checkpoint_policy, args.seed)
    fixed_ref = load_fixed_reference(args.fixed_reference_csv)
    compare_rows = compare(metric_rows, fixed_ref)
    summary_rows = summarize(compare_rows)
    best_rows = best_epoch_rows(training_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "phase5_timealign_hss_h0_metrics.csv", metric_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_h0_comparison.csv", compare_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_h0_summary.csv", summary_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_h0_training.csv", training_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_h0_best_epoch.csv", best_rows)
    (args.output_dir / "phase5_timealign_hss_h0_summary.json").write_text(
        json.dumps(
            {
                "metric_rows": len(metric_rows),
                "compare_rows": len(compare_rows),
                "training_rows": len(training_rows),
                "summary_rows": summary_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_report(args.output_dir / "phase5_timealign_hss_h0_prefix_gate_report.md", summary_rows, compare_rows)
    print(f"phase5_timealign_hss_h0_report={args.output_dir / 'phase5_timealign_hss_h0_prefix_gate_report.md'}")


if __name__ == "__main__":
    main()
