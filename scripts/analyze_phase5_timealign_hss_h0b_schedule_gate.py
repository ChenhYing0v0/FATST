from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATASETS = ["ETTh2", "ETTm2", "Weather"]
ARMS = ["stochastic_prefix_k2", "continuous_prefix_k2", "continuous_prefix_pool96"]
REFERENCE_MODES = ["full", "multi-prefix", "stochastic-prefix", "continuous-prefix"]
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


def run_dir(root: Path, checkpoint_policy: str, arm: str, dataset: str, seed: int) -> Path:
    run_name = f"TimeAlignOfficialUnified720_H0B_{arm}_{checkpoint_policy}"
    return root / checkpoint_policy / run_name / dataset / "mixed_h96_h192_h336_h720" / f"seed{seed}"


def collect_h0b_metrics(root: Path, checkpoint_policy: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for arm in ARMS:
            path = run_dir(root, checkpoint_policy, arm, dataset, seed) / "metrics_by_target_horizon.csv"
            for row in read_csv(path):
                rows.append(
                    {
                        "dataset": dataset,
                        "arm": arm,
                        "target_horizon": int(row["target_horizon"]),
                        "mse": float(row["mse"]),
                        "mae": float(row["mae"]),
                        "source_path": str(path),
                    }
                )
    return rows


def collect_h0b_training(root: Path, checkpoint_policy: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for arm in ARMS:
            path = run_dir(root, checkpoint_policy, arm, dataset, seed) / "training_log.csv"
            for row in read_csv(path):
                rows.append(
                    {
                        "dataset": dataset,
                        "arm": arm,
                        "epoch": int(row["epoch"]),
                        "train_loss": float(row["train_loss"]),
                        "train_prediction_l1": float(row["train_prediction_l1"]),
                        "train_prediction_full_l1": float(row.get("train_prediction_full_l1", row["train_prediction_l1"])),
                        "train_reconstruction_l1": float(row["train_reconstruction_l1"]),
                        "train_alignment_loss": float(row["train_alignment_loss"]),
                        "pred_loss_mode": row["pred_loss_mode"],
                        "prefix_samples": int(row.get("prefix_samples", 1)),
                        "continuous_min_prefix": int(row.get("continuous_min_prefix", 32)),
                        "continuous_prefix_step": int(row.get("continuous_prefix_step", 32)),
                        "val_mean_mse": float(row["val_mean_mse"]),
                        "lr": float(row["lr"]),
                        "epoch_seconds": float(row["epoch_seconds"]),
                    }
                )
    return rows


def load_reference_metrics(path: Path) -> dict[tuple[str, str, int], dict[str, float]]:
    ref: dict[tuple[str, str, int], dict[str, float]] = {}
    for row in read_csv(path):
        loss_mode = row["loss_mode"]
        if loss_mode not in REFERENCE_MODES:
            continue
        ref[(row["dataset"], loss_mode, int(row["target_horizon"]))] = {
            "mse": float(row["mse"]),
            "mae": float(row["mae"]),
        }
    return ref


def load_fixed_reference(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    ref: dict[tuple[str, int], dict[str, float]] = {}
    for row in read_csv(path):
        ref[(row["dataset"], int(row["target_horizon"]))] = {
            "fixed_mse": float(row["fixed_mse"]),
            "fixed_mae": float(row["fixed_mae"]),
        }
    return ref


def compare(
    metric_rows: list[dict[str, Any]],
    h0_ref: dict[tuple[str, str, int], dict[str, float]],
    fixed_ref: dict[tuple[str, int], dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in metric_rows:
        dataset = row["dataset"]
        horizon = row["target_horizon"]
        out: dict[str, Any] = dict(row)
        for mode in REFERENCE_MODES:
            ref = h0_ref.get((dataset, mode, horizon))
            if ref:
                safe_name = mode.replace("-", "_")
                out[f"relative_mse_vs_{safe_name}_pct"] = relative_pct(row["mse"], ref["mse"])
                out[f"beats_{safe_name}"] = row["mse"] < ref["mse"]
        fixed = fixed_ref.get((dataset, horizon))
        if fixed:
            out["fixed_mse"] = fixed["fixed_mse"]
            out["relative_mse_vs_fixed_pct"] = relative_pct(row["mse"], fixed["fixed_mse"])
            out["beats_fixed"] = row["mse"] < fixed["fixed_mse"]
        rows.append(out)
    return rows


def summarize(compare_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS + ["ALL"]:
        subset_ds = compare_rows if dataset == "ALL" else [row for row in compare_rows if row["dataset"] == dataset]
        for arm in ARMS:
            subset = [row for row in subset_ds if row["arm"] == arm]
            if not subset:
                continue
            out: dict[str, Any] = {
                "dataset": dataset,
                "arm": arm,
                "settings": len(subset),
            }
            for mode in REFERENCE_MODES:
                safe_name = mode.replace("-", "_")
                values = [row[f"relative_mse_vs_{safe_name}_pct"] for row in subset if f"relative_mse_vs_{safe_name}_pct" in row]
                if values:
                    out[f"wins_vs_{safe_name}"] = sum(1 for row in subset if row.get(f"beats_{safe_name}") is True)
                    out[f"mean_relative_mse_vs_{safe_name}_pct"] = sum(values) / len(values)
            fixed_values = [row["relative_mse_vs_fixed_pct"] for row in subset if "relative_mse_vs_fixed_pct" in row]
            if fixed_values:
                out["wins_vs_fixed"] = sum(1 for row in subset if row.get("beats_fixed") is True)
                out["mean_relative_mse_vs_fixed_pct"] = sum(fixed_values) / len(fixed_values)
            rows.append(out)
    return rows


def best_epoch_rows(training_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in training_rows:
        grouped.setdefault((row["dataset"], row["arm"]), []).append(row)
    for (dataset, arm), items in sorted(grouped.items()):
        best = min(items, key=lambda item: item["val_mean_mse"])
        last = max(items, key=lambda item: item["epoch"])
        rows.append(
            {
                "dataset": dataset,
                "arm": arm,
                "pred_loss_mode": last["pred_loss_mode"],
                "prefix_samples": last["prefix_samples"],
                "continuous_min_prefix": last["continuous_min_prefix"],
                "continuous_prefix_step": last["continuous_prefix_step"],
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
        "# Phase5 TimeAlign-HSS H0B Schedule Gate",
        "",
        "## Summary",
        "",
        "| dataset | arm | settings | vs_full | vs_multi_prefix | vs_stochastic_prefix | vs_continuous_prefix | wins_vs_fixed | vs_fixed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {arm} | {settings} | {full} | {multi} | {stoch} | {cont} | {wins_fixed} | {fixed} |".format(
                dataset=row["dataset"],
                arm=row["arm"],
                settings=row["settings"],
                full=fmt_pct(row.get("mean_relative_mse_vs_full_pct", float("nan"))),
                multi=fmt_pct(row.get("mean_relative_mse_vs_multi_prefix_pct", float("nan"))),
                stoch=fmt_pct(row.get("mean_relative_mse_vs_stochastic_prefix_pct", float("nan"))),
                cont=fmt_pct(row.get("mean_relative_mse_vs_continuous_prefix_pct", float("nan"))),
                wins_fixed=row.get("wins_vs_fixed", ""),
                fixed=fmt_pct(row.get("mean_relative_mse_vs_fixed_pct", float("nan"))),
            )
        )
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `stochastic_prefix_k2` passes if it improves ETTm2 or overall mean over H0 `stochastic-prefix` and approaches or beats `multi-prefix`.",
            "- `continuous_prefix_k2` passes if increasing sample count lets continuous scheduling approach `multi-prefix`.",
            "- `continuous_prefix_pool96` passes if removing very short prefixes improves over H0 `continuous-prefix`.",
            "",
            "## Per-Horizon Rows",
            "",
            "| dataset | horizon | arm | mse | vs_full | vs_multi_prefix | vs_stochastic_prefix | vs_continuous_prefix | vs_fixed |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in compare_rows:
        lines.append(
            "| {dataset} | {horizon} | {arm} | {mse:.6f} | {full} | {multi} | {stoch} | {cont} | {fixed} |".format(
                dataset=row["dataset"],
                horizon=row["target_horizon"],
                arm=row["arm"],
                mse=row["mse"],
                full=fmt_pct(row.get("relative_mse_vs_full_pct", float("nan"))),
                multi=fmt_pct(row.get("relative_mse_vs_multi_prefix_pct", float("nan"))),
                stoch=fmt_pct(row.get("relative_mse_vs_stochastic_prefix_pct", float("nan"))),
                cont=fmt_pct(row.get("relative_mse_vs_continuous_prefix_pct", float("nan"))),
                fixed=fmt_pct(row.get("relative_mse_vs_fixed_pct", float("nan"))),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase5 TimeAlign-HSS H0B schedule gate.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-policy", choices=["official-last", "best-val"], default="official-last")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--h0-metrics-csv",
        type=Path,
        default=Path("analysis/phase5_timealign_hss_h0_prefix_gate_20260630/phase5_timealign_hss_h0_metrics.csv"),
    )
    parser.add_argument(
        "--fixed-reference-csv",
        type=Path,
        default=Path("analysis/phase5_timealign_official_gate_20260626/phase5_timealign_official_unified_gap.csv"),
    )
    args = parser.parse_args()

    metric_rows = collect_h0b_metrics(args.root, args.checkpoint_policy, args.seed)
    training_rows = collect_h0b_training(args.root, args.checkpoint_policy, args.seed)
    h0_ref = load_reference_metrics(args.h0_metrics_csv)
    fixed_ref = load_fixed_reference(args.fixed_reference_csv)
    compare_rows = compare(metric_rows, h0_ref, fixed_ref)
    summary_rows = summarize(compare_rows)
    best_rows = best_epoch_rows(training_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "phase5_timealign_hss_h0b_metrics.csv", metric_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_h0b_comparison.csv", compare_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_h0b_summary.csv", summary_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_h0b_training.csv", training_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_h0b_best_epoch.csv", best_rows)
    (args.output_dir / "phase5_timealign_hss_h0b_summary.json").write_text(
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
    write_report(args.output_dir / "phase5_timealign_hss_h0b_schedule_gate_report.md", summary_rows, compare_rows)
    print(f"phase5_timealign_hss_h0b_report={args.output_dir / 'phase5_timealign_hss_h0b_schedule_gate_report.md'}")


if __name__ == "__main__":
    main()
