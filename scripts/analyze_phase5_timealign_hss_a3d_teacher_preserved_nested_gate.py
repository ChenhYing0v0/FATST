from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_phase5_timealign_hss_a2_interface_gate import (
    DATASETS,
    add_reference,
    best_epoch_rows,
    fmt_pct,
    load_fixed,
    load_reference,
    read_csv,
    write_csv,
)


ARMS = ["teacher_preserved_nested_w03", "teacher_preserved_nested_w10"]


def run_dir(root: Path, checkpoint_policy: str, arm: str, dataset: str, seed: int) -> Path:
    run_name = f"TimeAlignOfficialUnified720_A3D_{arm}_{checkpoint_policy}"
    return root / checkpoint_policy / run_name / dataset / "mixed_h96_h192_h336_h720" / f"seed{seed}"


def collect_metrics(root: Path, checkpoint_policy: str, seed: int) -> list[dict[str, Any]]:
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


def collect_training(root: Path, checkpoint_policy: str, seed: int) -> list[dict[str, Any]]:
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
                        "train_teacher_l1": float(row.get("train_teacher_l1", 0.0)),
                        "teacher_loss_weight": float(row.get("teacher_loss_weight", 0.0)),
                        "train_reconstruction_l1": float(row["train_reconstruction_l1"]),
                        "train_alignment_loss": float(row["train_alignment_loss"]),
                        "pred_loss_mode": row["pred_loss_mode"],
                        "prefix_samples": int(row.get("prefix_samples", 1)),
                        "val_mean_mse": float(row["val_mean_mse"]),
                        "lr": float(row["lr"]),
                        "epoch_seconds": float(row["epoch_seconds"]),
                        "warm_start_checkpoint": row.get("warm_start_checkpoint", ""),
                        "teacher_checkpoint": row.get("teacher_checkpoint", ""),
                        "teacher_readout_mode": row.get("teacher_readout_mode", ""),
                    }
                )
    return rows


def compare(metric_rows: list[dict[str, Any]], references: dict[str, dict[tuple[str, int], dict[str, float]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in metric_rows:
        out = dict(row)
        for name, ref in references.items():
            add_reference(out, row, ref, name)
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
            out: dict[str, Any] = {"dataset": dataset, "arm": arm, "settings": len(subset)}
            for name in ["a2_nested", "a3c_warm", "h1_target_set", "h1c_row_gated", "fixed"]:
                key = f"relative_mse_vs_{name}_pct"
                values = [row[key] for row in subset if key in row]
                if not values:
                    continue
                out[f"wins_vs_{name}"] = sum(1 for row in subset if row.get(f"beats_{name}") is True)
                out[f"mean_relative_mse_vs_{name}_pct"] = sum(values) / len(values)
            rows.append(out)
    return rows


def write_report(path: Path, summary_rows: list[dict[str, Any]], compare_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase5 TimeAlign-HSS A3D Teacher-Preserved Nested Gate",
        "",
        "## Summary",
        "",
        "| dataset | arm | settings | vs_a2_nested | vs_a3c_warm | vs_h1_target_set | wins_vs_h1c | vs_h1c_row_gated | wins_vs_fixed | vs_fixed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {arm} | {settings} | {a2} | {a3c} | {h1} | {wins_h1c} | {h1c} | {wins_fixed} | {fixed} |".format(
                dataset=row["dataset"],
                arm=row["arm"],
                settings=row["settings"],
                a2=fmt_pct(row.get("mean_relative_mse_vs_a2_nested_pct", float("nan"))),
                a3c=fmt_pct(row.get("mean_relative_mse_vs_a3c_warm_pct", float("nan"))),
                h1=fmt_pct(row.get("mean_relative_mse_vs_h1_target_set_pct", float("nan"))),
                wins_h1c=row.get("wins_vs_h1c_row_gated", ""),
                h1c=fmt_pct(row.get("mean_relative_mse_vs_h1c_row_gated_pct", float("nan"))),
                wins_fixed=row.get("wins_vs_fixed", ""),
                fixed=fmt_pct(row.get("mean_relative_mse_vs_fixed_pct", float("nan"))),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Horizon Rows",
            "",
            "| dataset | horizon | arm | mse | vs_a2_nested | vs_a3c_warm | vs_h1_target_set | vs_h1c_row_gated | vs_fixed |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in compare_rows:
        lines.append(
            "| {dataset} | {horizon} | {arm} | {mse:.6f} | {a2} | {a3c} | {h1} | {h1c} | {fixed} |".format(
                dataset=row["dataset"],
                horizon=row["target_horizon"],
                arm=row["arm"],
                mse=row["mse"],
                a2=fmt_pct(row.get("relative_mse_vs_a2_nested_pct", float("nan"))),
                a3c=fmt_pct(row.get("relative_mse_vs_a3c_warm_pct", float("nan"))),
                h1=fmt_pct(row.get("relative_mse_vs_h1_target_set_pct", float("nan"))),
                h1c=fmt_pct(row.get("relative_mse_vs_h1c_row_gated_pct", float("nan"))),
                fixed=fmt_pct(row.get("relative_mse_vs_fixed_pct", float("nan"))),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase5 TimeAlign-HSS A3D teacher-preserved nested gate.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-policy", choices=["official-last", "best-val"], default="official-last")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--a2-metrics-csv", type=Path, default=Path("analysis/phase5_timealign_hss_a2_interface_gate_20260701/phase5_timealign_hss_a2_metrics.csv"))
    parser.add_argument("--a3c-metrics-csv", type=Path, default=Path("analysis/phase5_timealign_hss_a3c_warm_started_nested_gate_20260701/phase5_timealign_hss_a3c_metrics.csv"))
    parser.add_argument("--h1-metrics-csv", type=Path, default=Path("analysis/phase5_timealign_hss_h1_readout_gate_20260630/phase5_timealign_hss_h1_metrics.csv"))
    parser.add_argument("--h1c-metrics-csv", type=Path, default=Path("analysis/phase5_timealign_hss_h1c_capacity_preserving_gate_20260701/phase5_timealign_hss_h1c_metrics.csv"))
    parser.add_argument("--fixed-reference-csv", type=Path, default=Path("analysis/phase5_timealign_official_gate_20260626/phase5_timealign_official_unified_gap.csv"))
    args = parser.parse_args()

    metric_rows = collect_metrics(args.root, args.checkpoint_policy, args.seed)
    training_rows = collect_training(args.root, args.checkpoint_policy, args.seed)
    references = {
        "a2_nested": load_reference(args.a2_metrics_csv, arm="nested_segment_decoder_multiprefix"),
        "a3c_warm": load_reference(args.a3c_metrics_csv, arm="checkpoint_initialized_nested_segment_decoder_multiprefix"),
        "h1_target_set": load_reference(args.h1_metrics_csv, arm="target_set_decoder_multiprefix"),
        "h1c_row_gated": load_reference(args.h1c_metrics_csv, arm="row_gated_dense_head_multiprefix"),
        "fixed": load_fixed(args.fixed_reference_csv),
    }
    compare_rows = compare(metric_rows, references)
    summary_rows = summarize(compare_rows)
    best_rows = best_epoch_rows(training_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "phase5_timealign_hss_a3d_metrics.csv", metric_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_a3d_comparison.csv", compare_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_a3d_summary.csv", summary_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_a3d_training.csv", training_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_a3d_best_epoch.csv", best_rows)
    (args.output_dir / "phase5_timealign_hss_a3d_summary.json").write_text(
        json.dumps({"metric_rows": len(metric_rows), "summary_rows": summary_rows}, indent=2, sort_keys=True) + "\n"
    )
    write_report(args.output_dir / "phase5_timealign_hss_a3d_teacher_preserved_nested_gate_report.md", summary_rows, compare_rows)
    print(f"phase5_timealign_hss_a3d_report={args.output_dir / 'phase5_timealign_hss_a3d_teacher_preserved_nested_gate_report.md'}")


if __name__ == "__main__":
    main()
