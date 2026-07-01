from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATASETS = ["ETTh2", "ETTm2", "Weather"]
ARMS = ["dense_prefix_residual_adapter_multiprefix", "row_gated_dense_head_multiprefix", "prefix_adapter_shared_dense_multiprefix"]


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
    run_name = f"TimeAlignOfficialUnified720_H1C_{arm}_{checkpoint_policy}"
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
                        "train_reconstruction_l1": float(row["train_reconstruction_l1"]),
                        "train_alignment_loss": float(row["train_alignment_loss"]),
                        "pred_loss_mode": row["pred_loss_mode"],
                        "prefix_samples": int(row.get("prefix_samples", 1)),
                        "val_mean_mse": float(row["val_mean_mse"]),
                        "lr": float(row["lr"]),
                        "epoch_seconds": float(row["epoch_seconds"]),
                    }
                )
    return rows


def load_h0_full(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    ref: dict[tuple[str, int], dict[str, float]] = {}
    for row in read_csv(path):
        if row["loss_mode"] != "full":
            continue
        ref[(row["dataset"], int(row["target_horizon"]))] = {
            "mse": float(row["mse"]),
            "mae": float(row["mae"]),
        }
    return ref


def load_h0b_stochastic(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    ref: dict[tuple[str, int], dict[str, float]] = {}
    for row in read_csv(path):
        if row["arm"] != "stochastic_prefix_k2":
            continue
        ref[(row["dataset"], int(row["target_horizon"]))] = {
            "mse": float(row["mse"]),
            "mae": float(row["mae"]),
        }
    return ref


def load_h1_target_set(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    ref: dict[tuple[str, int], dict[str, float]] = {}
    for row in read_csv(path):
        if row["arm"] != "target_set_decoder_multiprefix":
            continue
        ref[(row["dataset"], int(row["target_horizon"]))] = {
            "mse": float(row["mse"]),
            "mae": float(row["mae"]),
        }
    return ref


def load_fixed(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    ref: dict[tuple[str, int], dict[str, float]] = {}
    for row in read_csv(path):
        ref[(row["dataset"], int(row["target_horizon"]))] = {
            "mse": float(row["fixed_mse"]),
            "mae": float(row["fixed_mae"]),
        }
    return ref


def add_reference(
    out: dict[str, Any],
    row: dict[str, Any],
    ref: dict[tuple[str, int], dict[str, float]],
    name: str,
) -> None:
    item = ref.get((row["dataset"], row["target_horizon"]))
    if not item:
        return
    out[f"relative_mse_vs_{name}_pct"] = relative_pct(row["mse"], item["mse"])
    out[f"beats_{name}"] = row["mse"] < item["mse"]
    out[f"relative_mae_vs_{name}_pct"] = relative_pct(row["mae"], item["mae"])


def compare(
    metric_rows: list[dict[str, Any]],
    h0_full: dict[tuple[str, int], dict[str, float]],
    h0b_stochastic: dict[tuple[str, int], dict[str, float]],
    h1_target_set: dict[tuple[str, int], dict[str, float]],
    fixed: dict[tuple[str, int], dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in metric_rows:
        out = dict(row)
        add_reference(out, row, h0_full, "h0_full")
        add_reference(out, row, h0b_stochastic, "h0b_stochastic_k2")
        add_reference(out, row, h1_target_set, "h1_target_set")
        add_reference(out, row, fixed, "fixed")
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
            for name in ["h0_full", "h0b_stochastic_k2", "h1_target_set", "fixed"]:
                key = f"relative_mse_vs_{name}_pct"
                values = [row[key] for row in subset if key in row]
                if not values:
                    continue
                out[f"wins_vs_{name}"] = sum(1 for row in subset if row.get(f"beats_{name}") is True)
                out[f"mean_relative_mse_vs_{name}_pct"] = sum(values) / len(values)
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
        "# Phase5 TimeAlign-HSS H1C Capacity-Preserving Gate",
        "",
        "## Summary",
        "",
        "| dataset | arm | settings | vs_h0_full | vs_h0b_stochastic_k2 | vs_h1_target_set | wins_vs_fixed | vs_fixed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {arm} | {settings} | {h0} | {h0b} | {h1} | {wins_fixed} | {fixed} |".format(
                dataset=row["dataset"],
                arm=row["arm"],
                settings=row["settings"],
                h0=fmt_pct(row.get("mean_relative_mse_vs_h0_full_pct", float("nan"))),
                h0b=fmt_pct(row.get("mean_relative_mse_vs_h0b_stochastic_k2_pct", float("nan"))),
                h1=fmt_pct(row.get("mean_relative_mse_vs_h1_target_set_pct", float("nan"))),
                wins_fixed=row.get("wins_vs_fixed", ""),
                fixed=fmt_pct(row.get("mean_relative_mse_vs_fixed_pct", float("nan"))),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Horizon Rows",
            "",
            "| dataset | horizon | arm | mse | vs_h0_full | vs_h0b_stochastic_k2 | vs_h1_target_set | vs_fixed |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in compare_rows:
        lines.append(
            "| {dataset} | {horizon} | {arm} | {mse:.6f} | {h0} | {h0b} | {h1} | {fixed} |".format(
                dataset=row["dataset"],
                horizon=row["target_horizon"],
                arm=row["arm"],
                mse=row["mse"],
                h0=fmt_pct(row.get("relative_mse_vs_h0_full_pct", float("nan"))),
                h0b=fmt_pct(row.get("relative_mse_vs_h0b_stochastic_k2_pct", float("nan"))),
                h1=fmt_pct(row.get("relative_mse_vs_h1_target_set_pct", float("nan"))),
                fixed=fmt_pct(row.get("relative_mse_vs_fixed_pct", float("nan"))),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase5 TimeAlign-HSS H1C capacity preserving gate.")
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
        "--h0b-metrics-csv",
        type=Path,
        default=Path("analysis/phase5_timealign_hss_h0b_schedule_gate_20260630/phase5_timealign_hss_h0b_metrics.csv"),
    )
    parser.add_argument(
        "--h1-metrics-csv",
        type=Path,
        default=Path("analysis/phase5_timealign_hss_h1_readout_gate_20260630/phase5_timealign_hss_h1_metrics.csv"),
    )
    parser.add_argument(
        "--fixed-reference-csv",
        type=Path,
        default=Path("analysis/phase5_timealign_official_gate_20260626/phase5_timealign_official_unified_gap.csv"),
    )
    args = parser.parse_args()

    metric_rows = collect_metrics(args.root, args.checkpoint_policy, args.seed)
    training_rows = collect_training(args.root, args.checkpoint_policy, args.seed)
    compare_rows = compare(
        metric_rows,
        load_h0_full(args.h0_metrics_csv),
        load_h0b_stochastic(args.h0b_metrics_csv),
        load_h1_target_set(args.h1_metrics_csv),
        load_fixed(args.fixed_reference_csv),
    )
    summary_rows = summarize(compare_rows)
    best_rows = best_epoch_rows(training_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "phase5_timealign_hss_h1c_metrics.csv", metric_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_h1c_comparison.csv", compare_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_h1c_summary.csv", summary_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_h1c_training.csv", training_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_h1c_best_epoch.csv", best_rows)
    (args.output_dir / "phase5_timealign_hss_h1c_summary.json").write_text(
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
    write_report(args.output_dir / "phase5_timealign_hss_h1c_capacity_preserving_gate_report.md", summary_rows, compare_rows)
    print(f"phase5_timealign_hss_h1c_report={args.output_dir / 'phase5_timealign_hss_h1c_capacity_preserving_gate_report.md'}")


if __name__ == "__main__":
    main()
