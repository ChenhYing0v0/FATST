from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATASETS = ["ETTh2", "ETTm2", "Weather"]
LOSS_MODES = ["full", "multi-prefix"]
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


def fmt_pct(value: float) -> str:
    return f"{value:.2f}"


def relative_pct(value: float, baseline: float) -> float:
    if baseline == 0:
        return float("nan")
    return (value / baseline - 1.0) * 100.0


def run_dir(root: Path, checkpoint_policy: str, loss_mode: str, dataset: str, seed: int) -> Path:
    safe_loss_mode = loss_mode.replace("-", "_")
    run_name = f"TimeAlignOfficialUnified720_D0_{safe_loss_mode}_{checkpoint_policy}"
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
                for horizon in HORIZONS:
                    key = f"train_prediction_h{horizon}_l1"
                    if key in row and row[key] != "":
                        out[key] = float(row[key])
                rows.append(out)
    return rows


def compare_modes(metric_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {
        (row["dataset"], row["loss_mode"], row["target_horizon"]): row
        for row in metric_rows
    }
    gap_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            full = by_key.get((dataset, "full", horizon))
            multi = by_key.get((dataset, "multi-prefix", horizon))
            if not full or not multi:
                continue
            gap_rows.append(
                {
                    "dataset": dataset,
                    "target_horizon": horizon,
                    "full_mse": full["mse"],
                    "multi_prefix_mse": multi["mse"],
                    "relative_mse_pct": relative_pct(multi["mse"], full["mse"]),
                    "full_mae": full["mae"],
                    "multi_prefix_mae": multi["mae"],
                    "relative_mae_pct": relative_pct(multi["mae"], full["mae"]),
                    "multi_prefix_win": multi["mse"] < full["mse"],
                }
            )
    summary_rows: list[dict[str, Any]] = []
    for dataset in DATASETS + ["ALL"]:
        subset = gap_rows if dataset == "ALL" else [row for row in gap_rows if row["dataset"] == dataset]
        if not subset:
            continue
        summary_rows.append(
            {
                "dataset": dataset,
                "settings": len(subset),
                "multi_prefix_wins": sum(1 for row in subset if row["multi_prefix_win"]),
                "mean_relative_mse_pct": sum(row["relative_mse_pct"] for row in subset) / len(subset),
                "mean_relative_mae_pct": sum(row["relative_mae_pct"] for row in subset) / len(subset),
            }
        )
    return gap_rows, summary_rows


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


def write_report(
    path: Path,
    gap_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Phase5 TimeAlign-HSS D0 Head Gate",
        "",
        "## Decision Template",
        "",
        "[Question] Is the TimeAlign unified decrease mainly caused by the fixed `pred_len=720` head receiving only a full-horizon prediction loss?",
        "",
        "## Summary",
        "",
        "| dataset | settings | multi_prefix_wins | mean_relative_mse_pct | mean_relative_mae_pct |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {settings} | {wins} | {mse} | {mae} |".format(
                dataset=row["dataset"],
                settings=row["settings"],
                wins=row["multi_prefix_wins"],
                mse=fmt_pct(row["mean_relative_mse_pct"]),
                mae=fmt_pct(row["mean_relative_mae_pct"]),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Horizon Comparison",
            "",
            "| dataset | horizon | full_mse | multi_prefix_mse | relative_mse_pct | multi_prefix_win |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in gap_rows:
        lines.append(
            "| {dataset} | {horizon} | {full:.6f} | {multi:.6f} | {rel} | {win} |".format(
                dataset=row["dataset"],
                horizon=row["target_horizon"],
                full=row["full_mse"],
                multi=row["multi_prefix_mse"],
                rel=fmt_pct(row["relative_mse_pct"]),
                win=row["multi_prefix_win"],
            )
        )
    lines.extend(
        [
            "",
            "## Training Selector Diagnostic",
            "",
            "| dataset | loss_mode | best_epoch | best_val_mean_mse | last_epoch | last_val_mean_mse | last_gap_to_best_pct |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in best_rows:
        lines.append(
            "| {dataset} | {loss_mode} | {best_epoch} | {best_val:.6f} | {last_epoch} | {last_val:.6f} | {gap} |".format(
                dataset=row["dataset"],
                loss_mode=row["loss_mode"],
                best_epoch=row["best_epoch"],
                best_val=row["best_val_mean_mse"],
                last_epoch=row["last_epoch"],
                last_val=row["last_val_mean_mse"],
                gap=fmt_pct(row["last_gap_to_best_pct"]),
            )
        )
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- [Pass] If `multi-prefix` materially reduces ETTm2/Weather gaps without losing ETTh2, the head/interface confounder is strong and HSS should not start from supervision reliability alone.",
            "- [Partial] If `multi-prefix` helps only one degraded dataset, keep D1 but treat unified head design as a co-factor.",
            "- [Fail] If `multi-prefix` does not help, proceed to D1 supervision reliability diagnostic.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase5 TimeAlign-HSS D0 head/interface gate.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-policy", choices=["official-last", "best-val"], default="official-last")
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()

    metric_rows = collect_metrics(args.root, args.checkpoint_policy, args.seed)
    training_rows = collect_training(args.root, args.checkpoint_policy, args.seed)
    gap_rows, summary_rows = compare_modes(metric_rows)
    best_rows = best_epoch_rows(training_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "phase5_timealign_hss_d0_metrics.csv", metric_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_d0_multi_prefix_gap.csv", gap_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_d0_summary.csv", summary_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_d0_training.csv", training_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_d0_best_epoch.csv", best_rows)
    (args.output_dir / "phase5_timealign_hss_d0_summary.json").write_text(
        json.dumps(
            {
                "metric_rows": len(metric_rows),
                "gap_rows": len(gap_rows),
                "training_rows": len(training_rows),
                "summary_rows": summary_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_report(
        args.output_dir / "phase5_timealign_hss_d0_head_gate_report.md",
        gap_rows,
        summary_rows,
        best_rows,
    )
    print(f"phase5_timealign_hss_d0_report={args.output_dir / 'phase5_timealign_hss_d0_head_gate_report.md'}")


if __name__ == "__main__":
    main()
