from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATASETS = ["ETTh2", "ETTm1", "Weather"]
HORIZONS = [96, 192, 336, 720]
A3E_ARMS = ["target_conditioned_nested_warm", "target_conditioned_nested_scratch"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
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


def mixed_label() -> str:
    return "mixed_h96_h192_h336_h720"


def metric_path(root: Path, checkpoint_policy: str, run_name: str, dataset: str, seed: int) -> Path:
    return root / checkpoint_policy / run_name / dataset / mixed_label() / f"seed{seed}" / "metrics_by_target_horizon.csv"


def training_path(root: Path, checkpoint_policy: str, run_name: str, dataset: str, seed: int) -> Path:
    return root / checkpoint_policy / run_name / dataset / mixed_label() / f"seed{seed}" / "training_log.csv"


def collect_run_metrics(
    root: Path,
    checkpoint_policy: str,
    run_prefix: str,
    arms: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for arm in arms:
            run_name = f"{run_prefix}_{arm}_{checkpoint_policy}"
            path = metric_path(root, checkpoint_policy, run_name, dataset, seed)
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


def collect_a3e_training(root: Path, checkpoint_policy: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for arm in A3E_ARMS:
            run_name = f"TimeAlignOfficialUnified720_A3E_{arm}_{checkpoint_policy}"
            path = training_path(root, checkpoint_policy, run_name, dataset, seed)
            for row in read_csv(path):
                out: dict[str, Any] = {
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
                    "warm_start_checkpoint": row.get("warm_start_checkpoint", ""),
                }
                rows.append(out)
    return rows


def collect_fixed_reference(root: Path, checkpoint_policy: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            run_name = f"TimeAlignOfficialFixedH{horizon}_{checkpoint_policy}"
            path = root / checkpoint_policy / run_name / dataset / f"mixed_h{horizon}" / f"seed{seed}" / "metrics_by_target_horizon.csv"
            metric_rows = read_csv(path)
            if len(metric_rows) != 1:
                raise ValueError(f"Expected one fixed metric row in {path}, got {len(metric_rows)}")
            row = metric_rows[0]
            rows.append(
                {
                    "dataset": dataset,
                    "target_horizon": int(row["target_horizon"]),
                    "fixed_mse": float(row["mse"]),
                    "fixed_mae": float(row["mae"]),
                    "source_path": str(path),
                }
            )
    return rows


def reference_from_rows(rows: list[dict[str, Any]], arm: str | None = None, prefix: str = "") -> dict[tuple[str, int], dict[str, float]]:
    ref: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        if arm is not None and row.get("arm") != arm:
            continue
        mse_key = f"{prefix}mse" if prefix else "mse"
        mae_key = f"{prefix}mae" if prefix else "mae"
        ref[(row["dataset"], int(row["target_horizon"]))] = {
            "mse": float(row[mse_key]),
            "mae": float(row[mae_key]),
        }
    return ref


def add_reference(out: dict[str, Any], row: dict[str, Any], ref: dict[tuple[str, int], dict[str, float]], name: str) -> None:
    item = ref.get((row["dataset"], row["target_horizon"]))
    if item is None:
        raise KeyError((name, row["dataset"], row["target_horizon"]))
    out[f"relative_mse_vs_{name}_pct"] = relative_pct(row["mse"], item["mse"])
    out[f"beats_{name}"] = row["mse"] < item["mse"]
    out[f"relative_mae_vs_{name}_pct"] = relative_pct(row["mae"], item["mae"])


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
        for arm in A3E_ARMS:
            subset = [row for row in subset_ds if row["arm"] == arm]
            if not subset:
                continue
            out: dict[str, Any] = {"dataset": dataset, "arm": arm, "settings": len(subset)}
            for name in ["a2_nested", "a3c_warm", "a3d_teacher_w03", "h1_target_set", "h1c_row_gated", "fixed"]:
                key = f"relative_mse_vs_{name}_pct"
                values = [row[key] for row in subset]
                out[f"wins_vs_{name}"] = sum(1 for row in subset if row[f"beats_{name}"] is True)
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
        "# Phase5 TimeAlign-HSS A3E ETTm1 Replacement Gate",
        "",
        "Dataset universe: `ETTh2 + ETTm1 + Weather`. `ETTm1` replaces `ETTm2`; all ETTm1 references are rebuilt from remote raw metrics.",
        "",
        "## Summary",
        "",
        "| dataset | arm | settings | vs_a2_nested | vs_a3c_warm | vs_a3d_w03 | vs_h1_target_set | wins_vs_h1c | vs_h1c_row_gated | wins_vs_fixed | vs_fixed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {arm} | {settings} | {a2} | {a3c} | {a3d} | {h1} | {wins_h1c} | {h1c} | {wins_fixed} | {fixed} |".format(
                dataset=row["dataset"],
                arm=row["arm"],
                settings=row["settings"],
                a2=fmt_pct(row["mean_relative_mse_vs_a2_nested_pct"]),
                a3c=fmt_pct(row["mean_relative_mse_vs_a3c_warm_pct"]),
                a3d=fmt_pct(row["mean_relative_mse_vs_a3d_teacher_w03_pct"]),
                h1=fmt_pct(row["mean_relative_mse_vs_h1_target_set_pct"]),
                wins_h1c=row["wins_vs_h1c_row_gated"],
                h1c=fmt_pct(row["mean_relative_mse_vs_h1c_row_gated_pct"]),
                wins_fixed=row["wins_vs_fixed"],
                fixed=fmt_pct(row["mean_relative_mse_vs_fixed_pct"]),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Horizon Rows",
            "",
            "| dataset | horizon | arm | mse | vs_a2_nested | vs_a3c_warm | vs_a3d_w03 | vs_h1_target_set | vs_h1c_row_gated | vs_fixed |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in compare_rows:
        lines.append(
            "| {dataset} | {horizon} | {arm} | {mse:.6f} | {a2} | {a3c} | {a3d} | {h1} | {h1c} | {fixed} |".format(
                dataset=row["dataset"],
                horizon=row["target_horizon"],
                arm=row["arm"],
                mse=row["mse"],
                a2=fmt_pct(row["relative_mse_vs_a2_nested_pct"]),
                a3c=fmt_pct(row["relative_mse_vs_a3c_warm_pct"]),
                a3d=fmt_pct(row["relative_mse_vs_a3d_teacher_w03_pct"]),
                h1=fmt_pct(row["relative_mse_vs_h1_target_set_pct"]),
                h1c=fmt_pct(row["relative_mse_vs_h1c_row_gated_pct"]),
                fixed=fmt_pct(row["relative_mse_vs_fixed_pct"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase5 A3E with ETTm1 replacing ETTm2.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-policy", choices=["official-last", "best-val"], default="official-last")
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()

    official_rows = collect_fixed_reference(args.raw_root / "official", args.checkpoint_policy, args.seed)
    h1_rows = collect_run_metrics(
        args.raw_root / "h1",
        args.checkpoint_policy,
        "TimeAlignOfficialUnified720_H1",
        ["target_set_decoder_multiprefix"],
        args.seed,
    )
    h1c_rows = collect_run_metrics(
        args.raw_root / "h1c",
        args.checkpoint_policy,
        "TimeAlignOfficialUnified720_H1C",
        ["row_gated_dense_head_multiprefix"],
        args.seed,
    )
    a2_rows = collect_run_metrics(
        args.raw_root / "a2",
        args.checkpoint_policy,
        "TimeAlignOfficialUnified720_A2",
        ["nested_segment_decoder_multiprefix"],
        args.seed,
    )
    a3c_rows = collect_run_metrics(
        args.raw_root / "a3c",
        args.checkpoint_policy,
        "TimeAlignOfficialUnified720_A3C",
        ["checkpoint_initialized_nested_segment_decoder_multiprefix"],
        args.seed,
    )
    a3d_rows = collect_run_metrics(
        args.raw_root / "a3d",
        args.checkpoint_policy,
        "TimeAlignOfficialUnified720_A3D",
        ["teacher_preserved_nested_w03"],
        args.seed,
    )
    a3e_rows = collect_run_metrics(
        args.raw_root / "a3e",
        args.checkpoint_policy,
        "TimeAlignOfficialUnified720_A3E",
        A3E_ARMS,
        args.seed,
    )
    training_rows = collect_a3e_training(args.raw_root / "a3e", args.checkpoint_policy, args.seed)

    references = {
        "a2_nested": reference_from_rows(a2_rows, arm="nested_segment_decoder_multiprefix"),
        "a3c_warm": reference_from_rows(a3c_rows, arm="checkpoint_initialized_nested_segment_decoder_multiprefix"),
        "a3d_teacher_w03": reference_from_rows(a3d_rows, arm="teacher_preserved_nested_w03"),
        "h1_target_set": reference_from_rows(h1_rows, arm="target_set_decoder_multiprefix"),
        "h1c_row_gated": reference_from_rows(h1c_rows, arm="row_gated_dense_head_multiprefix"),
        "fixed": reference_from_rows(official_rows, prefix="fixed_"),
    }

    compare_rows = compare(a3e_rows, references)
    summary_rows = summarize(compare_rows)
    best_rows = best_epoch_rows(training_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "phase5_timealign_hss_a3e_ettm1_fixed_reference.csv", official_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_a3e_ettm1_h1_reference.csv", h1_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_a3e_ettm1_h1c_reference.csv", h1c_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_a3e_ettm1_a2_reference.csv", a2_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_a3e_ettm1_a3c_reference.csv", a3c_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_a3e_ettm1_a3d_reference.csv", a3d_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_a3e_ettm1_metrics.csv", a3e_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_a3e_ettm1_comparison.csv", compare_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_a3e_ettm1_summary.csv", summary_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_a3e_ettm1_training.csv", training_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_a3e_ettm1_best_epoch.csv", best_rows)
    (args.output_dir / "phase5_timealign_hss_a3e_ettm1_summary.json").write_text(
        json.dumps({"datasets": DATASETS, "metric_rows": len(a3e_rows), "summary_rows": summary_rows}, indent=2, sort_keys=True) + "\n"
    )
    write_report(args.output_dir / "phase5_timealign_hss_a3e_ettm1_replacement_gate_report.md", summary_rows, compare_rows)
    print(f"phase5_timealign_hss_a3e_ettm1_report={args.output_dir / 'phase5_timealign_hss_a3e_ettm1_replacement_gate_report.md'}")


if __name__ == "__main__":
    main()
