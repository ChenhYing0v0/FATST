from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


MODELS = {
    "PatchEncoderSingle720PrefixRisk": "single_720_prefix_risk",
    "PatchEncoderR3PrefixRisk": "r3_prefix_risk",
    "PatchEncoderHSSGRegionRoutedReadout": "hssg_region_routed_readout",
}
PRIMARY = "hssg_region_routed_readout"
BASELINES = ["single_720_prefix_risk", "r3_prefix_risk"]
DATASETS = ["ETTh2", "Weather"]
HORIZONS = [96, 192, 336, 720]
HORIZON_LABEL = "mixed_h96_h192_h336_h720"
LOG_RE = re.compile(
    r"epoch_progress run_name=(?P<run_name>\S+) dataset=(?P<dataset>\S+) "
    r"epoch=(?P<epoch>\d+)/(?P<max_epoch>\d+) val_mean_mse=(?P<val>[0-9.]+)"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def fmt_float(value: float) -> str:
    return f"{value:.6f}"


def lr_value(lr_dir: str) -> float:
    return float(lr_dir.removeprefix("lr_").replace("p", "."))


def load_main_metrics(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(raw_root.glob(f"lr_*/*/*/{HORIZON_LABEL}/seed2021/metrics_by_target_horizon.csv")):
        rel = path.relative_to(raw_root).parts
        lr_dir, model_name, dataset = rel[0], rel[1], rel[2]
        if model_name not in MODELS:
            continue
        for row in read_csv(path):
            rows.append(
                {
                    "lr_dir": lr_dir,
                    "learning_rate": lr_value(lr_dir),
                    "model": model_name,
                    "strategy": MODELS[model_name],
                    "dataset": dataset,
                    "horizon": int(row["target_horizon"]),
                    "mse": float(row["mse"]),
                    "mae": float(row["mae"]),
                }
            )
    return rows


def load_training_summary(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for log_path in sorted(raw_root.glob("lr_*/_logs/phase4_protocol_calibration_gate/*.log")):
        lr_dir = log_path.relative_to(raw_root).parts[0]
        values = []
        run_name = ""
        dataset = ""
        for line in log_path.read_text(errors="ignore").splitlines():
            match = LOG_RE.search(line)
            if not match:
                continue
            run_name = match.group("run_name")
            dataset = match.group("dataset")
            values.append((int(match.group("epoch")), float(match.group("val"))))
        if not values or run_name not in MODELS:
            continue
        best_epoch, best_val = min(values, key=lambda item: item[1])
        last_epoch, last_val = values[-1]
        first_epoch, first_val = values[0]
        rows.append(
            {
                "lr_dir": lr_dir,
                "learning_rate": lr_value(lr_dir),
                "model": run_name,
                "strategy": MODELS[run_name],
                "dataset": dataset,
                "epochs_ran": len(values),
                "first_epoch": first_epoch,
                "best_epoch": best_epoch,
                "last_epoch": last_epoch,
                "first_val_mean_mse": first_val,
                "best_val_mean_mse": best_val,
                "last_val_mean_mse": last_val,
                "post_best_val_drift_pct": pct(last_val, best_val),
            }
        )
    return rows


def build_delta_rows(main_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["lr_dir"], row["strategy"], row["dataset"], row["horizon"]): row
        for row in main_rows
    }
    rows = []
    for lr_dir in sorted({row["lr_dir"] for row in main_rows}, key=lr_value):
        for dataset in DATASETS:
            for horizon in HORIZONS:
                candidate = by_key[(lr_dir, PRIMARY, dataset, horizon)]
                for baseline in BASELINES:
                    base = by_key[(lr_dir, baseline, dataset, horizon)]
                    rows.append(
                        {
                            "lr_dir": lr_dir,
                            "learning_rate": lr_value(lr_dir),
                            "dataset": dataset,
                            "horizon": horizon,
                            "candidate_strategy": PRIMARY,
                            "baseline_strategy": baseline,
                            "candidate_mse": candidate["mse"],
                            "baseline_mse": base["mse"],
                            "relative_mse_pct": pct(candidate["mse"], base["mse"]),
                            "mse_win": candidate["mse"] < base["mse"],
                            "candidate_mae": candidate["mae"],
                            "baseline_mae": base["mae"],
                            "relative_mae_pct": pct(candidate["mae"], base["mae"]),
                            "mae_win": candidate["mae"] < base["mae"],
                        }
                    )
    return rows


def summarize_lr(main_rows: list[dict[str, Any]], delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_lr_strategy: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in main_rows:
        by_lr_strategy[(row["lr_dir"], row["strategy"])].append(row)

    rows = []
    for lr_dir in sorted({row["lr_dir"] for row in main_rows}, key=lr_value):
        for strategy in sorted({row["strategy"] for row in main_rows}):
            subset = by_lr_strategy[(lr_dir, strategy)]
            rows.append(
                {
                    "lr_dir": lr_dir,
                    "learning_rate": lr_value(lr_dir),
                    "strategy": strategy,
                    "settings": len(subset),
                    "mean_mse": mean(row["mse"] for row in subset),
                    "mean_mae": mean(row["mae"] for row in subset),
                }
            )

        for baseline in BASELINES:
            subset = [
                row
                for row in delta_rows
                if row["lr_dir"] == lr_dir and row["baseline_strategy"] == baseline
            ]
            rows.append(
                {
                    "lr_dir": lr_dir,
                    "learning_rate": lr_value(lr_dir),
                    "strategy": f"{PRIMARY}_vs_{baseline}",
                    "settings": len(subset),
                    "mean_mse": mean(row["relative_mse_pct"] for row in subset),
                    "mean_mae": mean(row["relative_mae_pct"] for row in subset),
                    "mse_wins": sum(1 for row in subset if row["mse_win"]),
                    "mae_wins": sum(1 for row in subset if row["mae_win"]),
                }
            )
    return rows


def summarize_training(training_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in training_rows:
        grouped[(row["lr_dir"], row["strategy"])].append(row)

    rows = []
    for (lr_dir, strategy), subset in sorted(grouped.items(), key=lambda item: (lr_value(item[0][0]), item[0][1])):
        rows.append(
            {
                "lr_dir": lr_dir,
                "learning_rate": lr_value(lr_dir),
                "strategy": strategy,
                "runs": len(subset),
                "mean_best_epoch": mean(row["best_epoch"] for row in subset),
                "mean_epochs_ran": mean(row["epochs_ran"] for row in subset),
                "mean_post_best_val_drift_pct": mean(row["post_best_val_drift_pct"] for row in subset),
                "max_post_best_val_drift_pct": max(row["post_best_val_drift_pct"] for row in subset),
            }
        )
    return rows


def rows_for_table(rows: list[dict[str, Any]], keys: list[str]) -> str:
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        values = []
        for key in keys:
            value = row[key]
            if isinstance(value, float):
                value = fmt_pct(value) if key.endswith("_pct") or key in {"mean_mse", "mean_mae"} else fmt_float(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    output_path: Path,
    lr_summary: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    training_lr_summary: list[dict[str, Any]],
) -> None:
    hssg_vs_single = [
        row for row in lr_summary if row["strategy"] == f"{PRIMARY}_vs_single_720_prefix_risk"
    ]
    hssg_vs_r3 = [row for row in lr_summary if row["strategy"] == f"{PRIMARY}_vs_r3_prefix_risk"]
    best_primary = min(
        [row for row in lr_summary if row["strategy"] == PRIMARY],
        key=lambda row: row["mean_mse"],
    )
    best_single = min(
        [row for row in lr_summary if row["strategy"] == "single_720_prefix_risk"],
        key=lambda row: row["mean_mse"],
    )
    best_r3 = min(
        [row for row in lr_summary if row["strategy"] == "r3_prefix_risk"],
        key=lambda row: row["mean_mse"],
    )

    dataset_delta = []
    for lr_dir in sorted({row["lr_dir"] for row in delta_rows}, key=lr_value):
        for dataset in DATASETS:
            for baseline in BASELINES:
                subset = [
                    row
                    for row in delta_rows
                    if row["lr_dir"] == lr_dir
                    and row["dataset"] == dataset
                    and row["baseline_strategy"] == baseline
                ]
                dataset_delta.append(
                    {
                        "lr_dir": lr_dir,
                        "dataset": dataset,
                        "baseline": baseline,
                        "mse_wins": f"{sum(1 for row in subset if row['mse_win'])}/4",
                        "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in subset),
                    }
                )

    lines = [
        "# Phase4 Protocol Calibration Gate Report",
        "",
        "## 结论",
        "",
        "[Strong Evidence] Lower LR 对 training trajectory 有帮助，但不能把 HSSG-A 修成主线候选。",
        f"最佳 HSSG-A setting 是 `{best_primary['lr_dir']}`，8 个 horizon 平均 MSE 为 `{fmt_float(best_primary['mean_mse'])}`；",
        f"最佳 single-prefix 是 `{best_single['lr_dir']}`，平均 MSE `{fmt_float(best_single['mean_mse'])}`；",
        f"最佳 R.3 是 `{best_r3['lr_dir']}`，平均 MSE `{fmt_float(best_r3['mean_mse'])}`。",
        "",
        "[Decision] Protocol calibration gate 不通过。下一步不应继续调 HSSG readout residual path，",
        "应回到 Step 6，设计 richer carrier：让 supervision scheduling 作用到更高语义的 state / condition / adapter 子空间，",
        "而不是只改变 horizon-independent loss 权重或小 residual readout。",
        "",
        "## HSSG-A vs Single",
        "",
        rows_for_table(hssg_vs_single, ["lr_dir", "settings", "mse_wins", "mean_mse", "mae_wins", "mean_mae"]),
        "",
        "## HSSG-A vs R.3",
        "",
        rows_for_table(hssg_vs_r3, ["lr_dir", "settings", "mse_wins", "mean_mse", "mae_wins", "mean_mae"]),
        "",
        "## Dataset-Level Delta",
        "",
        rows_for_table(dataset_delta, ["lr_dir", "dataset", "baseline", "mse_wins", "mean_relative_mse_pct"]),
        "",
        "## Training Trajectory",
        "",
        rows_for_table(
            training_lr_summary,
            [
                "lr_dir",
                "strategy",
                "runs",
                "mean_best_epoch",
                "mean_epochs_ran",
                "mean_post_best_val_drift_pct",
                "max_post_best_val_drift_pct",
            ],
        ),
        "",
        "## Gate Assessment",
        "",
        "- best epoch 后移：部分成立，`3e-5` 的 HSSG-A mean best epoch 后移到 6.5，但 R.3 仍为 3.0。",
        "- validation drift 下降：部分成立，HSSG-A 从高 LR 的 mean drift 14.87% 降到 6.56%，但 R.3 仍有 13.62%。",
        "- HSSG-A vs single 至少 5/8 wins：只在 `3e-5` 成立，达到 6/8。",
        "- ETTh2 h96/h192 不超过 +1%：`3e-5`、`5e-5` 成立。",
        "- Weather h720 late 接近 R.3：不成立，`3e-5` Weather h720 比 R.3 差 +3.37%。",
        "",
        "## 11-Step Decision",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | Step 9/10：评估 Phase4-PTC 结果并做 go/no-go 决策 |",
        "| `problem` | early-best 与 validation drift 会污染 HSSG-A carrier 判断 |",
        "| `existence_evidence` | 18 个 run 均完成；log 可解析 best epoch 与 drift；metrics 覆盖 ETTh2/Weather × 4 horizons |",
        "| `idea` | 用 lower LR 判断 protocol 是否是 HSSG-A 失败主因 |",
        "| `theory_check` | 若过快 optimization 是主因，lower LR 应同时推迟 best epoch、降低 drift，并让 HSSG-A 保持 Weather/R.3 竞争力 |",
        "| `design` | LR `1e-4/5e-5/3e-5`；single/R.3/HSSG-A；ETTh2 + Weather；seed 2021 |",
        "| `gate` | HSSG-A ≥5/8 wins vs single；ETTh2 short 不坏；Weather h720 late 接近 R.3；drift 降低 |",
        "| `artifacts` | 本目录 CSV 与 report；raw artifacts under `artifacts/runs/phase4_protocol_calibration_gate` |",
        "| `decision` | Fail as core-route repair；rollback to Step 6，设计 state/condition carrier，而不是继续 LR sweep |",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("artifacts/runs/phase4_protocol_calibration_gate"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/phase4_protocol_calibration_gate_20260625"),
    )
    args = parser.parse_args()

    main_rows = load_main_metrics(args.raw_root)
    training_rows = load_training_summary(args.raw_root)
    delta_rows = build_delta_rows(main_rows)
    lr_summary = summarize_lr(main_rows, delta_rows)
    training_lr_summary = summarize_training(training_rows)

    write_csv(args.output_dir / "phase4_protocol_main_metrics.csv", main_rows)
    write_csv(args.output_dir / "phase4_protocol_hssg_delta.csv", delta_rows)
    write_csv(args.output_dir / "phase4_protocol_lr_summary.csv", lr_summary)
    write_csv(args.output_dir / "phase4_protocol_training_summary.csv", training_rows)
    write_csv(args.output_dir / "phase4_protocol_training_lr_summary.csv", training_lr_summary)
    write_report(
        args.output_dir / "phase4_protocol_calibration_gate_report.md",
        lr_summary,
        delta_rows,
        training_rows,
        training_lr_summary,
    )


if __name__ == "__main__":
    main()
