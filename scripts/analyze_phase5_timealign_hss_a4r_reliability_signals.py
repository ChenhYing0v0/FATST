from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


SIGNAL_COLUMNS = [
    "best_val_mean_mse",
    "last_val_mean_mse",
    "last_gap_to_best_val_pct",
    "last_train_prediction_l1",
    "last_train_horizon_prediction_l1",
    "last_train_reconstruction_l1",
    "last_train_alignment_loss",
    "last_train_teacher_l1",
]


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


def parse_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def pearson(xs: list[float], ys: list[float]) -> float:
    valid = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(valid) < 3:
        return float("nan")
    x_vals, y_vals = zip(*valid)
    mean_x = sum(x_vals) / len(x_vals)
    mean_y = sum(y_vals) / len(y_vals)
    num = sum((x - mean_x) * (y - mean_y) for x, y in valid)
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_vals))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_vals))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + end - 1) / 2.0 + 1.0
        for original_index, _ in indexed[index:end]:
            result[original_index] = rank
        index = end
    return result


def spearman(xs: list[float], ys: list[float]) -> float:
    valid = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(valid) < 3:
        return float("nan")
    x_vals, y_vals = zip(*valid)
    return pearson(ranks(list(x_vals)), ranks(list(y_vals)))


def load_training_signals(metrics_path: Path, target_horizon: int) -> dict[str, Any]:
    training_path = metrics_path.parent / "training_log.csv"
    rows = read_csv(training_path)
    best = min(rows, key=lambda row: float(row["val_mean_mse"]))
    last = max(rows, key=lambda row: int(row["epoch"]))
    horizon_key = f"train_prediction_h{target_horizon}_l1"
    horizon_l1 = parse_float(last, horizon_key, parse_float(last, "train_prediction_l1"))
    return {
        "training_log_path": str(training_path),
        "best_epoch": int(best["epoch"]),
        "last_epoch": int(last["epoch"]),
        "best_val_mean_mse": float(best["val_mean_mse"]),
        "last_val_mean_mse": float(last["val_mean_mse"]),
        "last_gap_to_best_val_pct": relative_pct(float(last["val_mean_mse"]), float(best["val_mean_mse"])),
        "last_train_prediction_l1": parse_float(last, "train_prediction_l1"),
        "last_train_horizon_prediction_l1": horizon_l1,
        "last_train_reconstruction_l1": parse_float(last, "train_reconstruction_l1"),
        "last_train_alignment_loss": parse_float(last, "train_alignment_loss"),
        "last_train_teacher_l1": parse_float(last, "train_teacher_l1"),
    }


def build_signal_rows(all_paths_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(all_paths_path):
        target_horizon = int(row["target_horizon"])
        metrics_path = Path(row["source_path"])
        signals = load_training_signals(metrics_path, target_horizon)
        rows.append(
            {
                "dataset": row["dataset"],
                "target_horizon": target_horizon,
                "path_id": row["path_id"],
                "family": row["family"],
                "mse": float(row["mse"]),
                "relative_vs_setting_best_pct": float(row["relative_vs_setting_best_pct"]),
                "relative_vs_fixed_pct": float(row["relative_vs_fixed_pct"]),
                "relative_vs_h1_pct": float(row["relative_vs_h1_pct"]),
                "is_best": row["is_best"],
                "within_0p2pct_of_best": row["within_0p2pct_of_best"],
                **signals,
            }
        )
    return rows


def correlation_rows(signal_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in signal_rows}) + ["ALL"]:
        subset = signal_rows if dataset == "ALL" else [row for row in signal_rows if row["dataset"] == dataset]
        y = [float(row["relative_vs_setting_best_pct"]) for row in subset]
        for signal in SIGNAL_COLUMNS:
            x = [float(row[signal]) for row in subset]
            rows.append(
                {
                    "dataset": dataset,
                    "signal": signal,
                    "settings": len(subset),
                    "pearson_corr_with_gap_to_best": pearson(x, y),
                    "spearman_corr_with_gap_to_best": spearman(x, y),
                }
            )
    return rows


def best_signal_rows(correlations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in correlations}):
        subset = [row for row in correlations if row["dataset"] == dataset]
        ranked = sorted(
            subset,
            key=lambda row: abs(float(row["spearman_corr_with_gap_to_best"]))
            if math.isfinite(float(row["spearman_corr_with_gap_to_best"]))
            else -1.0,
            reverse=True,
        )
        rows.extend(ranked[:3])
    return rows


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.3f}"


def write_report(path: Path, correlations: list[dict[str, Any]], best_signals: list[dict[str, Any]]) -> None:
    all_rows = [row for row in correlations if row["dataset"] == "ALL"]
    all_ranked = sorted(
        all_rows,
        key=lambda row: abs(float(row["spearman_corr_with_gap_to_best"]))
        if math.isfinite(float(row["spearman_corr_with_gap_to_best"]))
        else -1.0,
        reverse=True,
    )
    lines = [
        "# Phase5 A4R Reliability Signal Diagnostic",
        "",
        "## 诊断目标",
        "",
        "[Step 3/4] A4 已证明 offline best path 分散，但这仍可能只是 test-oracle 现象。",
        "A4R 使用现有 training logs 中可观测的 validation/training signals，检查它们是否能解释 path 的 `gap_to_best`。",
        "",
        "## ALL-Level Signal Ranking",
        "",
        "| rank | signal | pearson | spearman |",
        "| --- | --- | --- | --- |",
    ]
    for idx, row in enumerate(all_ranked, start=1):
        lines.append(
            f"| {idx} | `{row['signal']}` | {fmt(float(row['pearson_corr_with_gap_to_best']))} | {fmt(float(row['spearman_corr_with_gap_to_best']))} |"
        )

    lines.extend(
        [
            "",
            "## Dataset-Level Top Signals",
            "",
            "| dataset | signal | pearson | spearman |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in best_signals:
        lines.append(
            f"| {row['dataset']} | `{row['signal']}` | {fmt(float(row['pearson_corr_with_gap_to_best']))} | {fmt(float(row['spearman_corr_with_gap_to_best']))} |"
        )

    strongest = all_ranked[0]
    strongest_abs = abs(float(strongest["spearman_corr_with_gap_to_best"]))
    decision = (
        "现有日志信号具备继续挖掘价值，但还不足以直接作为 routing method。"
        if strongest_abs >= 0.45
        else "现有日志信号较弱，不足以支撑 learned routing；需要设计更明确的 reliability signal export。"
    )
    lines.extend(
        [
            "",
            "## 机制判断",
            "",
            f"- [Fact] ALL-level 最强 signal 是 `{strongest['signal']}`，Spearman 相关为 `{fmt(float(strongest['spearman_corr_with_gap_to_best']))}`。",
            f"- [Decision] {decision}",
            "- [Limit] 当前 signals 主要来自 run-level training log；除 `train_prediction_h{H}_l1` 外，大多不是 horizon-specific，因此不能解释所有 per-horizon path choice。",
            "- [Next] 若继续 Stage A，应新增轻量诊断导出：teacher-student disagreement by prefix、validation prefix residual、prefix-wise validation MSE，而不是直接启动 routing head。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a4-root",
        type=Path,
        default=Path("analysis/phase5_timealign_hss_a4_interface_reliability_diagnostic_20260701"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis/phase5_timealign_hss_a4r_reliability_signal_diagnostic_20260701"),
    )
    args = parser.parse_args()

    signal_rows = build_signal_rows(args.a4_root / "phase5_timealign_hss_a4_all_paths.csv")
    correlations = correlation_rows(signal_rows)
    best_signals = best_signal_rows(correlations)

    write_csv(args.output_root / "phase5_timealign_hss_a4r_signal_rows.csv", signal_rows)
    write_csv(args.output_root / "phase5_timealign_hss_a4r_signal_correlations.csv", correlations)
    write_csv(args.output_root / "phase5_timealign_hss_a4r_best_signal_by_dataset.csv", best_signals)
    write_report(
        args.output_root / "phase5_timealign_hss_a4r_reliability_signal_diagnostic.md",
        correlations,
        best_signals,
    )


if __name__ == "__main__":
    main()
