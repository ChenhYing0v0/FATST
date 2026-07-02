from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


SIGNAL_COLUMNS = [
    "validation_prefix_mse",
    "validation_prefix_mae",
    "full_context_prefix_mse",
    "prefix_vs_full_mse",
    "teacher_student_mse",
    "teacher_student_mae",
    "residual_abs_mean",
    "residual_std",
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


def parse_float(value: str) -> float:
    if value == "" or value.lower() == "nan":
        return float("nan")
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
    output = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + end - 1) / 2.0 + 1.0
        for original_index, _value in indexed[index:end]:
            output[original_index] = rank
        index = end
    return output


def spearman(xs: list[float], ys: list[float]) -> float:
    valid = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(valid) < 3:
        return float("nan")
    x_vals, y_vals = zip(*valid)
    return pearson(ranks(list(x_vals)), ranks(list(y_vals)))


def load_a4_gap_rows(a4_root: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in read_csv(a4_root / "phase5_timealign_hss_a4_all_paths.csv"):
        path_id = row["path_id"]
        if path_id == "fixed_specialist":
            continue
        rows[(row["dataset"], int(row["target_horizon"]), path_id)] = {
            "relative_vs_setting_best_pct": float(row["relative_vs_setting_best_pct"]),
            "relative_vs_fixed_pct": float(row["relative_vs_fixed_pct"]),
            "relative_vs_h1_pct": float(row["relative_vs_h1_pct"]),
            "is_best": row["is_best"],
            "within_0p2pct_of_best": row["within_0p2pct_of_best"],
            "mse": float(row["mse"]),
            "family": row["family"],
        }
    return rows


def load_signal_rows(raw_root: Path, checkpoint_policy: str, seed: int, gap_rows: dict[tuple[str, int, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((raw_root / checkpoint_policy).glob(f"*/**/seed{seed}/validation_prefix_diagnostics.csv")):
        path_id = path.relative_to(raw_root / checkpoint_policy).parts[0]
        for row in read_csv(path):
            dataset = row["dataset"]
            horizon = int(row["target_horizon"])
            gap = gap_rows.get((dataset, horizon, path_id))
            if gap is None:
                continue
            out: dict[str, Any] = {
                "dataset": dataset,
                "target_horizon": horizon,
                "path_id": path_id,
                "family": gap["family"],
                "relative_vs_setting_best_pct": gap["relative_vs_setting_best_pct"],
                "relative_vs_fixed_pct": gap["relative_vs_fixed_pct"],
                "relative_vs_h1_pct": gap["relative_vs_h1_pct"],
                "is_best": gap["is_best"],
                "within_0p2pct_of_best": gap["within_0p2pct_of_best"],
                "test_mse": gap["mse"],
                "source_path": str(path),
            }
            for signal in SIGNAL_COLUMNS:
                out[signal] = parse_float(row.get(signal, "nan"))
            rows.append(out)
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
                    "settings": len([value for value in x if math.isfinite(value)]),
                    "pearson_corr_with_gap_to_best": pearson(x, y),
                    "spearman_corr_with_gap_to_best": spearman(x, y),
                }
            )
    return rows


def top_rows(correlations: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def write_report(path: Path, correlations: list[dict[str, Any]], best_signals: list[dict[str, Any]], signal_rows: list[dict[str, Any]]) -> None:
    all_rows = [row for row in correlations if row["dataset"] == "ALL"]
    all_ranked = sorted(
        all_rows,
        key=lambda row: abs(float(row["spearman_corr_with_gap_to_best"]))
        if math.isfinite(float(row["spearman_corr_with_gap_to_best"]))
        else -1.0,
        reverse=True,
    )
    strongest = all_ranked[0] if all_ranked else None
    lines = [
        "# Phase5 A4S Validation Prefix Signal Export",
        "",
        "## 诊断目标",
        "",
        "A4S 检查 prefix-wise validation diagnostics 是否比 A4R 的 run-level logs 更能解释 `gap_to_best`。",
        "本报告仍是 diagnostic-only，不是 routing method。",
        "",
        f"Rows analyzed: `{len(signal_rows)}`.",
        "",
        "## ALL-Level Signal Ranking",
        "",
        "| rank | signal | valid_settings | pearson | spearman |",
        "| --- | --- | --- | --- | --- |",
    ]
    for idx, row in enumerate(all_ranked, start=1):
        lines.append(
            "| {rank} | `{signal}` | {settings} | {pearson} | {spearman} |".format(
                rank=idx,
                signal=row["signal"],
                settings=row["settings"],
                pearson=fmt(float(row["pearson_corr_with_gap_to_best"])),
                spearman=fmt(float(row["spearman_corr_with_gap_to_best"])),
            )
        )

    lines.extend(
        [
            "",
            "## Dataset-Level Top Signals",
            "",
            "| dataset | signal | valid_settings | pearson | spearman |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in best_signals:
        lines.append(
            "| {dataset} | `{signal}` | {settings} | {pearson} | {spearman} |".format(
                dataset=row["dataset"],
                signal=row["signal"],
                settings=row["settings"],
                pearson=fmt(float(row["pearson_corr_with_gap_to_best"])),
                spearman=fmt(float(row["spearman_corr_with_gap_to_best"])),
            )
        )

    if strongest is None:
        decision = "未找到有效 signal rows；需要检查远程导出是否完整。"
        strongest_line = "- [Fact] no signal rows."
    else:
        strongest_abs = abs(float(strongest["spearman_corr_with_gap_to_best"]))
        decision = (
            "A4S 通过 signal-existence gate，可进入 routing method 的 narrative-gate 设计。"
            if strongest_abs >= 0.55
            else "A4S 未通过 signal-existence gate；不能进入 learned routing，应回 Step 2/3 重审 Stage A contribution。"
        )
        strongest_line = (
            f"- [Fact] ALL-level 最强 signal 是 `{strongest['signal']}`，"
            f"Spearman 为 `{fmt(float(strongest['spearman_corr_with_gap_to_best']))}`。"
        )

    lines.extend(
        [
            "",
            "## Gate Decision",
            "",
            strongest_line,
            f"- [Decision] {decision}",
            "- [Limit] 本诊断只覆盖 unified interface paths，不覆盖 fixed specialist；fixed 仍是 problem evidence，不是 routing 候选。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--a4-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-policy", default="official-last")
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()

    gap_rows = load_a4_gap_rows(args.a4_root)
    signal_rows = load_signal_rows(args.raw_root, args.checkpoint_policy, args.seed, gap_rows)
    correlations = correlation_rows(signal_rows)
    best_signals = top_rows(correlations)

    write_csv(args.output_dir / "phase5_timealign_hss_a4s_signal_rows.csv", signal_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_a4s_signal_correlations.csv", correlations)
    write_csv(args.output_dir / "phase5_timealign_hss_a4s_best_signal_by_dataset.csv", best_signals)
    write_report(
        args.output_dir / "phase5_timealign_hss_a4s_validation_prefix_signal_export.md",
        correlations,
        best_signals,
        signal_rows,
    )


if __name__ == "__main__":
    main()
