from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


MODEL_ORDER = [
    "PatchEncoderConditionedFutureUnitScheduling",
    "PatchEncoderFullTimeMSE720",
    "PatchEncoderR3PrefixRisk",
]

MODEL_LABELS = {
    "PatchEncoderConditionedFutureUnitScheduling": "S1_conditioned_future_unit_scheduling",
    "PatchEncoderFullTimeMSE720": "D0_full_time_mse",
    "PatchEncoderR3PrefixRisk": "D1_r3_prefix_risk",
}

DATASETS = ["ETTh2", "Weather"]
HORIZONS = [96, 192, 336, 720]
PRIMARY_MODEL = "PatchEncoderConditionedFutureUnitScheduling"
BASELINES = ["PatchEncoderFullTimeMSE720", "PatchEncoderR3PrefixRisk"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_dir(raw_root: Path, model: str, dataset: str) -> Path:
    return raw_root / model / dataset / "mixed_h96_h192_h336_h720" / "seed2021"


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def load_main_metrics(raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for dataset in DATASETS:
            path = run_dir(raw_root, model, dataset) / "metrics_by_target_horizon.csv"
            for row in read_csv(path):
                rows.append(
                    {
                        "model": model,
                        "strategy": MODEL_LABELS[model],
                        "dataset": dataset,
                        "horizon": int(row["target_horizon"]),
                        "mse": float(row["mse"]),
                        "mae": float(row["mae"]),
                    }
                )
    return rows


def add_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["model"], row["dataset"], row["horizon"]): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    for baseline in BASELINES:
        for row in rows:
            if row["model"] != PRIMARY_MODEL:
                continue
            base = by_key[(baseline, row["dataset"], row["horizon"])]
            output.append(
                {
                    **row,
                    "baseline_model": baseline,
                    "baseline_strategy": MODEL_LABELS[baseline],
                    "baseline_mse": base["mse"],
                    "baseline_mae": base["mae"],
                    "relative_mse_pct": pct(row["mse"], base["mse"]),
                    "relative_mae_pct": pct(row["mae"], base["mae"]),
                    "mse_win": row["mse"] < base["mse"],
                    "mae_win": row["mae"] < base["mae"],
                }
            )
    return output


def summarize_deltas(delta_rows: list[dict[str, Any]], key: str | None = None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for baseline in BASELINES:
        baseline_rows = [row for row in delta_rows if row["baseline_model"] == baseline]
        values = ["all"] if key is None else sorted({row[key] for row in baseline_rows})
        for value in values:
            subset = baseline_rows if key is None else [row for row in baseline_rows if row[key] == value]
            row: dict[str, Any] = {
                "strategy": MODEL_LABELS[PRIMARY_MODEL],
                "baseline_strategy": MODEL_LABELS[baseline],
                "settings": len(subset),
                "mse_wins": sum(1 for item in subset if item["mse_win"]),
                "mae_wins": sum(1 for item in subset if item["mae_win"]),
                "mean_relative_mse_pct": mean(item["relative_mse_pct"] for item in subset),
                "mean_relative_mae_pct": mean(item["relative_mae_pct"] for item in subset),
            }
            if key is not None:
                row[key] = value
            output.append(row)
    return output


def parse_segment(segment: str) -> tuple[int, int]:
    start, end = segment.split("-", maxsplit=1)
    return int(start), int(end)


def future_region(segment_end: int) -> str:
    if segment_end <= 96:
        return "early_1_96"
    if segment_end <= 336:
        return "middle_97_336"
    return "late_337_720"


def load_segment_metrics(raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                path = run_dir(raw_root, model, dataset) / f"h{horizon}" / "metrics_by_segment.csv"
                for item in read_csv(path):
                    start, end = parse_segment(item["segment"])
                    rows.append(
                        {
                            "model": model,
                            "strategy": MODEL_LABELS[model],
                            "dataset": dataset,
                            "horizon": horizon,
                            "segment": item["segment"],
                            "segment_start": start,
                            "segment_end": end,
                            "future_region": future_region(end),
                            "mse": float(item["mse"]),
                            "mae": float(item["mae"]),
                        }
                    )
    return rows


def add_segment_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["model"], row["dataset"], row["horizon"], row["segment"]): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    for baseline in BASELINES:
        for row in rows:
            if row["model"] != PRIMARY_MODEL:
                continue
            base = by_key[(baseline, row["dataset"], row["horizon"], row["segment"])]
            output.append(
                {
                    **row,
                    "baseline_model": baseline,
                    "baseline_strategy": MODEL_LABELS[baseline],
                    "baseline_mse": base["mse"],
                    "baseline_mae": base["mae"],
                    "relative_mse_pct": pct(row["mse"], base["mse"]),
                    "relative_mae_pct": pct(row["mae"], base["mae"]),
                    "mse_win": row["mse"] < base["mse"],
                    "mae_win": row["mae"] < base["mae"],
                }
            )
    return output


def summarize_segment_deltas(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for baseline in BASELINES:
        baseline_rows = [row for row in rows if row["baseline_model"] == baseline]
        for value in sorted({row[key] for row in baseline_rows}):
            subset = [row for row in baseline_rows if row[key] == value]
            output.append(
                {
                    "strategy": MODEL_LABELS[PRIMARY_MODEL],
                    "baseline_strategy": MODEL_LABELS[baseline],
                    key: value,
                    "segments": len(subset),
                    "mse_wins": sum(1 for item in subset if item["mse_win"]),
                    "mean_relative_mse_pct": mean(item["relative_mse_pct"] for item in subset),
                    "mean_relative_mae_pct": mean(item["relative_mae_pct"] for item in subset),
                }
            )
    return output


def load_trace_summary(raw_root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for dataset in DATASETS:
            base_dir = run_dir(raw_root, model, dataset)
            config = json.loads((base_dir / "effective_config.json").read_text())
            trace = read_csv(base_dir / "supervision_trace.csv")
            log = read_csv(base_dir / "training_log.csv")
            by_unit: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in trace:
                by_unit[row["unit_type"]].append(row)
            for unit_type, unit_rows in sorted(by_unit.items()):
                output.append(
                    {
                        "model": model,
                        "strategy": MODEL_LABELS[model],
                        "dataset": dataset,
                        "training_evaluation_decoupled": config.get("training_evaluation_decoupled", False),
                        "train_horizons_effective": ",".join(
                            str(item) for item in config.get("train_horizons_effective", [])
                        ),
                        "step_loss_weighting": config.get("step_loss_weighting", ""),
                        "unit_type": unit_type,
                        "epochs_ran": len(log),
                        "trace_rows": len(unit_rows),
                        "mean_active_steps": mean(float(row["active_steps"]) for row in unit_rows),
                        "mean_mask_ratio": mean(float(row["mask_ratio"]) for row in unit_rows),
                        "condition_types": ";".join(sorted({row.get("condition_type", "none") for row in unit_rows})),
                        "mean_condition_top_blocks": mean(
                            float(row.get("condition_top_blocks", 0) or 0) for row in unit_rows
                        ),
                        "mean_condition_score": mean(
                            float(row.get("condition_mean_score", 0) or 0) for row in unit_rows
                        ),
                        "mean_auxiliary_weight": mean(
                            float(row.get("auxiliary_weight", 0) or 0) for row in unit_rows
                        ),
                        "mean_time_loss": mean(float(row["loss_time"]) for row in unit_rows),
                        "mean_unit_loss": mean(float(row["loss_unit"]) for row in unit_rows),
                        "mean_total_loss": mean(float(row["loss_total"]) for row in unit_rows),
                    }
                )
    return output


def load_prefix_summary(raw_root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for dataset in DATASETS:
            rows = read_csv(run_dir(raw_root, model, dataset) / "prefix_consistency.csv")
            output.append(
                {
                    "model": model,
                    "strategy": MODEL_LABELS[model],
                    "dataset": dataset,
                    "rows": len(rows),
                    "max_prefix_mismatch_mse": max(float(row["prefix_mismatch_mse"]) for row in rows),
                    "mean_prefix_mismatch_mse": mean(float(row["prefix_mismatch_mse"]) for row in rows),
                }
            )
    return output


def format_pct(value: float) -> str:
    return f"{value:+.2f}%"


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                if "relative" in column:
                    values.append(format_pct(value))
                else:
                    values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    output_dir: Path,
    main_delta: list[dict[str, Any]],
    overall_summary: list[dict[str, Any]],
    dataset_summary: list[dict[str, Any]],
    horizon_summary: list[dict[str, Any]],
    segment_region_summary: list[dict[str, Any]],
    trace_summary: list[dict[str, Any]],
    prefix_summary: list[dict[str, Any]],
) -> None:
    vs_full = next(row for row in overall_summary if row["baseline_strategy"] == "D0_full_time_mse")
    vs_r3 = next(row for row in overall_summary if row["baseline_strategy"] == "D1_r3_prefix_risk")
    etth2_vs_r3 = next(
        row for row in dataset_summary
        if row["baseline_strategy"] == "D1_r3_prefix_risk" and row["dataset"] == "ETTh2"
    )
    weather_vs_r3 = next(
        row for row in dataset_summary
        if row["baseline_strategy"] == "D1_r3_prefix_risk" and row["dataset"] == "Weather"
    )
    lines = [
        "# Phase4-S CFUS Small Gate 决策报告",
        "",
        "## 11-step 记录",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | Step 9-11 |",
        "| `problem` | conditioned future-unit scheduling 是否能成为独立 HSS training strategy |",
        "| `existence_evidence` | ETTh2/Weather small remote gate, full-time and R.3 controls |",
        "| `idea` | full future dense anchor + train-side label-novelty sparse unit pressure |",
        "| `theory_check` | 若 condition 能定向补强 hard units，应优于 full-time 且缩小 R.3 gap，不应造成 h96/Weather collapse |",
        "| `design` | `conditioned_future_unit_scheduling` vs `full_time_mse` vs `r3_prefix_risk` on ETTh2/Weather |",
        "| `gate` | beat full-time, close R.3 gap, no h96/Weather collapse, trace confirms conditioned units |",
        "| `artifacts` | `analysis/phase4_s_cfus_gate_20260624` |",
        "| `decision` | fail as paper-core; pass only as weak evidence that conditioned auxiliary improves full-time anchor |",
        "",
        "## 主要结果",
        "",
        f"[Fact] CFUS vs `full_time_mse`: mean relative MSE {format_pct(vs_full['mean_relative_mse_pct'])}, MSE wins `{vs_full['mse_wins']}/{vs_full['settings']}`。",
        f"[Fact] CFUS vs `r3_prefix_risk`: mean relative MSE {format_pct(vs_r3['mean_relative_mse_pct'])}, MSE wins `{vs_r3['mse_wins']}/{vs_r3['settings']}`。",
        f"[Fact] CFUS vs R.3 on ETTh2: mean relative MSE {format_pct(etth2_vs_r3['mean_relative_mse_pct'])}, wins `{etth2_vs_r3['mse_wins']}/{etth2_vs_r3['settings']}`。",
        f"[Fact] CFUS vs R.3 on Weather: mean relative MSE {format_pct(weather_vs_r3['mean_relative_mse_pct'])}, wins `{weather_vs_r3['mse_wins']}/{weather_vs_r3['settings']}`。",
        "",
        "[Decision] 当前 CFUS 不通过 paper-core gate。它证明 conditioned sparse auxiliary 明显优于 plain full-time dense MSE，尤其 ETTh2；但仍未形成能替代或接近 R.3 的稳定 training strategy，Weather 相对 R.3 全面退化。",
        "",
        "## Main Metrics vs Controls",
        "",
        *markdown_table(
            main_delta,
            [
                "dataset",
                "horizon",
                "baseline_strategy",
                "mse",
                "baseline_mse",
                "relative_mse_pct",
                "mae",
                "baseline_mae",
                "relative_mae_pct",
                "mse_win",
            ],
        ),
        "",
        "## Overall Summary",
        "",
        *markdown_table(
            overall_summary,
            [
                "baseline_strategy",
                "settings",
                "mse_wins",
                "mae_wins",
                "mean_relative_mse_pct",
                "mean_relative_mae_pct",
            ],
        ),
        "",
        "## Dataset Summary",
        "",
        *markdown_table(
            dataset_summary,
            ["baseline_strategy", "dataset", "settings", "mse_wins", "mean_relative_mse_pct"],
        ),
        "",
        "## Horizon Summary",
        "",
        *markdown_table(
            horizon_summary,
            ["baseline_strategy", "horizon", "settings", "mse_wins", "mean_relative_mse_pct"],
        ),
        "",
        "## Segment Future-Region Summary",
        "",
        *markdown_table(
            segment_region_summary,
            ["baseline_strategy", "future_region", "segments", "mse_wins", "mean_relative_mse_pct"],
        ),
        "",
        "## Trace Summary",
        "",
        *markdown_table(
            trace_summary,
            [
                "strategy",
                "dataset",
                "training_evaluation_decoupled",
                "train_horizons_effective",
                "step_loss_weighting",
                "unit_type",
                "epochs_ran",
                "mean_mask_ratio",
                "condition_types",
                "mean_condition_top_blocks",
                "mean_auxiliary_weight",
            ],
        ),
        "",
        "## Prefix Consistency",
        "",
        *markdown_table(
            prefix_summary,
            ["strategy", "dataset", "rows", "max_prefix_mismatch_mse", "mean_prefix_mismatch_mse"],
        ),
        "",
        "## 机制判断",
        "",
        "[Strong Evidence] CFUS 的 train/eval 解耦实现是干净的：`train_horizons_effective=720`，`step_loss_weighting=uniform`，trace 中 `unit_type=conditioned_sparse`，并记录 `condition_type=label_novelty`。",
        "",
        "[Strong Evidence] CFUS 相比 full-time dense MSE 有实质收益：ETTh2 四个 horizons 全赢；Weather 基本持平，h336/h720 小幅赢。这说明 conditioned sparse pressure 不是完全无效。",
        "",
        "[Counter-Evidence] CFUS 相比 R.3 不稳定：ETTh2 在 h192/h336/h720 赢，但 h96 输；Weather 四个 horizons 全输。这直接触发 small gate 的 no Weather collapse / close R.3 gap 条件失败。",
        "",
        "[Diagnostic Gap] 当前 trace 只记录 `condition_top_blocks=4` 和 condition score，没有记录具体 selected block indices。因此它不能证明 selected units 没有退化为固定 late blocks。下一步若继续 CFUS，必须记录 selected block ranges 或 block-index histogram。",
        "",
        "## 下一步",
        "",
        "[Decision] 不进入 full matrix，不继续用当前 `label_novelty + top_ratio=0.25 + aux=0.1` 宽 sweep。",
        "",
        "建议回退到 Step 6，重新设计 condition 可观测性与 schedule：",
        "",
        "1. 先补 trace：记录 selected block indices / block ranges / per-block condition scores。",
        "2. 做 offline train-label condition diagnostic：判断 `label_novelty` 是否长期偏向 late blocks；若是，它只是 late weighting proxy。",
        "3. 设计 CFUS-v2：condition 需要同时保护 early/easy regions，可考虑 `balanced condition buckets` 或 `novelty within future-region groups`。",
        "4. 只有 CFUS-v2 local trace 证明不是固定 late weighting 后，再做 small gate。",
        "",
    ]
    (output_dir / "phase4_s_cfus_gate_decision_report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase4-S CFUS small gate.")
    parser.add_argument("--analysis-root", default="analysis/phase4_s_cfus_gate_20260624")
    args = parser.parse_args()

    output_dir = Path(args.analysis_root)
    raw_root = output_dir / "raw"
    main_rows = load_main_metrics(raw_root)
    main_delta = add_deltas(main_rows)
    overall_summary = summarize_deltas(main_delta)
    dataset_summary = summarize_deltas(main_delta, "dataset")
    horizon_summary = summarize_deltas(main_delta, "horizon")
    segment_rows = load_segment_metrics(raw_root)
    segment_delta = add_segment_deltas(segment_rows)
    segment_region_summary = summarize_segment_deltas(segment_delta, "future_region")
    trace_summary = load_trace_summary(raw_root)
    prefix_summary = load_prefix_summary(raw_root)

    write_csv(output_dir / "phase4_s_cfus_main_metrics_delta.csv", main_delta)
    write_csv(output_dir / "phase4_s_cfus_overall_summary.csv", overall_summary)
    write_csv(output_dir / "phase4_s_cfus_dataset_summary.csv", dataset_summary)
    write_csv(output_dir / "phase4_s_cfus_horizon_summary.csv", horizon_summary)
    write_csv(output_dir / "phase4_s_cfus_segment_delta.csv", segment_delta)
    write_csv(output_dir / "phase4_s_cfus_segment_region_summary.csv", segment_region_summary)
    write_csv(output_dir / "phase4_s_cfus_trace_summary.csv", trace_summary)
    write_csv(output_dir / "phase4_s_cfus_prefix_summary.csv", prefix_summary)
    write_report(
        output_dir,
        main_delta,
        overall_summary,
        dataset_summary,
        horizon_summary,
        segment_region_summary,
        trace_summary,
        prefix_summary,
    )


if __name__ == "__main__":
    main()
