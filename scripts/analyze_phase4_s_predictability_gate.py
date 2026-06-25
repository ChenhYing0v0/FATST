from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


MODEL_ORDER = [
    "PatchEncoderPredictabilityDownweight",
    "PatchEncoderFullTimeMSE720",
    "PatchEncoderR3PrefixRisk",
]

MODEL_LABELS = {
    "PatchEncoderPredictabilityDownweight": "S2_predictability_downweight",
    "PatchEncoderFullTimeMSE720": "D0_full_time_mse",
    "PatchEncoderR3PrefixRisk": "D1_r3_prefix_risk",
    "PatchEncoderConditionedFutureUnitScheduling": "S1_conditioned_future_unit_scheduling",
}

DATASETS = ["ETTh2", "Weather"]
HORIZONS = [96, 192, 336, 720]
PRIMARY_MODEL = "PatchEncoderPredictabilityDownweight"
BASELINES = ["PatchEncoderFullTimeMSE720", "PatchEncoderR3PrefixRisk"]
CFUS_MODEL = "PatchEncoderConditionedFutureUnitScheduling"


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


def load_main_metrics(raw_root: Path, models: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
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


def add_deltas(
    rows: list[dict[str, Any]],
    primary_model: str,
    baselines: list[str],
) -> list[dict[str, Any]]:
    by_key = {
        (row["model"], row["dataset"], row["horizon"]): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    for baseline in baselines:
        for row in rows:
            if row["model"] != primary_model:
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
    for baseline in sorted({row["baseline_model"] for row in delta_rows}):
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


def load_segment_metrics(raw_root: Path, models: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
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


def add_segment_deltas(
    rows: list[dict[str, Any]],
    primary_model: str,
    baselines: list[str],
) -> list[dict[str, Any]]:
    by_key = {
        (row["model"], row["dataset"], row["horizon"], row["segment"]): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    for baseline in baselines:
        for row in rows:
            if row["model"] != primary_model:
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
    for baseline in sorted({row["baseline_model"] for row in rows}):
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
                        "mean_auxiliary_weight": mean(
                            float(row.get("auxiliary_weight", 0) or 0) for row in unit_rows
                        ),
                        "mean_learnable_blocks": mean(
                            float(row.get("predictability_learnable_blocks", 0) or 0) for row in unit_rows
                        ),
                        "mean_noisy_blocks": mean(
                            float(row.get("predictability_noisy_blocks", 0) or 0) for row in unit_rows
                        ),
                        "mean_predictability_weight": mean(
                            float(row.get("predictability_mean_weight", 0) or 0) for row in unit_rows
                        ),
                        "mean_floor_weight": mean(
                            float(row.get("predictability_floor_weight", 0) or 0) for row in unit_rows
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


def summary_row(rows: list[dict[str, Any]], baseline: str, key: str | None = None, value: Any = None) -> dict[str, Any]:
    for row in rows:
        if row["baseline_strategy"] != baseline:
            continue
        if key is None or row[key] == value:
            return row
    raise KeyError((baseline, key, value))


def write_report(
    output_dir: Path,
    main_delta: list[dict[str, Any]],
    overall_summary: list[dict[str, Any]],
    dataset_summary: list[dict[str, Any]],
    horizon_summary: list[dict[str, Any]],
    segment_region_summary: list[dict[str, Any]],
    trace_summary: list[dict[str, Any]],
    prefix_summary: list[dict[str, Any]],
    cfus_delta: list[dict[str, Any]],
    cfus_summary: list[dict[str, Any]],
) -> None:
    vs_full = summary_row(overall_summary, "D0_full_time_mse")
    vs_r3 = summary_row(overall_summary, "D1_r3_prefix_risk")
    vs_cfus = summary_row(cfus_summary, "S1_conditioned_future_unit_scheduling")
    weather_vs_r3 = summary_row(dataset_summary, "D1_r3_prefix_risk", "dataset", "Weather")
    etth2_vs_r3 = summary_row(dataset_summary, "D1_r3_prefix_risk", "dataset", "ETTh2")
    trace_by_dataset = {
        row["dataset"]: row
        for row in trace_summary
        if row["strategy"] == MODEL_LABELS[PRIMARY_MODEL]
    }
    etth2_trace = trace_by_dataset["ETTh2"]
    weather_trace = trace_by_dataset["Weather"]
    lines = [
        "# Phase4-S Predictability Gate 决策报告",
        "",
        "## 11-step 记录",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | Step 9-11 |",
        "| `problem` | predictability-aware downweight 是否能修复 CFUS-S1 在 Weather 上的 noisy-hard collapse |",
        "| `existence_evidence` | train-only predictability diagnostic; ETTh2/Weather small remote gate; full-time, R.3, CFUS-S1 controls |",
        "| `idea` | low-predictability noisy-hard blocks 降权，learnable-hard blocks 继续 auxiliary emphasis |",
        "| `theory_check` | 若 Weather 失败来自 noisy-hard 污染 shared representation，则 downweight 应减少 Weather vs R.3 gap，同时保留 ETTh2 gain |",
        "| `design` | `predictability_downweight` vs `full_time_mse`, `r3_prefix_risk`, and prior CFUS-S1 on ETTh2/Weather |",
        "| `gate` | keep full-time gain; improve Weather vs R.3; no early/prefix collapse; trace confirms noisy/learnable split |",
        "| `artifacts` | `analysis/phase4_s_predictability_gate_20260625` |",
        "| `decision` | fail as paper-core; current predictability proxy/downweight is insufficient |",
        "",
        "## 主要结果",
        "",
        f"[Fact] S2 vs `full_time_mse`: mean relative MSE {format_pct(vs_full['mean_relative_mse_pct'])}, MSE wins `{vs_full['mse_wins']}/{vs_full['settings']}`。",
        f"[Fact] S2 vs `r3_prefix_risk`: mean relative MSE {format_pct(vs_r3['mean_relative_mse_pct'])}, MSE wins `{vs_r3['mse_wins']}/{vs_r3['settings']}`。",
        f"[Fact] S2 vs S1-CFUS: mean relative MSE {format_pct(vs_cfus['mean_relative_mse_pct'])}, MSE wins `{vs_cfus['mse_wins']}/{vs_cfus['settings']}`。",
        f"[Fact] S2 vs R.3 on ETTh2: mean relative MSE {format_pct(etth2_vs_r3['mean_relative_mse_pct'])}, wins `{etth2_vs_r3['mse_wins']}/{etth2_vs_r3['settings']}`。",
        f"[Fact] S2 vs R.3 on Weather: mean relative MSE {format_pct(weather_vs_r3['mean_relative_mse_pct'])}, wins `{weather_vs_r3['mse_wins']}/{weather_vs_r3['settings']}`。",
        "",
        "[Decision] 当前 `predictability_downweight` 没有通过 paper-core gate。它基本保留了 ETTh2 上相对 full-time / R.3 的收益，但没有修复 Weather：相对 R.3 仍然 `0/4` wins，并且相对 full-time、相对 CFUS-S1 都略差。",
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
        "## S2 vs Prior S1-CFUS",
        "",
        *markdown_table(
            cfus_delta,
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
            ["baseline_strategy", "settings", "mse_wins", "mae_wins", "mean_relative_mse_pct", "mean_relative_mae_pct"],
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
                "unit_type",
                "epochs_ran",
                "mean_active_steps",
                "mean_learnable_blocks",
                "mean_noisy_blocks",
                "mean_floor_weight",
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
        "[Strong Evidence] S2 的 train/eval 解耦实现是干净的：`train_horizons_effective=720`，evaluation horizons 仍只用于测试，prefix mismatch 保持 numerical-zero。",
        "",
        "[Strong Evidence] Trace 证明 noisy/learnable split 确实发生：ETTh2 平均约 `{:.2f}` learnable blocks、`{:.2f}` noisy blocks；Weather 平均约 `{:.2f}` learnable blocks、`{:.2f}` noisy blocks。这与 offline diagnostic 对 Weather noisy-hard 的判断一致。".format(
            etth2_trace["mean_learnable_blocks"],
            etth2_trace["mean_noisy_blocks"],
            weather_trace["mean_learnable_blocks"],
            weather_trace["mean_noisy_blocks"],
        ),
        "",
        "[Counter-Evidence] 尽管 split 正确发生，Weather 指标没有改善。说明当前简单 proxy `top_novelty ∩ top_variation` 加 `floor_weight=0.5` 不足以解决 Weather 的 interference，或者 Weather 的主差距不是仅靠降低 noisy-hard dense weight 能解决。",
        "",
        "[Counter-Evidence] S2 相比 S1-CFUS 没有带来实质改进：整体 mean relative MSE 为正，Weather 四个 horizons 都差于 S1。这说明当前 downweight formulation 把有效 hard-block emphasis 削弱了，但没有换来足够的 noise shielding。",
        "",
        "## 下一步",
        "",
        "[Decision] 不进入 full matrix，不继续 sweep 当前 `floor_weight`。当前失败的是 S2 的简单 downweight implementation，不是 predictability-conditioned scheduling 问题本身。",
        "",
        "建议回退到 Step 5/6：",
        "",
        "1. 重新评估 predictability proxy：仅用 local variation 过粗，需引入 train-only baseline residual / seasonal residual stability。",
        "2. 如果继续 shielding，优先考虑 `detached/isolated auxiliary path`，而不是在 shared dense loss 中简单降权。",
        "3. 保留 S1-CFUS 作为 evidence：hard-block emphasis 对 ETTh2 有效，但需要 dataset/state-aware gate 决定何时启用。",
        "4. 下一轮不应再只改 scalar weights，应先做 train-side residual predictability diagnostic，确认 Weather 的 low-predictability units 是否可由更强 proxy 分离。",
        "",
    ]
    (output_dir / "phase4_s_predictability_gate_decision_report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase4-S predictability small gate.")
    parser.add_argument("--analysis-root", default="analysis/phase4_s_predictability_gate_20260625")
    parser.add_argument("--cfus-analysis-root", default="analysis/phase4_s_cfus_gate_20260624")
    args = parser.parse_args()

    output_dir = Path(args.analysis_root)
    raw_root = output_dir / "raw"
    main_rows = load_main_metrics(raw_root, MODEL_ORDER)
    main_delta = add_deltas(main_rows, PRIMARY_MODEL, BASELINES)
    overall_summary = summarize_deltas(main_delta)
    dataset_summary = summarize_deltas(main_delta, "dataset")
    horizon_summary = summarize_deltas(main_delta, "horizon")
    segment_rows = load_segment_metrics(raw_root, MODEL_ORDER)
    segment_delta = add_segment_deltas(segment_rows, PRIMARY_MODEL, BASELINES)
    segment_region_summary = summarize_segment_deltas(segment_delta, "future_region")
    trace_summary = load_trace_summary(raw_root)
    prefix_summary = load_prefix_summary(raw_root)

    cfus_raw_root = Path(args.cfus_analysis_root) / "raw"
    cfus_rows = load_main_metrics(cfus_raw_root, [CFUS_MODEL])
    cfus_compare_rows = [row for row in main_rows if row["model"] == PRIMARY_MODEL] + cfus_rows
    cfus_delta = add_deltas(cfus_compare_rows, PRIMARY_MODEL, [CFUS_MODEL])
    cfus_summary = summarize_deltas(cfus_delta)

    write_csv(output_dir / "phase4_s_predictability_main_metrics_delta.csv", main_delta)
    write_csv(output_dir / "phase4_s_predictability_overall_summary.csv", overall_summary)
    write_csv(output_dir / "phase4_s_predictability_dataset_summary.csv", dataset_summary)
    write_csv(output_dir / "phase4_s_predictability_horizon_summary.csv", horizon_summary)
    write_csv(output_dir / "phase4_s_predictability_segment_delta.csv", segment_delta)
    write_csv(output_dir / "phase4_s_predictability_segment_region_summary.csv", segment_region_summary)
    write_csv(output_dir / "phase4_s_predictability_trace_summary.csv", trace_summary)
    write_csv(output_dir / "phase4_s_predictability_prefix_summary.csv", prefix_summary)
    write_csv(output_dir / "phase4_s_predictability_vs_cfus_delta.csv", cfus_delta)
    write_csv(output_dir / "phase4_s_predictability_vs_cfus_summary.csv", cfus_summary)
    write_report(
        output_dir,
        main_delta,
        overall_summary,
        dataset_summary,
        horizon_summary,
        segment_region_summary,
        trace_summary,
        prefix_summary,
        cfus_delta,
        cfus_summary,
    )


if __name__ == "__main__":
    main()
