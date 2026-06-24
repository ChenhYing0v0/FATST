from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


MODEL_ORDER = [
    "PatchEncoderR3PrefixRisk",
    "PatchEncoderFullTimeMSE720",
    "PatchEncoderRandomFutureMask",
    "PatchEncoderIntervalSupervision",
    "PatchEncoderComponentTop",
    "PatchEncoderComponentBalanced",
    "PatchEncoderCurriculumUnits",
]

MODEL_LABELS = {
    "PatchEncoderR3PrefixRisk": "D1_r3_prefix_risk",
    "PatchEncoderFullTimeMSE720": "D0_full_time_mse",
    "PatchEncoderRandomFutureMask": "D2_random_future_mask",
    "PatchEncoderIntervalSupervision": "D3_interval_supervision",
    "PatchEncoderComponentTop": "D4_component_basis_top",
    "PatchEncoderComponentBalanced": "D5_component_basis_balanced",
    "PatchEncoderCurriculumUnits": "D6_curriculum_units",
}

DATASETS = ["ETTh2", "ETTm1", "Weather"]
HORIZONS = [96, 192, 336, 720]
SEGMENT_TARGET_HORIZONS = {96, 720}


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
                horizon = int(row["target_horizon"])
                rows.append(
                    {
                        "model": model,
                        "strategy": MODEL_LABELS[model],
                        "dataset": dataset,
                        "horizon": horizon,
                        "mse": float(row["mse"]),
                        "mae": float(row["mae"]),
                    }
                )
    return rows


def add_baseline_deltas(rows: list[dict[str, Any]], baseline_model: str) -> list[dict[str, Any]]:
    baseline = {
        (row["dataset"], row["horizon"]): row
        for row in rows
        if row["model"] == baseline_model
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        base = baseline[(row["dataset"], row["horizon"])]
        new_row = dict(row)
        new_row["baseline_model"] = baseline_model
        new_row["baseline_mse"] = base["mse"]
        new_row["baseline_mae"] = base["mae"]
        new_row["relative_mse_pct"] = pct(row["mse"], base["mse"])
        new_row["relative_mae_pct"] = pct(row["mae"], base["mae"])
        new_row["mse_win"] = row["mse"] < base["mse"]
        new_row["mae_win"] = row["mae"] < base["mae"]
        out.append(new_row)
    return out


def summarize_main(delta_rows: list[dict[str, Any]], baseline_model: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        subset = [row for row in delta_rows if row["model"] == model]
        rows.append(
            {
                "model": model,
                "strategy": MODEL_LABELS[model],
                "baseline_model": baseline_model,
                "settings": len(subset),
                "mse_wins": sum(1 for row in subset if row["mse_win"]),
                "mae_wins": sum(1 for row in subset if row["mae_win"]),
                "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in subset),
                "mean_relative_mae_pct": mean(row["relative_mae_pct"] for row in subset),
                "max_dataset_mean_degradation_pct": max(
                    mean(row["relative_mse_pct"] for row in subset if row["dataset"] == dataset)
                    for dataset in DATASETS
                ),
            }
        )
    return rows


def summarize_by_key(
    delta_rows: list[dict[str, Any]],
    key: str,
    baseline_model: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        subset = [row for row in delta_rows if row["model"] == model]
        values = sorted({row[key] for row in subset})
        for value in values:
            value_rows = [row for row in subset if row[key] == value]
            rows.append(
                {
                    "model": model,
                    "strategy": MODEL_LABELS[model],
                    key: value,
                    "baseline_model": baseline_model,
                    "settings": len(value_rows),
                    "mse_wins": sum(1 for row in value_rows if row["mse_win"]),
                    "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in value_rows),
                    "mean_relative_mae_pct": mean(row["relative_mae_pct"] for row in value_rows),
                }
            )
    return rows


def load_segment_metrics(raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for dataset in DATASETS:
            base_dir = run_dir(raw_root, model, dataset)
            for horizon in SEGMENT_TARGET_HORIZONS:
                path = base_dir / f"h{horizon}" / "metrics_by_segment.csv"
                for row in read_csv(path):
                    rows.append(
                        {
                            "model": model,
                            "strategy": MODEL_LABELS[model],
                            "dataset": dataset,
                            "target_horizon": horizon,
                            "segment": row["segment"],
                            "mse": float(row["mse"]),
                            "mae": float(row["mae"]),
                        }
                    )
    return rows


def add_segment_deltas(rows: list[dict[str, Any]], baseline_model: str) -> list[dict[str, Any]]:
    baseline = {
        (row["dataset"], row["target_horizon"], row["segment"]): row
        for row in rows
        if row["model"] == baseline_model
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        base = baseline[(row["dataset"], row["target_horizon"], row["segment"])]
        new_row = dict(row)
        new_row["baseline_model"] = baseline_model
        new_row["baseline_mse"] = base["mse"]
        new_row["relative_mse_pct"] = pct(row["mse"], base["mse"])
        new_row["mse_win"] = row["mse"] < base["mse"]
        out.append(new_row)
    return out


def summarize_segments(delta_rows: list[dict[str, Any]], baseline_model: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        subset = [row for row in delta_rows if row["model"] == model]
        for horizon in sorted(SEGMENT_TARGET_HORIZONS):
            horizon_rows = [row for row in subset if row["target_horizon"] == horizon]
            rows.append(
                {
                    "model": model,
                    "strategy": MODEL_LABELS[model],
                    "target_horizon": horizon,
                    "baseline_model": baseline_model,
                    "segments": len(horizon_rows),
                    "segment_wins": sum(1 for row in horizon_rows if row["mse_win"]),
                    "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in horizon_rows),
                }
            )
    return rows


def load_prefix(raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for dataset in DATASETS:
            path = run_dir(raw_root, model, dataset) / "prefix_consistency.csv"
            for row in read_csv(path):
                rows.append(
                    {
                        "model": model,
                        "strategy": MODEL_LABELS[model],
                        "dataset": dataset,
                        "short_horizon": int(row["short_horizon"]),
                        "long_horizon": int(row["long_horizon"]),
                        "prefix_mismatch_mse": float(row["prefix_mismatch_mse"]),
                        "prefix_mismatch_mae": float(row["prefix_mismatch_mae"]),
                    }
                )
    return rows


def summarize_prefix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        subset = [row for row in rows if row["model"] == model]
        out.append(
            {
                "model": model,
                "strategy": MODEL_LABELS[model],
                "rows": len(subset),
                "max_prefix_mismatch_mse": max(row["prefix_mismatch_mse"] for row in subset),
                "mean_prefix_mismatch_mse": mean(row["prefix_mismatch_mse"] for row in subset),
            }
        )
    return out


def load_trace_summary(raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for dataset in DATASETS:
            base_dir = run_dir(raw_root, model, dataset)
            config = json.loads((base_dir / "effective_config.json").read_text())
            trace = read_csv(base_dir / "supervision_trace.csv")
            log = read_csv(base_dir / "training_log.csv")
            unit_counts: dict[str, int] = defaultdict(int)
            phase_counts: dict[str, int] = defaultdict(int)
            active_ratios = []
            unit_losses = []
            for row in trace:
                unit_counts[row["unit_type"]] += 1
                phase_counts[row["curriculum_phase"]] += 1
                active_ratios.append(float(row["mask_ratio"]))
                unit_losses.append(float(row["loss_unit"]))
            rows.append(
                {
                    "model": model,
                    "strategy": MODEL_LABELS[model],
                    "dataset": dataset,
                    "training_evaluation_decoupled": config.get("training_evaluation_decoupled", False),
                    "train_horizons_effective": ",".join(str(item) for item in config.get("train_horizons_effective", [])),
                    "epochs_ran": len(log),
                    "trace_rows": len(trace),
                    "unit_types": ";".join(f"{key}:{unit_counts[key]}" for key in sorted(unit_counts)),
                    "curriculum_phases": ";".join(f"{key}:{phase_counts[key]}" for key in sorted(phase_counts)),
                    "mean_active_step_ratio": mean(active_ratios) if active_ratios else 0.0,
                    "mean_unit_loss": mean(unit_losses) if unit_losses else 0.0,
                    "component_top_rank_variance": float(config.get("component_basis_stats", {}).get("component_top_rank_variance", 0.0)),
                }
            )
    return rows


def model_row(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    return next(row for row in rows if row["model"] == model)


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
                if "relative" in column or "degradation" in column:
                    values.append(format_pct(value))
                else:
                    values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    output_dir: Path,
    main_summary: list[dict[str, Any]],
    dataset_summary: list[dict[str, Any]],
    horizon_summary: list[dict[str, Any]],
    segment_summary: list[dict[str, Any]],
    prefix_summary: list[dict[str, Any]],
    trace_summary: list[dict[str, Any]],
) -> None:
    candidates = [row for row in main_summary if row["model"] != "PatchEncoderR3PrefixRisk"]
    best = min(candidates, key=lambda row: row["mean_relative_mse_pct"])
    r3 = model_row(main_summary, "PatchEncoderR3PrefixRisk")
    full_time = model_row(main_summary, "PatchEncoderFullTimeMSE720")
    random_mask = model_row(main_summary, "PatchEncoderRandomFutureMask")
    interval = model_row(main_summary, "PatchEncoderIntervalSupervision")
    component_top = model_row(main_summary, "PatchEncoderComponentTop")
    component_balanced = model_row(main_summary, "PatchEncoderComponentBalanced")
    curriculum = model_row(main_summary, "PatchEncoderCurriculumUnits")

    lines: list[str] = [
        "# Phase4 Horizon-Decoupled Gate 决策报告",
        "",
        "## 11-step 记录",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | Step 9-11 |",
        "| `problem` | training supervision units 是否应与 evaluation horizons 解耦 |",
        "| `existence_evidence` | 7 strategies x 3 datasets x 4 horizons remote gate |",
        "| `idea` | horizon-decoupled future supervision units |",
        "| `theory_check` | mask/interval/component/curriculum 应降低 task redundancy 或改善 optimization path |",
        "| `design` | R.3 carrier, unchanged evaluation horizons, train-side unit strategies |",
        "| `gate` | mean MSE vs R.3 < 0, wins >= 7/12, dataset degradation <= +0.5%, prefix stable, diagnostic support |",
        "| `artifacts` | `analysis/phase4_horizon_decoupled_gate_20260624` |",
        "| `decision` | fail as paper-core; rollback to Step 4/6 and redesign supervision strategy |",
        "",
        "## 主要结果",
        "",
        "[Fact] Primary baseline `PatchEncoderR3PrefixRisk` 的 relative MSE 为 `0.00%`；`mse_wins` 使用严格 `<` 计算，因此 baseline 自身为 `0/12`。",
        f"[Fact] 最好的非 R.3 candidate 是 `{best['strategy']}`，mean relative MSE {format_pct(best['mean_relative_mse_pct'])}，相对 R.3 有 `{best['mse_wins']}/12` 个 MSE wins。",
        f"[Fact] 相对 R.3，`full_time_mse` 为 {format_pct(full_time['mean_relative_mse_pct'])}，random mask 为 {format_pct(random_mask['mean_relative_mse_pct'])}，interval 为 {format_pct(interval['mean_relative_mse_pct'])}。",
        f"[Fact] 相对 R.3，component top / component balanced / curriculum 分别为 {format_pct(component_top['mean_relative_mse_pct'])}、{format_pct(component_balanced['mean_relative_mse_pct'])}、{format_pct(curriculum['mean_relative_mse_pct'])}。",
        "",
        "[Decision] 当前 Phase4-R 不通过 paper-core gate。没有任何 horizon-decoupled candidate 同时满足 mean relative MSE < 0、7/12 wins、dataset degradation <= +0.5%。",
        "",
        "## 统计口径",
        "",
        "本报告由 `scripts/analyze_phase4_horizon_decoupled_gate.py` 从 remote sync artifacts 生成。",
        "",
        "| Quantity | Source | Computation | Meaning |",
        "| --- | --- | --- | --- |",
        "| `mse`, `mae` | 每个 run 的 `metrics_by_target_horizon.csv` | 直接读取 per target horizon evaluation | 固定 evaluation horizons 上的预测误差 |",
        "| `relative_mse_pct` | candidate 与 R.3 同 dataset/horizon 的 `mse` | `(candidate / R.3 - 1) * 100` | 相对 R.3 的 MSE 变化，负数表示改进 |",
        "| `mse_wins` | `relative_mse_pct` 对应的 raw `mse` | candidate `mse < R.3 mse` 的计数 | 12 个 dataset-horizon 设置中的严格胜出数 |",
        "| `max_dataset_mean_degradation_pct` | dataset-level relative MSE | 每个 dataset 内取均值，再取最大 | 最坏 dataset 平均退化，用于避免单数据集被牺牲 |",
        "| `segment_wins` | `h96/h720/metrics_by_segment.csv` | candidate segment MSE 严格小于 R.3 的计数 | H96/H720 局部误差是否改善 |",
        "| `prefix_mismatch_mse` | `prefix_consistency.csv` | 同一 prefix 在不同 requested horizon 下的 prediction MSE mismatch | 检查 unified inference 的 prefix consistency 是否被破坏 |",
        "| `mean_active_step_ratio` | `supervision_trace.csv` | batch trace 中 `mask_ratio` 平均 | 每步训练实际监督的 future positions 比例 |",
        "",
        "## Strategy 汇总 vs R.3",
        "",
        *markdown_table(
            main_summary,
            [
                "strategy",
                "settings",
                "mse_wins",
                "mae_wins",
                "mean_relative_mse_pct",
                "mean_relative_mae_pct",
                "max_dataset_mean_degradation_pct",
            ],
        ),
        "",
        "## Dataset 汇总 vs R.3",
        "",
        *markdown_table(
            [row for row in dataset_summary if row["model"] != "PatchEncoderR3PrefixRisk"],
            ["strategy", "dataset", "mse_wins", "mean_relative_mse_pct", "mean_relative_mae_pct"],
        ),
        "",
        "## Horizon 汇总 vs R.3",
        "",
        *markdown_table(
            [row for row in horizon_summary if row["model"] != "PatchEncoderR3PrefixRisk"],
            ["strategy", "horizon", "mse_wins", "mean_relative_mse_pct", "mean_relative_mae_pct"],
        ),
        "",
        "## Segment 汇总 vs R.3",
        "",
        *markdown_table(
            [row for row in segment_summary if row["model"] != "PatchEncoderR3PrefixRisk"],
            ["strategy", "target_horizon", "segment_wins", "mean_relative_mse_pct"],
        ),
        "",
        "## Prefix Consistency 诊断",
        "",
        *markdown_table(
            prefix_summary,
            ["strategy", "rows", "max_prefix_mismatch_mse", "mean_prefix_mismatch_mse"],
        ),
        "",
        "[Fact] 所有 strategy 的 prefix mismatch 都保持在 numerical-zero 量级，因此失败不是 prefix consistency 被破坏导致。",
        "",
        "## Supervision Trace 汇总",
        "",
        *markdown_table(
            trace_summary,
            [
                "strategy",
                "dataset",
                "training_evaluation_decoupled",
                "train_horizons_effective",
                "epochs_ran",
                "unit_types",
                "curriculum_phases",
                "mean_active_step_ratio",
                "component_top_rank_variance",
            ],
        ),
        "",
        "## 结果解释",
        "",
        "[Strong Evidence] 简单 horizon-decoupled supervision 不能取代 R.3。`full_time_mse`、mask、interval、component、curriculum 均未超过 R.3，说明 R.3 的 prefix-risk pressure 不是一个容易被 horizon-free unit sampling 替代的弱 baseline。",
        "",
        "[Strong Evidence] 随机 mask / interval 的表现相对接近 R.3，但仍不满足 gate。这说明 stochastic future-unit scheduling 可能有 regularization 价值，但当前版本没有足够 paper-core 性能证据。",
        "",
        "[Strong Evidence] component-based routes 系统性较差，且 component top 与 curriculum 的结果非常接近，说明第一阶段 top-component supervision 主导了 curriculum early trajectory。结合 residual projection audit，当前 component-basis route 不应作为下一步主线。",
        "",
        "[Inference] 当前失败不是 evaluation/inference 接口问题：prefix consistency 没坏，evaluation horizons 也完整。更可能的问题是 train-side supervision unit 太粗糙：mask/interval 没有根据样本状态、future difficulty 或 error process 自适应分配 pressure；component supervision 又过度偏向 global variance basis。",
        "",
        "## 下一步方向",
        "",
        "[Decision] 回退到 Step 4/6，保留 training/evaluation 解耦问题，但停止当前 `D2-D6` 简单策略扩展。下一步不应继续调 mask ratio、interval length 或 component rank 的宽 sweep。",
        "",
        "推荐进入 `Phase4-S`: State/Difficulty-Conditioned Supervision Scheduling。",
        "",
        "核心问题：",
        "",
        "> Horizon-free supervision units 仍然成立，但 unit pressure 不能是全局静态或随机；它应由 train-side difficulty / future-label novelty / error-process proxy 条件化。",
        "",
        "最小下一步：",
        "",
        "1. 用本轮 artifacts 做 post-hoc diagnostic：哪些 samples/segments 在 R.3 下高 residual，mask/interval 是否覆盖这些 regions。",
        "2. 设计 `difficulty_conditioned_interval`：仍训练 720 future sequence，但 interval sampling probability 由 train-label novelty 或 running loss bucket 决定。",
        "3. 设计 `r3_plus_sparse_unit_aux`：保留 R.3 base loss，只加小权重 horizon-free auxiliary unit，而不是替换 R.3 objective。",
        "4. 先做 local diagnostic / small remote gate，再决定是否进入 full matrix。",
        "",
        "Rollback point: Step 4/6。当前 HSS 问题保留，具体 idea 从 static horizon-decoupled replacement 改为 conditioned auxiliary scheduling。",
        "",
    ]
    (output_dir / "phase4_horizon_decoupled_decision_report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase4 horizon-decoupled remote gate.")
    parser.add_argument("--analysis-root", default="analysis/phase4_horizon_decoupled_gate_20260624")
    args = parser.parse_args()

    output_dir = Path(args.analysis_root)
    raw_root = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    main_rows = load_main_metrics(raw_root)
    main_delta = add_baseline_deltas(main_rows, "PatchEncoderR3PrefixRisk")
    main_summary = summarize_main(main_delta, "PatchEncoderR3PrefixRisk")
    dataset_summary = summarize_by_key(main_delta, "dataset", "PatchEncoderR3PrefixRisk")
    horizon_summary = summarize_by_key(main_delta, "horizon", "PatchEncoderR3PrefixRisk")

    segment_rows = load_segment_metrics(raw_root)
    segment_delta = add_segment_deltas(segment_rows, "PatchEncoderR3PrefixRisk")
    segment_summary = summarize_segments(segment_delta, "PatchEncoderR3PrefixRisk")

    prefix_rows = load_prefix(raw_root)
    prefix_summary = summarize_prefix(prefix_rows)
    trace_summary = load_trace_summary(raw_root)

    write_csv(output_dir / "phase4_horizon_decoupled_main_metrics.csv", main_delta)
    write_csv(output_dir / "phase4_horizon_decoupled_strategy_summary.csv", main_summary)
    write_csv(output_dir / "phase4_horizon_decoupled_dataset_summary.csv", dataset_summary)
    write_csv(output_dir / "phase4_horizon_decoupled_horizon_summary.csv", horizon_summary)
    write_csv(output_dir / "phase4_horizon_decoupled_segment_metrics.csv", segment_delta)
    write_csv(output_dir / "phase4_horizon_decoupled_segment_summary.csv", segment_summary)
    write_csv(output_dir / "phase4_horizon_decoupled_prefix_summary.csv", prefix_summary)
    write_csv(output_dir / "phase4_horizon_decoupled_trace_summary.csv", trace_summary)
    write_report(output_dir, main_summary, dataset_summary, horizon_summary, segment_summary, prefix_summary, trace_summary)


if __name__ == "__main__":
    main()
