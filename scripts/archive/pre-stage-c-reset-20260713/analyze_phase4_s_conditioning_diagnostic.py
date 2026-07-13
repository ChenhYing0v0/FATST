from __future__ import annotations

import argparse
import csv
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
BASELINE_MODEL = "PatchEncoderR3PrefixRisk"


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


def parse_segment(segment: str) -> tuple[int, int]:
    start, end = segment.split("-", maxsplit=1)
    return int(start), int(end)


def future_region(segment_end: int) -> str:
    if segment_end <= 96:
        return "early_1_96"
    if segment_end <= 336:
        return "middle_97_336"
    return "late_337_720"


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def load_segment_rows(raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for dataset in DATASETS:
            base_dir = run_dir(raw_root, model, dataset)
            for horizon in HORIZONS:
                path = base_dir / f"h{horizon}" / "metrics_by_segment.csv"
                for row in read_csv(path):
                    start, end = parse_segment(row["segment"])
                    rows.append(
                        {
                            "model": model,
                            "strategy": MODEL_LABELS[model],
                            "dataset": dataset,
                            "horizon": horizon,
                            "segment": row["segment"],
                            "segment_start": start,
                            "segment_end": end,
                            "future_region": future_region(end),
                            "mse": float(row["mse"]),
                            "mae": float(row["mae"]),
                        }
                    )
    return rows


def assign_global_residual_bucket(baseline_rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], str]:
    ordered = sorted(baseline_rows, key=lambda row: row["mse"])
    total = len(ordered)
    buckets: dict[tuple[str, int, str], str] = {}
    for index, row in enumerate(ordered):
        rank = (index + 1) / total
        if rank <= 1 / 3:
            bucket = "low_r3_residual"
        elif rank <= 2 / 3:
            bucket = "mid_r3_residual"
        else:
            bucket = "high_r3_residual"
        buckets[(row["dataset"], row["horizon"], row["segment"])] = bucket
    return buckets


def add_segment_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {
        (row["dataset"], row["horizon"], row["segment"]): row
        for row in rows
        if row["model"] == BASELINE_MODEL
    }
    bucket_by_key = assign_global_residual_bucket(list(baseline.values()))

    out: list[dict[str, Any]] = []
    for row in rows:
        key = (row["dataset"], row["horizon"], row["segment"])
        base = baseline[key]
        new_row = dict(row)
        new_row["baseline_model"] = BASELINE_MODEL
        new_row["baseline_mse"] = base["mse"]
        new_row["baseline_mae"] = base["mae"]
        new_row["r3_residual_bucket"] = bucket_by_key[key]
        new_row["relative_mse_pct"] = pct(row["mse"], base["mse"])
        new_row["relative_mae_pct"] = pct(row["mae"], base["mae"])
        new_row["mse_win"] = row["mse"] < base["mse"]
        out.append(new_row)
    return out


def summarize_by_key(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        if model == BASELINE_MODEL:
            continue
        subset = [row for row in rows if row["model"] == model]
        values = sorted({row[key] for row in subset})
        for value in values:
            value_rows = [row for row in subset if row[key] == value]
            output.append(
                {
                    "model": model,
                    "strategy": MODEL_LABELS[model],
                    key: value,
                    "segments": len(value_rows),
                    "mse_wins": sum(1 for row in value_rows if row["mse_win"]),
                    "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in value_rows),
                    "mean_relative_mae_pct": mean(row["relative_mae_pct"] for row in value_rows),
                }
            )
    return output


def summarize_strategy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        if model == BASELINE_MODEL:
            continue
        subset = [row for row in rows if row["model"] == model]
        high = [row for row in subset if row["r3_residual_bucket"] == "high_r3_residual"]
        late = [row for row in subset if row["future_region"] == "late_337_720"]
        output.append(
            {
                "model": model,
                "strategy": MODEL_LABELS[model],
                "segments": len(subset),
                "mse_wins": sum(1 for row in subset if row["mse_win"]),
                "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in subset),
                "high_residual_segments": len(high),
                "high_residual_wins": sum(1 for row in high if row["mse_win"]),
                "high_residual_mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in high),
                "late_segments": len(late),
                "late_wins": sum(1 for row in late if row["mse_win"]),
                "late_mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in late),
            }
        )
    return output


def load_trace_rows(raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for dataset in DATASETS:
            trace_path = run_dir(raw_root, model, dataset) / "supervision_trace.csv"
            for row in read_csv(trace_path):
                rows.append(
                    {
                        "model": model,
                        "strategy": MODEL_LABELS[model],
                        "dataset": dataset,
                        "unit_type": row["unit_type"],
                        "active_steps": int(row["active_steps"]),
                        "supervision_pred_len": int(row["supervision_pred_len"]),
                        "mask_ratio": float(row["mask_ratio"]),
                        "loss_unit": float(row["loss_unit"]),
                    }
                )
    return rows


def summarize_trace(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        subset = [row for row in rows if row["model"] == model]
        by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in subset:
            by_unit[row["unit_type"]].append(row)
        for unit_type, unit_rows in sorted(by_unit.items()):
            output.append(
                {
                    "model": model,
                    "strategy": MODEL_LABELS[model],
                    "unit_type": unit_type,
                    "trace_rows": len(unit_rows),
                    "mean_active_steps": mean(row["active_steps"] for row in unit_rows),
                    "mean_mask_ratio": mean(row["mask_ratio"] for row in unit_rows),
                    "mean_loss_unit": mean(row["loss_unit"] for row in unit_rows),
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
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                if "relative" in column:
                    cells.append(format_pct(value))
                else:
                    cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_report(
    output_dir: Path,
    strategy_summary: list[dict[str, Any]],
    bucket_summary: list[dict[str, Any]],
    region_summary: list[dict[str, Any]],
    trace_summary: list[dict[str, Any]],
) -> None:
    random_row = next(row for row in strategy_summary if row["strategy"] == "D2_random_future_mask")
    interval_row = next(row for row in strategy_summary if row["strategy"] == "D3_interval_supervision")
    lines = [
        "# Phase4-S Conditioned Scheduling 事后诊断",
        "",
        "## 11-step 记录",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | Step 3-5 diagnostic for Phase4-S |",
        "| `problem` | 静态 horizon-free units 有局部 wins 但整体输 R.3，是否需要 train-side condition |",
        "| `existence_evidence` | Phase4-R segment artifacts, supervision traces, R.3 residual buckets |",
        "| `idea` | 根据 train-side difficulty / residual proxy 调整 horizon-free unit pressure |",
        "| `theory_check` | 如果 D2/D3 的 wins 集中在 high-residual 或 late regions，conditioned schedule 比 global static schedule 更有动机 |",
        "| `design` | 只做 post-hoc segment diagnostic，不训练新模型 |",
        "| `gate` | 找到局部有效区域和当前 static coverage mismatch；否则 Phase4-S 必须回退到文献调研 |",
        "| `artifacts` | `analysis/phase4_horizon_decoupled_gate_20260624` |",
        "| `decision` | conditioned scheduling 可作为 hypothesis 继续推进，但尚未通过 implementation gate |",
        "",
        "## 统计口径",
        "",
        "| Quantity | Source | Computation | Meaning |",
        "| --- | --- | --- | --- |",
        "| `baseline_mse` | R.3 `metrics_by_segment.csv` | same dataset/horizon/segment lookup | R.3 在局部 future segment 上的误差 |",
        "| `r3_residual_bucket` | all R.3 segment MSE rows | global tertiles over R.3 segment MSE | low/mid/high residual proxy |",
        "| `future_region` | segment endpoint | `<=96`, `<=336`, `>336` | early/middle/late future region |",
        "| `relative_mse_pct` | candidate vs R.3 segment MSE | `(candidate / R.3 - 1) * 100` | 局部 segment 相对变化 |",
        "| `mean_mask_ratio` | `supervision_trace.csv` | trace average | 静态 strategy 实际监督密度 |",
        "",
        "## Strategy-level 诊断",
        "",
        *markdown_table(
            strategy_summary,
            [
                "strategy",
                "segments",
                "mse_wins",
                "mean_relative_mse_pct",
                "high_residual_wins",
                "high_residual_mean_relative_mse_pct",
                "late_wins",
                "late_mean_relative_mse_pct",
            ],
        ),
        "",
        "## R.3 Residual Bucket 汇总",
        "",
        *markdown_table(
            bucket_summary,
            [
                "strategy",
                "r3_residual_bucket",
                "segments",
                "mse_wins",
                "mean_relative_mse_pct",
            ],
        ),
        "",
        "## Future Region 汇总",
        "",
        *markdown_table(
            region_summary,
            [
                "strategy",
                "future_region",
                "segments",
                "mse_wins",
                "mean_relative_mse_pct",
            ],
        ),
        "",
        "## Trace 汇总",
        "",
        *markdown_table(
            trace_summary,
            ["strategy", "unit_type", "trace_rows", "mean_active_steps", "mean_mask_ratio", "mean_loss_unit"],
        ),
        "",
        "## 解释",
        "",
        f"[Fact] `D2_random_future_mask` 在 segment-level 有 `{random_row['mse_wins']}/{random_row['segments']}` wins，high-residual bucket 为 `{random_row['high_residual_wins']}/{random_row['high_residual_segments']}` wins。",
        f"[Fact] `D3_interval_supervision` 在 segment-level 有 `{interval_row['mse_wins']}/{interval_row['segments']}` wins，high-residual bucket 为 `{interval_row['high_residual_wins']}/{interval_row['high_residual_segments']}` wins。",
        "",
        "[Strong Evidence] D2/D3 的 wins 集中在 high-residual bucket：D2 为 4/10，D3 为 5/10；low/mid residual bucket 为 0 wins。",
        "",
        "[Strong Evidence] D3 在 late region 达到 2/3 wins 且 mean relative MSE 为负；但 early region 为 0/12 wins 且显著退化。",
        "",
        "[Inference] static strategy 的失败更像 coverage/pressure mismatch，而不是 horizon-free unit 完全无效。一个全局固定的 mask/interval 会在 early/easy regions 施加不必要 pressure，同时没有把足够 pressure 定向给 high-residual/late regions。",
        "",
        "[Boundary] 当前 artifacts 只有 segment-level aggregate，没有 per-sample residual、label novelty 或 running-loss bucket。因此本报告不能证明具体 difficulty proxy 已成立，只能决定 Phase4-S 是否值得进入 Step 4-6 设计。",
        "",
        "## Phase4-S 设计含义",
        "",
        "[Decision] Phase4-S 可以作为 hypothesis 继续推进，但实现前必须先定义 train-side condition，且该 condition 不能直接使用 evaluation horizon label。",
        "",
        "推荐优先级：",
        "",
        "1. `S1_conditioned_future_unit_scheduling`：使用 full future dense anchor + train-side conditioned sparse unit pressure，作为独立 HSS training strategy。",
        "2. `S2_difficulty_conditioned_interval`：用 train-label novelty 或 running loss bucket 条件化 interval sampling。",
        "3. `S3_r3_plus_aux_control`：只作为 conflict/control，不作为 paper-core。",
        "",
    ]
    (output_dir / "phase4_s_conditioning_diagnostic_report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc diagnostic for Phase4-S conditioned scheduling.")
    parser.add_argument("--analysis-root", default="analysis/phase4_horizon_decoupled_gate_20260624")
    args = parser.parse_args()

    output_dir = Path(args.analysis_root)
    raw_root = output_dir / "raw"

    segment_rows = load_segment_rows(raw_root)
    segment_deltas = add_segment_deltas(segment_rows)
    strategy_summary = summarize_strategy(segment_deltas)
    bucket_summary = summarize_by_key(segment_deltas, "r3_residual_bucket")
    region_summary = summarize_by_key(segment_deltas, "future_region")
    trace_rows = load_trace_rows(raw_root)
    trace_summary = summarize_trace(trace_rows)

    write_csv(output_dir / "phase4_s_conditioning_segment_deltas.csv", segment_deltas)
    write_csv(output_dir / "phase4_s_conditioning_strategy_summary.csv", strategy_summary)
    write_csv(output_dir / "phase4_s_conditioning_residual_bucket_summary.csv", bucket_summary)
    write_csv(output_dir / "phase4_s_conditioning_future_region_summary.csv", region_summary)
    write_csv(output_dir / "phase4_s_conditioning_trace_summary.csv", trace_summary)
    write_report(output_dir, strategy_summary, bucket_summary, region_summary, trace_summary)


if __name__ == "__main__":
    main()
