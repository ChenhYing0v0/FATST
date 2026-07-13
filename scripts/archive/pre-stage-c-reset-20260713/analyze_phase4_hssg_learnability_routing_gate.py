from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PRIMARY = "PatchEncoderHSSGLearnabilityRegionRouting"
FULL = "PatchEncoderFullTimeMSE720"
SINGLE = "PatchEncoderSingle720PrefixRisk"
R3 = "PatchEncoderR3PrefixRisk"
HSSGA = "PatchEncoderHSSGRegionRoutedReadout"

MODELS = [FULL, SINGLE, R3, HSSGA, PRIMARY]
BASELINES = [FULL, SINGLE, R3, HSSGA]
LABELS = {
    FULL: "D0_full_time_mse",
    SINGLE: "D1_single_720_prefix_risk",
    R3: "D2_r3_prefix_risk",
    HSSGA: "HSSG-A_fixed_region_readout",
    PRIMARY: "HSSG-B_learnability_region_routing",
}
DATASETS = ["ETTh2", "Weather"]
HORIZONS = [96, 192, 336, 720]
HORIZON_LABEL = "mixed_h96_h192_h336_h720"


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


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def fmt_float(value: float) -> str:
    return f"{value:.6f}"


def run_dir(root: Path, model: str, dataset: str) -> Path:
    return root / model / dataset / HORIZON_LABEL / "seed2021"


def parse_segment(segment: str) -> tuple[int, int]:
    start, end = segment.split("-", maxsplit=1)
    return int(start), int(end)


def future_region(segment_end: int) -> str:
    if segment_end <= 96:
        return "early_1_96"
    if segment_end <= 336:
        return "middle_97_336"
    return "late_337_720"


def load_main_metrics(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            for row in read_csv(run_dir(raw_root, model, dataset) / "metrics_by_target_horizon.csv"):
                rows.append(
                    {
                        "model": model,
                        "strategy": LABELS[model],
                        "dataset": dataset,
                        "horizon": int(row["target_horizon"]),
                        "mse": float(row["mse"]),
                        "mae": float(row["mae"]),
                    }
                )
    return rows


def load_segment_metrics(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                path = run_dir(raw_root, model, dataset) / f"h{horizon}" / "metrics_by_segment.csv"
                for row in read_csv(path):
                    start, end = parse_segment(row["segment"])
                    rows.append(
                        {
                            "model": model,
                            "strategy": LABELS[model],
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


def add_main_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["model"], row["dataset"], row["horizon"]): row for row in rows}
    output = []
    for baseline in BASELINES:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                candidate = by_key[(PRIMARY, dataset, horizon)]
                base = by_key[(baseline, dataset, horizon)]
                output.append(
                    {
                        **candidate,
                        "baseline_model": baseline,
                        "baseline_strategy": LABELS[baseline],
                        "baseline_mse": base["mse"],
                        "baseline_mae": base["mae"],
                        "relative_mse_pct": pct(candidate["mse"], base["mse"]),
                        "relative_mae_pct": pct(candidate["mae"], base["mae"]),
                        "mse_win": candidate["mse"] < base["mse"],
                        "mae_win": candidate["mae"] < base["mae"],
                    }
                )
    return output


def add_segment_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["model"], row["dataset"], row["horizon"], row["segment"]): row for row in rows}
    output = []
    for baseline in BASELINES:
        for row in rows:
            if row["model"] != PRIMARY:
                continue
            base = by_key[(baseline, row["dataset"], row["horizon"], row["segment"])]
            output.append(
                {
                    **row,
                    "baseline_model": baseline,
                    "baseline_strategy": LABELS[baseline],
                    "baseline_mse": base["mse"],
                    "baseline_mae": base["mae"],
                    "relative_mse_pct": pct(row["mse"], base["mse"]),
                    "relative_mae_pct": pct(row["mae"], base["mae"]),
                    "mse_win": row["mse"] < base["mse"],
                    "mae_win": row["mae"] < base["mae"],
                }
            )
    return output


def summarize_main(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model in MODELS:
        subset = [row for row in rows if row["model"] == model]
        output.append(
            {
                "model": model,
                "strategy": LABELS[model],
                "settings": len(subset),
                "mean_mse": mean(row["mse"] for row in subset),
                "mean_mae": mean(row["mae"] for row in subset),
            }
        )
    return output


def summarize_deltas(rows: list[dict[str, Any]], group_key: str | None = None) -> list[dict[str, Any]]:
    output = []
    for baseline in BASELINES:
        baseline_rows = [row for row in rows if row["baseline_model"] == baseline]
        values = ["all"] if group_key is None else sorted({row[group_key] for row in baseline_rows})
        for value in values:
            subset = baseline_rows if group_key is None else [row for row in baseline_rows if row[group_key] == value]
            item = {
                "strategy": LABELS[PRIMARY],
                "baseline_strategy": LABELS[baseline],
                "settings": len(subset),
                "mse_wins": sum(1 for row in subset if row["mse_win"]),
                "mae_wins": sum(1 for row in subset if row["mae_win"]),
                "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in subset),
                "mean_relative_mae_pct": mean(row["relative_mae_pct"] for row in subset),
            }
            if group_key is not None:
                item[group_key] = value
            output.append(item)
    return output


def summarize_segments(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["baseline_model"], *[row[key] for key in group_keys])
        grouped[key].append(row)

    output = []
    for key, subset in sorted(grouped.items()):
        baseline = key[0]
        item = {
            "strategy": LABELS[PRIMARY],
            "baseline_strategy": LABELS[baseline],
            "segments": len(subset),
            "mse_wins": sum(1 for row in subset if row["mse_win"]),
            "mae_wins": sum(1 for row in subset if row["mae_win"]),
            "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in subset),
            "mean_relative_mae_pct": mean(row["relative_mae_pct"] for row in subset),
        }
        for field, value in zip(group_keys, key[1:]):
            item[field] = value
        output.append(item)
    return output


def load_trace_summary(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        trace = read_csv(run_dir(raw_root, PRIMARY, dataset) / "supervision_trace.csv")
        rows.append(
            {
                "model": PRIMARY,
                "strategy": LABELS[PRIMARY],
                "dataset": dataset,
                "trace_rows": len(trace),
                "mean_learnable_blocks": mean(float(row["residual_stability_learnable_blocks"]) for row in trace),
                "mean_noisy_blocks": mean(float(row["residual_stability_noisy_blocks"]) for row in trace),
                "mean_ambiguous_blocks": mean(float(row["residual_stability_ambiguous_blocks"]) for row in trace),
                "mean_noisy_suppression_ratio": mean(
                    float(row["residual_stability_noisy_suppression_ratio"]) for row in trace
                ),
                "mean_region_early_steps": mean(float(row["region_routed_early_steps"]) for row in trace),
                "mean_region_middle_steps": mean(float(row["region_routed_middle_steps"]) for row in trace),
                "mean_region_late_steps": mean(float(row["region_routed_late_steps"]) for row in trace),
                "mean_region_abs_residual": mean(float(row["region_routed_mean_abs_residual"]) for row in trace),
                "last_region_abs_residual": float(trace[-1]["region_routed_mean_abs_residual"]),
            }
        )
    return rows


def load_training_summary(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            log_rows = read_csv(run_dir(raw_root, model, dataset) / "training_log.csv")
            val_values = [float(row["val_mean_mse"]) for row in log_rows]
            train_values = [float(row["train_prediction_loss"]) for row in log_rows]
            best_index = min(range(len(val_values)), key=val_values.__getitem__)
            item = {
                "model": model,
                "strategy": LABELS[model],
                "dataset": dataset,
                "epochs_ran": len(log_rows),
                "best_epoch": best_index + 1,
                "first_val_mean_mse": val_values[0],
                "best_val_mean_mse": val_values[best_index],
                "last_val_mean_mse": val_values[-1],
                "post_best_val_drift_pct": pct(val_values[-1], val_values[best_index]),
                "train_loss_drop_pct": pct(train_values[-1], train_values[0]),
            }
            for col in ["train_region_grad_norm_early", "train_region_grad_norm_middle", "train_region_grad_norm_late"]:
                if col in log_rows[0]:
                    values = [float(row[col]) for row in log_rows]
                    item[f"mean_{col}"] = mean(values)
                    item[f"last_{col}"] = values[-1]
            rows.append(item)
    return rows


def load_region_summary(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            path = run_dir(raw_root, PRIMARY, dataset) / f"h{horizon}" / "region_readout_stats.csv"
            for row in read_csv(path):
                rows.append(
                    {
                        "model": PRIMARY,
                        "strategy": LABELS[PRIMARY],
                        "dataset": dataset,
                        "horizon": horizon,
                        "scope": row["scope"],
                        "residual_mse": float(row["residual_mse"]),
                        "residual_mae": float(row["residual_mae"]),
                        "max_abs_residual": float(row["max_abs_residual"]),
                    }
                )
    return rows


def load_prefix_summary(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            values = read_csv(run_dir(raw_root, model, dataset) / "prefix_consistency.csv")
            rows.append(
                {
                    "model": model,
                    "strategy": LABELS[model],
                    "dataset": dataset,
                    "max_prefix_mismatch_mse": max(float(row["prefix_mismatch_mse"]) for row in values),
                    "max_prefix_mismatch_mae": max(float(row["prefix_mismatch_mae"]) for row in values),
                    "max_truth_alignment_mse": max(float(row["truth_alignment_mse"]) for row in values),
                }
            )
    return rows


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            if isinstance(value, float):
                value = fmt_float(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(
    output_dir: Path,
    main_summary: list[dict[str, Any]],
    main_deltas: list[dict[str, Any]],
    dataset_summary: list[dict[str, Any]],
    segment_region_summary: list[dict[str, Any]],
    h720_segment_deltas: list[dict[str, Any]],
    trace_summary: list[dict[str, Any]],
    training_summary: list[dict[str, Any]],
    region_summary: list[dict[str, Any]],
    prefix_summary: list[dict[str, Any]],
) -> str:
    mean_by_model = {row["model"]: row["mean_mse"] for row in main_summary}
    by_delta = {(row["baseline_model"], row["dataset"], row["horizon"]): row for row in main_deltas}
    by_segment = {
        (row["baseline_model"], row["dataset"], row["horizon"], row["segment"]): row for row in h720_segment_deltas
    }
    primary_vs_single = [row for row in main_deltas if row["baseline_model"] == SINGLE]
    primary_vs_r3 = [row for row in main_deltas if row["baseline_model"] == R3]
    primary_vs_hssga = [row for row in main_deltas if row["baseline_model"] == HSSGA]
    single_wins = sum(1 for row in primary_vs_single if row["mse_win"])
    r3_wins = sum(1 for row in primary_vs_r3 if row["mse_win"])
    hssga_wins = sum(1 for row in primary_vs_hssga if row["mse_win"])

    etth2_h96 = by_delta[(SINGLE, "ETTh2", 96)]["relative_mse_pct"]
    etth2_h192 = by_delta[(SINGLE, "ETTh2", 192)]["relative_mse_pct"]
    weather_late_vs_single = by_segment[(SINGLE, "Weather", 720, "337-720")]["relative_mse_pct"]
    weather_late_vs_r3 = by_segment[(R3, "Weather", 720, "337-720")]["relative_mse_pct"]
    weather_late_vs_hssga = by_segment[(HSSGA, "Weather", 720, "337-720")]["relative_mse_pct"]

    max_prefix_mse = max(
        row["max_prefix_mismatch_mse"] for row in prefix_summary if row["model"] == PRIMARY
    )
    weather_trace = next(row for row in trace_summary if row["dataset"] == "Weather")
    etth2_trace = next(row for row in trace_summary if row["dataset"] == "ETTh2")
    weather_train = next(row for row in training_summary if row["model"] == PRIMARY and row["dataset"] == "Weather")
    etth2_train = next(row for row in training_summary if row["model"] == PRIMARY and row["dataset"] == "ETTh2")
    weather_h720_late_region = next(
        row for row in region_summary
        if row["dataset"] == "Weather" and row["horizon"] == 720 and row["scope"] == "337-720"
    )

    verdict = "fail_as_core_candidate"
    lines = [
        "# Phase4-HSSG-B/C Learnability Region Routing Gate Report",
        "",
        "## 11-step Record",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | Step 9/10：评估 learnability-conditioned region routing gate |",
        "| `problem` | HSSG-A fixed region path 修复部分 late/long 但牺牲 early；本轮验证 learnability mask 是否能保住 early 并修复 late |",
        "| `existence_evidence` | HSSG-A gate、RG-B residual-stability trace、remote artifacts |",
        "| `idea` | shared base 用 h720-only prefix-risk；learnable residual blocks 只更新 detached region readout path |",
        "| `theory_check` | 如果 proxy 有效，learnability mask 应减少 noisy/ambiguous auxiliary pressure，并避免 fixed region path 的 early 损伤 |",
        "| `design` | ETTh2 + Weather；比较 full-time、single-prefix、R.3、HSSG-A、HSSG-B/C；seed 2021 |",
        "| `gate` | vs single >=5/8 main wins；ETTh2 h96/h192 不超过 +1%；Weather h720 late vs R.3 <= +1%；trace/grad 非 collapse |",
        f"| `artifacts` | `{output_dir}` |",
        f"| `decision` | {verdict}；learnability mask 有审计信号，但性能和 story gate 均失败 |",
        "",
        "## Main Metrics",
        "",
        markdown_table(main_summary, ["strategy", "settings", "mean_mse", "mean_mae"]),
        "",
        "## HSSG-B/C vs Baselines",
        "",
        markdown_table(
            summarize_deltas(main_deltas),
            ["baseline_strategy", "settings", "mse_wins", "mean_relative_mse_pct", "mean_relative_mae_pct"],
        ),
        "",
        "## Dataset Split",
        "",
        markdown_table(
            dataset_summary,
            ["baseline_strategy", "dataset", "settings", "mse_wins", "mean_relative_mse_pct"],
        ),
        "",
        "## Gate Checks",
        "",
        f"- [Fact] Absolute mean MSE vs single-prefix `{fmt_pct(pct(mean_by_model[PRIMARY], mean_by_model[SINGLE]))}`；vs R.3 `{fmt_pct(pct(mean_by_model[PRIMARY], mean_by_model[R3]))}`；vs HSSG-A `{fmt_pct(pct(mean_by_model[PRIMARY], mean_by_model[HSSGA]))}`。",
        f"- [Fact] Main MSE wins: vs single `{single_wins}/8`，vs R.3 `{r3_wins}/8`，vs HSSG-A `{hssga_wins}/8`。",
        f"- [Fact] ETTh2 h96/h192 vs single: `{fmt_pct(etth2_h96)}` / `{fmt_pct(etth2_h192)}`，h96 仍超过 +1% gate。",
        f"- [Fact] Weather h720 late segment vs single `{fmt_pct(weather_late_vs_single)}`；vs R.3 `{fmt_pct(weather_late_vs_r3)}`；vs HSSG-A `{fmt_pct(weather_late_vs_hssga)}`。",
        f"- [Fact] Prefix mismatch max MSE `{max_prefix_mse:.3e}`，prefix consistency 没有问题。",
        f"- [Fact] Weather trace: mean learnable `{weather_trace['mean_learnable_blocks']:.2f}` blocks，noisy `{weather_trace['mean_noisy_blocks']:.2f}` blocks，late active steps `{weather_trace['mean_region_late_steps']:.1f}`。",
        f"- [Fact] Region path 非 collapse：Weather h720 late residual MAE `{weather_h720_late_region['residual_mae']:.6f}`；但它没有转化为 metric gain。",
        f"- [Fact] Best epoch: ETTh2 `{etth2_train['best_epoch']}/{etth2_train['epochs_ran']}`，Weather `{weather_train['best_epoch']}/{weather_train['epochs_ran']}`。",
        "",
        "## Segment/Region Evidence",
        "",
        markdown_table(
            [
                row for row in segment_region_summary
                if row["baseline_strategy"] in {LABELS[SINGLE], LABELS[R3], LABELS[HSSGA]}
            ],
            [
                "baseline_strategy",
                "dataset",
                "future_region",
                "segments",
                "mse_wins",
                "mean_relative_mse_pct",
            ],
        ),
        "",
        "## Trace Summary",
        "",
        markdown_table(
            trace_summary,
            [
                "dataset",
                "mean_learnable_blocks",
                "mean_noisy_blocks",
                "mean_noisy_suppression_ratio",
                "mean_region_early_steps",
                "mean_region_middle_steps",
                "mean_region_late_steps",
                "mean_region_abs_residual",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "[Strong Evidence] learnability router 本身没有 collapse。trace 显示 ETTh2/Weather 都能区分 learnable/noisy/ambiguous blocks，region grad 和 residual 也非零，并且 active steps 主要集中在 late region。",
        "",
        "[Counter-Evidence] 这条路没有修复 HSSG-A 的核心问题。HSSG-B/C absolute mean MSE 比 single-prefix 差 `"
        + fmt_pct(pct(mean_by_model[PRIMARY], mean_by_model[SINGLE]))
        + "`，比 HSSG-A 差 `"
        + fmt_pct(pct(mean_by_model[PRIMARY], mean_by_model[HSSGA]))
        + "`；Weather h720 late segment 比 R.3 差 `"
        + fmt_pct(weather_late_vs_r3)
        + "`，比 HSSG-A 差 `"
        + fmt_pct(weather_late_vs_hssga)
        + "`。这直接否定了“residual-stability learnability mask + detached region readout”作为 core method。",
        "",
        "[Inference] failure 不是实现未激活，而是 routing target 错了：learnability proxy 将大量 pressure 放到 late region，但 detached low-rank region path 不能承载 Weather 的 late structure，反而削弱了 fixed HSSG-A 已经获得的 late gain。换言之，当前 HSSG-B/C 解决了 noisy pressure 的形式问题，却没有解决 region path 的表达/优化能力问题。",
        "",
        "[Decision] 不继续 sweep `aux_weight`、`top_ratio` 或 rank。HSSG-B/C 作为 Step 7 候选失败；HSSG 主线需要回到 Step 5/6 重新评估 carrier，而不是继续在 target-set readout 旁增加 residual paths。",
        "",
        "## Next Direction",
        "",
        "[Rollback Point] 回到 Step 5/6。保留的事实是：h720-only prefix-risk base 仍强，HSSG-A 的 fixed region path 对 late/long 有 partial signal；但 learnability-conditioned detached region path 会破坏 Weather late。",
        "",
        "[Next] 下一步应停止增加 readout residual path，转向 training protocol / representation carrier：优先测试 prefix-risk stabilized base + non-detached richer target/readout subset，或学习率/early-best calibration；若仍坚持 routing，必须让 routing 更新 `condition_head/target_states` 的受控子空间，而不是只更新小 residual head。",
    ]
    return "\n".join(lines).rstrip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("analysis/phase4_hssg_learnability_routing_gate_20260625/raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/phase4_hssg_learnability_routing_gate_20260625"),
    )
    args = parser.parse_args()

    main_rows = load_main_metrics(args.raw_root)
    main_deltas = add_main_deltas(main_rows)
    segment_rows = load_segment_metrics(args.raw_root)
    segment_deltas = add_segment_deltas(segment_rows)
    h720_segment_deltas = [row for row in segment_deltas if row["horizon"] == 720]
    trace_summary = load_trace_summary(args.raw_root)
    training_summary = load_training_summary(args.raw_root)
    region_summary = load_region_summary(args.raw_root)
    prefix_summary = load_prefix_summary(args.raw_root)
    main_summary = summarize_main(main_rows)
    dataset_summary = summarize_deltas(main_deltas, "dataset")
    segment_region_summary = summarize_segments(segment_deltas, ("dataset", "future_region"))

    write_csv(args.output_dir / "phase4_hssg_learnability_main_metrics.csv", main_rows)
    write_csv(args.output_dir / "phase4_hssg_learnability_main_deltas.csv", main_deltas)
    write_csv(args.output_dir / "phase4_hssg_learnability_main_summary.csv", main_summary)
    write_csv(args.output_dir / "phase4_hssg_learnability_dataset_summary.csv", dataset_summary)
    write_csv(args.output_dir / "phase4_hssg_learnability_segment_deltas.csv", segment_deltas)
    write_csv(args.output_dir / "phase4_hssg_learnability_h720_segment_deltas.csv", h720_segment_deltas)
    write_csv(args.output_dir / "phase4_hssg_learnability_segment_region_summary.csv", segment_region_summary)
    write_csv(args.output_dir / "phase4_hssg_learnability_trace_summary.csv", trace_summary)
    write_csv(args.output_dir / "phase4_hssg_learnability_training_summary.csv", training_summary)
    write_csv(args.output_dir / "phase4_hssg_learnability_region_readout_summary.csv", region_summary)
    write_csv(args.output_dir / "phase4_hssg_learnability_prefix_consistency_summary.csv", prefix_summary)

    report = build_report(
        args.output_dir,
        main_summary,
        main_deltas,
        dataset_summary,
        segment_region_summary,
        h720_segment_deltas,
        trace_summary,
        training_summary,
        region_summary,
        prefix_summary,
    )
    report_path = args.output_dir / "phase4_hssg_learnability_routing_gate_report.md"
    report_path.write_text(report + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
