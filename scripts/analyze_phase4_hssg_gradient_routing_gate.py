from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PRIMARY = "PatchEncoderHSSGRegionRoutedReadout"
FULL = "PatchEncoderFullTimeMSE720"
SINGLE = "PatchEncoderSingle720PrefixRisk"
R3 = "PatchEncoderR3PrefixRisk"

MODELS = [FULL, SINGLE, R3, PRIMARY]
BASELINES = [FULL, SINGLE, R3]
LABELS = {
    FULL: "D0_full_time_mse",
    SINGLE: "D1_single_720_prefix_risk",
    R3: "D2_r3_prefix_risk",
    PRIMARY: "HSSG-A_region_routed_readout",
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
            path = run_dir(raw_root, model, dataset) / "metrics_by_target_horizon.csv"
            for row in read_csv(path):
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


def summarize_segments(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for baseline in BASELINES:
            if row["baseline_model"] == baseline:
                key = (baseline, *[row[key] for key in group_keys])
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
                "first_train_prediction_loss": train_values[0],
                "best_epoch_train_prediction_loss": train_values[best_index],
                "last_train_prediction_loss": train_values[-1],
                "post_best_val_drift_pct": pct(val_values[-1], val_values[best_index]),
                "train_loss_drop_pct": pct(train_values[-1], train_values[0]),
            }
            grad_cols = ["train_region_grad_norm_early", "train_region_grad_norm_middle", "train_region_grad_norm_late"]
            for col in grad_cols:
                if col in log_rows[0]:
                    values = [float(row[col]) for row in log_rows]
                    item[f"mean_{col}"] = mean(values)
                    item[f"last_{col}"] = values[-1]
            rows.append(item)
    return rows


def load_region_readout_summary(raw_root: Path) -> list[dict[str, Any]]:
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
            path = run_dir(raw_root, model, dataset) / "prefix_consistency.csv"
            values = read_csv(path)
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


def mean_mse_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    return "\n".join(lines).rstrip()


def build_report(
    output_dir: Path,
    main_summary: list[dict[str, Any]],
    main_dataset_summary: list[dict[str, Any]],
    main_deltas: list[dict[str, Any]],
    segment_region_summary: list[dict[str, Any]],
    h720_segment_deltas: list[dict[str, Any]],
    training_summary: list[dict[str, Any]],
    region_summary: list[dict[str, Any]],
    prefix_summary: list[dict[str, Any]],
) -> str:
    by_delta = {(row["baseline_model"], row["dataset"], row["horizon"]): row for row in main_deltas}
    mean_by_model = {row["model"]: row["mean_mse"] for row in main_summary}
    hssg_abs_vs_single = pct(mean_by_model[PRIMARY], mean_by_model[SINGLE])
    hssg_abs_vs_r3 = pct(mean_by_model[PRIMARY], mean_by_model[R3])
    single_rows = [row for row in main_deltas if row["baseline_model"] == SINGLE]
    r3_rows = [row for row in main_deltas if row["baseline_model"] == R3]
    single_wins = sum(1 for row in single_rows if row["mse_win"])
    r3_wins = sum(1 for row in r3_rows if row["mse_win"])
    single_mean_rel = mean(row["relative_mse_pct"] for row in single_rows)
    r3_mean_rel = mean(row["relative_mse_pct"] for row in r3_rows)

    etth2_h96 = by_delta[(SINGLE, "ETTh2", 96)]["relative_mse_pct"]
    etth2_h192 = by_delta[(SINGLE, "ETTh2", 192)]["relative_mse_pct"]
    weather_h720_r3 = by_delta[(R3, "Weather", 720)]["relative_mse_pct"]
    weather_h720_single = by_delta[(SINGLE, "Weather", 720)]["relative_mse_pct"]

    h720_late = [
        row for row in h720_segment_deltas
        if row["dataset"] == "Weather" and row["horizon"] == 720 and row["segment"] == "337-720"
    ]
    weather_late_vs_r3 = next(row for row in h720_late if row["baseline_model"] == R3)
    weather_late_vs_single = next(row for row in h720_late if row["baseline_model"] == SINGLE)

    hssg_training = [row for row in training_summary if row["model"] == PRIMARY]
    weather_training = next(row for row in hssg_training if row["dataset"] == "Weather")
    etth2_training = next(row for row in hssg_training if row["dataset"] == "ETTh2")
    region_all_h720_weather = next(
        row for row in region_summary
        if row["dataset"] == "Weather" and row["horizon"] == 720 and row["scope"] == "all"
    )
    region_late_h720_weather = next(
        row for row in region_summary
        if row["dataset"] == "Weather" and row["horizon"] == 720 and row["scope"] == "337-720"
    )
    prefix_hssg = [row for row in prefix_summary if row["model"] == PRIMARY]
    max_prefix_mse = max(row["max_prefix_mismatch_mse"] for row in prefix_hssg)

    verdict = "partial_pass"
    if single_wins < 5 or etth2_h96 > 1.0 or etth2_h192 > 1.0 or weather_late_vs_r3["relative_mse_pct"] > 1.0:
        verdict = "fail_as_core_candidate"

    lines = [
        "# Phase4-HSSG-A Gradient Routing Gate Report",
        "",
        "## 11-step Record",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | Step 9/10：评估远程结果并决定是否通过 gate |",
        "| `problem` | loss-only HSS 无法决定 future-unit gradient 应更新哪些参数子空间 |",
        "| `existence_evidence` | Phase4-R3D、SRP group/tuner 思想、前序 gradient-routing partial evidence |",
        "| `idea` | 在 h720-only prefix-risk objective 下，用 early/middle/late region-routed readout path 控制 gradient/update subspace |",
        "| `theory_check` | 若不同 future regions 的噪声/可预测性不同，region-specific update 应缓解 shared readout conflict；若只带来 tiny residual 或牺牲 short horizon，则理论实现不足 |",
        "| `design` | ETTh2 + Weather，seed2021；比较 full-time、single-prefix、R3、HSSG-A |",
        "| `gate` | vs single overall 改善且 >=5/8 main wins；Weather h720 late vs R.3 gap <= +1%；ETTh2 h96/h192 vs single 不超过 +1%；region path 不 collapse |",
        f"| `artifacts` | `{output_dir}` |",
        f"| `decision` | {verdict}；见下方证据与 rollback |",
        "",
        "## Main Metrics",
        "",
        markdown_table(main_summary, ["strategy", "settings", "mean_mse", "mean_mae"]),
        "",
        "## HSSG-A vs Baselines",
        "",
        markdown_table(
            summarize_deltas(main_deltas),
            ["baseline_strategy", "settings", "mse_wins", "mean_relative_mse_pct", "mean_relative_mae_pct"],
        ),
        "",
        "## Dataset Split",
        "",
        markdown_table(
            main_dataset_summary,
            ["baseline_strategy", "dataset", "settings", "mse_wins", "mean_relative_mse_pct"],
        ),
        "",
        "## Gate Checks",
        "",
        f"- [Fact] Absolute mean MSE vs `single_720_prefix_risk`: `{fmt_pct(hssg_abs_vs_single)}`；vs R.3: `{fmt_pct(hssg_abs_vs_r3)}`。",
        f"- [Fact] vs `single_720_prefix_risk`: HSSG-A main MSE wins `{single_wins}/8`, mean relative MSE `{fmt_pct(single_mean_rel)}`.",
        f"- [Fact] vs `r3_prefix_risk`: HSSG-A main MSE wins `{r3_wins}/8`, mean relative MSE `{fmt_pct(r3_mean_rel)}`.",
        f"- [Fact] ETTh2 h96/h192 vs single: `{fmt_pct(etth2_h96)}` / `{fmt_pct(etth2_h192)}`；h96 超过 +1% gate，h192 通过。",
        f"- [Fact] Weather h720 vs single: `{fmt_pct(weather_h720_single)}`；vs R.3: `{fmt_pct(weather_h720_r3)}`。",
        f"- [Fact] Weather h720 late segment `337-720` vs single: `{fmt_pct(weather_late_vs_single['relative_mse_pct'])}`；vs R.3: `{fmt_pct(weather_late_vs_r3['relative_mse_pct'])}`。",
        f"- [Fact] HSSG prefix mismatch max MSE `{max_prefix_mse:.3e}`，prefix consistency 没有问题。",
        f"- [Fact] HSSG Weather h720 region residual all MAE `{region_all_h720_weather['residual_mae']:.6f}`，late MAE `{region_late_h720_weather['residual_mae']:.6f}`；path 非零但 residual magnitude 很小。",
        f"- [Fact] HSSG best epoch: ETTh2 `{etth2_training['best_epoch']}/{etth2_training['epochs_ran']}`，Weather `{weather_training['best_epoch']}/{weather_training['epochs_ran']}`；Weather best 在 epoch 1。",
        "",
        "## Segment/Region Evidence",
        "",
        markdown_table(
            [
                row for row in segment_region_summary
                if row["baseline_strategy"] in {LABELS[SINGLE], LABELS[R3]}
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
        "## Interpretation",
        "",
        "[Strong Evidence] HSSG-A 不是无效机制。它的 absolute mean MSE 略优于 `single_720_prefix_risk`，在 ETTh2/Weather 的 h720 主指标上都改善 single-prefix，并且 Weather h720 late segment 已接近 R.3。这说明把一部分更新能力放到 region-routed readout path，确实能吸收 prefix-risk objective 中 shared path 难以处理的 residual。",
        "",
        "[Counter-Evidence] HSSG-A 不能作为当前 core candidate 直接推进。它只赢 `single_720_prefix_risk` 的 `4/8` main settings，没有达到 `5/8` gate；per-setting relative MSE 也略差于 single-prefix；ETTh2 h96 相对 single-prefix 变差超过 +1%；Weather early/middle regions 相对 single-prefix 和 R.3 都不稳。也就是说，HSSG-A 修到了一部分 late/long 问题，但把 short/early 优势让出去了。",
        "",
        "[Inference] 当前 region-routed readout 更像一个 small residual corrector，而不是充分的 gradient scheduler。证据是 residual path 非零但幅度很小，Weather best epoch 仍在 epoch 1，说明 optimization 仍是早期就被 validation 选择，后续训练主要 drift；这不支持继续简单扩大训练轮数或继续叠 loss weight。",
        "",
        "[Decision] HSSG-A 作为 Step 7 最小实现是 `partial evidence`，但 Step 10 gate 不通过。它支持“gradient/update subspace 是有价值轴”，但否定了“固定 early/middle/late low-rank readout residual 足够成为主线方法”。",
        "",
        "## Rollback And Next Direction",
        "",
        "[Rollback Point] 回到 Step 6，而不是 Step 4：核心 HSSG 主线不撤销，但当前 HSSG-A 设计需要重构。下一步不应 sweep rank/dropout/scale，因为失败点不是单纯 capacity；失败点是 fixed region path 不知道哪些 late residual 可学习、哪些应该被阻断。",
        "",
        "[Next] 进入 HSSG-B/C 的混合方向：`learnability-conditioned gradient routing`。具体应让 prefix-stable shared path 保留 single-prefix 的 short-horizon 优势，同时让 late/noisy units 根据 residual stability/predictability 决定进入 region path、shared path、或 no-update path。",
        "",
    ]
    return "\n".join(lines).rstrip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("analysis/phase4_hssg_gradient_routing_gate_20260625/raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/phase4_hssg_gradient_routing_gate_20260625"),
    )
    args = parser.parse_args()

    main_rows = load_main_metrics(args.raw_root)
    main_deltas = add_main_deltas(main_rows)
    segment_rows = load_segment_metrics(args.raw_root)
    segment_deltas = add_segment_deltas(segment_rows)
    h720_segment_deltas = [row for row in segment_deltas if row["horizon"] == 720]
    training_summary = load_training_summary(args.raw_root)
    region_summary = load_region_readout_summary(args.raw_root)
    prefix_summary = load_prefix_summary(args.raw_root)

    main_summary = mean_mse_by_model(main_rows)
    main_dataset_summary = summarize_deltas(main_deltas, "dataset")
    segment_region_summary = summarize_segments(segment_deltas, ("dataset", "future_region"))

    write_csv(args.output_dir / "phase4_hssg_main_metrics.csv", main_rows)
    write_csv(args.output_dir / "phase4_hssg_main_deltas.csv", main_deltas)
    write_csv(args.output_dir / "phase4_hssg_main_summary.csv", main_summary)
    write_csv(args.output_dir / "phase4_hssg_dataset_summary.csv", main_dataset_summary)
    write_csv(args.output_dir / "phase4_hssg_segment_deltas.csv", segment_deltas)
    write_csv(args.output_dir / "phase4_hssg_segment_region_summary.csv", segment_region_summary)
    write_csv(args.output_dir / "phase4_hssg_h720_segment_deltas.csv", h720_segment_deltas)
    write_csv(args.output_dir / "phase4_hssg_training_summary.csv", training_summary)
    write_csv(args.output_dir / "phase4_hssg_region_readout_summary.csv", region_summary)
    write_csv(args.output_dir / "phase4_hssg_prefix_consistency_summary.csv", prefix_summary)

    report = build_report(
        args.output_dir,
        main_summary,
        main_dataset_summary,
        main_deltas,
        segment_region_summary,
        h720_segment_deltas,
        training_summary,
        region_summary,
        prefix_summary,
    )
    report_path = args.output_dir / "phase4_hssg_gradient_routing_gate_report.md"
    report_path.write_text(report + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
