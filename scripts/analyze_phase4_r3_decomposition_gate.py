from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


FULL = "PatchEncoderFullTimeMSE720"
MIXED = "PatchEncoderHorizonMixedUniform"
SINGLE = "PatchEncoderSingle720PrefixRisk"
R3 = "PatchEncoderR3PrefixRisk"
MODELS = [FULL, MIXED, SINGLE, R3]
LABELS = {
    FULL: "full_time_mse720",
    MIXED: "mixed_horizon_uniform",
    SINGLE: "single_720_prefix_risk",
    R3: "r3_prefix_risk",
}
DATASETS = ["ETTh2", "Weather"]
HORIZONS = [96, 192, 336, 720]
HORIZON_LABEL = "mixed_h96_h192_h336_h720"
TRAIN_STEP_COLUMNS = {
    96: "train_steps_h96",
    192: "train_steps_h192",
    336: "train_steps_h336",
    720: "train_steps_h720",
}


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


def improvement_pct(candidate: float, baseline: float) -> float:
    return (1.0 - candidate / baseline) * 100.0


def fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def fmt_float(value: float) -> str:
    return f"{value:.6f}"


def run_dir(raw_root: Path, model: str, dataset: str) -> Path:
    return raw_root / model / dataset / HORIZON_LABEL / "seed2021"


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


def decompose_main(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["model"], row["dataset"], row["horizon"]): row for row in rows}
    output = []
    candidates = [MIXED, SINGLE, R3]
    for dataset in DATASETS:
        for horizon in HORIZONS:
            full = by_key[(FULL, dataset, horizon)]
            values = {model: by_key[(model, dataset, horizon)] for model in candidates}
            winner = min(candidates, key=lambda model: values[model]["mse"])
            mixed_imp = full["mse"] - values[MIXED]["mse"]
            single_imp = full["mse"] - values[SINGLE]["mse"]
            r3_imp = full["mse"] - values[R3]["mse"]
            output.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "full_mse": full["mse"],
                    "mixed_mse": values[MIXED]["mse"],
                    "single_mse": values[SINGLE]["mse"],
                    "r3_mse": values[R3]["mse"],
                    "winner_strategy": LABELS[winner],
                    "mixed_vs_full_mse_pct": pct(values[MIXED]["mse"], full["mse"]),
                    "single_vs_full_mse_pct": pct(values[SINGLE]["mse"], full["mse"]),
                    "r3_vs_full_mse_pct": pct(values[R3]["mse"], full["mse"]),
                    "r3_vs_mixed_mse_pct": pct(values[R3]["mse"], values[MIXED]["mse"]),
                    "r3_vs_single_mse_pct": pct(values[R3]["mse"], values[SINGLE]["mse"]),
                    "single_vs_mixed_mse_pct": pct(values[SINGLE]["mse"], values[MIXED]["mse"]),
                    "mixed_improvement_abs": mixed_imp,
                    "single_improvement_abs": single_imp,
                    "r3_improvement_abs": r3_imp,
                    "interaction_residual_abs": r3_imp - mixed_imp - single_imp,
                    "mixed_explains_r3_ratio": mixed_imp / r3_imp if r3_imp else 0.0,
                    "single_explains_r3_ratio": single_imp / r3_imp if r3_imp else 0.0,
                    "r3_beats_mixed": values[R3]["mse"] < values[MIXED]["mse"],
                    "r3_beats_single": values[R3]["mse"] < values[SINGLE]["mse"],
                }
            )
    return output


def summarize_main(decomp: list[dict[str, Any]], group_key: str | None = None) -> list[dict[str, Any]]:
    output = []
    values = ["all"] if group_key is None else sorted({row[group_key] for row in decomp})
    for value in values:
        subset = decomp if group_key is None else [row for row in decomp if row[group_key] == value]
        row = {
            "group": value if group_key is None else value,
            "settings": len(subset),
            "mixed_wins": sum(1 for item in subset if item["winner_strategy"] == LABELS[MIXED]),
            "single_wins": sum(1 for item in subset if item["winner_strategy"] == LABELS[SINGLE]),
            "r3_wins": sum(1 for item in subset if item["winner_strategy"] == LABELS[R3]),
            "mixed_beats_full": sum(1 for item in subset if item["mixed_vs_full_mse_pct"] < 0.0),
            "single_beats_full": sum(1 for item in subset if item["single_vs_full_mse_pct"] < 0.0),
            "r3_beats_full": sum(1 for item in subset if item["r3_vs_full_mse_pct"] < 0.0),
            "mean_mixed_vs_full_mse_pct": mean(item["mixed_vs_full_mse_pct"] for item in subset),
            "mean_single_vs_full_mse_pct": mean(item["single_vs_full_mse_pct"] for item in subset),
            "mean_r3_vs_full_mse_pct": mean(item["r3_vs_full_mse_pct"] for item in subset),
            "mean_r3_vs_mixed_mse_pct": mean(item["r3_vs_mixed_mse_pct"] for item in subset),
            "mean_r3_vs_single_mse_pct": mean(item["r3_vs_single_mse_pct"] for item in subset),
            "mean_interaction_residual_abs": mean(item["interaction_residual_abs"] for item in subset),
        }
        if group_key is not None:
            row[group_key] = value
        output.append(row)
    return output


def decompose_segments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["model"], row["dataset"], row["horizon"], row["segment"]): row for row in rows}
    output = []
    candidates = [MIXED, SINGLE, R3]
    for dataset in DATASETS:
        for horizon in HORIZONS:
            segments = sorted({row["segment"] for row in rows if row["dataset"] == dataset and row["horizon"] == horizon})
            for segment in segments:
                full = by_key[(FULL, dataset, horizon, segment)]
                values = {model: by_key[(model, dataset, horizon, segment)] for model in candidates}
                winner = min(candidates, key=lambda model: values[model]["mse"])
                start, end = parse_segment(segment)
                output.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "segment": segment,
                        "segment_start": start,
                        "segment_end": end,
                        "future_region": future_region(end),
                        "full_mse": full["mse"],
                        "mixed_mse": values[MIXED]["mse"],
                        "single_mse": values[SINGLE]["mse"],
                        "r3_mse": values[R3]["mse"],
                        "winner_strategy": LABELS[winner],
                        "mixed_vs_full_mse_pct": pct(values[MIXED]["mse"], full["mse"]),
                        "single_vs_full_mse_pct": pct(values[SINGLE]["mse"], full["mse"]),
                        "r3_vs_full_mse_pct": pct(values[R3]["mse"], full["mse"]),
                        "r3_vs_single_mse_pct": pct(values[R3]["mse"], values[SINGLE]["mse"]),
                        "r3_vs_mixed_mse_pct": pct(values[R3]["mse"], values[MIXED]["mse"]),
                    }
                )
    return output


def summarize_segments(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    output = []
    for value in sorted({row[group_key] for row in rows}):
        subset = [row for row in rows if row[group_key] == value]
        output.append(
            {
                group_key: value,
                "settings": len(subset),
                "mixed_wins": sum(1 for row in subset if row["winner_strategy"] == LABELS[MIXED]),
                "single_wins": sum(1 for row in subset if row["winner_strategy"] == LABELS[SINGLE]),
                "r3_wins": sum(1 for row in subset if row["winner_strategy"] == LABELS[R3]),
                "mixed_beats_full": sum(1 for row in subset if row["mixed_vs_full_mse_pct"] < 0.0),
                "single_beats_full": sum(1 for row in subset if row["single_vs_full_mse_pct"] < 0.0),
                "r3_beats_full": sum(1 for row in subset if row["r3_vs_full_mse_pct"] < 0.0),
                "mean_mixed_vs_full_mse_pct": mean(row["mixed_vs_full_mse_pct"] for row in subset),
                "mean_single_vs_full_mse_pct": mean(row["single_vs_full_mse_pct"] for row in subset),
                "mean_r3_vs_full_mse_pct": mean(row["r3_vs_full_mse_pct"] for row in subset),
            }
        )
    return output


def load_training_summary(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            base = run_dir(raw_root, model, dataset)
            with (base / "effective_config.json").open() as handle:
                config = json.load(handle)
            log_rows = read_csv(base / "training_log.csv")
            val_values = [float(row["val_mean_mse"]) for row in log_rows]
            train_values = [float(row["train_prediction_loss"]) for row in log_rows]
            best_index = min(range(len(val_values)), key=val_values.__getitem__)
            horizon_counts = {
                horizon: sum(int(float(row[column])) for row in log_rows)
                for horizon, column in TRAIN_STEP_COLUMNS.items()
            }
            observed_total = sum(horizon_counts.values())
            exposure_source = "train_steps_h*"
            if observed_total == 0:
                observed_total = sum(int(float(row["train_supervision_steps"])) for row in log_rows)
                horizon_counts = {horizon: 0 for horizon in HORIZONS}
                horizon_counts[int(config["supervision_pred_len"])] = observed_total
                exposure_source = "train_supervision_steps_proxy"
            row: dict[str, Any] = {
                "model": model,
                "strategy": LABELS[model],
                "dataset": dataset,
                "supervision_strategy": config["supervision_strategy"],
                "step_loss_weighting": config["step_loss_weighting"],
                "train_horizons_effective": ",".join(str(item) for item in config["train_horizons_effective"]),
                "training_evaluation_decoupled": config["training_evaluation_decoupled"],
                "epochs_ran": len(log_rows),
                "best_epoch": best_index + 1,
                "best_val_mean_mse": val_values[best_index],
                "last_val_mean_mse": val_values[-1],
                "post_best_val_drift_pct": pct(val_values[-1], val_values[best_index]),
                "train_loss_drop_pct": pct(train_values[-1], train_values[0]),
                "exposure_source": exposure_source,
                "observed_training_updates": observed_total,
            }
            for horizon in HORIZONS:
                row[f"updates_h{horizon}"] = horizon_counts[horizon]
                row[f"share_h{horizon}"] = horizon_counts[horizon] / observed_total if observed_total else 0.0
            rows.append(row)
    return rows


def load_objective_weights(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            path = run_dir(raw_root, model, dataset) / "objective_weight_stats.csv"
            for row in read_csv(path):
                rows.append(
                    {
                        "model": model,
                        "strategy": LABELS[model],
                        "dataset": dataset,
                        "scope": row["scope"],
                        "mode": row["mode"],
                        "mean_step_weight": float(row["mean_step_weight"]),
                        "weighted_pressure_share": float(row["weighted_pressure_share"]),
                        "pressure_share_delta_pct": float(row["pressure_share_delta_pct"]),
                    }
                )
    return rows


def pick(rows: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    for row in rows:
        if all(row[key] == value for key, value in criteria.items()):
            return row
    raise KeyError(criteria)


def write_report(
    path: Path,
    decomp: list[dict[str, Any]],
    overall: list[dict[str, Any]],
    dataset_summary: list[dict[str, Any]],
    horizon_summary: list[dict[str, Any]],
    segment_decomp: list[dict[str, Any]],
    segment_region_summary: list[dict[str, Any]],
    training_summary: list[dict[str, Any]],
) -> None:
    all_summary = overall[0]
    weather_rows = [row for row in decomp if row["dataset"] == "Weather"]
    etth2_rows = [row for row in decomp if row["dataset"] == "ETTh2"]
    single_weather_h720 = pick(decomp, dataset="Weather", horizon=720)
    etth2_h96 = pick(decomp, dataset="ETTh2", horizon=96)
    weather_late = pick(segment_decomp, dataset="Weather", horizon=720, segment="337-720")
    etth2_late = pick(segment_decomp, dataset="ETTh2", horizon=720, segment="337-720")
    lines = [
        "# Phase4-R3D Decomposition Gate 分析报告",
        "",
        "## 11-Step 记录",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | Step 9/10：评估 decomposition artifacts，并决定 rollback 与下一步方向 |",
        "| `problem` | R.3 是 compound baseline，需要拆开 mixed-horizon exposure 与 prefix-risk weighting |",
        "| `existence_evidence` | 远程 gate 包含 full-time、mixed-horizon uniform、single-720 prefix-risk、R.3 四组 controls |",
        "| `idea` | 将 R.3 分解为 exposure-only、weighting-only、compound 三类证据 |",
        "| `theory_check` | 在识别主导因素前，不能直接用 compound baseline 否定或肯定 HSS |",
        "| `design` | ETTh2/Weather，horizons 96/192/336/720，seed 2021，相同 target-set carrier |",
        "| `gate` | 判断 R.3 优势主要来自 exposure、weighting，还是二者 interaction |",
        f"| `artifacts` | `{path.parent}` |",
        "| `decision` | Prefix-risk pressure 是更可用的主导因素；mixed-horizon exposure 有收益但单独不足 |",
        "",
        "## 主结果",
        "",
        f"[Fact] `mixed_horizon_uniform` vs full-time: mean MSE `{fmt_pct(all_summary['mean_mixed_vs_full_mse_pct'])}`, beats full-time `{all_summary['mixed_beats_full']}/{all_summary['settings']}`, best among candidates `{all_summary['mixed_wins']}/{all_summary['settings']}`.",
        f"[Fact] `single_720_prefix_risk` vs full-time: mean MSE `{fmt_pct(all_summary['mean_single_vs_full_mse_pct'])}`, beats full-time `{all_summary['single_beats_full']}/{all_summary['settings']}`, best among candidates `{all_summary['single_wins']}/{all_summary['settings']}`.",
        f"[Fact] compound R.3 vs full-time: mean MSE `{fmt_pct(all_summary['mean_r3_vs_full_mse_pct'])}`, beats full-time `{all_summary['r3_beats_full']}/{all_summary['settings']}`, best among candidates `{all_summary['r3_wins']}/{all_summary['settings']}`.",
        f"[Fact] R.3 vs `single_720_prefix_risk`: mean MSE `{fmt_pct(all_summary['mean_r3_vs_single_mse_pct'])}`; lower is better for R.3.",
        f"[Fact] R.3 vs `mixed_horizon_uniform`: mean MSE `{fmt_pct(all_summary['mean_r3_vs_mixed_mse_pct'])}`.",
        "",
        "## 按 Horizon 分解",
        "",
        "| Dataset | Horizon | Full | Mixed uniform | Single prefix | R.3 | Winner | Mixed vs full | Single vs full | R.3 vs full |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in decomp:
        lines.append(
            f"| `{row['dataset']}` | {row['horizon']} | {fmt_float(row['full_mse'])} | "
            f"{fmt_float(row['mixed_mse'])} | {fmt_float(row['single_mse'])} | {fmt_float(row['r3_mse'])} | "
            f"`{row['winner_strategy']}` | {fmt_pct(row['mixed_vs_full_mse_pct'])} | "
            f"{fmt_pct(row['single_vs_full_mse_pct'])} | {fmt_pct(row['r3_vs_full_mse_pct'])} |"
        )
    lines += [
        "",
        "## 按 Dataset 汇总",
        "",
        "| Dataset | Mixed vs full | Single vs full | R.3 vs full | R.3 vs single | Winner counts M/S/R3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in dataset_summary:
        lines.append(
            f"| `{row['dataset']}` | {fmt_pct(row['mean_mixed_vs_full_mse_pct'])} | "
            f"{fmt_pct(row['mean_single_vs_full_mse_pct'])} | {fmt_pct(row['mean_r3_vs_full_mse_pct'])} | "
            f"{fmt_pct(row['mean_r3_vs_single_mse_pct'])} | "
            f"{row['mixed_wins']}/{row['single_wins']}/{row['r3_wins']} |"
        )
    lines += [
        "",
        "## 按 Horizon 汇总",
        "",
        "| Horizon | Mixed vs full | Single vs full | R.3 vs full | R.3 vs single | Winner counts M/S/R3 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in horizon_summary:
        lines.append(
            f"| {row['horizon']} | {fmt_pct(row['mean_mixed_vs_full_mse_pct'])} | "
            f"{fmt_pct(row['mean_single_vs_full_mse_pct'])} | {fmt_pct(row['mean_r3_vs_full_mse_pct'])} | "
            f"{fmt_pct(row['mean_r3_vs_single_mse_pct'])} | "
            f"{row['mixed_wins']}/{row['single_wins']}/{row['r3_wins']} |"
        )
    lines += [
        "",
        "## Segment 证据",
        "",
        f"[Fact] Weather h720 late segment `337-720`: mixed `{fmt_pct(weather_late['mixed_vs_full_mse_pct'])}`, single `{fmt_pct(weather_late['single_vs_full_mse_pct'])}`, R.3 `{fmt_pct(weather_late['r3_vs_full_mse_pct'])}` vs full-time.",
        f"[Fact] ETTh2 h720 late segment `337-720`: mixed `{fmt_pct(etth2_late['mixed_vs_full_mse_pct'])}`, single `{fmt_pct(etth2_late['single_vs_full_mse_pct'])}`, R.3 `{fmt_pct(etth2_late['r3_vs_full_mse_pct'])}` vs full-time.",
        "",
        "| Future region | Mixed vs full | Single vs full | R.3 vs full | Winner counts M/S/R3 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in segment_region_summary:
        lines.append(
            f"| `{row['future_region']}` | {fmt_pct(row['mean_mixed_vs_full_mse_pct'])} | "
            f"{fmt_pct(row['mean_single_vs_full_mse_pct'])} | {fmt_pct(row['mean_r3_vs_full_mse_pct'])} | "
            f"{row['mixed_wins']}/{row['single_wins']}/{row['r3_wins']} |"
        )
    lines += [
        "",
        "## Training Dynamics",
        "",
        "| Dataset | Strategy | Train horizons | Weighting | Best epoch | Best val | Last drift | Train loss change |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in training_summary:
        lines.append(
            f"| `{row['dataset']}` | `{row['strategy']}` | `{row['train_horizons_effective']}` | "
            f"`{row['step_loss_weighting']}` | {row['best_epoch']} | {fmt_float(row['best_val_mean_mse'])} | "
            f"{fmt_pct(row['post_best_val_drift_pct'])} | {fmt_pct(row['train_loss_drop_pct'])} |"
        )
    lines += [
        "",
        "## 解释",
        "",
        "[Strong Evidence] `mixed_horizon_uniform` 有用但不足。它在所有 main horizons 上都优于 h720 full-time，但只在 1 个 setting 中成为最优，并且在 Weather 上被 compound R.3 稳定压过。",
        "",
        "[Strong Evidence] `single_720_prefix_risk` 在不把 training 绑定到 evaluation horizons 的前提下承载了大量有效信号。它在 ETTh2 h96/h192 最优，并且在 Weather 上接近 R.3，同时保持 h720-only training。",
        "",
        f"[Fact] Weather h720: `single_720_prefix_risk` vs full-time is `{fmt_pct(single_weather_h720['single_vs_full_mse_pct'])}`, while R.3 vs full-time is `{fmt_pct(single_weather_h720['r3_vs_full_mse_pct'])}`.",
        f"[Fact] ETTh2 h96 is the main exception: R.3 beats single-prefix by `{fmt_pct(etth2_h96['r3_vs_single_mse_pct'])}`, indicating mixed exposure can help short horizon under ETTh2.",
        "",
        "[Inference] 需要重新解释 R.3 的优势：mixed-horizon exposure 确实有帮助，但 prefix-weighted supervision pressure 是更适合 horizon-agnostic 叙事的可操作因素。compound protocol 的额外价值主要体现在 Weather 和 long horizons。",
        "",
        "[Decision] 不把 R.3 作为 core story，也不把 mixed-horizon exposure 作为默认主线。下一步应基于 horizon-agnostic、h720-only 的 prefix/stability pressure，进一步从 scalar loss reweighting 推进到控制 gradient 更新位置。",
        "",
        "[Next Direction] 在 h720-only prefix-risk base 下测试 architecture-level HSS：允许 prefix-weighted supervision 更新受限的 future-state/readout subspace，同时保护 late/noisy regions，而不是继续采样 target horizons 或 repair R.3。",
        "",
        "[Rollback Point] 回到 Step 4/6：将 HSS 重新定义为 horizon-agnostic supervision pressure + gradient routing，而不是 mixed-horizon training schedule。",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase4 R.3 decomposition gate.")
    parser.add_argument("--raw-root", default="analysis/phase4_r3_decomposition_gate_20260625/raw")
    parser.add_argument("--output-root", default="analysis/phase4_r3_decomposition_gate_20260625")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    main_metrics = load_main_metrics(raw_root)
    main_decomp = decompose_main(main_metrics)
    overall = summarize_main(main_decomp)
    dataset_summary = summarize_main(main_decomp, "dataset")
    horizon_summary = summarize_main(main_decomp, "horizon")
    segment_metrics = load_segment_metrics(raw_root)
    segment_decomp = decompose_segments(segment_metrics)
    segment_region_summary = summarize_segments(segment_decomp, "future_region")
    training_summary = load_training_summary(raw_root)
    objective_weights = load_objective_weights(raw_root)

    write_csv(output_root / "phase4_r3_decomposition_main_metrics.csv", main_metrics)
    write_csv(output_root / "phase4_r3_decomposition_main_delta.csv", main_decomp)
    write_csv(output_root / "phase4_r3_decomposition_overall_summary.csv", overall)
    write_csv(output_root / "phase4_r3_decomposition_dataset_summary.csv", dataset_summary)
    write_csv(output_root / "phase4_r3_decomposition_horizon_summary.csv", horizon_summary)
    write_csv(output_root / "phase4_r3_decomposition_segment_delta.csv", segment_decomp)
    write_csv(output_root / "phase4_r3_decomposition_segment_region_summary.csv", segment_region_summary)
    write_csv(output_root / "phase4_r3_decomposition_training_summary.csv", training_summary)
    write_csv(output_root / "phase4_r3_decomposition_objective_weight_summary.csv", objective_weights)
    write_report(
        output_root / "phase4_r3_decomposition_gate_report.md",
        main_decomp,
        overall,
        dataset_summary,
        horizon_summary,
        segment_decomp,
        segment_region_summary,
        training_summary,
    )
    print(output_root / "phase4_r3_decomposition_gate_report.md")


if __name__ == "__main__":
    main()
