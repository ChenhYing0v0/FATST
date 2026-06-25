from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


FULL = "PatchEncoderFullTimeMSE720"
R3 = "PatchEncoderR3PrefixRisk"
MODELS = [FULL, R3]
LABELS = {
    FULL: "D0_full_time_mse720",
    R3: "D1_r3_prefix_risk",
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


def fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


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


def load_main_deltas(raw_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = []
    for model in MODELS:
        for dataset in DATASETS:
            path = run_dir(raw_root, model, dataset) / "metrics_by_target_horizon.csv"
            for row in read_csv(path):
                metrics.append(
                    {
                        "model": model,
                        "strategy": LABELS[model],
                        "dataset": dataset,
                        "horizon": int(row["target_horizon"]),
                        "mse": float(row["mse"]),
                        "mae": float(row["mae"]),
                    }
                )
    by_key = {(row["model"], row["dataset"], row["horizon"]): row for row in metrics}
    deltas = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            r3 = by_key[(R3, dataset, horizon)]
            full = by_key[(FULL, dataset, horizon)]
            deltas.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "r3_mse": r3["mse"],
                    "full_mse": full["mse"],
                    "r3_mae": r3["mae"],
                    "full_mae": full["mae"],
                    "r3_vs_full_mse_pct": pct(r3["mse"], full["mse"]),
                    "r3_vs_full_mae_pct": pct(r3["mae"], full["mae"]),
                    "r3_mse_win": r3["mse"] < full["mse"],
                    "r3_mae_win": r3["mae"] < full["mae"],
                }
            )
    return metrics, deltas


def load_segment_deltas(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            full_path = run_dir(raw_root, FULL, dataset) / f"h{horizon}" / "metrics_by_segment.csv"
            r3_path = run_dir(raw_root, R3, dataset) / f"h{horizon}" / "metrics_by_segment.csv"
            full_rows = {row["segment"]: row for row in read_csv(full_path)}
            for row in read_csv(r3_path):
                segment = row["segment"]
                start, end = parse_segment(segment)
                full = full_rows[segment]
                r3_mse = float(row["mse"])
                full_mse = float(full["mse"])
                r3_mae = float(row["mae"])
                full_mae = float(full["mae"])
                rows.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "segment": segment,
                        "segment_start": start,
                        "segment_end": end,
                        "future_region": future_region(end),
                        "r3_mse": r3_mse,
                        "full_mse": full_mse,
                        "r3_mae": r3_mae,
                        "full_mae": full_mae,
                        "r3_vs_full_mse_pct": pct(r3_mse, full_mse),
                        "r3_vs_full_mae_pct": pct(r3_mae, full_mae),
                        "r3_mse_win": r3_mse < full_mse,
                        "r3_mae_win": r3_mae < full_mae,
                    }
                )
    return rows


def summarize_deltas(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        groups.setdefault(key, []).append(row)
    summary = []
    for key, subset in sorted(groups.items()):
        item = {group_key: value for group_key, value in zip(group_keys, key, strict=True)}
        item.update(
            {
                "settings": len(subset),
                "r3_mse_wins": sum(1 for row in subset if row["r3_mse_win"]),
                "r3_mae_wins": sum(1 for row in subset if row["r3_mae_win"]),
                "mean_r3_vs_full_mse_pct": mean(row["r3_vs_full_mse_pct"] for row in subset),
                "mean_r3_vs_full_mae_pct": mean(row["r3_vs_full_mae_pct"] for row in subset),
            }
        )
        summary.append(item)
    return summary


def load_training_exposure(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            base = run_dir(raw_root, model, dataset)
            with (base / "effective_config.json").open() as handle:
                config = json.load(handle)
            log_rows = read_csv(base / "training_log.csv")
            observed = {
                horizon: sum(int(float(row[column])) for row in log_rows)
                for horizon, column in TRAIN_STEP_COLUMNS.items()
            }
            observed_total = sum(observed.values())
            supervised_steps = sum(int(float(row["train_supervision_steps"])) for row in log_rows)
            exposure_source = "train_steps_h*"
            if observed_total == 0:
                observed = {horizon: 0 for horizon in HORIZONS}
                observed[int(config["supervision_pred_len"])] = supervised_steps
                observed_total = supervised_steps
                exposure_source = "train_supervision_steps_proxy"
            val_values = [float(row["val_mean_mse"]) for row in log_rows]
            train_values = [float(row["train_prediction_loss"]) for row in log_rows]
            best_index = min(range(len(val_values)), key=val_values.__getitem__)
            row: dict[str, Any] = {
                "model": model,
                "strategy": LABELS[model],
                "dataset": dataset,
                "supervision_strategy": config["supervision_strategy"],
                "step_loss_weighting": config["step_loss_weighting"],
                "step_loss_alpha": config["step_loss_alpha"],
                "epochs_ran": len(log_rows),
                "best_epoch": best_index + 1,
                "best_val_mean_mse": val_values[best_index],
                "last_val_mean_mse": val_values[-1],
                "post_best_val_drift_pct": pct(val_values[-1], val_values[best_index]),
                "train_loss_drop_pct": pct(train_values[-1], train_values[0]),
                "exposure_source": exposure_source,
                "observed_training_updates": observed_total,
                "uses_mixed_horizon_training": any(observed[horizon] > 0 for horizon in [96, 192, 336]),
            }
            for horizon in HORIZONS:
                row[f"updates_h{horizon}"] = observed[horizon]
                row[f"share_h{horizon}"] = observed[horizon] / observed_total if observed_total else 0.0
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
                        "min_step_weight": float(row["min_step_weight"]),
                        "max_step_weight": float(row["max_step_weight"]),
                        "uniform_pressure_share": float(row["uniform_pressure_share"]),
                        "weighted_pressure_share": float(row["weighted_pressure_share"]),
                        "pressure_share_delta_pct": float(row["pressure_share_delta_pct"]),
                    }
                )
    return rows


def load_phase2_reference(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main_path = root / "r3_vs_uniform_main.csv"
    pressure_path = root / "objective_pressure_summary.csv"
    main_rows: list[dict[str, Any]] = []
    pressure_rows: list[dict[str, Any]] = []
    if main_path.exists():
        for row in read_csv(main_path):
            main_rows.append(
                {
                    "dataset": row["dataset"],
                    "horizon": int(row["horizon"]),
                    "r3_vs_uniform_mse_pct": float(row["r3_vs_uniform_mse_pct"]),
                    "r3_improves_uniform": row["r3_improves_uniform"] == "True",
                }
            )
    if pressure_path.exists():
        for row in read_csv(pressure_path):
            pressure_rows.append(
                {
                    "segment": row["segment"],
                    "uniform_pressure_share": float(row["uniform_pressure_share"]),
                    "prefix_pressure_share": float(row["prefix_pressure_share"]),
                    "pressure_share_delta_pct": float(row["pressure_share_delta_pct"]),
                    "raw_pressure_ratio": float(row["raw_pressure_ratio"]),
                }
            )
    return main_rows, pressure_rows


def pick(rows: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    for row in rows:
        if all(row[key] == value for key, value in criteria.items()):
            return row
    raise KeyError(criteria)


def write_report(
    path: Path,
    main_deltas: list[dict[str, Any]],
    segment_deltas: list[dict[str, Any]],
    dataset_summary: list[dict[str, Any]],
    horizon_summary: list[dict[str, Any]],
    segment_summary: list[dict[str, Any]],
    training_exposure: list[dict[str, Any]],
    objective_weights: list[dict[str, Any]],
    phase2_main: list[dict[str, Any]],
    phase2_pressure: list[dict[str, Any]],
) -> None:
    all_summary = summarize_deltas(main_deltas, [])[0]
    weather_rows = [row for row in main_deltas if row["dataset"] == "Weather"]
    etth2_rows = [row for row in main_deltas if row["dataset"] == "ETTh2"]
    weather_late = pick(
        segment_deltas,
        dataset="Weather",
        horizon=720,
        segment="337-720",
    )
    etth2_late = pick(
        segment_deltas,
        dataset="ETTh2",
        horizon=720,
        segment="337-720",
    )
    phase2_mean = mean(row["r3_vs_uniform_mse_pct"] for row in phase2_main) if phase2_main else 0.0
    r3_weight_rows = [
        row for row in objective_weights
        if row["model"] == R3 and row["dataset"] == "ETTh2" and not row["scope"].startswith("horizon_")
    ]

    lines = [
        "# Phase4-R3D R.3 Mechanism Diagnostic",
        "",
        "## 11-Step Record",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | Step 5/6: evaluate why R.3 is strong before designing the next method |",
        "| `problem` | R.3 dominates current Phase4 candidates, but its advantage source is confounded |",
        "| `existence_evidence` | Phase4 RG-B and OP-A returned artifacts; Phase2 objective-pressure diagnostic |",
        "| `idea` | Treat R.3 as a compound protocol: mixed-horizon exposure plus prefix-risk pressure |",
        "| `theory_check` | A method cannot claim horizon-agnostic supervision improvement until this compound baseline is decomposed |",
        "| `design` | Compare R.3 vs h720 full-time; audit actual train horizon exposure and objective pressure; use Phase2 as historical reference |",
        "| `gate` | If R.3 strength is compound, run decomposition controls before adding routing/architecture complexity |",
        f"| `artifacts` | `{path.parent}` |",
        "| `decision` | R.3 is not a single simple step-weight baseline; next work should decompose protocol factors |",
        "",
        "## Main Finding",
        "",
        f"[Fact] Current Phase4 R.3 vs `full_time_mse720`: MSE wins `{all_summary['r3_mse_wins']}/{all_summary['settings']}`, mean relative MSE `{fmt_pct(all_summary['mean_r3_vs_full_mse_pct'])}`.",
        f"[Fact] ETTh2 R.3 vs full-time: `{sum(1 for row in etth2_rows if row['r3_mse_win'])}/4`, mean `{fmt_pct(mean(row['r3_vs_full_mse_pct'] for row in etth2_rows))}`.",
        f"[Fact] Weather R.3 vs full-time: `{sum(1 for row in weather_rows if row['r3_mse_win'])}/4`, mean `{fmt_pct(mean(row['r3_vs_full_mse_pct'] for row in weather_rows))}`.",
        f"[Fact] Weather h720 late segment `337-720`: R.3 vs full-time `{fmt_pct(weather_late['r3_vs_full_mse_pct'])}`; ETTh2 h720 late segment `{fmt_pct(etth2_late['r3_vs_full_mse_pct'])}`.",
        "",
        "## Per-Horizon Delta",
        "",
        "| Dataset | Horizon | R.3 MSE | Full-time MSE | R.3 vs full |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in main_deltas:
        lines.append(
            f"| `{row['dataset']}` | {row['horizon']} | {row['r3_mse']:.6f} | "
            f"{row['full_mse']:.6f} | {fmt_pct(row['r3_vs_full_mse_pct'])} |"
        )
    lines += [
        "",
        "## Dataset And Region Summary",
        "",
        "| Dataset | Settings | Wins | Mean R.3 vs full MSE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in dataset_summary:
        lines.append(
            f"| `{row['dataset']}` | {row['settings']} | {row['r3_mse_wins']} | "
            f"{fmt_pct(row['mean_r3_vs_full_mse_pct'])} |"
        )
    lines += [
        "",
        "| Horizon | Settings | Wins | Mean R.3 vs full MSE |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in horizon_summary:
        lines.append(
            f"| {row['horizon']} | {row['settings']} | {row['r3_mse_wins']} | "
            f"{fmt_pct(row['mean_r3_vs_full_mse_pct'])} |"
        )
    lines += [
        "",
        "| Future region | Segments | Wins | Mean R.3 vs full MSE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in segment_summary:
        lines.append(
            f"| `{row['future_region']}` | {row['settings']} | {row['r3_mse_wins']} | "
            f"{fmt_pct(row['mean_r3_vs_full_mse_pct'])} |"
        )
    lines += [
        "",
        "## Actual Training Exposure",
        "",
        "[Fact] The directory label contains all evaluation horizons, but the training log determines actual supervision exposure.",
        "",
        "| Dataset | Strategy | Step weighting | Mixed train | Exposure source | h96 | h192 | h336 | h720 | Best epoch | Train loss change |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in training_exposure:
        lines.append(
            f"| `{row['dataset']}` | `{row['supervision_strategy']}` | `{row['step_loss_weighting']}` | "
            f"{str(row['uses_mixed_horizon_training'])} | `{row['exposure_source']}` | "
            f"{row['share_h96']:.2f} | {row['share_h192']:.2f} | {row['share_h336']:.2f} | "
            f"{row['share_h720']:.2f} | {row['best_epoch']} | {fmt_pct(row['train_loss_drop_pct'])} |"
        )
    lines += [
        "",
        "## Objective Pressure",
        "",
        "| Region | Mean step weight | Uniform pressure | Weighted pressure | Pressure delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in r3_weight_rows:
        lines.append(
            f"| `{row['scope']}` | {row['mean_step_weight']:.3f} | {row['uniform_pressure_share']:.4f} | "
            f"{row['weighted_pressure_share']:.4f} | {fmt_pct(row['pressure_share_delta_pct'])} |"
        )
    lines += [
        "",
        "## Historical Reference",
        "",
        f"[Fact] Phase2 compared R.3 against uniform target-set training and found mean R.3 vs uniform MSE `{fmt_pct(phase2_mean)}` over `{len(phase2_main)}` settings.",
        "",
        "| Scope | Phase2 pressure delta | Raw pressure ratio |",
        "| --- | ---: | ---: |",
    ]
    for row in phase2_pressure:
        lines.append(
            f"| `{row['segment']}` | {fmt_pct(row['pressure_share_delta_pct'])} | {row['raw_pressure_ratio']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "[Strong Evidence] 当前 R.3 的优势不能解释为“简单 step 权重”单因素。它同时改变了训练 horizon 暴露分布和 step pressure：R.3 每个 epoch 近似均匀采样 h96/h192/h336/h720，而 full-time 实际只用 h720 supervision。",
        "",
        "[Strong Evidence] `prefix_risk` 又把 expected pressure 从 late region 推向 early prefix：`1-96` pressure share 增加约 `+50%`，`337-720` 降低约 `-65%`。这解释了为什么 R.3 对 Weather 全 horizon 都强，但也暴露了它对 late future 的叙事风险。",
        "",
        "[Counter-Evidence] Phase2 里 R.3 相对 uniform target-set 的均值收益只有约 `-1%`。因此 Phase4 里 R.3 相对 h720 full-time 的更大优势，很可能来自 compound protocol，而不是 prefix-risk 单独强大。",
        "",
        "[Decision] 下一步不应继续 repair R.3，也不应把 R.3 包装成 core story。必须先做 decomposition controls：`mixed_horizon_uniform` 用来隔离 mixed-horizon exposure，`single_720_prefix_risk` 用来隔离 prefix-risk weighting，当前 R.3 保留为 compound reference。",
        "",
        "[Rollback Point] 回退到 Step 6 设计实验。只有当 decomposition 证明 architecture/gradient-routing 仍能在同等 exposure/objective 条件下带来稳定收益，才继续推进 HSS 主线升级。",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose why R.3 is strong in Phase4.")
    parser.add_argument("--raw-root", default="analysis/phase4_dynamic_residual_stability_gate_20260625/raw")
    parser.add_argument("--output-root", default="analysis/phase4_r3_mechanism_diagnostic_20260625")
    parser.add_argument(
        "--phase2-root",
        default="analysis/phase2_objective_pressure_diagnostic_20260623",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metrics, main_deltas = load_main_deltas(raw_root)
    segment_deltas = load_segment_deltas(raw_root)
    dataset_summary = summarize_deltas(main_deltas, ["dataset"])
    horizon_summary = summarize_deltas(main_deltas, ["horizon"])
    segment_summary = summarize_deltas(segment_deltas, ["future_region"])
    training_exposure = load_training_exposure(raw_root)
    objective_weights = load_objective_weights(raw_root)
    phase2_main, phase2_pressure = load_phase2_reference(Path(args.phase2_root))

    write_csv(output_root / "phase4_r3_main_metrics.csv", metrics)
    write_csv(output_root / "phase4_r3_vs_full_main_delta.csv", main_deltas)
    write_csv(output_root / "phase4_r3_vs_full_dataset_summary.csv", dataset_summary)
    write_csv(output_root / "phase4_r3_vs_full_horizon_summary.csv", horizon_summary)
    write_csv(output_root / "phase4_r3_vs_full_segment_delta.csv", segment_deltas)
    write_csv(output_root / "phase4_r3_vs_full_segment_region_summary.csv", segment_summary)
    write_csv(output_root / "phase4_r3_training_exposure.csv", training_exposure)
    write_csv(output_root / "phase4_r3_objective_weight_summary.csv", objective_weights)
    write_csv(output_root / "phase4_r3_phase2_reference_main.csv", phase2_main)
    write_csv(output_root / "phase4_r3_phase2_reference_pressure.csv", phase2_pressure)
    write_report(
        output_root / "phase4_r3_mechanism_diagnostic_report.md",
        main_deltas,
        segment_deltas,
        dataset_summary,
        horizon_summary,
        segment_summary,
        training_exposure,
        objective_weights,
        phase2_main,
        phase2_pressure,
    )
    print(output_root / "phase4_r3_mechanism_diagnostic_report.md")


if __name__ == "__main__":
    main()
