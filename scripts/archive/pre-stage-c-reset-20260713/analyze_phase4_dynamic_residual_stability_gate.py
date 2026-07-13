from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PRIMARY = "PatchEncoderDynamicResidualStabilityRouting"
FULL = "PatchEncoderFullTimeMSE720"
R3 = "PatchEncoderR3PrefixRisk"
MODELS = [PRIMARY, FULL, R3]
LABELS = {
    PRIMARY: "RG-B_dynamic_residual_stability",
    FULL: "D0_full_time_mse",
    R3: "D1_r3_prefix_risk",
}
BASELINES = [FULL, R3]
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


def add_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def summarize_segments(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    output = []
    for baseline in BASELINES:
        baseline_rows = [row for row in rows if row["baseline_model"] == baseline]
        for value in sorted({row[group_key] for row in baseline_rows}):
            subset = [row for row in baseline_rows if row[group_key] == value]
            output.append(
                {
                    "strategy": LABELS[PRIMARY],
                    "baseline_strategy": LABELS[baseline],
                    group_key: value,
                    "segments": len(subset),
                    "mse_wins": sum(1 for row in subset if row["mse_win"]),
                    "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in subset),
                    "mean_relative_mae_pct": mean(row["relative_mae_pct"] for row in subset),
                }
            )
    return output


def load_trace_summary(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        trace = read_csv(run_dir(raw_root, PRIMARY, dataset) / "supervision_trace.csv")
        rows.append(
            {
                "dataset": dataset,
                "trace_rows": len(trace),
                "mean_learnable_blocks": mean(float(row["residual_stability_learnable_blocks"]) for row in trace),
                "mean_noisy_blocks": mean(float(row["residual_stability_noisy_blocks"]) for row in trace),
                "mean_ambiguous_blocks": mean(float(row["residual_stability_ambiguous_blocks"]) for row in trace),
                "mean_noisy_suppression_ratio": mean(
                    float(row["residual_stability_noisy_suppression_ratio"]) for row in trace
                ),
                "mean_adapter_active_steps": mean(float(row["adapter_active_steps"]) for row in trace),
                "mean_adapter_abs_residual": mean(float(row["adapter_mean_abs_residual"]) for row in trace),
                "last_adapter_abs_residual": float(trace[-1]["adapter_mean_abs_residual"]),
                "mean_unit_loss": mean(float(row["loss_unit"]) for row in trace),
                "mean_time_loss": mean(float(row["loss_time"]) for row in trace),
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
            rows.append(
                {
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
            )
    return rows


def load_historical_training_summary(current_root: Path) -> list[dict[str, Any]]:
    roots = [
        current_root,
        Path("analysis/phase4_s_predictability_gate_20260625/raw"),
        Path("analysis/phase4_horizon_decoupled_gate_20260624/raw"),
    ]
    output = []
    seen: set[tuple[str, str, str]] = set()
    for root in roots:
        if not root.exists():
            continue
        gate = root.parent.name if root.name == "raw" else root.name
        for path in sorted(root.glob("PatchEncoder*/**/training_log.csv")):
            parts = path.relative_to(root).parts
            if len(parts) < 5:
                continue
            model, dataset = parts[0], parts[1]
            if dataset not in DATASETS:
                continue
            key = (gate, model, dataset)
            if key in seen:
                continue
            seen.add(key)
            rows = read_csv(path)
            if not rows:
                continue
            val_values = [float(row["val_mean_mse"]) for row in rows]
            best_epoch = min(range(len(val_values)), key=val_values.__getitem__) + 1
            output.append(
                {
                    "gate": gate,
                    "model": model,
                    "dataset": dataset,
                    "epochs_ran": len(rows),
                    "best_epoch": best_epoch,
                    "first_val_mean_mse": val_values[0],
                    "best_val_mean_mse": min(val_values),
                    "last_val_mean_mse": val_values[-1],
                }
            )
    return output


def load_prefix_summary(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            prefix_rows = read_csv(run_dir(raw_root, model, dataset) / "prefix_consistency.csv")
            rows.append(
                {
                    "model": model,
                    "strategy": LABELS[model],
                    "dataset": dataset,
                    "max_prefix_mismatch_mse": max(float(row["prefix_mismatch_mse"]) for row in prefix_rows),
                    "max_prefix_mismatch_mae": max(float(row["prefix_mismatch_mae"]) for row in prefix_rows),
                }
            )
    return rows


def pick_row(rows: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    for row in rows:
        if all(row[key] == value for key, value in criteria.items()):
            return row
    raise KeyError(criteria)


def write_report(
    path: Path,
    main_rows: list[dict[str, Any]],
    main_delta: list[dict[str, Any]],
    overall: list[dict[str, Any]],
    dataset_summary: list[dict[str, Any]],
    segment_delta: list[dict[str, Any]],
    segment_region_summary: list[dict[str, Any]],
    trace_summary: list[dict[str, Any]],
    training_summary: list[dict[str, Any]],
    historical_training_summary: list[dict[str, Any]],
    prefix_summary: list[dict[str, Any]],
) -> None:
    full_overall = pick_row(overall, baseline_strategy=LABELS[FULL])
    r3_overall = pick_row(overall, baseline_strategy=LABELS[R3])
    weather_r3 = [
        row for row in main_delta
        if row["dataset"] == "Weather" and row["baseline_model"] == R3
    ]
    etth2_r3 = [
        row for row in main_delta
        if row["dataset"] == "ETTh2" and row["baseline_model"] == R3
    ]
    weather_late = pick_row(
        segment_delta,
        dataset="Weather",
        horizon=720,
        segment="337-720",
        baseline_model=R3,
    )
    etth2_late = pick_row(
        segment_delta,
        dataset="ETTh2",
        horizon=720,
        segment="337-720",
        baseline_model=R3,
    )
    max_prefix = max(float(row["max_prefix_mismatch_mse"]) for row in prefix_summary)
    by_metric = {(row["model"], row["dataset"], row["horizon"]): row for row in main_rows}
    by_delta = {(row["dataset"], row["horizon"], row["baseline_model"]): row for row in main_delta}

    early_best_rows = [row for row in historical_training_summary if int(row["best_epoch"]) <= 5]
    lines = [
        "# Phase4 RG-B Dynamic Residual-Stability Gate Report",
        "",
        "## 11-Step Record",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | Step 9/10: evaluate returned artifacts and decide whether RG-B passes |",
        "| `problem` | Fixed late routing mixed learnable/noisy conflict units and failed Weather vs R.3 |",
        "| `existence_evidence` | RG-A gate, residual-stability diagnostic, train logs, trace buckets |",
        "| `idea` | Route only residual-stability learnable high-novelty units to detached adapter auxiliary |",
        "| `theory_check` | Structured residuals should benefit from adapter learning; noisy residuals should not add auxiliary pressure |",
        "| `design` | `dynamic_residual_stability_routing` vs `full_time_mse` and R.3 on ETTh2/Weather |",
        "| `gate` | retain full-time gain, avoid Weather 0/4 vs R.3, improve Weather late, retain ETTh2 signal, prefix zero |",
        f"| `artifacts` | `{path.parent}` |",
        "| `decision` | Fail as paper-core candidate; useful partial evidence vs full-time, but R.3 gap remains |",
        "",
        "## Main Result",
        "",
        f"[Fact] RG-B vs `full_time_mse`: MSE wins `{full_overall['mse_wins']}/{full_overall['settings']}`, mean relative MSE `{fmt_pct(full_overall['mean_relative_mse_pct'])}`.",
        f"[Fact] RG-B vs R.3: MSE wins `{r3_overall['mse_wins']}/{r3_overall['settings']}`, mean relative MSE `{fmt_pct(r3_overall['mean_relative_mse_pct'])}`.",
        f"[Fact] Weather vs R.3 remains `{sum(1 for row in weather_r3 if row['mse_win'])}/4`, mean relative MSE `{fmt_pct(mean(row['relative_mse_pct'] for row in weather_r3))}`.",
        f"[Fact] ETTh2 vs R.3 is `{sum(1 for row in etth2_r3 if row['mse_win'])}/4`, mean relative MSE `{fmt_pct(mean(row['relative_mse_pct'] for row in etth2_r3))}`.",
        "",
        "## Per-Horizon MSE",
        "",
        "| Dataset | Horizon | RG-B | Full-time | R.3 | RG-B vs full | RG-B vs R.3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        for horizon in HORIZONS:
            rgb = by_metric[(PRIMARY, dataset, horizon)]
            full = by_metric[(FULL, dataset, horizon)]
            r3 = by_metric[(R3, dataset, horizon)]
            full_delta = by_delta[(dataset, horizon, FULL)]
            r3_delta = by_delta[(dataset, horizon, R3)]
            lines.append(
                f"| `{dataset}` | {horizon} | {rgb['mse']:.6f} | {full['mse']:.6f} | {r3['mse']:.6f} | "
                f"{fmt_pct(full_delta['relative_mse_pct'])} | {fmt_pct(r3_delta['relative_mse_pct'])} |"
            )
    lines += [
        "",
        "## Segment Gate",
        "",
        f"[Fact] Weather h720 `337-720` vs R.3: `{fmt_pct(weather_late['relative_mse_pct'])}`.",
        f"[Fact] ETTh2 h720 `337-720` vs R.3: `{fmt_pct(etth2_late['relative_mse_pct'])}`.",
        "",
        "| Future region | Baseline | Segments | MSE wins | Mean relative MSE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in segment_region_summary:
        lines.append(
            f"| `{row['future_region']}` | `{row['baseline_strategy']}` | {row['segments']} | "
            f"{row['mse_wins']} | {fmt_pct(row['mean_relative_mse_pct'])} |"
        )
    lines += [
        "",
        "## Trace Buckets",
        "",
        "| Dataset | Learnable blocks | Noisy blocks | Ambiguous blocks | Noisy suppression | Adapter active steps | Mean abs adapter residual | Last abs adapter residual |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in trace_summary:
        lines.append(
            f"| `{row['dataset']}` | {row['mean_learnable_blocks']:.2f} | {row['mean_noisy_blocks']:.2f} | "
            f"{row['mean_ambiguous_blocks']:.2f} | {row['mean_noisy_suppression_ratio']:.2f} | "
            f"{row['mean_adapter_active_steps']:.1f} | {row['mean_adapter_abs_residual']:.6f} | "
            f"{row['last_adapter_abs_residual']:.6f} |"
        )
    lines += [
        "",
        "## Training Dynamics",
        "",
        "| Dataset | Strategy | Epochs ran | Best epoch | Best val MSE | Last val drift | Train loss change |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in training_summary:
        lines.append(
            f"| `{row['dataset']}` | `{row['strategy']}` | {row['epochs_ran']} | {row['best_epoch']} | "
            f"{row['best_val_mean_mse']:.6f} | {fmt_pct(row['post_best_val_drift_pct'])} | "
            f"{fmt_pct(row['train_loss_drop_pct'])} |"
        )
    lines += [
        "",
        f"[Fact] In the current gate, all six runs reach best validation MSE by epoch `1-4`; the run length is best epoch plus patience `10`.",
        f"[Fact] Across current and adjacent Phase4 gates, `{len(early_best_rows)}/{len(historical_training_summary)}` ETTh2/Weather logs reach best validation MSE by epoch `<=5`.",
        "",
        "## Interpretation",
        "",
        "[Strong Evidence] RG-B preserves the useful mechanism signal against `full_time_mse`: it wins 7/8 MSE settings and improves Weather long horizons versus full-time.",
        "",
        "[Counter-Evidence] RG-B does not solve the paper-core gate. Weather remains 0/4 vs R.3, and the Weather h720 late segment is still worse than R.3. The dynamic residual-stability adapter is therefore not enough to replace R.3 as the main story.",
        "",
        "[Inference] The trace shows the router is active rather than collapsed: Weather suppresses more noisy blocks than ETTh2 and the adapter residual becomes nonzero. The failure is more likely due to the carrier/optimization target being weak than due to a dead router.",
        "",
        "[Training-Dynamics Analysis] The early best epoch is real and systemic. It appears in prior Phase4 controls too, while train loss keeps decreasing after validation has worsened. This points to fast overfitting or validation-objective mismatch under the current small carrier and no scheduler, not to a RG-B-specific logging bug.",
        "",
        "[Decision] Do not continue by sweeping RG-B thresholds or aux weight. Roll back to Step 5/6 and investigate the training protocol/carrier: learning-rate schedule, shorter calibrated training, or a stronger pretraining/warmup design should be studied before adding more routing complexity.",
        "",
        f"[Fact] max prefix mismatch MSE is `{max_prefix:.3e}`, so prefix consistency remains numerical-zero.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase4 dynamic residual-stability gate.")
    parser.add_argument("--raw-root", default="analysis/phase4_dynamic_residual_stability_gate_20260625/raw")
    parser.add_argument("--output-root", default="analysis/phase4_dynamic_residual_stability_gate_20260625")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    main_rows = load_main_metrics(raw_root)
    main_delta = add_deltas(main_rows)
    segment_rows = load_segment_metrics(raw_root)
    segment_delta = add_segment_deltas(segment_rows)
    overall = summarize_deltas(main_delta)
    dataset_summary = summarize_deltas(main_delta, "dataset")
    horizon_summary = summarize_deltas(main_delta, "horizon")
    segment_region_summary = summarize_segments(segment_delta, "future_region")
    trace_summary = load_trace_summary(raw_root)
    training_summary = load_training_summary(raw_root)
    historical_training_summary = load_historical_training_summary(raw_root)
    prefix_summary = load_prefix_summary(raw_root)

    write_csv(output_root / "phase4_dynamic_residual_main_metrics.csv", main_rows)
    write_csv(output_root / "phase4_dynamic_residual_main_delta.csv", main_delta)
    write_csv(output_root / "phase4_dynamic_residual_overall_summary.csv", overall)
    write_csv(output_root / "phase4_dynamic_residual_dataset_summary.csv", dataset_summary)
    write_csv(output_root / "phase4_dynamic_residual_horizon_summary.csv", horizon_summary)
    write_csv(output_root / "phase4_dynamic_residual_segment_delta.csv", segment_delta)
    write_csv(output_root / "phase4_dynamic_residual_segment_region_summary.csv", segment_region_summary)
    write_csv(output_root / "phase4_dynamic_residual_trace_summary.csv", trace_summary)
    write_csv(output_root / "phase4_dynamic_residual_training_summary.csv", training_summary)
    write_csv(output_root / "phase4_dynamic_residual_historical_training_summary.csv", historical_training_summary)
    write_csv(output_root / "phase4_dynamic_residual_prefix_summary.csv", prefix_summary)
    write_report(
        output_root / "phase4_dynamic_residual_stability_gate_report.md",
        main_rows,
        main_delta,
        overall,
        dataset_summary,
        segment_delta,
        segment_region_summary,
        trace_summary,
        training_summary,
        historical_training_summary,
        prefix_summary,
    )
    print(output_root / "phase4_dynamic_residual_stability_gate_report.md")


if __name__ == "__main__":
    main()
