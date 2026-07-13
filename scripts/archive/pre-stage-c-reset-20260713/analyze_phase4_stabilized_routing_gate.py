from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


PRIMARY = "PatchEncoderStabilizedDynamicResidualRouting"
PRETRAIN = "PatchEncoderFullTimeMSE720Pretrain"
R3 = "PatchEncoderR3PrefixRisk"
RGB = "PatchEncoderDynamicResidualStabilityRouting"
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


def metrics_by_horizon(root: Path, model: str, dataset: str) -> dict[int, dict[str, float]]:
    path = run_dir(root, model, dataset) / "metrics_by_target_horizon.csv"
    return {
        int(row["target_horizon"]): {"mse": float(row["mse"]), "mae": float(row["mae"])}
        for row in read_csv(path)
    }


def segment_metrics(root: Path, model: str, dataset: str, horizon: int) -> dict[str, dict[str, float]]:
    path = run_dir(root, model, dataset) / f"h{horizon}" / "metrics_by_segment.csv"
    return {
        row["segment"]: {"mse": float(row["mse"]), "mae": float(row["mae"])}
        for row in read_csv(path)
    }


def training_summary(root: Path, model: str, dataset: str, label: str) -> dict[str, Any]:
    rows = read_csv(run_dir(root, model, dataset) / "training_log.csv")
    val_values = [float(row["val_mean_mse"]) for row in rows]
    train_values = [float(row["train_prediction_loss"]) for row in rows]
    best_index = min(range(len(val_values)), key=val_values.__getitem__)
    return {
        "dataset": dataset,
        "strategy": label,
        "epochs_ran": len(rows),
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
            }
        )
    return rows


def load_audit(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        base = run_dir(raw_root, PRIMARY, dataset)
        config = json.loads((base / "effective_config.json").read_text())
        env = json.loads((base / "environment.json").read_text())
        init_info = config["init_checkpoint_info"]
        rows.append(
            {
                "dataset": dataset,
                "freeze_non_adapter_effective": config["freeze_non_adapter_effective"],
                "parameter_count": env["parameter_count"],
                "trainable_parameter_count": env["trainable_parameter_count"],
                "trainable_parameter_pct": env["trainable_parameter_count"] / env["parameter_count"] * 100.0,
                "init_checkpoint_path": init_info["path"],
                "missing_keys": len(init_info["missing_keys"]),
                "unexpected_keys": len(init_info["unexpected_keys"]),
                "missing_key_sample": ";".join(init_info["missing_keys"][:4]),
            }
        )
    return rows


def build_metric_delta(raw_root: Path, previous_root: Path) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        primary = metrics_by_horizon(raw_root, PRIMARY, dataset)
        pretrain = metrics_by_horizon(raw_root, PRETRAIN, dataset)
        r3 = metrics_by_horizon(previous_root, R3, dataset)
        rgb = metrics_by_horizon(previous_root, RGB, dataset)
        for horizon in HORIZONS:
            for baseline_name, baseline_label, baseline in [
                (PRETRAIN, "OP-A_pretrain", pretrain),
                (R3, "D1_r3_prefix_risk", r3),
                (RGB, "RG-B_from_scratch", rgb),
            ]:
                rows.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "candidate_model": PRIMARY,
                        "baseline_model": baseline_name,
                        "baseline_strategy": baseline_label,
                        "candidate_mse": primary[horizon]["mse"],
                        "baseline_mse": baseline[horizon]["mse"],
                        "relative_mse_pct": pct(primary[horizon]["mse"], baseline[horizon]["mse"]),
                        "mse_win": primary[horizon]["mse"] < baseline[horizon]["mse"],
                        "candidate_mae": primary[horizon]["mae"],
                        "baseline_mae": baseline[horizon]["mae"],
                        "relative_mae_pct": pct(primary[horizon]["mae"], baseline[horizon]["mae"]),
                        "mae_win": primary[horizon]["mae"] < baseline[horizon]["mae"],
                    }
                )
    return rows


def summarize_delta(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for baseline in sorted({row["baseline_strategy"] for row in rows}):
        subset = [row for row in rows if row["baseline_strategy"] == baseline]
        output.append(
            {
                "baseline_strategy": baseline,
                "settings": len(subset),
                "mse_wins": sum(1 for row in subset if row["mse_win"]),
                "mae_wins": sum(1 for row in subset if row["mae_win"]),
                "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in subset),
                "mean_relative_mae_pct": mean(row["relative_mae_pct"] for row in subset),
            }
        )
    return output


def build_segment_delta(raw_root: Path, previous_root: Path) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        primary = segment_metrics(raw_root, PRIMARY, dataset, 720)
        pretrain = segment_metrics(raw_root, PRETRAIN, dataset, 720)
        r3 = segment_metrics(previous_root, R3, dataset, 720)
        rgb = segment_metrics(previous_root, RGB, dataset, 720)
        for segment, candidate in primary.items():
            for baseline_name, baseline_label, baseline in [
                (PRETRAIN, "OP-A_pretrain", pretrain),
                (R3, "D1_r3_prefix_risk", r3),
                (RGB, "RG-B_from_scratch", rgb),
            ]:
                base = baseline[segment]
                rows.append(
                    {
                        "dataset": dataset,
                        "horizon": 720,
                        "segment": segment,
                        "baseline_model": baseline_name,
                        "baseline_strategy": baseline_label,
                        "candidate_mse": candidate["mse"],
                        "baseline_mse": base["mse"],
                        "relative_mse_pct": pct(candidate["mse"], base["mse"]),
                        "mse_win": candidate["mse"] < base["mse"],
                    }
                )
    return rows


def write_report(
    output_path: Path,
    metric_delta: list[dict[str, Any]],
    overall: list[dict[str, Any]],
    segment_delta: list[dict[str, Any]],
    trace_summary: list[dict[str, Any]],
    audit: list[dict[str, Any]],
    train_summary: list[dict[str, Any]],
) -> None:
    by_delta = {
        (row["dataset"], row["horizon"], row["baseline_strategy"]): row
        for row in metric_delta
    }
    weather_r3 = [row for row in metric_delta if row["dataset"] == "Weather" and row["baseline_strategy"] == "D1_r3_prefix_risk"]
    etth2_r3 = [row for row in metric_delta if row["dataset"] == "ETTh2" and row["baseline_strategy"] == "D1_r3_prefix_risk"]
    weather_late_r3 = next(
        row for row in segment_delta
        if row["dataset"] == "Weather" and row["segment"] == "337-720" and row["baseline_strategy"] == "D1_r3_prefix_risk"
    )
    weather_late_rgb = next(
        row for row in segment_delta
        if row["dataset"] == "Weather" and row["segment"] == "337-720" and row["baseline_strategy"] == "RG-B_from_scratch"
    )
    lines = [
        "# Phase4 OP-A Stabilized Routing Gate Report",
        "",
        "## 11-Step Record",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | Step 9/10: evaluate stabilized-base adapter-only routing |",
        "| `problem` | RG-B router works against full-time but fails R.3; Phase4 logs show systemic early best epoch |",
        "| `idea` | Pretrain full-time base, freeze base, then train only dynamic residual-stability adapter |",
        "| `gate` | audit freeze, delay early-best collapse, improve Weather vs R.3 and RG-B late segment |",
        f"| `artifacts` | `{output_path.parent}` |",
        "| `decision` | Fail; protocol stabilizes training timing but damages metrics |",
        "",
        "## Overall Result",
        "",
        "| Baseline | Settings | MSE wins | MAE wins | Mean relative MSE | Mean relative MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall:
        lines.append(
            f"| `{row['baseline_strategy']}` | {row['settings']} | {row['mse_wins']} | {row['mae_wins']} | "
            f"{fmt_pct(row['mean_relative_mse_pct'])} | {fmt_pct(row['mean_relative_mae_pct'])} |"
        )
    lines += [
        "",
        f"[Fact] Weather vs R.3 remains `{sum(1 for row in weather_r3 if row['mse_win'])}/4`, mean relative MSE `{fmt_pct(mean(row['relative_mse_pct'] for row in weather_r3))}`.",
        f"[Fact] ETTh2 vs R.3 is `{sum(1 for row in etth2_r3 if row['mse_win'])}/4`, mean relative MSE `{fmt_pct(mean(row['relative_mse_pct'] for row in etth2_r3))}`.",
        "",
        "## Per-Horizon MSE Delta",
        "",
        "| Dataset | Horizon | vs pretrain | vs R.3 | vs RG-B |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        for horizon in HORIZONS:
            pre = by_delta[(dataset, horizon, "OP-A_pretrain")]
            r3 = by_delta[(dataset, horizon, "D1_r3_prefix_risk")]
            rgb = by_delta[(dataset, horizon, "RG-B_from_scratch")]
            lines.append(
                f"| `{dataset}` | {horizon} | {fmt_pct(pre['relative_mse_pct'])} | "
                f"{fmt_pct(r3['relative_mse_pct'])} | {fmt_pct(rgb['relative_mse_pct'])} |"
            )
    lines += [
        "",
        "## Audit",
        "",
        "| Dataset | Freeze | Trainable params | Missing keys | Unexpected keys |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in audit:
        lines.append(
            f"| `{row['dataset']}` | `{row['freeze_non_adapter_effective']}` | "
            f"{row['trainable_parameter_count']} / {row['parameter_count']} ({row['trainable_parameter_pct']:.2f}%) | "
            f"{row['missing_keys']} | {row['unexpected_keys']} |"
        )
    lines += [
        "",
        "## Trace",
        "",
        "| Dataset | Learnable blocks | Noisy blocks | Noisy suppression | Adapter active steps | Mean abs adapter residual |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in trace_summary:
        lines.append(
            f"| `{row['dataset']}` | {row['mean_learnable_blocks']:.2f} | {row['mean_noisy_blocks']:.2f} | "
            f"{row['mean_noisy_suppression_ratio']:.2f} | {row['mean_adapter_active_steps']:.1f} | "
            f"{row['mean_adapter_abs_residual']:.6f} |"
        )
    lines += [
        "",
        "## Training Dynamics",
        "",
        "| Dataset | Strategy | Epochs | Best epoch | Best val MSE | Last val drift | Train loss change |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in train_summary:
        lines.append(
            f"| `{row['dataset']}` | `{row['strategy']}` | {row['epochs_ran']} | {row['best_epoch']} | "
            f"{row['best_val_mean_mse']:.6f} | {fmt_pct(row['post_best_val_drift_pct'])} | "
            f"{fmt_pct(row['train_loss_drop_pct'])} |"
        )
    lines += [
        "",
        "## Segment Gate",
        "",
        f"[Fact] Weather h720 `337-720` vs R.3: `{fmt_pct(weather_late_r3['relative_mse_pct'])}`.",
        f"[Fact] Weather h720 `337-720` vs RG-B: `{fmt_pct(weather_late_rgb['relative_mse_pct'])}`.",
        "",
        "## Interpretation",
        "",
        "[Strong Evidence] OP-A validates the audit side: checkpoint loading, base freezing, and adapter-only optimization are functioning. Trainable parameters are only 0.63% of total parameters, and missing keys are the expected adapter head.",
        "",
        "[Counter-Evidence] OP-A fails the performance gate. Adapter-only finetune is worse than the full-time pretrain at every horizon, worse than R.3 at every horizon, and mostly worse than from-scratch RG-B.",
        "",
        "[Inference] Stabilizing the base delays early-best collapse, but the current adapter-only residual path cannot improve the frozen base. This suggests the problem is not just simultaneous base/routing optimization; the adapter capacity/objective or base objective is insufficient.",
        "",
        "[Decision] Do not pursue adapter-only stabilized routing with full-time base. Roll back to Step 5/6 and test whether the base objective itself must be R.3/prefix-risk stabilized, or whether routing needs to update a richer target/readout subset rather than only the small adapter head.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase4 stabilized routing gate.")
    parser.add_argument("--raw-root", default="analysis/phase4_stabilized_routing_gate_20260625/raw")
    parser.add_argument("--previous-root", default="analysis/phase4_dynamic_residual_stability_gate_20260625/raw")
    parser.add_argument("--output-root", default="analysis/phase4_stabilized_routing_gate_20260625")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    previous_root = Path(args.previous_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metric_delta = build_metric_delta(raw_root, previous_root)
    overall = summarize_delta(metric_delta)
    segment_delta = build_segment_delta(raw_root, previous_root)
    trace_summary = load_trace_summary(raw_root)
    audit = load_audit(raw_root)
    train_rows = []
    for dataset in DATASETS:
        train_rows.append(training_summary(raw_root, PRETRAIN, dataset, "OP-A_pretrain"))
        train_rows.append(training_summary(raw_root, PRIMARY, dataset, "OP-A_adapter_only"))

    write_csv(output_root / "phase4_stabilized_metric_delta.csv", metric_delta)
    write_csv(output_root / "phase4_stabilized_overall_summary.csv", overall)
    write_csv(output_root / "phase4_stabilized_segment_delta.csv", segment_delta)
    write_csv(output_root / "phase4_stabilized_trace_summary.csv", trace_summary)
    write_csv(output_root / "phase4_stabilized_audit.csv", audit)
    write_csv(output_root / "phase4_stabilized_training_summary.csv", train_rows)
    write_report(
        output_root / "phase4_stabilized_routing_gate_report.md",
        metric_delta,
        overall,
        segment_delta,
        trace_summary,
        audit,
        train_rows,
    )
    print(output_root / "phase4_stabilized_routing_gate_report.md")


if __name__ == "__main__":
    main()
