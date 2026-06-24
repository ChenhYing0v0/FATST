from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = ["ETTh2", "ETTm1", "Weather"]
HORIZONS = [96, 720]
SEED = 2021
RUN = "mixed_h96_h720"
FULL_RUN = "mixed_h96_h192_h336_h720"
R3_NAME = "PatchEncoderPrefixRiskWeighted"
CONTROL_NAME = "PatchEncoderPrefixRiskWeightedH96H720"
OPERATOR_NAME = "PatchEncoderRegimeSegmentTargetOperatorHistoryOnly"
FULL_OPERATOR_NAME = "PatchEncoderRegimeSegmentTargetOperatorHistoryOnlyFull"
OBSERVED_GAPS = {
    ("ETTm1", 96, "aggregate"),
    ("Weather", 96, "aggregate"),
    ("ETTh2", 720, "193-336"),
    ("ETTh2", 720, "337-720"),
    ("ETTm1", 720, "337-720"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def run_dir(raw_root: Path, model_name: str, dataset: str, run_name: str) -> Path:
    return raw_root / model_name / dataset / run_name / f"seed{SEED}"


def metrics_by_target(run_path: Path) -> dict[int, dict[str, str]]:
    return {int(row["target_horizon"]): row for row in read_csv(run_path / "metrics_by_target_horizon.csv")}


def collect_vs_r3(control_raw_root: Path, r3_raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        control_metrics = metrics_by_target(run_dir(control_raw_root, CONTROL_NAME, dataset, RUN))
        r3_metrics = metrics_by_target(run_dir(r3_raw_root, R3_NAME, dataset, FULL_RUN))
        for horizon in HORIZONS:
            control = control_metrics[horizon]
            r3 = r3_metrics[horizon]
            control_mse = float(control["mse"])
            r3_mse = float(r3["mse"])
            control_mae = float(control["mae"])
            r3_mae = float(r3["mae"])
            rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "control_mse": control_mse,
                    "r3_full_mse": r3_mse,
                    "relative_mse_vs_r3_full_pct": pct(control_mse, r3_mse),
                    "control_wins_mse_vs_r3_full": control_mse < r3_mse,
                    "control_mae": control_mae,
                    "r3_full_mae": r3_mae,
                    "relative_mae_vs_r3_full_pct": pct(control_mae, r3_mae),
                    "control_wins_mae_vs_r3_full": control_mae < r3_mae,
                    "is_observed_gap": (dataset, horizon, "aggregate") in OBSERVED_GAPS,
                }
            )
    return rows


def collect_operator_increment(
    control_raw_root: Path,
    operator_raw_root: Path,
    full_operator_raw_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        control_metrics = metrics_by_target(run_dir(control_raw_root, CONTROL_NAME, dataset, RUN))
        operator_metrics = metrics_by_target(run_dir(operator_raw_root, OPERATOR_NAME, dataset, RUN))
        full_operator_metrics = metrics_by_target(
            run_dir(full_operator_raw_root, FULL_OPERATOR_NAME, dataset, FULL_RUN)
        )
        for horizon in HORIZONS:
            control_mse = float(control_metrics[horizon]["mse"])
            operator_mse = float(operator_metrics[horizon]["mse"])
            full_operator_mse = float(full_operator_metrics[horizon]["mse"])
            rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "control_reduced_mse": control_mse,
                    "operator_reduced_mse": operator_mse,
                    "operator_increment_vs_control_pct": pct(operator_mse, control_mse),
                    "operator_reduced_beats_control": operator_mse < control_mse,
                    "operator_full_mse": full_operator_mse,
                    "full_operator_vs_reduced_operator_pct": pct(full_operator_mse, operator_mse),
                    "full_operator_beats_reduced_operator": full_operator_mse < operator_mse,
                    "is_observed_gap": (dataset, horizon, "aggregate") in OBSERVED_GAPS,
                }
            )
    return rows


def collect_segment_vs_r3(control_raw_root: Path, r3_raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        control_rows = {
            row["segment"]: row
            for row in read_csv(run_dir(control_raw_root, CONTROL_NAME, dataset, RUN) / "h720" / "metrics_by_segment.csv")
        }
        r3_rows = {
            row["segment"]: row
            for row in read_csv(run_dir(r3_raw_root, R3_NAME, dataset, FULL_RUN) / "h720" / "metrics_by_segment.csv")
        }
        for segment, control in control_rows.items():
            r3 = r3_rows[segment]
            control_mse = float(control["mse"])
            r3_mse = float(r3["mse"])
            rows.append(
                {
                    "dataset": dataset,
                    "segment": segment,
                    "control_mse": control_mse,
                    "r3_full_mse": r3_mse,
                    "relative_mse_vs_r3_full_pct": pct(control_mse, r3_mse),
                    "control_wins_mse_vs_r3_full": control_mse < r3_mse,
                    "is_observed_gap": (dataset, 720, segment) in OBSERVED_GAPS,
                }
            )
    return rows


def collect_objective_pressure(control_raw_root: Path, r3_objective_raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        control_rows = {
            row["scope"]: row
            for row in read_csv(run_dir(control_raw_root, CONTROL_NAME, dataset, RUN) / "objective_weight_stats.csv")
        }
        r3_rows = {
            row["scope"]: row
            for row in read_csv(run_dir(r3_objective_raw_root, R3_NAME, dataset, FULL_RUN) / "objective_weight_stats.csv")
        }
        for segment in ["1-96", "97-192", "193-336", "337-720", "horizon_96", "horizon_720"]:
            control = control_rows[segment]
            r3 = r3_rows[segment]
            control_pressure = float(control["weighted_pressure_share"])
            r3_pressure = float(r3["weighted_pressure_share"])
            rows.append(
                {
                    "dataset": dataset,
                    "segment": segment,
                    "control_exposure_share": float(control["uniform_pressure_share"]),
                    "r3_full_exposure_share": float(r3["uniform_pressure_share"]),
                    "exposure_share_delta": float(control["uniform_pressure_share"]) - float(r3["uniform_pressure_share"]),
                    "control_effective_pressure": control_pressure,
                    "r3_full_effective_pressure": r3_pressure,
                    "effective_pressure_delta_pct": pct(control_pressure, r3_pressure),
                }
            )
    return rows


def collect_prefix_rows(control_raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for row in read_csv(run_dir(control_raw_root, CONTROL_NAME, dataset, RUN) / "prefix_consistency.csv"):
            rows.append(
                {
                    "dataset": dataset,
                    "short_horizon": int(row["short_horizon"]),
                    "long_horizon": int(row["long_horizon"]),
                    "prefix_mismatch_mse": float(row["prefix_mismatch_mse"]),
                    "prefix_mismatch_mae": float(row["prefix_mismatch_mae"]),
                }
            )
    return rows


def collect_training_rows(control_raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        path = run_dir(control_raw_root, CONTROL_NAME, dataset, RUN)
        log_rows = read_csv(path / "training_log.csv")
        config = read_json(path / "effective_config.json")
        rows.append(
            {
                "dataset": dataset,
                "epochs_recorded": len(log_rows),
                "best_val_mean_mse": min(float(row["val_mean_mse"]) for row in log_rows),
                "last_val_mean_mse": float(log_rows[-1]["val_mean_mse"]),
                "steps_per_epoch_effective": int(config["steps_per_epoch_effective"]),
                "target_horizons": ",".join(str(item) for item in config["target_horizons"]),
                "use_window_position": bool(config.get("use_window_position", False)),
                "model_variant": config.get("model_variant", ""),
                "step_loss_weighting": config.get("step_loss_weighting", ""),
            }
        )
    return rows


def summarize(
    vs_r3_rows: list[dict[str, Any]],
    operator_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    pressure_rows: list[dict[str, Any]],
    prefix_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    h96_rows = [row for row in vs_r3_rows if row["horizon"] == 96]
    h720_rows = [row for row in vs_r3_rows if row["horizon"] == 720]
    aggregate_gaps = [row for row in vs_r3_rows if row["is_observed_gap"]]
    segment_gaps = [row for row in segment_rows if row["is_observed_gap"]]
    operator_gap_rows = [row for row in operator_rows if row["is_observed_gap"]]
    long_pressure = [row for row in pressure_rows if row["segment"] == "337-720"]
    summary = {
        "control_wins_vs_r3_full": sum(row["control_wins_mse_vs_r3_full"] for row in vs_r3_rows),
        "control_total_vs_r3_full": len(vs_r3_rows),
        "control_mean_relative_mse_vs_r3_full_pct": mean(
            row["relative_mse_vs_r3_full_pct"] for row in vs_r3_rows
        ),
        "control_h96_mean_relative_mse_vs_r3_full_pct": mean(
            row["relative_mse_vs_r3_full_pct"] for row in h96_rows
        ),
        "control_h720_mean_relative_mse_vs_r3_full_pct": mean(
            row["relative_mse_vs_r3_full_pct"] for row in h720_rows
        ),
        "control_observed_aggregate_gap_wins": sum(row["control_wins_mse_vs_r3_full"] for row in aggregate_gaps),
        "control_observed_aggregate_gap_total": len(aggregate_gaps),
        "control_h720_segment_wins_vs_r3_full": sum(row["control_wins_mse_vs_r3_full"] for row in segment_rows),
        "control_h720_segment_total_vs_r3_full": len(segment_rows),
        "control_observed_h720_segment_gap_wins": sum(
            row["control_wins_mse_vs_r3_full"] for row in segment_gaps
        ),
        "control_observed_h720_segment_gap_total": len(segment_gaps),
        "operator_reduced_beats_control": sum(row["operator_reduced_beats_control"] for row in operator_rows),
        "operator_reduced_total": len(operator_rows),
        "operator_increment_mean_pct": mean(row["operator_increment_vs_control_pct"] for row in operator_rows),
        "operator_gap_increment_mean_pct": mean(row["operator_increment_vs_control_pct"] for row in operator_gap_rows),
        "full_operator_degradation_vs_reduced_operator_mean_pct": mean(
            row["full_operator_vs_reduced_operator_pct"] for row in operator_rows
        ),
        "long_segment_pressure_delta_pct_mean": mean(row["effective_pressure_delta_pct"] for row in long_pressure),
        "max_prefix_mismatch_mse": max(row["prefix_mismatch_mse"] for row in prefix_rows),
        "epochs_recorded": {row["dataset"]: row["epochs_recorded"] for row in training_rows},
        "uses_window_position": any(row["use_window_position"] for row in training_rows),
    }
    summary["decision"] = {
        "reduced_horizon_set_alone_is_sufficient": (
            summary["control_wins_vs_r3_full"] >= 4
            and summary["control_mean_relative_mse_vs_r3_full_pct"] <= 0.0
            and summary["control_observed_aggregate_gap_wins"] == summary["control_observed_aggregate_gap_total"]
        ),
        "reduced_horizon_set_is_material_factor": (
            summary["control_h720_mean_relative_mse_vs_r3_full_pct"] <= 0.0
            and summary["long_segment_pressure_delta_pct_mean"] > 25.0
        ),
        "operator_only_claim_is_rejected": summary["full_operator_degradation_vs_reduced_operator_mean_pct"] > 1.0,
        "horizon_set_interference_mainline_supported": (
            summary["control_h720_mean_relative_mse_vs_r3_full_pct"] <= 0.0
            and summary["full_operator_degradation_vs_reduced_operator_mean_pct"] > 1.0
        ),
    }
    return summary


def fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def write_report(
    path: Path,
    summary: dict[str, Any],
    vs_r3_rows: list[dict[str, Any]],
    operator_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    pressure_rows: list[dict[str, Any]],
) -> None:
    decision = summary["decision"]
    lines = [
        "# Phase3 Horizon-Set Interference Analysis",
        "",
        "## What This Tests",
        "",
        "[Fact] This report analyzes `PatchEncoderPrefixRiskWeightedH96H720`, an R.3 carrier trained only on horizons `96,720` with no Phase3-C operator and no `window_index_norm`.",
        "",
        "[Question] Does the earlier positive `h96,h720` result come from the Phase3-C operator, or from removing intermediate horizons `192/336` from the training objective?",
        "",
        "## Statistic Definitions",
        "",
        "- `relative_mse_vs_r3_full_pct`: `(control_mse / R3_full_mse - 1) * 100`; negative is better than full-horizon R.3.",
        "- `operator_increment_vs_control_pct`: `(operator_reduced_mse / control_reduced_mse - 1) * 100`; negative means the Phase3-C operator improves over the reduced-set carrier.",
        "- `full_operator_vs_reduced_operator_pct`: `(operator_full_mse / operator_reduced_mse - 1) * 100`; positive means the operator degrades when `192/336` are restored.",
        "- `effective_pressure_delta_pct`: `(control_effective_pressure / R3_full_effective_pressure - 1) * 100`; positive means the reduced horizon set gives that segment more objective pressure.",
        "- `observed gap`: previously flagged aggregate or H720-segment settings where R.3 still had a notable gap.",
        "",
        "## Decision",
        "",
    ]
    if decision["horizon_set_interference_mainline_supported"]:
        lines.append("[Decision] The new mainline should be `horizon-set interference`, not a pure operator story.")
    else:
        lines.append("[Decision] The current control does not yet support `horizon-set interference` as the mainline.")
    lines.extend(
        [
            "",
            f"- reduced horizon set alone sufficient: `{decision['reduced_horizon_set_alone_is_sufficient']}`",
            f"- reduced horizon set is material factor: `{decision['reduced_horizon_set_is_material_factor']}`",
            f"- operator-only claim rejected: `{decision['operator_only_claim_is_rejected']}`",
            f"- horizon-set interference mainline supported: `{decision['horizon_set_interference_mainline_supported']}`",
            "",
            "## Key Results",
            "",
            f"- Control wins vs full-horizon R.3: `{summary['control_wins_vs_r3_full']}/{summary['control_total_vs_r3_full']}`.",
            f"- Control mean relative MSE vs full-horizon R.3: `{fmt_pct(summary['control_mean_relative_mse_vs_r3_full_pct'])}`.",
            f"- H96 mean relative MSE: `{fmt_pct(summary['control_h96_mean_relative_mse_vs_r3_full_pct'])}`.",
            f"- H720 mean relative MSE: `{fmt_pct(summary['control_h720_mean_relative_mse_vs_r3_full_pct'])}`.",
            f"- Observed aggregate-gap wins: `{summary['control_observed_aggregate_gap_wins']}/{summary['control_observed_aggregate_gap_total']}`.",
            f"- Observed H720 segment-gap wins: `{summary['control_observed_h720_segment_gap_wins']}/{summary['control_observed_h720_segment_gap_total']}`.",
            f"- Operator reduced-set wins over this control: `{summary['operator_reduced_beats_control']}/{summary['operator_reduced_total']}`.",
            f"- Operator reduced-set mean increment vs control: `{fmt_pct(summary['operator_increment_mean_pct'])}`.",
            f"- Full-set operator degradation vs reduced-set operator: `{fmt_pct(summary['full_operator_degradation_vs_reduced_operator_mean_pct'])}`.",
            f"- Mean 337-720 effective-pressure delta from removing 192/336: `{fmt_pct(summary['long_segment_pressure_delta_pct_mean'])}`.",
            f"- Max prefix mismatch MSE: `{summary['max_prefix_mismatch_mse']:.3e}`.",
            f"- Epochs recorded: `{summary['epochs_recorded']}`.",
            "",
            "## Interpretation",
            "",
            "[Fact] The reduced-set carrier improves the H720 average but hurts the H96 average. Therefore, removing `192/336` is not sufficient by itself to reproduce the full Phase3-C `h96,h720` result.",
            "",
            "[Fact] The Phase3-C history-only operator still beats this reduced-set carrier on part of the reduced-set matrix. Therefore, the operator has conditional value when the horizon set is sparse.",
            "",
            "[Strong Evidence] The same operator fails after restoring `96,192,336,720`. This rejects a simple operator-only paper story and points to interaction between the operator and horizon-set/objective pressure.",
            "",
            "[Inference] The paper-worthy problem should be reframed as: multi-horizon training creates conflicting objective pressure across future steps; useful horizon-specific adaptation must control that interference rather than merely add a target-conditioned module.",
            "",
            "## Control vs Full-Horizon R.3",
            "",
            "| Dataset | Horizon | Control MSE | R.3 full MSE | Rel MSE | Win | Observed gap |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in vs_r3_rows:
        lines.append(
            "| {dataset} | {horizon} | {control:.6f} | {r3:.6f} | {rel} | {win} | {gap} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                control=row["control_mse"],
                r3=row["r3_full_mse"],
                rel=fmt_pct(row["relative_mse_vs_r3_full_pct"]),
                win=row["control_wins_mse_vs_r3_full"],
                gap=row["is_observed_gap"],
            )
        )
    lines.extend(
        [
            "",
            "## Operator Increment Under Reduced Horizon Set",
            "",
            "| Dataset | Horizon | Control MSE | Operator reduced MSE | Operator increment | Full operator degradation |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in operator_rows:
        lines.append(
            "| {dataset} | {horizon} | {control:.6f} | {operator:.6f} | {inc} | {full} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                control=row["control_reduced_mse"],
                operator=row["operator_reduced_mse"],
                inc=fmt_pct(row["operator_increment_vs_control_pct"]),
                full=fmt_pct(row["full_operator_vs_reduced_operator_pct"]),
            )
        )
    lines.extend(
        [
            "",
            "## H720 Segment Control vs R.3",
            "",
            "| Dataset | Segment | Control MSE | R.3 full MSE | Rel MSE | Win | Observed gap |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in segment_rows:
        lines.append(
            "| {dataset} | {segment} | {control:.6f} | {r3:.6f} | {rel} | {win} | {gap} |".format(
                dataset=row["dataset"],
                segment=row["segment"],
                control=row["control_mse"],
                r3=row["r3_full_mse"],
                rel=fmt_pct(row["relative_mse_vs_r3_full_pct"]),
                win=row["control_wins_mse_vs_r3_full"],
                gap=row["is_observed_gap"],
            )
        )
    lines.extend(
        [
            "",
            "## Objective Pressure Shift",
            "",
            "| Dataset | Segment | Control exposure | R.3 full exposure | Control pressure | R.3 full pressure | Pressure delta |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in pressure_rows:
        lines.append(
            "| {dataset} | {segment} | {ce:.6f} | {re:.6f} | {cp:.6f} | {rp:.6f} | {delta} |".format(
                dataset=row["dataset"],
                segment=row["segment"],
                ce=row["control_exposure_share"],
                re=row["r3_full_exposure_share"],
                cp=row["control_effective_pressure"],
                rp=row["r3_full_effective_pressure"],
                delta=fmt_pct(row["effective_pressure_delta_pct"]),
            )
        )
    lines.extend(
        [
            "",
            "## Next Research Direction",
            "",
            "[Decision] Return to Step 2-3/6 of the 11-step loop: define and validate `horizon-set interference` as the problem before adding another complex architecture.",
            "",
            "[Plan] The next minimal experiment should build a horizon-interference map with pair controls: `96,192`, `96,336`, `96,720`, `192,720`, and `336,720` on the R.3 carrier. The purpose is to identify which neighboring or distant horizons create the destructive pressure observed in the full four-horizon run.",
            "",
            "[Plan] Only after that map is clear should we design a mechanism, likely a conflict-aware objective/sampler or horizon-clustered training schedule. A new MoE/router should be delayed until the interference source is measured.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase3 horizon-set interference control.")
    parser.add_argument("--analysis-root", default="analysis/phase3_horizon_set_interference_20260624")
    parser.add_argument(
        "--r3-analysis-root",
        default="analysis/phase1_prefix_risk_weighted_gate_20260622",
    )
    parser.add_argument(
        "--r3-objective-root",
        default="analysis/phase2_qdf_alignment_diagnostic_20260623/raw",
        help="Raw R.3 run that contains objective_weight_stats.csv.",
    )
    parser.add_argument(
        "--operator-analysis-root",
        default="analysis/phase3_regime_segment_operator_history_only_20260624",
    )
    parser.add_argument(
        "--full-operator-analysis-root",
        default="analysis/phase3_regime_segment_operator_history_only_full_20260624",
    )
    parser.add_argument("--output-prefix", default="phase3_horizon_set_interference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root = Path(args.analysis_root)
    control_raw_root = analysis_root / "raw"
    r3_raw_root = Path(args.r3_analysis_root) / "raw"
    r3_objective_raw_root = Path(args.r3_objective_root)
    operator_raw_root = Path(args.operator_analysis_root) / "raw"
    full_operator_raw_root = Path(args.full_operator_analysis_root) / "raw"

    vs_r3_rows = collect_vs_r3(control_raw_root, r3_raw_root)
    operator_rows = collect_operator_increment(control_raw_root, operator_raw_root, full_operator_raw_root)
    segment_rows = collect_segment_vs_r3(control_raw_root, r3_raw_root)
    pressure_rows = collect_objective_pressure(control_raw_root, r3_objective_raw_root)
    prefix_rows = collect_prefix_rows(control_raw_root)
    training_rows = collect_training_rows(control_raw_root)
    summary = summarize(vs_r3_rows, operator_rows, segment_rows, pressure_rows, prefix_rows, training_rows)

    write_csv(analysis_root / f"{args.output_prefix}_vs_r3_full.csv", vs_r3_rows)
    write_csv(analysis_root / f"{args.output_prefix}_operator_increment.csv", operator_rows)
    write_csv(analysis_root / f"{args.output_prefix}_h720_segments_vs_r3_full.csv", segment_rows)
    write_csv(analysis_root / f"{args.output_prefix}_objective_pressure.csv", pressure_rows)
    write_csv(analysis_root / f"{args.output_prefix}_prefix_consistency.csv", prefix_rows)
    write_csv(analysis_root / f"{args.output_prefix}_training_summary.csv", training_rows)
    (analysis_root / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(
        analysis_root / f"{args.output_prefix}_report.md",
        summary,
        vs_r3_rows,
        operator_rows,
        segment_rows,
        pressure_rows,
    )


if __name__ == "__main__":
    main()
