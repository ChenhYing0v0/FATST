from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = ["ETTh2", "ETTm1", "Weather"]
SEED = 2021
CANDIDATE_RUN = "mixed_h96_h720"
R3_RUN = "mixed_h96_h192_h336_h720"
R3_NAME = "PatchEncoderPrefixRiskWeighted"
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


def candidate_dir(raw_root: Path, model_name: str, candidate_run: str, dataset: str) -> Path:
    return raw_root / model_name / dataset / candidate_run / f"seed{SEED}"


def r3_dir(r3_raw_root: Path, dataset: str) -> Path:
    return r3_raw_root / R3_NAME / dataset / R3_RUN / f"seed{SEED}"


def collect_horizons(raw_root: Path, model_name: str, candidate_run: str) -> list[int]:
    metrics = read_csv(
        candidate_dir(raw_root, model_name, candidate_run, DATASETS[0])
        / "metrics_by_target_horizon.csv"
    )
    return [int(row["target_horizon"]) for row in metrics]


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def collect_main_rows(
    raw_root: Path,
    r3_raw_root: Path,
    model_name: str,
    candidate_run: str,
    horizons: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        candidate_metrics = {
            int(row["target_horizon"]): row
            for row in read_csv(
                candidate_dir(raw_root, model_name, candidate_run, dataset) / "metrics_by_target_horizon.csv"
            )
        }
        r3_metrics = {
            int(row["target_horizon"]): row
            for row in read_csv(r3_dir(r3_raw_root, dataset) / "metrics_by_target_horizon.csv")
        }
        for horizon in horizons:
            candidate = candidate_metrics[horizon]
            r3 = r3_metrics[horizon]
            candidate_mse = float(candidate["mse"])
            r3_mse = float(r3["mse"])
            candidate_mae = float(candidate["mae"])
            r3_mae = float(r3["mae"])
            rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "candidate_mse": candidate_mse,
                    "r3_mse": r3_mse,
                    "relative_mse_vs_r3_pct": pct(candidate_mse, r3_mse),
                    "candidate_wins_mse_vs_r3": candidate_mse < r3_mse,
                    "candidate_mae": candidate_mae,
                    "r3_mae": r3_mae,
                    "relative_mae_vs_r3_pct": pct(candidate_mae, r3_mae),
                    "candidate_wins_mae_vs_r3": candidate_mae < r3_mae,
                    "is_observed_gap": (dataset, horizon, "aggregate") in OBSERVED_GAPS,
                }
            )
    return rows


def collect_segment_rows(
    raw_root: Path,
    r3_raw_root: Path,
    model_name: str,
    candidate_run: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        candidate_rows = {
            row["segment"]: row
            for row in read_csv(
                candidate_dir(raw_root, model_name, candidate_run, dataset)
                / "h720"
                / "metrics_by_segment.csv"
            )
        }
        r3_rows = {
            row["segment"]: row
            for row in read_csv(r3_dir(r3_raw_root, dataset) / "h720" / "metrics_by_segment.csv")
        }
        for segment, candidate in candidate_rows.items():
            r3 = r3_rows[segment]
            candidate_mse = float(candidate["mse"])
            r3_mse = float(r3["mse"])
            rows.append(
                {
                    "dataset": dataset,
                    "segment": segment,
                    "candidate_mse": candidate_mse,
                    "r3_mse": r3_mse,
                    "relative_mse_vs_r3_pct": pct(candidate_mse, r3_mse),
                    "candidate_wins_mse_vs_r3": candidate_mse < r3_mse,
                    "is_observed_gap": (dataset, 720, segment) in OBSERVED_GAPS,
                }
            )
    return rows


def collect_prefix_rows(raw_root: Path, model_name: str, candidate_run: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for row in read_csv(candidate_dir(raw_root, model_name, candidate_run, dataset) / "prefix_consistency.csv"):
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


def collect_operator_rows(
    raw_root: Path,
    model_name: str,
    candidate_run: str,
    horizons: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    operator_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for horizon in horizons:
            horizon_dir = candidate_dir(raw_root, model_name, candidate_run, dataset) / f"h{horizon}"
            for row in read_csv(horizon_dir / "regime_segment_operator_stats.csv"):
                operator_rows.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "scope": row["scope"],
                        "mean_abs_scale": float(row["mean_abs_scale"]),
                        "mean_abs_shift": float(row["mean_abs_shift"]),
                        "max_abs_scale": float(row["max_abs_scale"]),
                        "max_abs_shift": float(row["max_abs_shift"]),
                    }
                )
            for row in read_csv(horizon_dir / "regime_feature_stats.csv"):
                feature_rows.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "feature": row["feature"],
                        "mean": float(row["mean"]),
                        "std": float(row["std"]),
                        "mean_abs": float(row["mean_abs"]),
                        "max_abs": float(row["max_abs"]),
                    }
                )
    return operator_rows, feature_rows


def collect_training_rows(raw_root: Path, model_name: str, candidate_run: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        log_rows = read_csv(candidate_dir(raw_root, model_name, candidate_run, dataset) / "training_log.csv")
        config = read_json(candidate_dir(raw_root, model_name, candidate_run, dataset) / "effective_config.json")
        best = min(float(row["val_mean_mse"]) for row in log_rows)
        last = float(log_rows[-1]["val_mean_mse"])
        rows.append(
            {
                "dataset": dataset,
                "epochs_recorded": len(log_rows),
                "best_val_mean_mse": best,
                "last_val_mean_mse": last,
                "steps_per_epoch_effective": int(config["steps_per_epoch_effective"]),
                "target_horizons": ",".join(str(item) for item in config["target_horizons"]),
                "use_window_position": bool(config.get("use_window_position", False)),
                "model_variant": config.get("model_variant", ""),
            }
        )
    return rows


def summarize(
    main_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    prefix_rows: list[dict[str, Any]],
    operator_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    horizons: list[int],
) -> dict[str, Any]:
    gap_main = [row for row in main_rows if row["is_observed_gap"]]
    non_gap_main = [row for row in main_rows if not row["is_observed_gap"]]
    gap_segments = [row for row in segment_rows if row["is_observed_gap"]]
    non_gap_segments = [row for row in segment_rows if not row["is_observed_gap"]]
    operator_all = [row for row in operator_rows if row["scope"] == "all"]
    window_features = [row for row in feature_rows if row["feature"] == "window_index_norm"]
    target_horizon_confounded = horizons != [96, 192, 336, 720]
    summary = {
        "mse_wins_vs_r3": sum(row["candidate_wins_mse_vs_r3"] for row in main_rows),
        "mse_total_vs_r3": len(main_rows),
        "mean_relative_mse_vs_r3_pct": mean(row["relative_mse_vs_r3_pct"] for row in main_rows),
        "observed_aggregate_gap_wins": sum(row["candidate_wins_mse_vs_r3"] for row in gap_main),
        "observed_aggregate_gap_total": len(gap_main),
        "observed_h720_segment_gap_wins": sum(row["candidate_wins_mse_vs_r3"] for row in gap_segments),
        "observed_h720_segment_gap_total": len(gap_segments),
        "non_gap_mean_relative_mse_vs_r3_pct": mean(row["relative_mse_vs_r3_pct"] for row in non_gap_main),
        "non_gap_segment_mean_relative_mse_vs_r3_pct": mean(
            row["relative_mse_vs_r3_pct"] for row in non_gap_segments
        ),
        "max_prefix_mismatch_mse": max(row["prefix_mismatch_mse"] for row in prefix_rows),
        "max_prefix_mismatch_mae": max(row["prefix_mismatch_mae"] for row in prefix_rows),
        "max_operator_abs_scale": max(row["max_abs_scale"] for row in operator_all),
        "max_operator_abs_shift": max(row["max_abs_shift"] for row in operator_all),
        "mean_operator_abs_scale": mean(row["mean_abs_scale"] for row in operator_all),
        "mean_operator_abs_shift": mean(row["mean_abs_shift"] for row in operator_all),
        "window_index_feature_mean_std": mean(row["std"] for row in window_features),
        "candidate_horizons": horizons,
        "target_horizon_confounded": target_horizon_confounded,
        "uses_window_position": all(row["use_window_position"] for row in training_rows),
    }
    summary["gate"] = {
        "performance_gate_pass": (
            summary["mse_wins_vs_r3"] >= 4
            and summary["observed_aggregate_gap_wins"] >= 1
            and summary["observed_h720_segment_gap_wins"] >= 2
            and summary["non_gap_mean_relative_mse_vs_r3_pct"] <= 0.2
            and summary["max_prefix_mismatch_mse"] <= 1e-10
        ),
        "prefix_consistency_pass": summary["max_prefix_mismatch_mse"] <= 1e-10,
        "window_index_concern_blocks_claim": summary["uses_window_position"],
        "needs_no_window_position_ablation": summary["uses_window_position"],
        "needs_same_horizon_set_control": target_horizon_confounded,
    }
    return summary


def fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def write_report(
    path: Path,
    summary: dict[str, Any],
    main_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    prefix_rows: list[dict[str, Any]],
    operator_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    horizons: list[int],
) -> None:
    lines = [
        "# Phase3-C Regime/Segment Operator Gate Report",
        "",
        "## Decision",
        "",
    ]
    if summary["gate"]["performance_gate_pass"]:
        if summary["uses_window_position"]:
            lines.append("[Decision] performance gate passes numerically, but the mechanism claim is blocked by controls.")
        else:
            lines.append("[Decision] history-only performance gate passes numerically; next control must align horizon set.")
    else:
        lines.append("[Decision] performance gate does not pass as a paper-story candidate.")
    lines.extend(
        [
            "",
            f"[Fact] This run used `TARGET_HORIZONS={','.join(str(item) for item in horizons)}`.",
            f"[Fact] Horizon-set confound vs R.3: `{summary['target_horizon_confounded']}`.",
            f"[Fact] This run used `window_index_norm`: `{summary['uses_window_position']}`.",
            "[Decision] The result can be considered clean only when window-position and horizon-set controls pass.",
            "",
            "## Summary",
            "",
            f"- MSE wins vs R.3: `{summary['mse_wins_vs_r3']}/{summary['mse_total_vs_r3']}`.",
            f"- Mean relative MSE vs R.3: `{fmt_pct(summary['mean_relative_mse_vs_r3_pct'])}`.",
            f"- Observed aggregate-gap wins: `{summary['observed_aggregate_gap_wins']}/{summary['observed_aggregate_gap_total']}`.",
            f"- Observed H720 segment-gap wins: `{summary['observed_h720_segment_gap_wins']}/{summary['observed_h720_segment_gap_total']}`.",
            f"- Non-gap mean relative MSE vs R.3: `{fmt_pct(summary['non_gap_mean_relative_mse_vs_r3_pct'])}`.",
            f"- Max prefix mismatch MSE: `{summary['max_prefix_mismatch_mse']:.3e}`.",
            f"- Mean operator abs scale: `{summary['mean_operator_abs_scale']:.6f}`.",
            f"- Mean operator abs shift: `{summary['mean_operator_abs_shift']:.6f}`.",
            "",
            "## Main Metrics vs R.3",
            "",
            "| Dataset | Horizon | Candidate MSE | R.3 MSE | Rel MSE | Win | Observed gap |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in main_rows:
        lines.append(
            "| {dataset} | {horizon} | {candidate:.6f} | {r3:.6f} | {rel} | {win} | {gap} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                candidate=row["candidate_mse"],
                r3=row["r3_mse"],
                rel=fmt_pct(row["relative_mse_vs_r3_pct"]),
                win=row["candidate_wins_mse_vs_r3"],
                gap=row["is_observed_gap"],
            )
        )
    lines.extend(
        [
            "",
            "## H720 Segment Metrics vs R.3",
            "",
            "| Dataset | Segment | Candidate MSE | R.3 MSE | Rel MSE | Win | Observed gap |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in segment_rows:
        lines.append(
            "| {dataset} | {segment} | {candidate:.6f} | {r3:.6f} | {rel} | {win} | {gap} |".format(
                dataset=row["dataset"],
                segment=row["segment"],
                candidate=row["candidate_mse"],
                r3=row["r3_mse"],
                rel=fmt_pct(row["relative_mse_vs_r3_pct"]),
                win=row["candidate_wins_mse_vs_r3"],
                gap=row["is_observed_gap"],
            )
        )
    lines.extend(
        [
            "",
            "## Prefix Consistency",
            "",
            "| Dataset | Prefix MSE | Prefix MAE |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in prefix_rows:
        lines.append(
            f"| {row['dataset']} | {row['prefix_mismatch_mse']:.3e} | {row['prefix_mismatch_mae']:.3e} |"
        )
    lines.extend(
        [
            "",
            "## Operator Magnitude",
            "",
            "| Dataset | Horizon | Mean abs scale | Mean abs shift | Max abs scale | Max abs shift |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in [item for item in operator_rows if item["scope"] == "all"]:
        lines.append(
            "| {dataset} | {horizon} | {scale:.6f} | {shift:.6f} | {max_scale:.6f} | {max_shift:.6f} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                scale=row["mean_abs_scale"],
                shift=row["mean_abs_shift"],
                max_scale=row["max_abs_scale"],
                max_shift=row["max_abs_shift"],
            )
        )
    lines.extend(
        [
            "",
            "## Training Log",
            "",
            "| Dataset | Epoch rows | Best val mean MSE | Last val mean MSE | Target horizons | Window index |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in training_rows:
        lines.append(
            "| {dataset} | {epochs} | {best:.6f} | {last:.6f} | {horizons} | {window} |".format(
                dataset=row["dataset"],
                epochs=row["epochs_recorded"],
                best=row["best_val_mean_mse"],
                last=row["last_val_mean_mse"],
                horizons=row["target_horizons"],
                window=row["use_window_position"],
            )
        )
    lines.extend(
        [
            "",
            "## Window-Index Concern",
            "",
            "[Concern] `window_index_norm` is prediction-before, but it is not a robust causal or calendar variable.",
            "It is normalized inside each split, so it can encode train/val/test split position rather than a deployable regime.",
            "",
            "[Decision] Before claiming a mechanism, Phase3-C needs controls:",
            "",
            "1. same architecture without `window_index_norm`, using history-only regime features;",
            "2. same target horizon set as R.3, using `96,192,336,720`.",
            "",
            "[Next] If the no-window-position control keeps most gains, continue the regime-token route.",
            "If gains disappear, treat the current result as split-position shortcut evidence and rollback to Step 4-6.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase3-C regime/segment operator gate.")
    parser.add_argument("--analysis-root", default="analysis/phase3_regime_segment_operator_20260624")
    parser.add_argument(
        "--raw-root",
        default="analysis/phase3_regime_segment_operator_20260624/raw",
    )
    parser.add_argument(
        "--r3-raw-root",
        default="analysis/phase2_qdf_alignment_diagnostic_20260623/raw",
    )
    parser.add_argument("--model-name", default="PatchEncoderRegimeSegmentTargetOperator")
    parser.add_argument("--candidate-run", default=CANDIDATE_RUN)
    parser.add_argument("--output-prefix", default="phase3_regime_segment_operator")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root = Path(args.analysis_root)
    analysis_root.mkdir(parents=True, exist_ok=True)
    raw_root = Path(args.raw_root)
    r3_raw_root = Path(args.r3_raw_root)
    horizons = collect_horizons(raw_root, args.model_name, args.candidate_run)
    main_rows = collect_main_rows(raw_root, r3_raw_root, args.model_name, args.candidate_run, horizons)
    segment_rows = collect_segment_rows(raw_root, r3_raw_root, args.model_name, args.candidate_run)
    prefix_rows = collect_prefix_rows(raw_root, args.model_name, args.candidate_run)
    operator_rows, feature_rows = collect_operator_rows(raw_root, args.model_name, args.candidate_run, horizons)
    training_rows = collect_training_rows(raw_root, args.model_name, args.candidate_run)
    summary = summarize(main_rows, segment_rows, prefix_rows, operator_rows, feature_rows, training_rows, horizons)

    prefix = args.output_prefix
    write_csv(analysis_root / f"{prefix}_vs_r3.csv", main_rows)
    write_csv(analysis_root / f"{prefix}_h720_segments_vs_r3.csv", segment_rows)
    write_csv(analysis_root / f"{prefix}_prefix_consistency.csv", prefix_rows)
    write_csv(analysis_root / f"{prefix}_stats.csv", operator_rows)
    write_csv(analysis_root / f"{prefix}_feature_stats.csv", feature_rows)
    write_csv(analysis_root / f"{prefix}_training_summary.csv", training_rows)
    (analysis_root / f"{prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(
        analysis_root / f"{prefix}_report.md",
        summary,
        main_rows,
        segment_rows,
        prefix_rows,
        operator_rows,
        training_rows,
        horizons,
    )
    print(f"{prefix}_report={analysis_root / f'{prefix}_report.md'}")


if __name__ == "__main__":
    main()
