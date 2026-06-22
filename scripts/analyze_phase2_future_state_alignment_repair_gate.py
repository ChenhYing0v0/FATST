from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


DATASETS = ["ETTh2", "ETTm1", "Weather"]
HORIZONS = [96, 192, 336, 720]
SEED = 2021
TARGET_RUN = "mixed_h96_h192_h336_h720"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, float]:
    return json.loads(path.read_text())


def run_dir(raw_root: Path, model_name: str, dataset: str) -> Path:
    return raw_root / model_name / dataset / TARGET_RUN / f"seed{SEED}"


def format_pct(value: float) -> str:
    return f"{value:+.2f}%"


def r3_reference(reference_csv: Path) -> dict[tuple[str, int], dict[str, float]]:
    rows = read_csv(reference_csv)
    reference: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        key = (row["dataset"], int(row["horizon"]))
        reference[key] = {
            "r3_mse": float(row["target_mse"]),
            "r3_mae": float(row["target_mae"]),
            "fixed_mse": float(row["fixed_mse"]),
            "fixed_mae": float(row["fixed_mae"]),
        }
    return reference


def collect_main_rows(
    raw_root: Path,
    model_name: str,
    reference: dict[tuple[str, int], dict[str, float]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    vs_fixed_rows: list[dict[str, object]] = []
    vs_r3_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        metrics = read_csv(run_dir(raw_root, model_name, dataset) / "metrics_by_target_horizon.csv")
        by_horizon = {int(row["target_horizon"]): row for row in metrics}
        for horizon in HORIZONS:
            target = by_horizon[horizon]
            ref = reference[(dataset, horizon)]
            repair_mse = float(target["mse"])
            repair_mae = float(target["mae"])
            fixed_mse = ref["fixed_mse"]
            fixed_mae = ref["fixed_mae"]
            r3_mse = ref["r3_mse"]
            r3_mae = ref["r3_mae"]
            vs_fixed_rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "repair_mse": repair_mse,
                    "repair_mae": repair_mae,
                    "fixed_mse": fixed_mse,
                    "fixed_mae": fixed_mae,
                    "relative_mse_vs_fixed_pct": (repair_mse / fixed_mse - 1.0) * 100.0,
                    "relative_mae_vs_fixed_pct": (repair_mae / fixed_mae - 1.0) * 100.0,
                    "repair_wins_mse_vs_fixed": repair_mse < fixed_mse,
                    "repair_wins_mae_vs_fixed": repair_mae < fixed_mae,
                }
            )
            vs_r3_rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "repair_mse": repair_mse,
                    "repair_mae": repair_mae,
                    "r3_mse": r3_mse,
                    "r3_mae": r3_mae,
                    "relative_mse_vs_r3_pct": (repair_mse / r3_mse - 1.0) * 100.0,
                    "relative_mae_vs_r3_pct": (repair_mae / r3_mae - 1.0) * 100.0,
                    "repair_wins_mse_vs_r3": repair_mse < r3_mse,
                    "repair_wins_mae_vs_r3": repair_mae < r3_mae,
                }
            )
    return vs_fixed_rows, vs_r3_rows


def collect_prefix_rows(raw_root: Path, model_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for row in read_csv(run_dir(raw_root, model_name, dataset) / "prefix_consistency.csv"):
            rows.append(
                {
                    "dataset": dataset,
                    "short_horizon": int(row["short_horizon"]),
                    "long_horizon": int(row["long_horizon"]),
                    "prefix_mismatch_mse": float(row["prefix_mismatch_mse"]),
                    "prefix_mismatch_mae": float(row["prefix_mismatch_mae"]),
                    "truth_alignment_mse": float(row["truth_alignment_mse"]),
                }
            )
    return rows


def collect_alignment_rows(raw_root: Path, model_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            path = run_dir(raw_root, model_name, dataset) / f"h{horizon}" / "future_alignment_stats.csv"
            stats = read_csv(path)[0]
            rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "future_local_alignment_loss": float(stats["future_local_alignment_loss"]),
                    "future_relation_alignment_loss": float(stats["future_relation_alignment_loss"]),
                    "future_reconstruction_loss": float(stats["future_reconstruction_loss"]),
                    "future_raw_reconstruction_loss": float(
                        stats.get("future_raw_reconstruction_loss", stats["future_reconstruction_loss"])
                    ),
                    "future_normalized_reconstruction_loss": float(
                        stats.get(
                            "future_normalized_reconstruction_loss",
                            stats["future_reconstruction_loss"],
                        )
                    ),
                    "future_alignment_confidence_mean": float(
                        stats.get("future_alignment_confidence_mean", "1.0")
                    ),
                    "future_alignment_confidence_min": float(
                        stats.get("future_alignment_confidence_min", "1.0")
                    ),
                    "future_alignment_confidence_max": float(
                        stats.get("future_alignment_confidence_max", "1.0")
                    ),
                    "teacher_student_cosine": float(stats["teacher_student_cosine"]),
                    "prediction_leakage_max_abs": float(stats["prediction_leakage_max_abs"]),
                }
            )
    return rows


def collect_config_rows(raw_root: Path, model_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        config = read_json(run_dir(raw_root, model_name, dataset) / "effective_config.json")
        rows.append(
            {
                "dataset": dataset,
                "future_recon_normalization": config.get("future_recon_normalization", "none"),
                "future_align_weighting": config.get("future_align_weighting", "uniform"),
                "future_confidence_temperature": config.get("future_confidence_temperature", 1.0),
                "future_confidence_floor": config.get("future_confidence_floor", 0.0),
                "future_align_weight": config.get("future_align_weight", 0.0),
                "future_relation_weight": config.get("future_relation_weight", 0.0),
                "future_recon_weight": config.get("future_recon_weight", 0.0),
                "epochs": config.get("epochs", 0),
                "steps_per_epoch_effective": config.get("steps_per_epoch_effective", 0),
            }
        )
    return rows


def summarize(
    vs_fixed_rows: list[dict[str, object]],
    vs_r3_rows: list[dict[str, object]],
    prefix_rows: list[dict[str, object]],
    alignment_rows: list[dict[str, object]],
) -> dict[str, object]:
    fixed_rel = [float(row["relative_mse_vs_fixed_pct"]) for row in vs_fixed_rows]
    r3_rel = [float(row["relative_mse_vs_r3_pct"]) for row in vs_r3_rows]
    dataset_mean_vs_r3 = {
        dataset: mean(float(row["relative_mse_vs_r3_pct"]) for row in vs_r3_rows if row["dataset"] == dataset)
        for dataset in DATASETS
    }
    dataset_mean_vs_fixed = {
        dataset: mean(
            float(row["relative_mse_vs_fixed_pct"]) for row in vs_fixed_rows if row["dataset"] == dataset
        )
        for dataset in DATASETS
    }
    horizon_mean_vs_r3 = {
        str(horizon): mean(
            float(row["relative_mse_vs_r3_pct"]) for row in vs_r3_rows if row["horizon"] == horizon
        )
        for horizon in HORIZONS
    }
    max_prefix_mismatch = max(float(row["prefix_mismatch_mse"]) for row in prefix_rows)
    max_leakage = max(float(row["prediction_leakage_max_abs"]) for row in alignment_rows)
    gate = {
        "prediction_leakage_pass": max_leakage <= 1e-7,
        "prefix_consistency_pass": max_prefix_mismatch <= 1e-10,
        "etth2_conflict_repaired": dataset_mean_vs_r3["ETTh2"] <= 0.3,
        "ettm1_signal_preserved": dataset_mean_vs_r3["ETTm1"] <= 0.0,
        "weather_signal_preserved": dataset_mean_vs_r3["Weather"] <= 0.0,
    }
    gate["pass"] = all(gate.values())
    return {
        "mse_wins_vs_fixed": sum(1 for row in vs_fixed_rows if row["repair_wins_mse_vs_fixed"]),
        "mae_wins_vs_fixed": sum(1 for row in vs_fixed_rows if row["repair_wins_mae_vs_fixed"]),
        "mean_relative_mse_vs_fixed_pct": mean(fixed_rel),
        "dataset_mean_relative_mse_vs_fixed_pct": dataset_mean_vs_fixed,
        "mse_wins_vs_r3": sum(1 for row in vs_r3_rows if row["repair_wins_mse_vs_r3"]),
        "mae_wins_vs_r3": sum(1 for row in vs_r3_rows if row["repair_wins_mae_vs_r3"]),
        "mean_relative_mse_vs_r3_pct": mean(r3_rel),
        "dataset_mean_relative_mse_vs_r3_pct": dataset_mean_vs_r3,
        "horizon_mean_relative_mse_vs_r3_pct": horizon_mean_vs_r3,
        "max_prefix_mismatch_mse": max_prefix_mismatch,
        "max_prediction_leakage_abs": max_leakage,
        "mean_teacher_student_cosine": mean(float(row["teacher_student_cosine"]) for row in alignment_rows),
        "mean_future_alignment_confidence": mean(
            float(row["future_alignment_confidence_mean"]) for row in alignment_rows
        ),
        "mean_future_normalized_reconstruction_loss": mean(
            float(row["future_normalized_reconstruction_loss"]) for row in alignment_rows
        ),
        "mean_future_raw_reconstruction_loss": mean(
            float(row["future_raw_reconstruction_loss"]) for row in alignment_rows
        ),
        "gate": gate,
    }


def write_report(
    path: Path,
    summary: dict[str, object],
    vs_r3_rows: list[dict[str, object]],
    alignment_rows: list[dict[str, object]],
    model_name: str,
) -> None:
    dataset_mean = summary["dataset_mean_relative_mse_vs_r3_pct"]
    horizon_mean = summary["horizon_mean_relative_mse_vs_r3_pct"]
    gate = summary["gate"]
    decision = "passes" if gate["pass"] else "fails"
    lines = [
        "# Phase2-R.1 Confidence-Weighted Future Alignment Gate Report",
        "",
        "## Decision",
        "",
        f"[Decision] `{model_name}` {decision} the Phase2-R.1 repair gate.",
        "",
    ]
    if gate["pass"]:
        lines.append(
            "[Inference] Confidence-weighted future alignment repairs the ETTh2 conflict while preserving useful ETTm1/Weather signal. It can remain a future-aware decoder candidate, but still needs a paper-story check against stronger baselines and ablations."
        )
    else:
        lines.append(
            "[Inference] The repair does not yet prove future-state alignment is a paper-core mechanism. If leakage and prefix checks pass but MSE/MAE do not, the likely issue is semantic mismatch between teacher state and forecasting objective rather than implementation safety."
        )
    lines += [
        "",
        "## Main Metrics vs R.3",
        "",
        f"- MSE wins vs `PatchEncoderPrefixRiskWeighted`: `{summary['mse_wins_vs_r3']}/12`.",
        f"- MAE wins vs `PatchEncoderPrefixRiskWeighted`: `{summary['mae_wins_vs_r3']}/12`.",
        f"- Mean relative MSE vs R.3: `{format_pct(float(summary['mean_relative_mse_vs_r3_pct']))}`.",
        f"- ETTh2 mean relative MSE vs R.3: `{format_pct(float(dataset_mean['ETTh2']))}`.",
        f"- ETTm1 mean relative MSE vs R.3: `{format_pct(float(dataset_mean['ETTm1']))}`.",
        f"- Weather mean relative MSE vs R.3: `{format_pct(float(dataset_mean['Weather']))}`.",
        "",
        "## Main Metrics vs FixedHead",
        "",
        f"- MSE wins vs horizon-specific `PatchEncoderFixedHead`: `{summary['mse_wins_vs_fixed']}/12`.",
        f"- MAE wins vs horizon-specific `PatchEncoderFixedHead`: `{summary['mae_wins_vs_fixed']}/12`.",
        f"- Mean relative MSE vs FixedHead: `{format_pct(float(summary['mean_relative_mse_vs_fixed_pct']))}`.",
        "",
        "## Gate",
        "",
        f"- prediction leakage <= `1e-7`: `{gate['prediction_leakage_pass']}` (`{summary['max_prediction_leakage_abs']:.8g}`).",
        f"- prefix mismatch numerical zero: `{gate['prefix_consistency_pass']}` (`{summary['max_prefix_mismatch_mse']:.8g}`).",
        f"- ETTh2 conflict repaired <= `+0.3%`: `{gate['etth2_conflict_repaired']}` (`{format_pct(float(dataset_mean['ETTh2']))}`).",
        f"- ETTm1 signal preserved: `{gate['ettm1_signal_preserved']}` (`{format_pct(float(dataset_mean['ETTm1']))}`).",
        f"- Weather signal preserved: `{gate['weather_signal_preserved']}` (`{format_pct(float(dataset_mean['Weather']))}`).",
        "",
        "## Confidence Diagnostics",
        "",
        f"- mean teacher/student cosine: `{float(summary['mean_teacher_student_cosine']):.6f}`.",
        f"- mean alignment confidence: `{float(summary['mean_future_alignment_confidence']):.6f}`.",
        f"- mean normalized reconstruction loss: `{float(summary['mean_future_normalized_reconstruction_loss']):.6f}`.",
        f"- mean raw reconstruction loss: `{float(summary['mean_future_raw_reconstruction_loss']):.6f}`.",
        "",
        "| Dataset | Horizon | Relative MSE vs R.3 | Repair MSE | R.3 MSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in vs_r3_rows:
        lines.append(
            "| {dataset} | {horizon} | {rel} | {repair:.6f} | {r3:.6f} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                rel=format_pct(float(row["relative_mse_vs_r3_pct"])),
                repair=float(row["repair_mse"]),
                r3=float(row["r3_mse"]),
            )
        )
    lines += [
        "",
        "## Alignment Rows",
        "",
        "| Dataset | Horizon | Cosine | Confidence | Norm Recon | Raw Recon | Leakage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in alignment_rows:
        lines.append(
            "| {dataset} | {horizon} | {cos:.4f} | {conf:.4f} | {norm:.4f} | {raw:.4f} | {leak:.2g} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                cos=float(row["teacher_student_cosine"]),
                conf=float(row["future_alignment_confidence_mean"]),
                norm=float(row["future_normalized_reconstruction_loss"]),
                raw=float(row["future_raw_reconstruction_loss"]),
                leak=float(row["prediction_leakage_max_abs"]),
            )
        )
    lines += [
        "",
        "## Rollback Rule",
        "",
        "If this report fails only on performance while leakage/prefix pass, return to loop step 2-3 and reconsider whether future-state alignment is the right decoder problem. Do not stack MoE on this state before that rollback assessment.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase2-R.1 future-state alignment repair gate.")
    parser.add_argument("--analysis-root", default="analysis/phase2_future_state_alignment_repair_gate_20260623")
    parser.add_argument("--model-name", default="PatchEncoderFutureStateAlignmentConfWeighted")
    parser.add_argument("--output-prefix", default="phase2_future_state_alignment_repair")
    parser.add_argument(
        "--r3-reference",
        default="analysis/phase1_prefix_risk_weighted_gate_20260622/phase1_prefix_risk_weighted_vs_fixed.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root = Path(args.analysis_root)
    raw_root = analysis_root / "raw"
    reference = r3_reference(Path(args.r3_reference))
    vs_fixed_rows, vs_r3_rows = collect_main_rows(raw_root, args.model_name, reference)
    prefix_rows = collect_prefix_rows(raw_root, args.model_name)
    alignment_rows = collect_alignment_rows(raw_root, args.model_name)
    config_rows = collect_config_rows(raw_root, args.model_name)
    summary = summarize(vs_fixed_rows, vs_r3_rows, prefix_rows, alignment_rows)

    write_csv(analysis_root / f"{args.output_prefix}_vs_fixed.csv", vs_fixed_rows)
    write_csv(analysis_root / f"{args.output_prefix}_vs_r3.csv", vs_r3_rows)
    write_csv(analysis_root / f"{args.output_prefix}_prefix_consistency.csv", prefix_rows)
    write_csv(analysis_root / f"{args.output_prefix}_alignment_stats.csv", alignment_rows)
    write_csv(analysis_root / f"{args.output_prefix}_effective_configs.csv", config_rows)
    (analysis_root / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(
        analysis_root / f"{args.output_prefix}_decision_report.md",
        summary,
        vs_r3_rows,
        alignment_rows,
        args.model_name,
    )


if __name__ == "__main__":
    main()
