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
FOCUS_REGIONS = [
    ("ETTh2", "193-336"),
    ("ETTh2", "337-720"),
    ("ETTm1", "337-720"),
    ("Weather", "1-96"),
]


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


def read_json(path: Path) -> dict[str, object]:
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
            candidate_mse = float(target["mse"])
            candidate_mae = float(target["mae"])
            fixed_mse = ref["fixed_mse"]
            fixed_mae = ref["fixed_mae"]
            r3_mse = ref["r3_mse"]
            r3_mae = ref["r3_mae"]
            vs_fixed_rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "candidate_mse": candidate_mse,
                    "candidate_mae": candidate_mae,
                    "fixed_mse": fixed_mse,
                    "fixed_mae": fixed_mae,
                    "relative_mse_vs_fixed_pct": (candidate_mse / fixed_mse - 1.0) * 100.0,
                    "relative_mae_vs_fixed_pct": (candidate_mae / fixed_mae - 1.0) * 100.0,
                    "candidate_wins_mse_vs_fixed": candidate_mse < fixed_mse,
                    "candidate_wins_mae_vs_fixed": candidate_mae < fixed_mae,
                }
            )
            vs_r3_rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "candidate_mse": candidate_mse,
                    "candidate_mae": candidate_mae,
                    "r3_mse": r3_mse,
                    "r3_mae": r3_mae,
                    "relative_mse_vs_r3_pct": (candidate_mse / r3_mse - 1.0) * 100.0,
                    "relative_mae_vs_r3_pct": (candidate_mae / r3_mae - 1.0) * 100.0,
                    "candidate_wins_mse_vs_r3": candidate_mse < r3_mse,
                    "candidate_wins_mae_vs_r3": candidate_mae < r3_mae,
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


def collect_error_process_rows(raw_root: Path, model_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            path = run_dir(raw_root, model_name, dataset) / f"h{horizon}" / "error_process_stats.csv"
            for row in read_csv(path):
                rows.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "scope": row["scope"],
                        "residual_base_mae_ratio": float(row["residual_base_mae_ratio"]),
                        "residual_energy": float(row["residual_energy"]),
                        "residual_second_diff_smoothness": float(
                            row["residual_second_diff_smoothness"]
                        ),
                        "error_process_state_norm": float(row["error_process_state_norm"]),
                        "segment_state_cosine": float(row["segment_state_cosine"]),
                        "base_prediction_mse": float(row["base_prediction_mse"]),
                        "final_prediction_mse": float(row["final_prediction_mse"]),
                        "residual_gain_mse_pct": float(row["residual_gain_mse_pct"]),
                        "prediction_decomposition_max_abs": float(
                            row.get("prediction_decomposition_max_abs", "0")
                        ),
                    }
                )
    return rows


def collect_h720_region_rows(
    raw_root: Path,
    model_name: str,
    r3_raw_root: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        candidate_rows = {
            row["segment"]: row
            for row in read_csv(run_dir(raw_root, model_name, dataset) / "h720" / "metrics_by_segment.csv")
        }
        r3_rows = {
            row["segment"]: row
            for row in read_csv(run_dir(r3_raw_root, "PatchEncoderPrefixRiskWeighted", dataset) / "h720" / "metrics_by_segment.csv")
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
                    "relative_mse_vs_r3_pct": (candidate_mse / r3_mse - 1.0) * 100.0,
                    "candidate_wins_mse_vs_r3": candidate_mse < r3_mse,
                    "is_focus_region": (dataset, segment) in FOCUS_REGIONS,
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
                "model_variant": config.get("model_variant", ""),
                "error_process_dim": config.get("error_process_dim", 0),
                "error_process_layers": config.get("error_process_layers", 0),
                "error_residual_gate_init": config.get("error_residual_gate_init", 0.0),
                "error_energy_weight": config.get("error_energy_weight", 0.0),
                "error_smoothness_weight": config.get("error_smoothness_weight", 0.0),
                "step_loss_weighting": config.get("step_loss_weighting", ""),
                "step_loss_alpha": config.get("step_loss_alpha", 0.0),
                "epochs": config.get("epochs", 0),
                "steps_per_epoch_effective": config.get("steps_per_epoch_effective", 0),
            }
        )
    return rows


def summarize(
    vs_fixed_rows: list[dict[str, object]],
    vs_r3_rows: list[dict[str, object]],
    prefix_rows: list[dict[str, object]],
    error_rows: list[dict[str, object]],
    region_rows: list[dict[str, object]],
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
    all_error_rows = [row for row in error_rows if row["scope"] == "all"]
    focus_rows = [row for row in region_rows if row["is_focus_region"]]
    max_prefix_mismatch = max(float(row["prefix_mismatch_mse"]) for row in prefix_rows)
    max_decomposition = max(float(row["prediction_decomposition_max_abs"]) for row in error_rows)
    mean_residual_ratio = mean(float(row["residual_base_mae_ratio"]) for row in all_error_rows)
    mean_residual_energy = mean(float(row["residual_energy"]) for row in all_error_rows)
    mean_residual_gain = mean(float(row["residual_gain_mse_pct"]) for row in all_error_rows)
    gate = {
        "mean_mse_vs_r3_improves": mean(r3_rel) < 0.0,
        "mse_wins_vs_r3_at_least_7": sum(1 for row in vs_r3_rows if row["candidate_wins_mse_vs_r3"]) >= 7,
        "no_dataset_degrades_over_0_3pct": max(dataset_mean_vs_r3.values()) <= 0.3,
        "prefix_consistency_pass": max_prefix_mismatch <= 1e-10,
        "focus_region_wins_at_least_2": sum(1 for row in focus_rows if row["candidate_wins_mse_vs_r3"]) >= 2,
        "residual_controlled": 1e-6 < mean_residual_ratio < 0.2,
        "decomposition_pass": max_decomposition <= 1e-5,
    }
    gate["pass"] = all(gate.values())
    return {
        "mse_wins_vs_fixed": sum(1 for row in vs_fixed_rows if row["candidate_wins_mse_vs_fixed"]),
        "mae_wins_vs_fixed": sum(1 for row in vs_fixed_rows if row["candidate_wins_mae_vs_fixed"]),
        "mean_relative_mse_vs_fixed_pct": mean(fixed_rel),
        "dataset_mean_relative_mse_vs_fixed_pct": dataset_mean_vs_fixed,
        "mse_wins_vs_r3": sum(1 for row in vs_r3_rows if row["candidate_wins_mse_vs_r3"]),
        "mae_wins_vs_r3": sum(1 for row in vs_r3_rows if row["candidate_wins_mae_vs_r3"]),
        "mean_relative_mse_vs_r3_pct": mean(r3_rel),
        "dataset_mean_relative_mse_vs_r3_pct": dataset_mean_vs_r3,
        "horizon_mean_relative_mse_vs_r3_pct": horizon_mean_vs_r3,
        "max_prefix_mismatch_mse": max_prefix_mismatch,
        "focus_region_wins_vs_r3": sum(1 for row in focus_rows if row["candidate_wins_mse_vs_r3"]),
        "focus_region_total": len(focus_rows),
        "mean_residual_base_mae_ratio": mean_residual_ratio,
        "mean_residual_energy": mean_residual_energy,
        "mean_residual_gain_mse_pct": mean_residual_gain,
        "max_prediction_decomposition_abs": max_decomposition,
        "gate": gate,
    }


def write_report(
    path: Path,
    summary: dict[str, object],
    vs_r3_rows: list[dict[str, object]],
    region_rows: list[dict[str, object]],
    error_rows: list[dict[str, object]],
    model_name: str,
) -> None:
    dataset_mean = summary["dataset_mean_relative_mse_vs_r3_pct"]
    gate = summary["gate"]
    decision = "passes" if gate["pass"] else "fails"
    lines = [
        "# Phase2-B Error-Process Decoder Gate Report",
        "",
        "## Decision",
        "",
        f"[Decision] `{model_name}` {decision} the Phase2-B error-process gate.",
        "",
    ]
    if gate["pass"]:
        lines.append(
            "[Inference] The target-conditioned error-process decoder improves the one-model target-set carrier while keeping prefix consistency and a controlled residual path. It is a plausible paper-core decoder mechanism, pending ablations."
        )
    else:
        lines.append(
            "[Inference] The error-process mechanism is not yet a paper-core candidate. If residual diagnostics are active but MSE does not improve, the rollback should target objective design or base architecture rather than simply increasing residual capacity."
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
    ]
    gate_lines = [
        ("mean MSE vs R.3 improves", "mean_mse_vs_r3_improves"),
        ("MSE wins vs R.3 >= 7/12", "mse_wins_vs_r3_at_least_7"),
        ("no dataset degrades over +0.3%", "no_dataset_degrades_over_0_3pct"),
        ("prefix mismatch numerical zero", "prefix_consistency_pass"),
        ("focus H720 regions win >= 2/4", "focus_region_wins_at_least_2"),
        ("residual controlled", "residual_controlled"),
        ("base + residual decomposition", "decomposition_pass"),
    ]
    for label, key in gate_lines:
        lines.append(f"- {label}: `{gate[key]}`")
    lines += [
        "",
        "## Error-Process Diagnostics",
        "",
        f"- mean residual/base MAE ratio: `{float(summary['mean_residual_base_mae_ratio']):.6g}`.",
        f"- mean residual energy: `{float(summary['mean_residual_energy']):.6g}`.",
        f"- mean residual gain MSE: `{format_pct(float(summary['mean_residual_gain_mse_pct']))}`.",
        f"- max decomposition abs: `{float(summary['max_prediction_decomposition_abs']):.6g}`.",
        "",
        "## Per-Horizon Metrics vs R.3",
        "",
        "| Dataset | Horizon | Relative MSE vs R.3 | Candidate MSE | R.3 MSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in vs_r3_rows:
        lines.append(
            "| {dataset} | {horizon} | {rel} | {candidate:.6f} | {r3:.6f} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                rel=format_pct(float(row["relative_mse_vs_r3_pct"])),
                candidate=float(row["candidate_mse"]),
                r3=float(row["r3_mse"]),
            )
        )
    lines += [
        "",
        "## H720 Focus Regions",
        "",
        "| Dataset | Segment | Focus | Relative MSE vs R.3 | Candidate MSE | R.3 MSE |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in region_rows:
        lines.append(
            "| {dataset} | {segment} | {focus} | {rel} | {candidate:.6f} | {r3:.6f} |".format(
                dataset=row["dataset"],
                segment=row["segment"],
                focus=row["is_focus_region"],
                rel=format_pct(float(row["relative_mse_vs_r3_pct"])),
                candidate=float(row["candidate_mse"]),
                r3=float(row["r3_mse"]),
            )
        )
    lines += [
        "",
        "## Residual Rows",
        "",
        "| Dataset | Horizon | Scope | Residual/Base MAE | Residual Gain MSE | Decomp Max Abs |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in error_rows:
        if row["scope"] != "all":
            continue
        lines.append(
            "| {dataset} | {horizon} | {scope} | {ratio:.6g} | {gain} | {decomp:.3g} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                scope=row["scope"],
                ratio=float(row["residual_base_mae_ratio"]),
                gain=format_pct(float(row["residual_gain_mse_pct"])),
                decomp=float(row["prediction_decomposition_max_abs"]),
            )
        )
    lines += [
        "",
        "## Rollback Rule",
        "",
        "If this report fails while residual activity is controlled, return to loop step 2-3 and consider objective-level modeling such as step covariance weighting. Do not add MoE to this residual state unless this gate passes or a specific failure mode is repaired.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase2-B error-process decoder gate.")
    parser.add_argument("--analysis-root", default="analysis/phase2_error_process_decoder_gate_20260623")
    parser.add_argument("--model-name", default="PatchEncoderErrorProcessDecoder")
    parser.add_argument("--output-prefix", default="phase2_error_process_decoder")
    parser.add_argument(
        "--r3-reference",
        default="analysis/phase1_prefix_risk_weighted_gate_20260622/phase1_prefix_risk_weighted_vs_fixed.csv",
    )
    parser.add_argument(
        "--r3-raw-root",
        default="analysis/phase1_prefix_risk_weighted_gate_20260622/raw",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root = Path(args.analysis_root)
    raw_root = analysis_root / "raw"
    reference = r3_reference(Path(args.r3_reference))
    vs_fixed_rows, vs_r3_rows = collect_main_rows(raw_root, args.model_name, reference)
    prefix_rows = collect_prefix_rows(raw_root, args.model_name)
    error_rows = collect_error_process_rows(raw_root, args.model_name)
    region_rows = collect_h720_region_rows(raw_root, args.model_name, Path(args.r3_raw_root))
    config_rows = collect_config_rows(raw_root, args.model_name)
    summary = summarize(vs_fixed_rows, vs_r3_rows, prefix_rows, error_rows, region_rows)

    write_csv(analysis_root / f"{args.output_prefix}_vs_fixed.csv", vs_fixed_rows)
    write_csv(analysis_root / f"{args.output_prefix}_vs_r3.csv", vs_r3_rows)
    write_csv(analysis_root / f"{args.output_prefix}_prefix_consistency.csv", prefix_rows)
    write_csv(analysis_root / f"{args.output_prefix}_error_process_stats.csv", error_rows)
    write_csv(analysis_root / f"{args.output_prefix}_h720_regions_vs_r3.csv", region_rows)
    write_csv(analysis_root / f"{args.output_prefix}_effective_configs.csv", config_rows)
    (analysis_root / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(
        analysis_root / f"{args.output_prefix}_decision_report.md",
        summary,
        vs_r3_rows,
        region_rows,
        error_rows,
        args.model_name,
    )
    print(f"decision_report={analysis_root / f'{args.output_prefix}_decision_report.md'}")


if __name__ == "__main__":
    main()
