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
SPECIALIST_GAPS = [
    ("ETTm1", 96),
    ("ETTm1", 720),
    ("ETTh2", 720),
    ("Weather", 96),
]
H720_STABILITY_REGIONS = [
    ("ETTh2", "193-336"),
    ("ETTh2", "337-720"),
    ("ETTm1", "337-720"),
    ("Weather", "337-720"),
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


def format_pct(value: float) -> str:
    return f"{value:+.2f}%"


def run_dir(raw_root: Path, model_name: str, dataset: str) -> Path:
    return raw_root / model_name / dataset / TARGET_RUN / f"seed{SEED}"


def reference_rows(path: Path, mse_name: str, mae_name: str) -> dict[tuple[str, int], dict[str, float]]:
    rows = {}
    for row in read_csv(path):
        key = (row["dataset"], int(row["horizon"]))
        rows[key] = {
            mse_name: float(row["target_mse"]),
            mae_name: float(row["target_mae"]),
            "fixed_mse": float(row["fixed_mse"]),
            "fixed_mae": float(row["fixed_mae"]),
        }
    return rows


def collect_main_rows(
    raw_root: Path,
    model_name: str,
    r3_ref: dict[tuple[str, int], dict[str, float]],
    uniform_ref: dict[tuple[str, int], dict[str, float]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    vs_r3: list[dict[str, object]] = []
    vs_uniform: list[dict[str, object]] = []
    vs_fixed: list[dict[str, object]] = []
    for dataset in DATASETS:
        metrics = read_csv(run_dir(raw_root, model_name, dataset) / "metrics_by_target_horizon.csv")
        by_horizon = {int(row["target_horizon"]): row for row in metrics}
        for horizon in HORIZONS:
            row = by_horizon[horizon]
            candidate_mse = float(row["mse"])
            candidate_mae = float(row["mae"])
            r3 = r3_ref[(dataset, horizon)]
            uniform = uniform_ref[(dataset, horizon)]
            fixed_mse = r3["fixed_mse"]
            fixed_mae = r3["fixed_mae"]
            vs_r3.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "candidate_mse": candidate_mse,
                    "candidate_mae": candidate_mae,
                    "r3_mse": r3["r3_mse"],
                    "r3_mae": r3["r3_mae"],
                    "relative_mse_vs_r3_pct": (candidate_mse / r3["r3_mse"] - 1.0) * 100.0,
                    "relative_mae_vs_r3_pct": (candidate_mae / r3["r3_mae"] - 1.0) * 100.0,
                    "candidate_wins_mse_vs_r3": candidate_mse < r3["r3_mse"],
                    "candidate_wins_mae_vs_r3": candidate_mae < r3["r3_mae"],
                    "is_specialist_gap": (dataset, horizon) in SPECIALIST_GAPS,
                }
            )
            vs_uniform.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "candidate_mse": candidate_mse,
                    "candidate_mae": candidate_mae,
                    "uniform_mse": uniform["uniform_mse"],
                    "uniform_mae": uniform["uniform_mae"],
                    "relative_mse_vs_uniform_pct": (candidate_mse / uniform["uniform_mse"] - 1.0)
                    * 100.0,
                    "relative_mae_vs_uniform_pct": (candidate_mae / uniform["uniform_mae"] - 1.0)
                    * 100.0,
                    "candidate_wins_mse_vs_uniform": candidate_mse < uniform["uniform_mse"],
                    "candidate_wins_mae_vs_uniform": candidate_mae < uniform["uniform_mae"],
                }
            )
            vs_fixed.append(
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
    return vs_r3, vs_uniform, vs_fixed


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


def collect_h720_region_rows(raw_root: Path, model_name: str, r3_raw_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        candidate_rows = {
            row["segment"]: row
            for row in read_csv(run_dir(raw_root, model_name, dataset) / "h720" / "metrics_by_segment.csv")
        }
        r3_rows = {
            row["segment"]: row
            for row in read_csv(
                run_dir(r3_raw_root, "PatchEncoderPrefixRiskWeighted", dataset)
                / "h720"
                / "metrics_by_segment.csv"
            )
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
                    "is_stability_region": (dataset, segment) in H720_STABILITY_REGIONS,
                }
            )
    return rows


def collect_objective_rows(raw_root: Path, model_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for row in read_csv(run_dir(raw_root, model_name, dataset) / "objective_weight_stats.csv"):
            rows.append(
                {
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


def collect_config_rows(raw_root: Path, model_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        config = read_json(run_dir(raw_root, model_name, dataset) / "effective_config.json")
        rows.append(
            {
                "dataset": dataset,
                "model_variant": config.get("model_variant", ""),
                "step_loss_weighting": config.get("step_loss_weighting", ""),
                "step_loss_alpha": config.get("step_loss_alpha", 0.0),
                "target_horizons": ",".join(str(item) for item in config.get("target_horizons", [])),
                "epochs": config.get("epochs", 0),
                "steps_per_epoch_effective": config.get("steps_per_epoch_effective", 0),
            }
        )
    return rows


def summarize(
    vs_r3: list[dict[str, object]],
    vs_uniform: list[dict[str, object]],
    vs_fixed: list[dict[str, object]],
    prefix_rows: list[dict[str, object]],
    region_rows: list[dict[str, object]],
) -> dict[str, object]:
    r3_rel = [float(row["relative_mse_vs_r3_pct"]) for row in vs_r3]
    uniform_rel = [float(row["relative_mse_vs_uniform_pct"]) for row in vs_uniform]
    fixed_rel = [float(row["relative_mse_vs_fixed_pct"]) for row in vs_fixed]
    dataset_mean_vs_r3 = {
        dataset: mean(float(row["relative_mse_vs_r3_pct"]) for row in vs_r3 if row["dataset"] == dataset)
        for dataset in DATASETS
    }
    horizon_mean_vs_r3 = {
        str(horizon): mean(float(row["relative_mse_vs_r3_pct"]) for row in vs_r3 if row["horizon"] == horizon)
        for horizon in HORIZONS
    }
    specialist_rows = [row for row in vs_r3 if row["is_specialist_gap"]]
    stability_rows = [row for row in region_rows if row["is_stability_region"]]
    max_prefix_mismatch = max(float(row["prefix_mismatch_mse"]) for row in prefix_rows)
    gate = {
        "mean_mse_vs_r3_improves": mean(r3_rel) < 0.0,
        "mse_wins_vs_r3_at_least_7": sum(1 for row in vs_r3 if row["candidate_wins_mse_vs_r3"]) >= 7,
        "no_dataset_degrades_over_0_3pct": max(dataset_mean_vs_r3.values()) <= 0.3,
        "specialist_gap_wins_at_least_2": sum(1 for row in specialist_rows if row["candidate_wins_mse_vs_r3"]) >= 2,
        "h720_stability_regions_not_worse": all(
            float(row["relative_mse_vs_r3_pct"]) <= 0.3 for row in stability_rows
        ),
        "prefix_consistency_pass": max_prefix_mismatch <= 1e-10,
    }
    gate["pass"] = all(gate.values())
    return {
        "mse_wins_vs_r3": sum(1 for row in vs_r3 if row["candidate_wins_mse_vs_r3"]),
        "mae_wins_vs_r3": sum(1 for row in vs_r3 if row["candidate_wins_mae_vs_r3"]),
        "mean_relative_mse_vs_r3_pct": mean(r3_rel),
        "dataset_mean_relative_mse_vs_r3_pct": dataset_mean_vs_r3,
        "horizon_mean_relative_mse_vs_r3_pct": horizon_mean_vs_r3,
        "mse_wins_vs_uniform": sum(1 for row in vs_uniform if row["candidate_wins_mse_vs_uniform"]),
        "mean_relative_mse_vs_uniform_pct": mean(uniform_rel),
        "mse_wins_vs_fixed": sum(1 for row in vs_fixed if row["candidate_wins_mse_vs_fixed"]),
        "mean_relative_mse_vs_fixed_pct": mean(fixed_rel),
        "specialist_gap_wins_vs_r3": sum(1 for row in specialist_rows if row["candidate_wins_mse_vs_r3"]),
        "h720_stability_region_wins_vs_r3": sum(1 for row in stability_rows if row["candidate_wins_mse_vs_r3"]),
        "max_prefix_mismatch_mse": max_prefix_mismatch,
        "gate": gate,
    }


def write_report(
    path: Path,
    model_name: str,
    summary: dict[str, object],
    vs_r3: list[dict[str, object]],
    region_rows: list[dict[str, object]],
    objective_rows: list[dict[str, object]],
) -> None:
    gate = summary["gate"]
    decision = "passes" if gate["pass"] else "fails"
    dataset_mean = summary["dataset_mean_relative_mse_vs_r3_pct"]
    lines = [
        "# Phase2-C Region-Balanced Objective Gate Report",
        "",
        "## Decision",
        "",
        f"[Decision] `{model_name}` {decision} the Phase2-C region-balanced objective gate.",
        "",
    ]
    if gate["pass"]:
        lines.append(
            "[Inference] Coverage-balanced objective improves over R.3 and can become the first objective-side mechanism candidate, pending covariance/novelty ablations."
        )
    else:
        lines.append(
            "[Inference] Coverage balance alone is not sufficient as a paper-core objective. If it fails while prefix consistency remains intact, the next repair must add target covariance/novelty evidence or stop the objective-only path."
        )
    lines += [
        "",
        "## Main Metrics vs R.3",
        "",
        f"- MSE wins vs R.3: `{summary['mse_wins_vs_r3']}/12`.",
        f"- MAE wins vs R.3: `{summary['mae_wins_vs_r3']}/12`.",
        f"- Mean relative MSE vs R.3: `{format_pct(float(summary['mean_relative_mse_vs_r3_pct']))}`.",
        f"- ETTh2 mean relative MSE vs R.3: `{format_pct(float(dataset_mean['ETTh2']))}`.",
        f"- ETTm1 mean relative MSE vs R.3: `{format_pct(float(dataset_mean['ETTm1']))}`.",
        f"- Weather mean relative MSE vs R.3: `{format_pct(float(dataset_mean['Weather']))}`.",
        "",
        "## Secondary Metrics",
        "",
        f"- MSE wins vs uniform target-set: `{summary['mse_wins_vs_uniform']}/12`.",
        f"- Mean relative MSE vs uniform target-set: `{format_pct(float(summary['mean_relative_mse_vs_uniform_pct']))}`.",
        f"- MSE wins vs FixedHead: `{summary['mse_wins_vs_fixed']}/12`.",
        f"- Mean relative MSE vs FixedHead: `{format_pct(float(summary['mean_relative_mse_vs_fixed_pct']))}`.",
        "",
        "## Gate",
        "",
    ]
    for key, value in gate.items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "",
        "## Per-Horizon Metrics vs R.3",
        "",
        "| Dataset | Horizon | Specialist gap | Relative MSE vs R.3 | Candidate MSE | R.3 MSE |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in vs_r3:
        lines.append(
            "| {dataset} | {horizon} | {gap} | {rel} | {candidate:.6f} | {r3:.6f} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                gap=row["is_specialist_gap"],
                rel=format_pct(float(row["relative_mse_vs_r3_pct"])),
                candidate=float(row["candidate_mse"]),
                r3=float(row["r3_mse"]),
            )
        )
    lines += [
        "",
        "## H720 Region Stability vs R.3",
        "",
        "| Dataset | Segment | Stability region | Relative MSE vs R.3 | Candidate MSE | R.3 MSE |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in region_rows:
        lines.append(
            "| {dataset} | {segment} | {region} | {rel} | {candidate:.6f} | {r3:.6f} |".format(
                dataset=row["dataset"],
                segment=row["segment"],
                region=row["is_stability_region"],
                rel=format_pct(float(row["relative_mse_vs_r3_pct"])),
                candidate=float(row["candidate_mse"]),
                r3=float(row["r3_mse"]),
            )
        )
    lines += [
        "",
        "## Objective Weights",
        "",
        "| Dataset | Scope | Mean weight | Weighted pressure share | Shift vs uniform |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in objective_rows:
        if str(row["scope"]).startswith("horizon_"):
            continue
        lines.append(
            "| {dataset} | {scope} | {weight:.6f} | {share:.4f} | {shift} |".format(
                dataset=row["dataset"],
                scope=row["scope"],
                weight=float(row["mean_step_weight"]),
                share=float(row["weighted_pressure_share"]),
                shift=format_pct(float(row["pressure_share_delta_pct"])),
            )
        )
    lines += [
        "",
        "## Rollback Rule",
        "",
        "If region-balanced fails, do not tune region multipliers by hand. Either add a source-grounded covariance/novelty prior and test it as a distinct Phase2-C repair, or stop the objective-only path and return to base architecture selection.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase2-C region-balanced objective gate.")
    parser.add_argument("--analysis-root", default="analysis/phase2_region_balanced_gate_20260623")
    parser.add_argument("--model-name", default="PatchEncoderRegionBalanced")
    parser.add_argument("--output-prefix", default="phase2_region_balanced")
    parser.add_argument(
        "--r3-reference",
        default="analysis/phase1_prefix_risk_weighted_gate_20260622/phase1_prefix_risk_weighted_vs_fixed.csv",
    )
    parser.add_argument(
        "--uniform-reference",
        default="analysis/phase1_target_set_decoder_gate_20260622/phase1_target_set_decoder_vs_fixed.csv",
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
    r3_ref = reference_rows(Path(args.r3_reference), "r3_mse", "r3_mae")
    uniform_ref = reference_rows(Path(args.uniform_reference), "uniform_mse", "uniform_mae")
    vs_r3, vs_uniform, vs_fixed = collect_main_rows(raw_root, args.model_name, r3_ref, uniform_ref)
    prefix_rows = collect_prefix_rows(raw_root, args.model_name)
    region_rows = collect_h720_region_rows(raw_root, args.model_name, Path(args.r3_raw_root))
    objective_rows = collect_objective_rows(raw_root, args.model_name)
    config_rows = collect_config_rows(raw_root, args.model_name)
    summary = summarize(vs_r3, vs_uniform, vs_fixed, prefix_rows, region_rows)
    write_csv(analysis_root / f"{args.output_prefix}_vs_r3.csv", vs_r3)
    write_csv(analysis_root / f"{args.output_prefix}_vs_uniform.csv", vs_uniform)
    write_csv(analysis_root / f"{args.output_prefix}_vs_fixed.csv", vs_fixed)
    write_csv(analysis_root / f"{args.output_prefix}_prefix_consistency.csv", prefix_rows)
    write_csv(analysis_root / f"{args.output_prefix}_h720_regions_vs_r3.csv", region_rows)
    write_csv(analysis_root / f"{args.output_prefix}_objective_weight_stats.csv", objective_rows)
    write_csv(analysis_root / f"{args.output_prefix}_effective_configs.csv", config_rows)
    (analysis_root / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(
        analysis_root / f"{args.output_prefix}_decision_report.md",
        args.model_name,
        summary,
        vs_r3,
        region_rows,
        objective_rows,
    )
    print(f"decision_report={analysis_root / f'{args.output_prefix}_decision_report.md'}")


if __name__ == "__main__":
    main()
