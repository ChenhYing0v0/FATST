#!/usr/bin/env python3
"""Aggregate SC1-D4 structured-basis and dense-horizon diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
CHECKPOINT_SEEDS = (2021, 2022, 2023)
STRUCTURE_SEEDS = (3101, 3102, 3103)
CONTROLS = (
    "identity",
    "dct2",
    "pca_fit",
    "permuted_interval",
    "random_interval_tree",
    "random_orthogonal",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--d4-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if args.synthetic_smoke:
        return args
    for name in ("input_root", "d4_config", "output_dir"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean requires values")
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def reduction(log_effect: float) -> float:
    return 1.0 - math.exp(-log_effect)


def build_checkpoint_comparisons(
    metrics: list[dict[str, str]], horizons: list[int]
) -> list[dict[str, Any]]:
    index: dict[tuple[str, int, str, int], dict[str, str]] = {}
    for row in metrics:
        key = (
            row["dataset"],
            int(row["checkpoint_seed"]),
            row["family"],
            int(row["structure_seed"]),
        )
        if key in index:
            raise ValueError(f"duplicate D4 row: {key}")
        index[key] = row
    outputs: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for checkpoint_seed in CHECKPOINT_SEEDS:
            for control in CONTROLS:
                for horizon in horizons:
                    mse_effects = []
                    mae_effects = []
                    for structure_seed in STRUCTURE_SEEDS:
                        candidate = index[
                            (dataset, checkpoint_seed, "balanced_interval", structure_seed)
                        ]
                        baseline = index[(dataset, checkpoint_seed, control, structure_seed)]
                        candidate_mse = float(candidate[f"val_mse_eval_h{horizon}"])
                        control_mse = float(baseline[f"val_mse_eval_h{horizon}"])
                        candidate_mae = float(candidate[f"val_mae_eval_h{horizon}"])
                        control_mae = float(baseline[f"val_mae_eval_h{horizon}"])
                        if min(candidate_mse, control_mse, candidate_mae, control_mae) <= 0.0:
                            raise ValueError("non-positive metric in D4")
                        mse_effects.append(math.log(control_mse / candidate_mse))
                        mae_effects.append(math.log(control_mae / candidate_mae))
                    mse_log = mean(mse_effects)
                    mae_log = mean(mae_effects)
                    outputs.append(
                        {
                            "dataset": dataset,
                            "checkpoint_seed": checkpoint_seed,
                            "control_family": control,
                            "horizon": horizon,
                            "structure_seed_count": len(STRUCTURE_SEEDS),
                            "mse_log_effect": mse_log,
                            "mse_reduction": reduction(mse_log),
                            "mae_log_effect": mae_log,
                            "mae_reduction": reduction(mae_log),
                        }
                    )
    return outputs


def build_dataset_horizon_summary(
    checkpoints: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in checkpoints:
        grouped[(row["dataset"], row["control_family"], int(row["horizon"]))].append(row)
    outputs = []
    for (dataset, control, horizon), rows in sorted(grouped.items()):
        mse = [float(row["mse_log_effect"]) for row in rows]
        mae = [float(row["mae_log_effect"]) for row in rows]
        outputs.append(
            {
                "dataset": dataset,
                "control_family": control,
                "horizon": horizon,
                "checkpoint_count": len(rows),
                "mean_mse_log_effect": mean(mse),
                "mse_reduction": reduction(mean(mse)),
                "std_mse_log_effect": std(mse),
                "positive_mse_checkpoints": sum(value > 0.0 for value in mse),
                "mean_mae_log_effect": mean(mae),
                "mae_reduction": reduction(mean(mae)),
            }
        )
    return outputs


def build_family_summary(
    checkpoints: list[dict[str, Any]], horizons: list[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped_checkpoint: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in checkpoints:
        grouped_checkpoint[
            (row["dataset"], int(row["checkpoint_seed"]), row["control_family"])
        ].append(row)
    checkpoint_overall = []
    for (dataset, checkpoint_seed, control), rows in sorted(grouped_checkpoint.items()):
        if {int(row["horizon"]) for row in rows} != set(horizons):
            raise ValueError("incomplete horizons in checkpoint summary")
        mse_log = mean([float(row["mse_log_effect"]) for row in rows])
        mae_log = mean([float(row["mae_log_effect"]) for row in rows])
        h720 = next(row for row in rows if int(row["horizon"]) == 720)
        checkpoint_overall.append(
            {
                "dataset": dataset,
                "checkpoint_seed": checkpoint_seed,
                "control_family": control,
                "mean_horizon_mse_log_effect": mse_log,
                "mean_horizon_mse_reduction": reduction(mse_log),
                "mean_horizon_mae_log_effect": mae_log,
                "mean_horizon_mae_reduction": reduction(mae_log),
                "h720_mse_log_effect": h720["mse_log_effect"],
                "h720_mse_reduction": h720["mse_reduction"],
            }
        )
    grouped_dataset: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in checkpoint_overall:
        grouped_dataset[(row["dataset"], row["control_family"])].append(row)
    dataset_family = []
    for (dataset, control), rows in sorted(grouped_dataset.items()):
        mse = [float(row["mean_horizon_mse_log_effect"]) for row in rows]
        mae = [float(row["mean_horizon_mae_log_effect"]) for row in rows]
        h720 = [float(row["h720_mse_log_effect"]) for row in rows]
        dataset_family.append(
            {
                "dataset": dataset,
                "control_family": control,
                "mean_horizon_mse_reduction": reduction(mean(mse)),
                "positive_mse_checkpoints": sum(value > 0.0 for value in mse),
                "mean_horizon_mae_reduction": reduction(mean(mae)),
                "h720_mse_reduction": reduction(mean(h720)),
                "positive_h720_checkpoints": sum(value > 0.0 for value in h720),
            }
        )
    return checkpoint_overall, dataset_family


def macro_horizon_summary(
    checkpoints: list[dict[str, Any]], horizons: list[int]
) -> list[dict[str, Any]]:
    outputs = []
    for control in CONTROLS:
        for horizon in horizons:
            rows = [
                row
                for row in checkpoints
                if row["control_family"] == control and int(row["horizon"]) == horizon
            ]
            mse_log = mean([float(row["mse_log_effect"]) for row in rows])
            mae_log = mean([float(row["mae_log_effect"]) for row in rows])
            outputs.append(
                {
                    "control_family": control,
                    "horizon": horizon,
                    "primary_unit_count": len(rows),
                    "mse_reduction": reduction(mse_log),
                    "mae_reduction": reduction(mae_log),
                }
            )
    return outputs


def summarize_geometry(rows: list[dict[str, str]], horizons: list[int]) -> list[dict[str, Any]]:
    fields = [
        "covariance_offdiag_ratio",
        "mean_atom_support_fraction",
        "variance_capture_top16",
        "variance_capture_top64",
        "variance_capture_top144",
        "variance_capture_top256",
        *[f"active_atoms_h{horizon}" for horizon in horizons],
    ]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    outputs = []
    for family in sorted(grouped):
        output: dict[str, Any] = {"family": family, "row_count": len(grouped[family])}
        for field in fields:
            output[f"mean_{field}"] = mean([float(row[field]) for row in grouped[family]])
        outputs.append(output)
    return outputs


def rank_values(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def pearson(values_x: list[float], values_y: list[float]) -> float:
    center_x = mean(values_x)
    center_y = mean(values_y)
    numerator = sum(
        (value_x - center_x) * (value_y - center_y)
        for value_x, value_y in zip(values_x, values_y, strict=True)
    )
    denominator_x = math.sqrt(sum((value - center_x) ** 2 for value in values_x))
    denominator_y = math.sqrt(sum((value - center_y) ** 2 for value in values_y))
    if denominator_x == 0.0 or denominator_y == 0.0:
        return float("nan")
    return numerator / (denominator_x * denominator_y)


def mechanism_correlations(
    metrics: list[dict[str, str]],
    geometry: list[dict[str, str]],
    horizons: list[int],
) -> list[dict[str, Any]]:
    metric_groups: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    geometry_groups: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in metrics:
        metric_groups[(row["dataset"], int(row["checkpoint_seed"]), row["family"])].append(row)
    for row in geometry:
        geometry_groups[(row["dataset"], int(row["checkpoint_seed"]), row["family"])].append(row)
    geometry_fields = (
        "covariance_offdiag_ratio",
        "variance_capture_top16",
        "variance_capture_top64",
        "active_atoms_h48",
    )
    outputs = []
    families = ("balanced_interval", *CONTROLS)
    for dataset in DATASETS:
        for checkpoint_seed in CHECKPOINT_SEEDS:
            errors = []
            geometry_values = {field: [] for field in geometry_fields}
            for family in families:
                metric_rows = metric_groups[(dataset, checkpoint_seed, family)]
                geometry_rows = geometry_groups[(dataset, checkpoint_seed, family)]
                log_errors = [
                    math.log(float(row[f"val_mse_eval_h{horizon}"]))
                    for row in metric_rows
                    for horizon in horizons
                ]
                errors.append(mean(log_errors))
                for field in geometry_fields:
                    geometry_values[field].append(
                        mean([float(row[field]) for row in geometry_rows])
                    )
            error_ranks = rank_values(errors)
            output: dict[str, Any] = {
                "dataset": dataset,
                "checkpoint_seed": checkpoint_seed,
                "family_count": len(families),
            }
            for field in geometry_fields:
                output[f"spearman_log_mse_vs_{field}"] = pearson(
                    error_ranks, rank_values(geometry_values[field])
                )
            outputs.append(output)
    return outputs


def invariant_gate(metadata: list[dict[str, Any]], metrics: list[dict[str, str]]) -> dict[str, Any]:
    hashes = {item["contract_hash"] for item in metadata}
    config_hashes = {item["d4_config_hash"] for item in metadata}
    passed = (
        len(metadata) == 15
        and len(metrics) == 315
        and len(hashes) == 1
        and len(config_hashes) == 1
        and all(
            not item["uses_test_split"]
            and not item["forecast_model_updated"]
            and not item["official_validation_used_for_early_stopping"]
            and item["pca_uses_fit_targets_only"]
            and float(item["basis_orthogonality_max_abs"]) <= 2e-5
            for item in metadata
        )
    )
    return {"pass": passed, "metadata_count": len(metadata), "fit_count": len(metrics)}


def build_summary(
    checkpoints: list[dict[str, Any]],
    dataset_family: list[dict[str, Any]],
    macro_horizons: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    metrics: list[dict[str, str]],
    config: dict[str, Any],
) -> dict[str, Any]:
    gate = config["gate"]

    def family_values(family: str, field: str) -> list[float]:
        return [float(row[field]) for row in checkpoints if row["control_family"] == family]

    def positive_datasets(family: str, field: str) -> int:
        count_field = "positive_h720_checkpoints" if field == "h720_mse_log_effect" else "positive_mse_checkpoints"
        return sum(
            int(row[count_field]) >= 2
            for row in dataset_family
            if row["control_family"] == family
        )

    def macro_reduction(family: str, field: str) -> float:
        return reduction(mean(family_values(family, field)))

    invariant = invariant_gate(metadata, metrics)
    random_h720 = macro_reduction("random_orthogonal", "h720_mse_log_effect")
    random_pass = (
        random_h720 >= float(gate["minimum_random_replication_reduction_h720"])
        and positive_datasets("random_orthogonal", "h720_mse_log_effect")
        >= int(gate["minimum_datasets_with_two_of_three_positive_checkpoints"])
    )
    global_details = {}
    global_pass = True
    for family in ("identity", "dct2", "pca_fit"):
        macro = macro_reduction(family, "mean_horizon_mse_log_effect")
        noninferior_datasets = sum(
            float(row["mean_horizon_mse_reduction"])
            >= float(gate["dataset_noninferiority_floor"])
            for row in dataset_family
            if row["control_family"] == family
        )
        noninferior_horizons = sum(
            float(row["mse_reduction"]) >= float(gate["horizon_noninferiority_margin"])
            for row in macro_horizons
            if row["control_family"] == family
        )
        passed = (
            macro >= float(gate["global_basis_noninferiority_margin"])
            and noninferior_datasets >= int(gate["minimum_noninferior_datasets"])
            and noninferior_horizons >= int(gate["minimum_noninferior_horizons"])
        )
        global_details[family] = {
            "macro_reduction": macro,
            "noninferior_datasets": noninferior_datasets,
            "noninferior_horizons": noninferior_horizons,
            "pass": passed,
        }
        global_pass = global_pass and passed

    def positive_gate(family: str, minimum: float) -> dict[str, Any]:
        mse_macro = macro_reduction(family, "mean_horizon_mse_log_effect")
        mae_macro = macro_reduction(family, "mean_horizon_mae_log_effect")
        datasets = positive_datasets(family, "mean_horizon_mse_log_effect")
        positive_horizons = sum(
            float(row["mse_reduction"]) > 0.0
            for row in macro_horizons
            if row["control_family"] == family
        )
        passed = (
            mse_macro >= minimum
            and mae_macro >= float(gate["mae_guard"])
            and datasets >= int(gate["minimum_datasets_with_two_of_three_positive_checkpoints"])
            and positive_horizons >= int(gate["minimum_noninferior_horizons"])
        )
        return {
            "mse_macro_reduction": mse_macro,
            "mae_macro_reduction": mae_macro,
            "positive_datasets": datasets,
            "positive_horizons": positive_horizons,
            "pass": passed,
        }

    locality = positive_gate("permuted_interval", float(gate["minimum_locality_reduction"]))
    specificity = positive_gate(
        "random_interval_tree", float(gate["minimum_balance_specificity_reduction"])
    )
    if not invariant["pass"]:
        decision = "diagnostic_invalid_for_direction_rejection"
    elif not random_pass:
        decision = "d3_signal_not_replicated_reaudit_step2"
    elif not global_pass:
        decision = "standard_structured_basis_explains_gain_return_step2"
    elif not locality["pass"]:
        decision = "locality_not_supported_coordinate_effect_only_step2"
    elif not specificity["pass"]:
        decision = "interval_local_family_supported_refine_step4"
    else:
        decision = "balanced_interval_specificity_supported_step5"
    return {
        "candidate": "SC1-D4",
        "role": "diagnostic_only",
        "invariant_gate": invariant,
        "random_replication_gate": {
            "h720_reduction": random_h720,
            "positive_datasets": positive_datasets("random_orthogonal", "h720_mse_log_effect"),
            "pass": random_pass,
        },
        "global_noninferiority_gate": {"details": global_details, "pass": global_pass},
        "locality_gate": locality,
        "balance_specificity_gate": specificity,
        "decision": decision,
        "method_training_authorized": False,
        "authorization_if_passed": "step5_theory_feasibility_only",
    }


def render_report(summary: dict[str, Any], dataset_family: list[dict[str, Any]]) -> str:
    lines = [
        "# SC1-D4 Structured-Basis Diagnostic Report",
        "",
        f"- `decision`: `{summary['decision']}`",
        "- `method_training_authorized`: `false`",
        "",
        "## Eight-Horizon Balanced-vs-Control Effects",
        "",
        "Positive means balanced interval has lower error.",
        "",
        "| Dataset | Control | MSE reduction | MAE reduction | H720 reduction | Positive checkpoints |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in dataset_family:
        lines.append(
            f"| {row['dataset']} | {row['control_family']} | "
            f"{float(row['mean_horizon_mse_reduction']):.4%} | "
            f"{float(row['mean_horizon_mae_reduction']):.4%} | "
            f"{float(row['h720_mse_reduction']):.4%} | {row['positive_mse_checkpoints']}/3 |"
        )
    lines.extend(
        [
            "",
            "## Gates",
            "",
            f"- random replication: `{summary['random_replication_gate']}`",
            f"- global noninferiority: `{summary['global_noninferiority_gate']}`",
            f"- locality: `{summary['locality_gate']}`",
            f"- exact balance specificity: `{summary['balance_specificity_gate']}`",
            f"- invariants: `{summary['invariant_gate']}`",
            "",
        ]
    )
    return "\n".join(lines)


def synthetic_smoke() -> None:
    config = {
        "horizons": [48, 96, 144, 192, 288, 336, 512, 720],
    }
    metrics = []
    for dataset in DATASETS:
        for checkpoint_seed in CHECKPOINT_SEEDS:
            for family in ("balanced_interval", *CONTROLS):
                for structure_seed in STRUCTURE_SEEDS:
                    row = {
                        "dataset": dataset,
                        "checkpoint_seed": str(checkpoint_seed),
                        "family": family,
                        "structure_seed": str(structure_seed),
                    }
                    for horizon in config["horizons"]:
                        value = 0.97 if family == "balanced_interval" else 1.0
                        row[f"val_mse_eval_h{horizon}"] = str(value)
                        row[f"val_mae_eval_h{horizon}"] = str(value)
                    metrics.append(row)
    checkpoints = build_checkpoint_comparisons(metrics, config["horizons"])
    datasets = build_dataset_horizon_summary(checkpoints)
    overall, family = build_family_summary(checkpoints, config["horizons"])
    if len(metrics) != 315 or len(checkpoints) != 720 or len(datasets) != 240:
        raise RuntimeError("D4 synthetic aggregation counts changed")
    if len(overall) != 90 or len(family) != 30:
        raise RuntimeError("D4 family aggregation counts changed")
    print("stage_c_sc1_d4_analyzer_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    config = json.loads(args.d4_config.read_text(encoding="utf-8"))
    horizons = [int(value) for value in config["horizons"]]
    metrics: list[dict[str, str]] = []
    geometry: list[dict[str, str]] = []
    metadata: list[dict[str, Any]] = []
    for dataset in DATASETS:
        metrics.extend(read_csv(args.input_root / dataset / "d4_probe_metrics.csv"))
        geometry.extend(read_csv(args.input_root / dataset / "d4_basis_geometry.csv"))
        metadata.extend(
            json.loads((args.input_root / dataset / "d4_metadata.json").read_text(encoding="utf-8"))
        )
    checkpoints = build_checkpoint_comparisons(metrics, horizons)
    dataset_horizons = build_dataset_horizon_summary(checkpoints)
    checkpoint_overall, dataset_family = build_family_summary(checkpoints, horizons)
    macro_horizons = macro_horizon_summary(checkpoints, horizons)
    geometry_summary = summarize_geometry(geometry, horizons)
    correlations = mechanism_correlations(metrics, geometry, horizons)
    summary = build_summary(
        checkpoint_overall,
        dataset_family,
        macro_horizons,
        metadata,
        metrics,
        config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "d4_checkpoint_horizon_comparisons.csv", checkpoints)
    write_csv(args.output_dir / "d4_dataset_horizon_summary.csv", dataset_horizons)
    write_csv(args.output_dir / "d4_checkpoint_family_summary.csv", checkpoint_overall)
    write_csv(args.output_dir / "d4_dataset_family_summary.csv", dataset_family)
    write_csv(args.output_dir / "d4_macro_horizon_summary.csv", macro_horizons)
    write_csv(args.output_dir / "d4_geometry_summary.csv", geometry_summary)
    write_csv(args.output_dir / "d4_mechanism_correlations.csv", correlations)
    (args.output_dir / "d4_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "d4_diagnostic_report.md").write_text(
        render_report(summary, dataset_family), encoding="utf-8"
    )
    print(
        f"stage_c_sc1_d4_analysis_done decision={summary['decision']} "
        f"output_dir={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
