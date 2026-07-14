#!/usr/bin/env python3
"""Confirm the SC1-D6 horizon-by-support-scale interaction."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from analyze_stage_c_sc1_d4_structured_basis import mean, reduction, write_csv
    from analyze_stage_c_sc1_d5_conditioning_locality import (
        compare_local_families,
        load_artifacts,
        summarize_checkpoint_families,
        summarize_local_family_details,
    )
except ModuleNotFoundError:
    from scripts.analyze_stage_c_sc1_d4_structured_basis import mean, reduction, write_csv
    from scripts.analyze_stage_c_sc1_d5_conditioning_locality import (
        compare_local_families,
        load_artifacts,
        summarize_checkpoint_families,
        summarize_local_family_details,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--d6-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if args.synthetic_smoke:
        return args
    for name in ("input_root", "d6_config", "output_dir"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    return args


def window_effect(
    comparisons: list[dict[str, Any]], control: str, horizons: set[int], metric: str
) -> float:
    values = [
        float(row[f"{metric}_log_effect"])
        for row in comparisons
        if row["control_family"] == control and int(row["horizon"]) in horizons
    ]
    return reduction(mean(values))


def directional_datasets(
    comparisons: list[dict[str, Any]],
    control: str,
    horizons: set[int],
    positive: bool,
) -> int:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in comparisons:
        if row["control_family"] == control and int(row["horizon"]) in horizons:
            grouped[(row["dataset"], int(row["checkpoint_seed"]))].append(
                float(row["mse_log_effect"])
            )
    count = 0
    for dataset in {key[0] for key in grouped}:
        checkpoint_effects = [
            mean(values)
            for (row_dataset, _seed), values in grouped.items()
            if row_dataset == dataset
        ]
        count += sum((value > 0.0) == positive for value in checkpoint_effects) >= 2
    return count


def crossed_units(
    comparisons: list[dict[str, Any]], short: set[int], long: set[int]
) -> int:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        grouped[
            (row["dataset"], int(row["checkpoint_seed"]), row["control_family"])
        ].append(row)
    count = 0
    for (_dataset, _seed, control), rows in grouped.items():
        if control != "dct2":
            continue
        short_effect = mean(
            [float(row["mse_log_effect"]) for row in rows if int(row["horizon"]) in short]
        )
        long_effect = mean(
            [float(row["mse_log_effect"]) for row in rows if int(row["horizon"]) in long]
        )
        count += short_effect > 0.0 and long_effect < 0.0
    return count


def build_summary(
    comparisons: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    metrics: list[dict[str, str]],
    config: dict[str, Any],
) -> dict[str, Any]:
    short = {int(value) for value in config["short_horizons"]}
    long = {int(value) for value in config["long_horizons"]}
    gate = config["gate"]
    short_mse_dct = window_effect(comparisons, "dct2", short, "mse")
    long_mse_dct = window_effect(comparisons, "dct2", long, "mse")
    short_mae_dct = window_effect(comparisons, "dct2", short, "mae")
    long_mae_dct = window_effect(comparisons, "dct2", long, "mae")
    short_mse_balanced = window_effect(comparisons, "balanced_interval", short, "mse")
    cross_count = crossed_units(comparisons, short, long)
    short_datasets_dct = directional_datasets(comparisons, "dct2", short, True)
    long_datasets_dct = directional_datasets(comparisons, "dct2", long, False)
    short_datasets_balanced = directional_datasets(
        comparisons, "balanced_interval", short, True
    )
    invariant_pass = (
        len(metrics) == int(config["fit_count"])
        and len(metadata) == 15
        and len({item["contract_hash"] for item in metadata}) == 1
        and len({item["d4_config_hash"] for item in metadata}) == 1
        and {
            (
                row["dataset"],
                int(row["checkpoint_seed"]),
                row["family"],
                int(row["structure_seed"]),
            )
            for row in metrics
        }
        == {
            (dataset, int(checkpoint_seed), family, int(structure_seed))
            for dataset in config["datasets"]
            for checkpoint_seed in config["checkpoint_seeds"]
            for family in config["basis_families"]
            for structure_seed in config["structure_seeds"]
        }
        and all(
            not item["uses_test_split"]
            and not item["forecast_model_updated"]
            and not item["official_validation_used_for_early_stopping"]
            and int(item["validation_batch_offset"]) == 8
            and float(item["basis_orthogonality_max_abs"]) <= 2e-5
            for item in metadata
        )
        and all(
            math.isfinite(float(row[f"val_{metric}_eval_h{horizon}"]))
            and float(row[f"val_{metric}_eval_h{horizon}"]) > 0.0
            for row in metrics
            for horizon in config["horizons"]
            for metric in ("mse", "mae")
        )
    )
    interaction_pass = (
        short_mse_dct >= float(gate["minimum_short_reduction_vs_dct"])
        and long_mse_dct <= float(gate["maximum_long_reduction_vs_dct"])
        and short_mse_balanced >= float(gate["minimum_short_reduction_vs_balanced"])
        and short_mae_dct >= float(gate["short_mae_guard_vs_dct"])
        and long_mae_dct <= float(gate["maximum_long_mae_reduction_vs_dct"])
        and cross_count >= int(gate["minimum_crossed_primary_units"])
        and short_datasets_dct >= int(gate["minimum_directional_datasets"])
        and long_datasets_dct >= int(gate["minimum_directional_datasets"])
        and short_datasets_balanced >= int(gate["minimum_directional_datasets"])
    )
    if not invariant_pass:
        decision = "diagnostic_invalid_for_direction_rejection"
    elif interaction_pass:
        decision = "horizon_support_scale_interaction_supported_return_step4"
    else:
        decision = "horizon_support_scale_interaction_not_confirmed_step2"
    return {
        "candidate": "SC1-D6",
        "role": "diagnostic_only_confirmation",
        "invariant_gate": {
            "pass": invariant_pass,
            "fit_count": len(metrics),
            "metadata_count": len(metadata),
        },
        "interaction_gate": {
            "short_mse_reduction_vs_dct": short_mse_dct,
            "long_mse_reduction_vs_dct": long_mse_dct,
            "short_mae_reduction_vs_dct": short_mae_dct,
            "long_mae_reduction_vs_dct": long_mae_dct,
            "short_mse_reduction_vs_balanced": short_mse_balanced,
            "crossed_primary_units": cross_count,
            "short_positive_datasets_vs_dct": short_datasets_dct,
            "long_negative_datasets_vs_dct": long_datasets_dct,
            "short_positive_datasets_vs_balanced": short_datasets_balanced,
            "pass": interaction_pass,
        },
        "decision": decision,
        "method_training_authorized": False,
        "authorization_if_passed": config["authorization_if_passed"],
    }


def render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# SC1-D6 Horizon-Support Interaction Confirmation",
            "",
            f"- `decision`: `{summary['decision']}`",
            "- `method_training_authorized`: `false`",
            f"- invariants: `{summary['invariant_gate']}`",
            f"- interaction: `{summary['interaction_gate']}`",
            "",
        ]
    )


def synthetic_smoke() -> None:
    comparisons = []
    for dataset in ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"):
        for checkpoint_seed in (2021, 2022, 2023):
            for control in ("balanced_interval", "dct2"):
                for horizon in (48, 96, 144, 336, 512, 720):
                    effect = 0.02 if horizon <= 144 else -0.02
                    comparisons.append(
                        {
                            "dataset": dataset,
                            "checkpoint_seed": checkpoint_seed,
                            "control_family": control,
                            "horizon": horizon,
                            "mse_log_effect": effect,
                            "mae_log_effect": effect,
                        }
                    )
    if crossed_units(comparisons, {48, 96, 144}, {336, 512, 720}) != 15:
        raise RuntimeError("D6 crossed-unit synthetic check failed")
    print("stage_c_sc1_d6_analyzer_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    config = json.loads(args.d6_config.read_text(encoding="utf-8"))
    metrics, _geometry, metadata = load_artifacts(args.input_root, config["datasets"])
    families = summarize_checkpoint_families(metrics, config["horizons"])
    comparisons = compare_local_families(
        families,
        [config["candidate_family"]],
        config["horizons"],
        controls=("balanced_interval", "dct2"),
    )
    dataset_summary, horizon_summary = summarize_local_family_details(comparisons)
    summary = build_summary(comparisons, metadata, metrics, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "d6_comparisons.csv", comparisons)
    write_csv(args.output_dir / "d6_dataset_summary.csv", dataset_summary)
    write_csv(args.output_dir / "d6_horizon_summary.csv", horizon_summary)
    (args.output_dir / "d6_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "d6_diagnostic_report.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    print(
        f"stage_c_sc1_d6_analysis_done decision={summary['decision']} "
        f"output_dir={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
