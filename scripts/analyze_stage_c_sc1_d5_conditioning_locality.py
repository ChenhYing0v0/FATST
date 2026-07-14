#!/usr/bin/env python3
"""Aggregate SC1-D5 conditioning-locality frontier diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from analyze_stage_c_sc1_d4_structured_basis import mean, read_csv, reduction, write_csv
except ModuleNotFoundError:
    from scripts.analyze_stage_c_sc1_d4_structured_basis import (
        mean,
        read_csv,
        reduction,
        write_csv,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--d5-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if args.synthetic_smoke:
        return args
    for name in ("input_root", "d5_config", "output_dir"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    return args


def load_artifacts(
    root: Path, datasets: list[str]
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]]]:
    metrics: list[dict[str, str]] = []
    geometry: list[dict[str, str]] = []
    metadata: list[dict[str, Any]] = []
    for dataset in datasets:
        metrics.extend(read_csv(root / dataset / "d4_probe_metrics.csv"))
        geometry.extend(read_csv(root / dataset / "d4_basis_geometry.csv"))
        metadata.extend(
            json.loads((root / dataset / "d4_metadata.json").read_text(encoding="utf-8"))
        )
    return metrics, geometry, metadata


def summarize_checkpoint_families(
    metrics: list[dict[str, str]], horizons: list[int]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in metrics:
        grouped[(row["dataset"], int(row["checkpoint_seed"]), row["family"])].append(row)
    outputs = []
    for (dataset, checkpoint_seed, family), rows in sorted(grouped.items()):
        output: dict[str, Any] = {
            "dataset": dataset,
            "checkpoint_seed": checkpoint_seed,
            "family": family,
            "structure_seed_count": len(rows),
        }
        mse_logs = []
        mae_logs = []
        for horizon in horizons:
            mse = math.exp(
                mean([math.log(float(row[f"val_mse_eval_h{horizon}"])) for row in rows])
            )
            mae = math.exp(
                mean([math.log(float(row[f"val_mae_eval_h{horizon}"])) for row in rows])
            )
            output[f"mean_val_mse_h{horizon}"] = mse
            output[f"mean_val_mae_h{horizon}"] = mae
            mse_logs.append(math.log(mse))
            mae_logs.append(math.log(mae))
        output["mean_horizon_log_mse"] = mean(mse_logs)
        output["mean_horizon_log_mae"] = mean(mae_logs)
        outputs.append(output)
    return outputs


def summarize_checkpoint_geometry(
    geometry: list[dict[str, str]], horizons: list[int]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in geometry:
        grouped[(row["dataset"], int(row["checkpoint_seed"]), row["family"])].append(row)
    fields = [
        "covariance_offdiag_ratio",
        "variance_capture_top16",
        "variance_capture_top64",
        "mean_atom_support_fraction",
        *[f"active_atoms_h{horizon}" for horizon in horizons],
    ]
    outputs = []
    for (dataset, checkpoint_seed, family), rows in sorted(grouped.items()):
        output: dict[str, Any] = {
            "dataset": dataset,
            "checkpoint_seed": checkpoint_seed,
            "family": family,
        }
        for field in fields:
            output[field] = mean([float(row[field]) for row in rows])
        outputs.append(output)
    return outputs


def select_local_families(
    geometry: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    local = set(config["local_basis_families"])
    cap = int(config["selection"]["maximum_active_atoms_h48"])
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in geometry:
        if row["family"] in local and float(row["active_atoms_h48"]) <= cap:
            grouped[(row["dataset"], int(row["checkpoint_seed"]))].append(row)
    outputs = []
    for (dataset, checkpoint_seed), rows in sorted(grouped.items()):
        selected = min(
            rows,
            key=lambda row: (
                float(row["covariance_offdiag_ratio"]),
                -float(row["variance_capture_top16"]),
                row["family"],
            ),
        )
        outputs.append(
            {
                "dataset": dataset,
                "checkpoint_seed": checkpoint_seed,
                "selected_family": selected["family"],
                "covariance_offdiag_ratio": selected["covariance_offdiag_ratio"],
                "variance_capture_top16": selected["variance_capture_top16"],
                "variance_capture_top64": selected["variance_capture_top64"],
                "mean_atom_support_fraction": selected["mean_atom_support_fraction"],
                "active_atoms_h48": selected["active_atoms_h48"],
            }
        )
    return outputs


def compare_selected(
    families: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    horizons: list[int],
) -> list[dict[str, Any]]:
    index = {
        (row["dataset"], int(row["checkpoint_seed"]), row["family"]): row
        for row in families
    }
    outputs = []
    for selection in selections:
        dataset = selection["dataset"]
        checkpoint_seed = int(selection["checkpoint_seed"])
        selected_family = selection["selected_family"]
        selected = index[(dataset, checkpoint_seed, selected_family)]
        for control_family in ("balanced_interval", "dct2", "pca_fit"):
            control = index[(dataset, checkpoint_seed, control_family)]
            for horizon in horizons:
                mse_log_effect = math.log(
                    float(control[f"mean_val_mse_h{horizon}"])
                    / float(selected[f"mean_val_mse_h{horizon}"])
                )
                mae_log_effect = math.log(
                    float(control[f"mean_val_mae_h{horizon}"])
                    / float(selected[f"mean_val_mae_h{horizon}"])
                )
                outputs.append(
                    {
                        "dataset": dataset,
                        "checkpoint_seed": checkpoint_seed,
                        "selected_family": selected_family,
                        "control_family": control_family,
                        "horizon": horizon,
                        "mse_log_effect": mse_log_effect,
                        "mse_reduction": reduction(mse_log_effect),
                        "mae_log_effect": mae_log_effect,
                        "mae_reduction": reduction(mae_log_effect),
                    }
                )
    return outputs


def compare_local_families(
    families: list[dict[str, Any]],
    local_families: list[str],
    horizons: list[int],
    controls: tuple[str, ...] = ("balanced_interval", "dct2", "pca_fit"),
) -> list[dict[str, Any]]:
    index = {
        (row["dataset"], int(row["checkpoint_seed"]), row["family"]): row
        for row in families
    }
    outputs = []
    primary_units = sorted({(row["dataset"], int(row["checkpoint_seed"])) for row in families})
    for dataset, checkpoint_seed in primary_units:
        for candidate_family in local_families:
            candidate = index[(dataset, checkpoint_seed, candidate_family)]
            for control_family in controls:
                control = index[(dataset, checkpoint_seed, control_family)]
                for horizon in horizons:
                    mse_log_effect = math.log(
                        float(control[f"mean_val_mse_h{horizon}"])
                        / float(candidate[f"mean_val_mse_h{horizon}"])
                    )
                    mae_log_effect = math.log(
                        float(control[f"mean_val_mae_h{horizon}"])
                        / float(candidate[f"mean_val_mae_h{horizon}"])
                    )
                    outputs.append(
                        {
                            "dataset": dataset,
                            "checkpoint_seed": checkpoint_seed,
                            "candidate_family": candidate_family,
                            "control_family": control_family,
                            "horizon": horizon,
                            "mse_log_effect": mse_log_effect,
                            "mse_reduction": reduction(mse_log_effect),
                            "mae_log_effect": mae_log_effect,
                            "mae_reduction": reduction(mae_log_effect),
                        }
                    )
    return outputs


def summarize_local_families(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        grouped[(row["candidate_family"], row["control_family"])].append(row)
    outputs = []
    for (candidate, control), rows in sorted(grouped.items()):
        dataset_checkpoint: dict[tuple[str, int], list[float]] = defaultdict(list)
        for row in rows:
            dataset_checkpoint[(row["dataset"], int(row["checkpoint_seed"]))].append(
                float(row["mse_log_effect"])
            )
        positive_datasets = 0
        for dataset in {key[0] for key in dataset_checkpoint}:
            checkpoint_effects = [
                mean(values)
                for (row_dataset, _seed), values in dataset_checkpoint.items()
                if row_dataset == dataset
            ]
            positive_datasets += sum(value > 0.0 for value in checkpoint_effects) >= 2
        mse = mean([float(row["mse_log_effect"]) for row in rows])
        mae = mean([float(row["mae_log_effect"]) for row in rows])
        outputs.append(
            {
                "candidate_family": candidate,
                "control_family": control,
                "primary_horizon_units": len(rows),
                "mse_reduction": reduction(mse),
                "mae_reduction": reduction(mae),
                "positive_datasets": positive_datasets,
            }
        )
    return outputs


def summarize_local_family_details(
    comparisons: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    horizon_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        dataset_groups[
            (row["candidate_family"], row["control_family"], row["dataset"])
        ].append(row)
        horizon_groups[
            (row["candidate_family"], row["control_family"], int(row["horizon"]))
        ].append(row)
    dataset_outputs = []
    for (candidate, control, dataset), rows in sorted(dataset_groups.items()):
        checkpoint_groups: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            checkpoint_groups[int(row["checkpoint_seed"])].append(
                float(row["mse_log_effect"])
            )
        mse = mean([float(row["mse_log_effect"]) for row in rows])
        mae = mean([float(row["mae_log_effect"]) for row in rows])
        dataset_outputs.append(
            {
                "candidate_family": candidate,
                "control_family": control,
                "dataset": dataset,
                "mse_reduction": reduction(mse),
                "mae_reduction": reduction(mae),
                "positive_checkpoints": sum(
                    mean(values) > 0.0 for values in checkpoint_groups.values()
                ),
            }
        )
    horizon_outputs = []
    for (candidate, control, horizon), rows in sorted(horizon_groups.items()):
        mse = mean([float(row["mse_log_effect"]) for row in rows])
        mae = mean([float(row["mae_log_effect"]) for row in rows])
        horizon_outputs.append(
            {
                "candidate_family": candidate,
                "control_family": control,
                "horizon": horizon,
                "mse_reduction": reduction(mse),
                "mae_reduction": reduction(mae),
            }
        )
    return dataset_outputs, horizon_outputs


def macro_comparisons(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int | str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        grouped[(row["control_family"], "all")].append(row)
        grouped[(row["control_family"], int(row["horizon"]))].append(row)
    outputs = []
    for (control, horizon), rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        mse = mean([float(row["mse_log_effect"]) for row in rows])
        mae = mean([float(row["mae_log_effect"]) for row in rows])
        outputs.append(
            {
                "control_family": control,
                "horizon": horizon,
                "unit_count": len(rows),
                "mse_log_effect": mse,
                "mse_reduction": reduction(mse),
                "mae_log_effect": mae,
                "mae_reduction": reduction(mae),
            }
        )
    return outputs


def build_summary(
    comparisons: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    metrics: list[dict[str, str]],
    config: dict[str, Any],
) -> dict[str, Any]:
    gate = config["gate"]

    def effects(control: str, horizon: int | None = None, metric: str = "mse") -> list[float]:
        field = f"{metric}_log_effect"
        return [
            float(row[field])
            for row in comparisons
            if row["control_family"] == control
            and (horizon is None or int(row["horizon"]) == horizon)
        ]

    def dataset_positive_count(control: str) -> int:
        count = 0
        for dataset in config["datasets"]:
            checkpoint_effects = []
            for checkpoint_seed in config["checkpoint_seeds"]:
                rows = [
                    row
                    for row in comparisons
                    if row["dataset"] == dataset
                    and int(row["checkpoint_seed"]) == int(checkpoint_seed)
                    and row["control_family"] == control
                ]
                checkpoint_effects.append(mean([float(row["mse_log_effect"]) for row in rows]))
            count += sum(value > 0.0 for value in checkpoint_effects) >= 2
        return count

    invariant_pass = (
        len(metrics) == int(config["fit_count"])
        and len(metadata) == 15
        and len(selections) == 15
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
            math.isfinite(float(row[f"val_{metric}_eval_h{horizon}"]))
            and float(row[f"val_{metric}_eval_h{horizon}"]) > 0.0
            for row in metrics
            for horizon in config["horizons"]
            for metric in ("mse", "mae")
        )
        and all(
            not item["uses_test_split"]
            and not item["forecast_model_updated"]
            and not item["official_validation_used_for_early_stopping"]
            and item["pca_uses_fit_targets_only"]
            and float(item["basis_orthogonality_max_abs"]) <= 2e-5
            for item in metadata
        )
    )
    selected_vs_balanced = reduction(mean(effects("balanced_interval")))
    selected_mae_vs_balanced = reduction(mean(effects("balanced_interval", metric="mae")))
    selected_vs_dct = reduction(mean(effects("dct2")))
    selected_vs_pca = reduction(mean(effects("pca_fit")))
    dct_gain_over_balanced_log = mean(effects("balanced_interval")) - mean(effects("dct2"))
    selected_gain_over_balanced_log = mean(effects("balanced_interval"))
    gap_closure = (
        selected_gain_over_balanced_log / dct_gain_over_balanced_log
        if dct_gain_over_balanced_log > 0.0
        else float("nan")
    )
    noninferior_horizons = sum(
        reduction(mean(effects("dct2", horizon)))
        >= float(gate["horizon_noninferiority_margin_vs_dct"])
        for horizon in config["horizons"]
    )
    headroom_pass = (
        selected_vs_balanced >= float(gate["minimum_selected_vs_balanced_reduction"])
        and selected_mae_vs_balanced >= float(gate["mae_guard_vs_balanced"])
        and dataset_positive_count("balanced_interval") >= int(gate["minimum_positive_datasets"])
    )
    global_pass = (
        selected_vs_dct >= float(gate["selected_vs_dct_noninferiority_margin"])
        and selected_vs_pca >= float(gate["selected_vs_pca_noninferiority_margin"])
        and noninferior_horizons >= int(gate["minimum_noninferior_horizons"])
        and gap_closure >= float(gate["minimum_dct_gap_closure"])
    )
    if not invariant_pass:
        decision = "diagnostic_invalid_for_direction_rejection"
    elif headroom_pass and global_pass:
        decision = "conditioning_locality_gap_supported_return_step4"
    elif headroom_pass:
        decision = "local_conditioning_headroom_partial_repeat_step2"
    else:
        decision = "local_family_headroom_not_supported_basis_component_only"
    return {
        "candidate": "SC1-D5",
        "role": "diagnostic_only",
        "invariant_gate": {
            "pass": invariant_pass,
            "fit_count": len(metrics),
            "metadata_count": len(metadata),
            "selection_count": len(selections),
        },
        "selected_family_counts": dict(
            sorted(
                {
                    family: sum(row["selected_family"] == family for row in selections)
                    for family in {row["selected_family"] for row in selections}
                }.items()
            )
        ),
        "headroom_gate": {
            "selected_vs_balanced_mse_reduction": selected_vs_balanced,
            "selected_vs_balanced_mae_reduction": selected_mae_vs_balanced,
            "positive_datasets": dataset_positive_count("balanced_interval"),
            "pass": headroom_pass,
        },
        "global_conditioning_gate": {
            "selected_vs_dct_mse_reduction": selected_vs_dct,
            "selected_vs_pca_mse_reduction": selected_vs_pca,
            "dct_gap_closure": gap_closure,
            "noninferior_horizons_vs_dct": noninferior_horizons,
            "pass": global_pass,
        },
        "decision": decision,
        "method_training_authorized": False,
        "authorization_if_passed": config["authorization_if_passed"],
    }


def render_report(summary: dict[str, Any], selections: list[dict[str, Any]]) -> str:
    lines = [
        "# SC1-D5 Conditioning-Locality Frontier Diagnostic Report",
        "",
        f"- `decision`: `{summary['decision']}`",
        "- `method_training_authorized`: `false`",
        f"- selected family counts: `{summary['selected_family_counts']}`",
        "",
        "## Gates",
        "",
        f"- invariants: `{summary['invariant_gate']}`",
        f"- local headroom: `{summary['headroom_gate']}`",
        f"- global conditioning: `{summary['global_conditioning_gate']}`",
        "",
        "## Fit-Only Geometry Selection",
        "",
        "| Dataset | Checkpoint | Family | Offdiag | Top16 | H48 active |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in selections:
        lines.append(
            f"| {row['dataset']} | {row['checkpoint_seed']} | {row['selected_family']} | "
            f"{float(row['covariance_offdiag_ratio']):.6f} | "
            f"{float(row['variance_capture_top16']):.6f} | "
            f"{float(row['active_atoms_h48']):.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def synthetic_smoke() -> None:
    horizons = [48, 96]
    metrics = []
    geometry = []
    for dataset in ("ETTh1",):
        for checkpoint_seed in (2021,):
            for family, value, offdiag, active in (
                ("balanced_interval", 1.00, 0.55, 55),
                ("dct2", 0.98, 0.52, 720),
                ("pca_fit", 0.97, 0.00, 720),
                ("block_pca_fit_b48", 0.975, 0.20, 48),
            ):
                for structure_seed in (3101, 3102, 3103):
                    row = {
                        "dataset": dataset,
                        "checkpoint_seed": str(checkpoint_seed),
                        "family": family,
                        "structure_seed": str(structure_seed),
                    }
                    for horizon in horizons:
                        row[f"val_mse_eval_h{horizon}"] = str(value)
                        row[f"val_mae_eval_h{horizon}"] = str(value)
                    metrics.append(row)
                    geometry.append(
                        {
                            "dataset": dataset,
                            "checkpoint_seed": str(checkpoint_seed),
                            "family": family,
                            "covariance_offdiag_ratio": str(offdiag),
                            "variance_capture_top16": "0.5",
                            "variance_capture_top64": "0.8",
                            "mean_atom_support_fraction": "0.1",
                            "active_atoms_h48": str(active),
                            "active_atoms_h96": str(max(active, 96)),
                        }
                    )
    families = summarize_checkpoint_families(metrics, horizons)
    geometries = summarize_checkpoint_geometry(geometry, horizons)
    config = {
        "local_basis_families": ["block_pca_fit_b48"],
        "selection": {"maximum_active_atoms_h48": 96},
    }
    selections = select_local_families(geometries, config)
    comparisons = compare_selected(families, selections, horizons)
    local_comparisons = compare_local_families(
        families, config["local_basis_families"], horizons
    )
    local_summary = summarize_local_families(local_comparisons)
    local_datasets, local_horizons = summarize_local_family_details(local_comparisons)
    if (
        len(families) != 4
        or len(selections) != 1
        or len(comparisons) != 6
        or len(local_comparisons) != 6
        or len(local_summary) != 3
        or len(local_datasets) != 3
        or len(local_horizons) != 6
    ):
        raise RuntimeError("D5 synthetic aggregation counts changed")
    print("stage_c_sc1_d5_analyzer_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    config = json.loads(args.d5_config.read_text(encoding="utf-8"))
    horizons = [int(value) for value in config["horizons"]]
    metrics, geometry, metadata = load_artifacts(args.input_root, config["datasets"])
    families = summarize_checkpoint_families(metrics, horizons)
    geometries = summarize_checkpoint_geometry(geometry, horizons)
    selections = select_local_families(geometries, config)
    comparisons = compare_selected(families, selections, horizons)
    local_comparisons = compare_local_families(
        families, config["local_basis_families"], horizons
    )
    local_summary = summarize_local_families(local_comparisons)
    local_datasets, local_horizons = summarize_local_family_details(local_comparisons)
    macros = macro_comparisons(comparisons)
    summary = build_summary(comparisons, selections, metadata, metrics, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "d5_checkpoint_family_summary.csv", families)
    write_csv(args.output_dir / "d5_checkpoint_geometry_summary.csv", geometries)
    write_csv(args.output_dir / "d5_selected_families.csv", selections)
    write_csv(args.output_dir / "d5_selected_comparisons.csv", comparisons)
    write_csv(args.output_dir / "d5_macro_comparisons.csv", macros)
    write_csv(args.output_dir / "d5_local_family_comparisons.csv", local_comparisons)
    write_csv(args.output_dir / "d5_local_family_summary.csv", local_summary)
    write_csv(args.output_dir / "d5_local_family_dataset_summary.csv", local_datasets)
    write_csv(args.output_dir / "d5_local_family_horizon_summary.csv", local_horizons)
    (args.output_dir / "d5_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "d5_diagnostic_report.md").write_text(
        render_report(summary, selections), encoding="utf-8"
    )
    print(
        f"stage_c_sc1_d5_analysis_done decision={summary['decision']} "
        f"output_dir={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
