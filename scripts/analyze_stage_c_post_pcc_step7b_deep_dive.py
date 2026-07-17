#!/usr/bin/env python3
"""Produce Step 9/10 attribution diagnostics for the SIFF/MCCA screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
NEW_ARMS = (
    "pcsd_mcca",
    "siff_equal",
    "siff_pcc",
    "siff_mcca",
    "siff_const_mcca",
    "siff_permuted_mcca",
    "pcsd_q1_wide_mcca",
    "independent_scope_matched_mcca",
    "dense_siff_matched",
    "pcsd_pointwise_mcca",
    "pcsd_uniform_balanced_ot",
)
PCSD_REFERENCES = ("a6", "pcsd_direct", "dense_matched")
PCC_REFERENCES = ("equal_skill", "pcc_transport_full")
HORIZON_BINS = (
    ("h1_48", 1, 48),
    ("h49_96", 49, 96),
    ("h97_192", 97, 192),
    ("h193_336", 193, 336),
    ("h337_720", 337, 720),
)
COMPARISONS = {
    "joint_over_a6": ("siff_mcca", "a6"),
    "architecture_mcca": ("siff_mcca", "pcsd_mcca"),
    "ordered_over_constant": ("siff_mcca", "siff_const_mcca"),
    "ordered_over_permuted": ("siff_mcca", "siff_permuted_mcca"),
    "mcca_over_pcc": ("pcsd_mcca", "pcc_transport_full"),
    "transport_over_pointwise": ("pcsd_mcca", "pcsd_pointwise_mcca"),
    "capability_over_uniform_ot": (
        "pcsd_mcca",
        "pcsd_uniform_balanced_ot",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--pcsd-reference-root", type=Path, required=True)
    parser.add_argument("--pcc-reference-root", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_dir(root: Path, arm: str, dataset: str, seed: int) -> Path:
    return root / arm / dataset / "h720_full" / f"seed{seed}"


def arm_root(args: argparse.Namespace, arm: str) -> Path:
    if arm in PCSD_REFERENCES:
        return args.pcsd_reference_root
    if arm in PCC_REFERENCES:
        return args.pcc_reference_root
    return args.raw_root


def metric_curve(
    args: argparse.Namespace,
    arm: str,
    dataset: str,
) -> tuple[list[float], list[float]]:
    path = (
        run_dir(arm_root(args, arm), arm, dataset, args.seed)
        / "metrics_by_target_horizon.csv"
    )
    rows = read_csv(path)
    by_horizon = {int(row["target_horizon"]): row for row in rows}
    if sorted(by_horizon) != list(range(1, 721)):
        raise ValueError(f"incomplete dense horizons: {path}")
    mse = [float(by_horizon[horizon]["mse"]) for horizon in range(1, 721)]
    mae = [float(by_horizon[horizon]["mae"]) for horizon in range(1, 721)]
    if not all(math.isfinite(value) for value in (*mse, *mae)):
        raise ValueError(f"non-finite metric: {path}")
    return mse, mae


def gain_percent(candidate: float, reference: float) -> float:
    return 100.0 * (1.0 - candidate / reference)


def harmonic_target_weights(max_horizon: int = 720) -> list[float]:
    return [
        sum(1.0 / horizon for horizon in range(target, max_horizon + 1))
        / max_horizon
        for target in range(1, max_horizon + 1)
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = read_csv(args.run_summary)
    summary = {
        (row["dataset"], row["arm"]): row
        for row in summary_rows
        if row["status"] == "ok"
    }
    required = {
        (dataset, arm)
        for dataset in DATASETS
        for arm in (*NEW_ARMS, *PCSD_REFERENCES, *PCC_REFERENCES)
    }
    if not required.issubset(summary):
        missing = sorted(required - set(summary))
        raise ValueError(f"missing summary rows: {missing}")

    curves = {
        (dataset, arm): metric_curve(args, arm, dataset)
        for dataset in DATASETS
        for arm in (*NEW_ARMS, *PCSD_REFERENCES, *PCC_REFERENCES)
    }

    scoreboard: list[dict[str, Any]] = []
    for dataset in DATASETS:
        a6_mse = float(summary[(dataset, "a6")]["dense_mse_auc"])
        for arm in (*NEW_ARMS, *PCSD_REFERENCES, *PCC_REFERENCES):
            row = summary[(dataset, arm)]
            mse_curve, _ = curves[(dataset, arm)]
            scoreboard.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "dense_mse_auc": float(row["dense_mse_auc"]),
                    "dense_mae_auc": float(row["dense_mae_auc"]),
                    "gain_over_a6_percent": gain_percent(
                        float(row["dense_mse_auc"]),
                        a6_mse,
                    ),
                    "h1_mse": mse_curve[0],
                    "h48_mse": mse_curve[47],
                    "h96_mse": mse_curve[95],
                    "h192_mse": mse_curve[191],
                    "h336_mse": mse_curve[335],
                    "h720_mse": mse_curve[719],
                }
            )

    bin_rows: list[dict[str, Any]] = []
    for effect, (candidate, reference) in COMPARISONS.items():
        for dataset in DATASETS:
            candidate_mse, candidate_mae = curves[(dataset, candidate)]
            reference_mse, reference_mae = curves[(dataset, reference)]
            for bin_name, start, end in HORIZON_BINS:
                bin_rows.append(
                    {
                        "effect": effect,
                        "candidate": candidate,
                        "reference": reference,
                        "dataset": dataset,
                        "horizon_bin": bin_name,
                        "start_horizon": start,
                        "end_horizon": end,
                        "mse_gain_percent": gain_percent(
                            mean(candidate_mse[start - 1 : end]),
                            mean(reference_mse[start - 1 : end]),
                        ),
                        "mae_gain_percent": gain_percent(
                            mean(candidate_mae[start - 1 : end]),
                            mean(reference_mae[start - 1 : end]),
                        ),
                    }
                )
        for bin_name, start, end in HORIZON_BINS:
            selected = [
                row
                for row in bin_rows
                if row["effect"] == effect
                and row["horizon_bin"] == bin_name
                and row["dataset"] != "macro"
            ]
            bin_rows.append(
                {
                    "effect": effect,
                    "candidate": candidate,
                    "reference": reference,
                    "dataset": "macro",
                    "horizon_bin": bin_name,
                    "start_horizon": start,
                    "end_horizon": end,
                    "mse_gain_percent": mean(
                        float(row["mse_gain_percent"]) for row in selected
                    ),
                    "mae_gain_percent": mean(
                        float(row["mae_gain_percent"]) for row in selected
                    ),
                }
            )

    training_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for arm in NEW_ARMS:
            path = (
                run_dir(args.raw_root, arm, dataset, args.seed)
                / "training_log.csv"
            )
            rows = read_csv(path)
            val = [float(row["val_mean_mse"]) for row in rows]
            last = rows[-1]
            training_rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "epochs_ran": len(rows),
                    "best_epoch": int(float(last["best_epoch_so_far"])),
                    "last_epoch": int(float(last["epoch"])),
                    "best_h720_val_mse": min(val),
                    "last_h720_val_mse": val[-1],
                    "last_over_best_fraction": val[-1] / min(val) - 1.0,
                    "stop_triggered": last["stop_triggered"],
                    "all_finite": all(
                        math.isfinite(float(value))
                        for value in val
                    ),
                }
            )

    architecture_by_dataset: dict[str, float] = {}
    mcca_by_dataset: dict[str, float] = {}
    joint_by_dataset: dict[str, float] = {}
    for dataset in DATASETS:
        architecture_by_dataset[dataset] = mean(
            gain_percent(
                float(summary[(dataset, candidate)]["dense_mse_auc"]),
                float(summary[(dataset, reference)]["dense_mse_auc"]),
            )
            for candidate, reference in (
                ("siff_equal", "equal_skill"),
                ("siff_pcc", "pcc_transport_full"),
                ("siff_mcca", "pcsd_mcca"),
            )
        )
        mcca_by_dataset[dataset] = mean(
            gain_percent(
                float(summary[(dataset, candidate)]["dense_mse_auc"]),
                float(summary[(dataset, reference)]["dense_mse_auc"]),
            )
            for candidate, reference in (
                ("pcsd_mcca", "pcc_transport_full"),
                ("siff_mcca", "siff_pcc"),
            )
        )
        joint_by_dataset[dataset] = gain_percent(
            float(summary[(dataset, "siff_mcca")]["dense_mse_auc"]),
            float(summary[(dataset, "a6")]["dense_mse_auc"]),
        )

    leave_one_out: list[dict[str, Any]] = []
    for omitted in ("none", *DATASETS):
        included = [
            dataset for dataset in DATASETS if dataset != omitted
        ]
        for effect, values in (
            ("architecture_main_effect", architecture_by_dataset),
            ("mcca_main_effect", mcca_by_dataset),
            ("joint_over_a6", joint_by_dataset),
        ):
            selected = [values[dataset] for dataset in included]
            leave_one_out.append(
                {
                    "effect": effect,
                    "omitted_dataset": omitted,
                    "datasets_included": len(included),
                    "macro_gain_percent": mean(selected),
                    "dataset_wins": sum(value > 0.0 for value in selected),
                }
            )

    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    siff_ettm2, _ = curves[("ETTm2", "siff_mcca")]
    pcsd_ettm2, _ = curves[("ETTm2", "pcsd_mcca")]
    weights = harmonic_target_weights()
    result = {
        "seed": args.seed,
        "evaluation_split": "validation",
        "test_used": False,
        "valid_new_runs": gate["valid_new_runs"],
        "valid_reference_runs": gate["valid_reference_runs"],
        "formal_method_pass": gate["method_pass"],
        "architecture_main_effect_macro_gain_percent": (
            100.0 * gate["architecture_main_effect_macro_gain"]
        ),
        "mcca_main_effect_macro_gain_percent": (
            100.0 * gate["mcca_main_effect_macro_gain"]
        ),
        "joint_over_a6_macro_gain_percent": mean(joint_by_dataset.values()),
        "joint_over_a6_dataset_wins": sum(
            value > 0.0 for value in joint_by_dataset.values()
        ),
        "joint_over_a6_excluding_ettm2_macro_gain_percent": mean(
            value
            for dataset, value in joint_by_dataset.items()
            if dataset != "ETTm2"
        ),
        "joint_over_a6_excluding_ettm2_wins": sum(
            value > 0.0
            for dataset, value in joint_by_dataset.items()
            if dataset != "ETTm2"
        ),
        "ettm2_siff_vs_pcsd_h1_gain_percent": gain_percent(
            siff_ettm2[0],
            pcsd_ettm2[0],
        ),
        "ettm2_siff_vs_pcsd_h720_gain_percent": gain_percent(
            siff_ettm2[-1],
            pcsd_ettm2[-1],
        ),
        "dense_auc_target_weight_h1": weights[0],
        "dense_auc_target_weight_h720": weights[-1],
        "dense_auc_h1_to_h720_weight_ratio": weights[0] / weights[-1],
        "dense_auc_h1_to_flat_weight_ratio": weights[0] * 720.0,
        "dense_auc_h720_to_flat_weight_ratio": weights[-1] * 720.0,
        "coupling_training_target_measure": "exact_dense_prefix_harmonic",
        "coupling_training_error_norm": "l1",
        "checkpoint_selection_measure": "h720_mse",
        "primary_screen_measure": "dense_prefix_mse_auc",
        "failure_attribution": {
            "hypothesis_false": (
                "false_for_order_information_but_exact_global_linear_field"
                "_not_supported_as_general_method"
            ),
            "intervention_point_wrong": (
                "possible_rigid_scale_field_redistributes_error_toward_early"
                "_targets_on_ettm2"
            ),
            "readout_or_head_design_wrong": (
                "supported_q2_parameterization_has_short_prefix_pathology"
            ),
            "optimization_or_numeric_pathology": (
                "true_ettm2_h1_degradation_exceeds_100_percent_while_h720"
                "_improves_all_values_finite_checkpoint_trajectory_unknown"
            ),
            "capacity_control_explains": (
                "independent_scope_and_q1_width_controls_explain_ettm2"
                "_recovery_but_not_ordered_over_permuted_5_of_5"
            ),
            "mcca_exact_hypothesis": (
                "not_supported_same_mass_competition_loses_to_pcc_on_pcsd"
                "_5_of_5"
            ),
        },
        "research_decision": (
            "close_exact_siff_v1_and_mcca_v1_return_step4_keep_order"
            "_transport_signals_and_audit_dense_checkpoint_trajectory"
        ),
    }

    write_csv(args.output_dir / "arm_scoreboard.csv", scoreboard)
    write_csv(args.output_dir / "horizon_bin_effects.csv", bin_rows)
    write_csv(args.output_dir / "training_stability.csv", training_rows)
    write_csv(args.output_dir / "leave_one_dataset_out.csv", leave_one_out)
    (args.output_dir / "step9_attribution.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
