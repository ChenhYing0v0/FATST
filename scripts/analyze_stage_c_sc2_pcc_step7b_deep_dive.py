#!/usr/bin/env python3
"""Produce Step9/10 deep diagnostics for the PCC validation screen."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
MODES = (
    "measure_only",
    "equal_skill",
    "pointwise_route_only",
    "pointwise_capability_skill_only",
    "pointwise_prior_composed",
    "pointwise_pcc_v0",
    "transport_skill_only",
    "transport_route_only",
    "pcc_transport_full",
)
REFERENCES = (
    "a6",
    "pcsd_direct",
    "equal_skill",
    "pointwise_prior_composed",
    "pointwise_pcc_v0",
)
HORIZON_BINS = (
    ("h1_48", 1, 48),
    ("h49_96", 49, 96),
    ("h97_144", 97, 144),
    ("h145_192", 145, 192),
    ("h193_288", 193, 288),
    ("h289_336", 289, 336),
    ("h337_512", 337, 512),
    ("h513_720", 513, 720),
)
COLORS = {
    "a6": "#111827",
    "pcsd_direct": "#dc2626",
    "equal_skill": "#2563eb",
    "pointwise_prior_composed": "#7c3aed",
    "pointwise_pcc_v0": "#059669",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalize_generated_svg(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        "\n".join(line.rstrip() for line in lines) + "\n",
        encoding="utf-8",
    )


def run_dir(root: Path, arm: str, dataset: str, seed: int) -> Path:
    return root / arm / dataset / "h720_full" / f"seed{seed}"


def metric_curve(
    root: Path,
    arm: str,
    dataset: str,
    seed: int,
) -> tuple[list[float], list[float]]:
    path = run_dir(root, arm, dataset, seed) / "metrics_by_target_horizon.csv"
    rows = read_csv(path)
    by_horizon = {int(row["target_horizon"]): row for row in rows}
    if sorted(by_horizon) != list(range(1, 721)):
        raise ValueError(f"incomplete dense horizons: {path}")
    mse = [float(by_horizon[horizon]["mse"]) for horizon in range(1, 721)]
    mae = [float(by_horizon[horizon]["mae"]) for horizon in range(1, 721)]
    return mse, mae


def gain_percent(candidate: float, reference: float) -> float:
    return 100.0 * (1.0 - candidate / reference)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.run_summary)
    summary = {
        (row["dataset"], row["arm"]): row
        for row in rows
        if row["status"] == "ok"
    }
    required = {
        (dataset, arm)
        for dataset in DATASETS
        for arm in ("a6", "pcsd_direct", *MODES)
    }
    if not required.issubset(summary):
        missing = sorted(required - set(summary))
        raise ValueError(f"missing run summary rows: {missing}")

    scoreboard: list[dict[str, Any]] = []
    for arm in ("a6", "pcsd_direct", *MODES):
        mse = [float(summary[(dataset, arm)]["dense_mse_auc"]) for dataset in DATASETS]
        mae = [float(summary[(dataset, arm)]["dense_mae_auc"]) for dataset in DATASETS]
        a6 = [float(summary[(dataset, "a6")]["dense_mse_auc"]) for dataset in DATASETS]
        plain = [
            float(summary[(dataset, "pcsd_direct")]["dense_mse_auc"])
            for dataset in DATASETS
        ]
        scoreboard.append(
            {
                "arm": arm,
                "macro_absolute_dense_mse_auc": mean(mse),
                "macro_absolute_dense_mae_auc": mean(mae),
                "macro_gain_over_a6_percent": mean(
                    gain_percent(value, baseline)
                    for value, baseline in zip(mse, a6)
                ),
                "wins_over_a6": sum(
                    value < baseline for value, baseline in zip(mse, a6)
                ),
                "macro_gain_over_plain_percent": mean(
                    gain_percent(value, baseline)
                    for value, baseline in zip(mse, plain)
                ),
                "wins_over_plain": sum(
                    value < baseline for value, baseline in zip(mse, plain)
                ),
            }
        )

    curves: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
    for dataset in DATASETS:
        for arm in REFERENCES:
            root = (
                args.reference_root
                if arm in {"a6", "pcsd_direct"}
                else args.raw_root
            )
            curves[(dataset, arm)] = metric_curve(root, arm, dataset, args.seed)
        curves[(dataset, "pcc_transport_full")] = metric_curve(
            args.raw_root,
            "pcc_transport_full",
            dataset,
            args.seed,
        )

    horizon_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        candidate_mse, candidate_mae = curves[(dataset, "pcc_transport_full")]
        for reference in REFERENCES:
            reference_mse, reference_mae = curves[(dataset, reference)]
            for horizon in range(1, 721):
                horizon_rows.append(
                    {
                        "dataset": dataset,
                        "reference": reference,
                        "target_horizon": horizon,
                        "pcc_mse": candidate_mse[horizon - 1],
                        "reference_mse": reference_mse[horizon - 1],
                        "pcc_mse_gain_percent": gain_percent(
                            candidate_mse[horizon - 1],
                            reference_mse[horizon - 1],
                        ),
                        "pcc_mae_gain_percent": gain_percent(
                            candidate_mae[horizon - 1],
                            reference_mae[horizon - 1],
                        ),
                    }
                )
            for name, start, end in HORIZON_BINS:
                candidate_mse_mean = mean(candidate_mse[start - 1 : end])
                reference_mse_mean = mean(reference_mse[start - 1 : end])
                candidate_mae_mean = mean(candidate_mae[start - 1 : end])
                reference_mae_mean = mean(reference_mae[start - 1 : end])
                bin_rows.append(
                    {
                        "dataset": dataset,
                        "reference": reference,
                        "horizon_bin": name,
                        "start_horizon": start,
                        "end_horizon": end,
                        "pcc_mse_gain_percent": gain_percent(
                            candidate_mse_mean,
                            reference_mse_mean,
                        ),
                        "pcc_mae_gain_percent": gain_percent(
                            candidate_mae_mean,
                            reference_mae_mean,
                        ),
                    }
                )
    for reference in REFERENCES:
        for name, start, end in HORIZON_BINS:
            selected = [
                row
                for row in bin_rows
                if row["dataset"] != "macro"
                and row["reference"] == reference
                and row["horizon_bin"] == name
            ]
            bin_rows.append(
                {
                    "dataset": "macro",
                    "reference": reference,
                    "horizon_bin": name,
                    "start_horizon": start,
                    "end_horizon": end,
                    "pcc_mse_gain_percent": mean(
                        float(row["pcc_mse_gain_percent"]) for row in selected
                    ),
                    "pcc_mae_gain_percent": mean(
                        float(row["pcc_mae_gain_percent"]) for row in selected
                    ),
                }
            )

    mechanism_rows: list[dict[str, Any]] = []
    for arm in MODES:
        for dataset in DATASETS:
            row = summary[(dataset, arm)]
            plain = summary[(dataset, "pcsd_direct")]
            mechanism_rows.append(
                {
                    "arm": arm,
                    "dataset": dataset,
                    "same_run_oracle_headroom_percent": 100.0
                    * float(row["same_run_oracle_headroom"]),
                    "minimum_pairwise_probe_nrmse": float(
                        row["minimum_pairwise_probe_nrmse"]
                    ),
                    "pairwise_nrmse_retention_fraction": float(
                        row["minimum_pairwise_probe_nrmse"]
                    )
                    / max(float(plain["minimum_pairwise_probe_nrmse"]), 1e-12),
                    "policy_normalized_entropy": float(
                        row["policy_normalized_entropy"]
                    ),
                    "policy_usage_max": float(row["policy_usage_max"]),
                    "shared_gradient_cosine_mean": float(
                        row["shared_gradient_cosine_mean"]
                    ),
                    "shared_gradient_cosine_min": float(
                        row["shared_gradient_cosine_min"]
                    ),
                }
            )

    training_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        path = (
            run_dir(args.raw_root, "pcc_transport_full", dataset, args.seed)
            / "training_log.csv"
        )
        log = read_csv(path)
        last = log[-1]
        training_rows.append(
            {
                "dataset": dataset,
                "epochs_ran": len(log),
                "best_epoch": int(float(last["best_epoch_so_far"])),
                "stop_triggered": last["stop_triggered"],
                "last_credit_argmax_accuracy": float(
                    last["train_pcc_credit_argmax_accuracy"]
                ),
                "last_credit_normalized_entropy": float(
                    last["train_pcc_credit_normalized_entropy"]
                ),
                "last_credit_policy_kl": float(
                    last["train_pcc_credit_policy_kl"]
                ),
                "last_policy_normalized_entropy": float(
                    last["train_pcc_policy_normalized_entropy"]
                ),
                "last_policy_usage_max": float(
                    last["train_pcc_policy_usage_max"]
                ),
            }
        )

    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    score = {row["arm"]: row for row in scoreboard}
    full_gain = float(score["pcc_transport_full"]["macro_gain_over_a6_percent"])
    equal_gain = float(score["equal_skill"]["macro_gain_over_a6_percent"])
    deep_dive = {
        "seed": args.seed,
        "evaluation_split": "validation",
        "expected_runs": 45,
        "valid_runs": gate["valid_runs"],
        "paired_pcc_initialization": gate["paired_pcc_initialization"],
        "test_used": False,
        "formal_method_pass": gate["method_pass"],
        "formal_decision": gate["decision"],
        "pcc_macro_gain_over_a6_percent": full_gain,
        "equal_skill_macro_gain_over_a6_percent": equal_gain,
        "equal_skill_fraction_of_pcc_a6_gain": equal_gain / max(full_gain, 1e-12),
        "pcc_gain_over_equal_skill_percent": mean(
            gain_percent(
                float(summary[(dataset, "pcc_transport_full")]["dense_mse_auc"]),
                float(summary[(dataset, "equal_skill")]["dense_mse_auc"]),
            )
            for dataset in DATASETS
        ),
        "pcc_gain_over_prior_composed_percent": 100.0
        * float(gate["macro_comparisons"]["pointwise_prior_composed"]["gain_fraction"]),
        "pcc_gain_over_pointwise_pcc_percent": 100.0
        * float(gate["macro_comparisons"]["pointwise_pcc_v0"]["gain_fraction"]),
        "arm_pairs_improved": gate["arm_pairs_improved"],
        "arm_degradation_median_relative_reduction": gate[
            "arm_degradation_median_relative_reduction"
        ],
        "pairwise_nrmse_retention_min": gate[
            "pairwise_nrmse_retention_min"
        ],
        "credit_argmax_accuracy_min": min(
            float(row["last_credit_argmax_accuracy"]) for row in training_rows
        ),
        "credit_argmax_accuracy_max": max(
            float(row["last_credit_argmax_accuracy"]) for row in training_rows
        ),
        "failure_attribution": {
            "hypothesis_false": False,
            "intervention_point_wrong": (
                "partially_supported_skill_supervision_homogenizes_scopes"
            ),
            "readout_or_head_design_wrong": (
                "remains_possible_shared_field_specialization_not_preserved"
            ),
            "optimization_or_numeric_pathology": False,
            "capacity_control_explains": (
                "generic_equal_skill_and_prior_composition_explain_most_gain"
            ),
        },
        "research_decision": (
            "exact_pcc_v1_ti_fail_retain_arm_recovery_signal_return_step4"
        ),
    }

    write_csv(args.output_dir / "objective_scoreboard.csv", scoreboard)
    write_csv(args.output_dir / "horizon_gain_by_reference.csv", horizon_rows)
    write_csv(args.output_dir / "horizon_bin_gain.csv", bin_rows)
    write_csv(args.output_dir / "mechanism_control_summary.csv", mechanism_rows)
    write_csv(args.output_dir / "pcc_training_diagnostics.csv", training_rows)
    (args.output_dir / "deep_dive_gate.json").write_text(
        json.dumps(deep_dive, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    for axis, dataset in zip(axes.flat, DATASETS):
        rows_for_dataset = [
            row for row in horizon_rows if row["dataset"] == dataset
        ]
        for reference in REFERENCES:
            selected = [
                row
                for row in rows_for_dataset
                if row["reference"] == reference
            ]
            axis.plot(
                [row["target_horizon"] for row in selected],
                [row["pcc_mse_gain_percent"] for row in selected],
                label=reference,
                color=COLORS[reference],
                linewidth=1.1,
            )
        axis.axhline(0.0, color="#6b7280", linewidth=0.8, linestyle="--")
        axis.set_title(dataset)
        axis.set_ylabel("PCC gain (%)")
        axis.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.93, 0.08))
    figure.supxlabel("Requested prefix horizon H")
    figure.suptitle("PCC validation MSE gain by horizon")
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.97))
    svg_path = args.output_dir / "horizon_gain_curves.svg"
    figure.savefig(svg_path)
    normalize_generated_svg(svg_path)
    figure.savefig(args.output_dir / "horizon_gain_curves.png", dpi=160)
    plt.close(figure)

    print(json.dumps(deep_dive, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
