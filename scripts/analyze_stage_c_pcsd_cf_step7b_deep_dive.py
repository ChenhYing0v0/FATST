#!/usr/bin/env python3
"""Produce horizon-level and mechanism-level PCSD-CF Step7B diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
REFERENCES = (
    "a6",
    "pcsd_equal",
    "pcsd_static",
    "dense_matched",
    "pcsd_random",
)
FIXED_ARMS = (
    "pcsd_fixed_1",
    "pcsd_fixed_48",
    "pcsd_fixed_144",
    "pcsd_fixed_360",
    "pcsd_fixed_720",
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
    "pcsd_equal": "#2563eb",
    "pcsd_static": "#7c3aed",
    "dense_matched": "#dc2626",
    "pcsd_random": "#059669",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--evaluation-split",
        choices=("validation", "test-audit"),
        default="validation",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalize_generated_svg(path: Path) -> None:
    """Remove backend-added trailing spaces for stable versioned artifacts."""
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def run_dir(root: Path, arm: str, dataset: str, seed: int) -> Path:
    return root / arm / dataset / "h720_full" / f"seed{seed}"


def horizon_mse(
    root: Path,
    arm: str,
    dataset: str,
    seed: int,
    evaluation_split: str,
) -> list[float]:
    filename = (
        "test_audit_metrics_by_target_horizon.csv"
        if evaluation_split == "test-audit"
        else "metrics_by_target_horizon.csv"
    )
    path = run_dir(root, arm, dataset, seed) / filename
    rows = read_csv(path)
    by_horizon = {int(row["target_horizon"]): float(row["mse"]) for row in rows}
    expected = list(range(1, 721))
    if sorted(by_horizon) != expected:
        raise ValueError(f"incomplete dense horizons: {path}")
    return [by_horizon[horizon] for horizon in expected]


def gain_percent(candidate: float, reference: float) -> float:
    return 100.0 * (1.0 - candidate / reference)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = read_csv(args.run_summary)
    summary = {(row["dataset"], row["arm"]): row for row in summary_rows}

    horizon_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    horizon_cache: dict[tuple[str, str], list[float]] = {}
    for dataset in DATASETS:
        for arm in ("pcsd_direct", *REFERENCES):
            horizon_cache[(dataset, arm)] = horizon_mse(
                args.raw_root,
                arm,
                dataset,
                args.seed,
                args.evaluation_split,
            )
        direct = horizon_cache[(dataset, "pcsd_direct")]
        for reference in REFERENCES:
            reference_values = horizon_cache[(dataset, reference)]
            gains = [
                gain_percent(candidate, baseline)
                for candidate, baseline in zip(direct, reference_values)
            ]
            for horizon, (candidate, baseline, gain) in enumerate(
                zip(direct, reference_values, gains), start=1
            ):
                horizon_rows.append(
                    {
                        "dataset": dataset,
                        "reference": reference,
                        "target_horizon": horizon,
                        "direct_mse": candidate,
                        "reference_mse": baseline,
                        "direct_gain_percent": gain,
                    }
                )
            for bin_name, start, end in HORIZON_BINS:
                candidate_mean = mean(direct[start - 1 : end])
                reference_mean = mean(reference_values[start - 1 : end])
                bin_rows.append(
                    {
                        "dataset": dataset,
                        "reference": reference,
                        "horizon_bin": bin_name,
                        "start_horizon": start,
                        "end_horizon": end,
                        "direct_mse_mean": candidate_mean,
                        "reference_mse_mean": reference_mean,
                        "direct_gain_percent": gain_percent(
                            candidate_mean, reference_mean
                        ),
                    }
                )

    fixed_rows: list[dict[str, Any]] = []
    best_fixed_over_a6 = 0
    for dataset in DATASETS:
        a6_auc = float(summary[(dataset, "a6")]["dense_mse_auc"])
        candidates = []
        for arm in FIXED_ARMS:
            auc = float(summary[(dataset, arm)]["dense_mse_auc"])
            candidates.append((auc, arm))
        best_auc, best_arm = min(candidates)
        if best_auc < a6_auc:
            best_fixed_over_a6 += 1
        for auc, arm in candidates:
            fixed_rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "scope": int(arm.rsplit("_", 1)[-1]),
                    "dense_mse_auc": auc,
                    "a6_dense_mse_auc": a6_auc,
                    "gain_over_a6_percent": gain_percent(auc, a6_auc),
                    "is_best_fixed_scope": arm == best_arm,
                }
            )

    mechanism_rows: list[dict[str, Any]] = []
    oracle_bin_rows: list[dict[str, Any]] = []
    arm_training_rows: list[dict[str, Any]] = []
    target_usage_ranges: list[float] = []
    positive_oracle_datasets = 0
    for dataset in DATASETS:
        row = summary[(dataset, "pcsd_direct")]
        oracle_headroom = float(row["same_run_oracle_headroom"])
        if oracle_headroom > 0.0:
            positive_oracle_datasets += 1
        usage = json.loads(row["policy_usage"])
        mechanism_rows.append(
            {
                "dataset": dataset,
                "same_run_oracle_headroom_percent": 100.0 * oracle_headroom,
                "best_arm_gain_over_fused_percent": 100.0
                * float(row["best_same_run_arm_gain_over_fused"]),
                "best_arm_gain_over_persistence_percent": 100.0
                * float(row["best_same_run_arm_gain_over_persistence"]),
                "minimum_pairwise_probe_nrmse": float(
                    row["minimum_pairwise_probe_nrmse"]
                ),
                "policy_entropy": float(row["policy_entropy"]),
                "policy_usage_max": float(row["policy_usage_max"]),
                "usage_scope_1": usage[0],
                "usage_scope_48": usage[1],
                "usage_scope_144": usage[2],
                "usage_scope_360": usage[3],
                "usage_scope_720": usage[4],
            }
        )
        diagnostic_filename = (
            "pcsd_test_audit_diagnostics.npz"
            if args.evaluation_split == "test-audit"
            else "pcsd_validation_diagnostics.npz"
        )
        diagnostic_path = (
            run_dir(args.raw_root, "pcsd_direct", dataset, args.seed)
            / diagnostic_filename
        )
        with np.load(diagnostic_path, allow_pickle=False) as payload:
            arm_losses = payload["arm_row_bin_mse"].astype(np.float64)
            fused_losses = payload["fused_row_bin_mse"].astype(np.float64)
            usage_by_row = payload["policy_row_bin_usage"].astype(np.float64)
        bin_mean_usage = usage_by_row.mean(axis=0)
        target_usage_l1_range = max(
            float(np.abs(bin_mean_usage[left] - bin_mean_usage[right]).sum())
            for left in range(bin_mean_usage.shape[0])
            for right in range(bin_mean_usage.shape[0])
        )
        row_mean_usage = usage_by_row.mean(axis=1)
        instance_usage_l1_deviation = float(
            np.abs(row_mean_usage - row_mean_usage.mean(axis=0)).sum(axis=1).mean()
        )
        target_usage_ranges.append(target_usage_l1_range)
        mechanism_rows[-1]["target_usage_max_pairwise_l1"] = (
            target_usage_l1_range
        )
        mechanism_rows[-1]["instance_usage_mean_l1_deviation"] = (
            instance_usage_l1_deviation
        )
        for arm_index, scope in enumerate((1, 48, 144, 360, 720)):
            fixed_path = (
                run_dir(
                    args.raw_root,
                    f"pcsd_fixed_{scope}",
                    dataset,
                    args.seed,
                )
                / diagnostic_filename
            )
            with np.load(fixed_path, allow_pickle=False) as fixed_payload:
                fixed_fused_mse = float(
                    fixed_payload["fused_row_bin_mse"].astype(np.float64).mean()
                )
            direct_arm_mse = float(arm_losses[..., arm_index].mean())
            arm_training_rows.append(
                {
                    "dataset": dataset,
                    "scope": scope,
                    "direct_run_arm_mse": direct_arm_mse,
                    "independently_trained_fixed_arm_mse": fixed_fused_mse,
                    "direct_arm_degradation_percent": 100.0
                    * (direct_arm_mse / fixed_fused_mse - 1.0),
                }
            )
        for bin_index, (bin_name, start, end) in enumerate(HORIZON_BINS):
            arm_means = arm_losses[:, bin_index, :].mean(axis=0)
            fused_mean = float(fused_losses[:, bin_index].mean())
            oracle_mean = float(
                arm_losses[:, bin_index, :].min(axis=-1).mean()
            )
            best_arm_index = int(arm_means.argmin())
            usage_mean = usage_by_row[:, bin_index, :].mean(axis=0)
            oracle_bin_rows.append(
                {
                    "dataset": dataset,
                    "horizon_bin": bin_name,
                    "start_horizon": start,
                    "end_horizon": end,
                    "fused_mse": fused_mean,
                    "row_oracle_mse": oracle_mean,
                    "row_oracle_gain_over_fused_percent": gain_percent(
                        oracle_mean, fused_mean
                    ),
                    "best_mean_arm_scope": (1, 48, 144, 360, 720)[
                        best_arm_index
                    ],
                    "best_mean_arm_mse": float(arm_means[best_arm_index]),
                    "best_mean_arm_gain_over_fused_percent": gain_percent(
                        float(arm_means[best_arm_index]), fused_mean
                    ),
                    "usage_scope_1": float(usage_mean[0]),
                    "usage_scope_48": float(usage_mean[1]),
                    "usage_scope_144": float(usage_mean[2]),
                    "usage_scope_360": float(usage_mean[3]),
                    "usage_scope_720": float(usage_mean[4]),
                }
            )

    a6_m0_max_auc_gap = max(
        abs(
            float(summary[(dataset, "a6")]["dense_mse_auc"])
            - float(summary[(dataset, "pcsd_m0")]["dense_mse_auc"])
        )
        for dataset in DATASETS
    )
    arm_degradations = [
        float(row["direct_arm_degradation_percent"])
        for row in arm_training_rows
    ]
    posthoc_credit_gate = bool(
        positive_oracle_datasets >= 4 and best_fixed_over_a6 >= 4
    )
    deep_dive = {
        "seed": args.seed,
        "evaluation_split": args.evaluation_split,
        "positive_same_run_oracle_headroom_datasets": positive_oracle_datasets,
        "best_fixed_scope_over_a6_datasets": best_fixed_over_a6,
        "a6_m0_max_dense_mse_auc_absolute_gap": a6_m0_max_auc_gap,
        "direct_arm_undertrained_pairs": sum(
            value > 0.0 for value in arm_degradations
        ),
        "direct_arm_total_pairs": len(arm_degradations),
        "direct_arm_degradation_percent_min": min(arm_degradations),
        "direct_arm_degradation_percent_median": float(
            np.median(arm_degradations)
        ),
        "direct_arm_degradation_percent_max": max(arm_degradations),
        "target_usage_max_pairwise_l1_min": min(target_usage_ranges),
        "target_usage_max_pairwise_l1_max": max(target_usage_ranges),
        "posthoc_cross_dataset_arm_competitiveness_gate": posthoc_credit_gate,
        "formal_preregistered_gate_remains": (
            "direct_credit_problem_supported_sc2_step2_4_only"
        ),
        "research_interpretation": (
            "joint_fused_loss_arm_credit_starvation_supported_but_"
            "target_specific_credit_mechanism_unproven"
        ),
    }

    write_csv(args.output_dir / "horizon_gain_by_reference.csv", horizon_rows)
    write_csv(args.output_dir / "horizon_bin_gain.csv", bin_rows)
    write_csv(args.output_dir / "fixed_scope_summary.csv", fixed_rows)
    write_csv(args.output_dir / "mechanism_by_dataset.csv", mechanism_rows)
    write_csv(args.output_dir / "same_run_oracle_by_bin.csv", oracle_bin_rows)
    write_csv(
        args.output_dir / "direct_arm_vs_fixed_training.csv",
        arm_training_rows,
    )
    (args.output_dir / "deep_dive_gate.json").write_text(
        json.dumps(deep_dive, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    for axis, dataset in zip(axes.flat, DATASETS):
        for reference in REFERENCES:
            rows = [
                row
                for row in horizon_rows
                if row["dataset"] == dataset and row["reference"] == reference
            ]
            axis.plot(
                [row["target_horizon"] for row in rows],
                [row["direct_gain_percent"] for row in rows],
                label=reference,
                color=COLORS[reference],
                linewidth=1.1,
            )
        axis.axhline(0.0, color="#6b7280", linewidth=0.8, linestyle="--")
        axis.set_title(dataset)
        axis.set_ylabel("DIRECT gain (%)")
        axis.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.93, 0.08))
    figure.supxlabel("Requested prefix horizon H")
    split_label = (
        "test" if args.evaluation_split == "test-audit" else "validation"
    )
    figure.suptitle(
        f"PCSD-CF DIRECT relative {split_label} MSE by horizon"
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.97))
    svg_path = args.output_dir / "horizon_gain_curves.svg"
    figure.savefig(svg_path)
    normalize_generated_svg(svg_path)
    figure.savefig(args.output_dir / "horizon_gain_curves.png", dpi=160)
    plt.close(figure)

    print(json.dumps(deep_dive, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
