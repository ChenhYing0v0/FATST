#!/usr/bin/env python3
"""Audit the frozen three-seed ISCF-BSCA-v1 confirmation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_stage_c_iscf_bsca_v1 import (
    DATASETS,
    HORIZONS,
    gain,
    internal_metrics,
    selected_metrics,
    sha256,
    write_rows,
)

SEEDS = (2021, 2022, 2023)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_iscf_bsca_v1_confirmation.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run_dir(root: Path, dataset: str, seed: int, *, candidate: bool) -> Path:
    prefix = root / "iscf_bsca_v1" if candidate else root
    return prefix / dataset / "h720_full" / f"seed{seed}"


def root_pair(
    config: dict[str, Any],
    seed: int,
) -> tuple[Path, Path]:
    contract = config["reference_contract"]
    if seed == 2021:
        return (
            Path(contract["seed2021_candidate_root"]),
            Path(contract["seed2021_equal_root"]),
        )
    return (
        Path(contract["confirmation_candidate_root"]),
        Path(contract["confirmation_equal_root"]),
    )


def mean_gain(
    rows: list[dict[str, Any]],
    field: str,
) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def grouped_means(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, float]:
    values = sorted({row[key] for row in rows}, key=str)
    return {
        str(value): mean_gain(
            [row for row in rows if row[key] == value],
            "mse_gain_percent",
        )
        for value in values
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    health: list[dict[str, Any]] = []

    for seed in SEEDS:
        candidate_root, reference_root = root_pair(config, seed)
        for dataset in DATASETS:
            candidate = run_dir(
                candidate_root,
                dataset,
                seed,
                candidate=True,
            )
            reference = run_dir(
                reference_root,
                dataset,
                seed,
                candidate=False,
            )
            required = (
                "checkpoint.pt",
                "training_log.csv",
                "metrics_by_target_horizon.csv",
                "effective_config.json",
                "initialization_contract.json",
                "model_diagnostics.json",
                "pcsd_validation_diagnostics.npz",
                "trained_invariants.json",
                "test_audit_metrics_by_target_horizon.csv",
                "test_audit_invariants.json",
                "pcsd_test_audit_diagnostics.npz",
            )
            missing = [name for name in required if not (candidate / name).is_file()]
            invariant = json.loads(
                (candidate / "test_audit_invariants.json").read_text(
                    encoding="utf-8"
                )
            )
            effective = json.loads(
                (candidate / "effective_config.json").read_text(
                    encoding="utf-8"
                )
            )
            candidate_initialization = json.loads(
                (candidate / "initialization_contract.json").read_text(
                    encoding="utf-8"
                )
            )
            reference_initialization = json.loads(
                (reference / "initialization_contract.json").read_text(
                    encoding="utf-8"
                )
            )
            initialization_hashes = [
                key for key in candidate_initialization if "hash" in key
            ]
            checkpoint_hash = sha256(candidate / "checkpoint.pt")
            audit.append(
                {
                    "seed": seed,
                    "dataset": dataset,
                    "missing_count": len(missing),
                    "checkpoint_sha256": checkpoint_hash,
                    "checkpoint_hash_matches_test_invariant": (
                        checkpoint_hash == invariant.get("checkpoint_sha256")
                    ),
                    "objective": effective["adapter"]["pcc_objective_mode"],
                    "initialization_paired_all_hashes": bool(
                        initialization_hashes
                        and all(
                            candidate_initialization.get(key)
                            == reference_initialization.get(key)
                            for key in initialization_hashes
                        )
                    ),
                    "test_split": invariant.get("evaluation_split"),
                    "uses_test_split": invariant.get("uses_test_split"),
                    "test_invariant_pass": invariant.get("pass"),
                    "checkpoint_retrained_for_candidate": invariant.get(
                        "checkpoint_retrained"
                    ),
                }
            )
            for arm, directory in (
                ("iscf_bsca_v1", candidate),
                ("iscf_equal", reference),
            ):
                health.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "dataset": dataset,
                        **internal_metrics(directory),
                    }
                )
            for split in ("val", "test"):
                candidate_metrics = selected_metrics(candidate, split)
                reference_metrics = selected_metrics(reference, split)
                for horizon in HORIZONS:
                    candidate_row = candidate_metrics[horizon]
                    reference_row = reference_metrics[horizon]
                    cells.append(
                        {
                            "split": split,
                            "seed": seed,
                            "dataset": dataset,
                            "horizon": horizon,
                            "candidate_mse": float(candidate_row["mse"]),
                            "reference_mse": float(reference_row["mse"]),
                            "mse_gain_percent": gain(
                                float(reference_row["mse"]),
                                float(candidate_row["mse"]),
                            ),
                            "candidate_mae": float(candidate_row["mae"]),
                            "reference_mae": float(reference_row["mae"]),
                            "mae_gain_percent": gain(
                                float(reference_row["mae"]),
                                float(candidate_row["mae"]),
                            ),
                        }
                    )

    test_rows = [row for row in cells if row["split"] == "test"]
    macro_mse = mean_gain(test_rows, "mse_gain_percent")
    macro_mae = mean_gain(test_rows, "mae_gain_percent")
    seed_means = grouped_means(test_rows, "seed")
    dataset_means = grouped_means(test_rows, "dataset")
    horizon_means = grouped_means(test_rows, "horizon")
    seed_wins = sum(value > 0.0 for value in seed_means.values())
    dataset_wins = sum(value > 0.0 for value in dataset_means.values())
    horizon_wins = sum(value > 0.0 for value in horizon_means.values())

    artifact_pass = all(
        row["missing_count"] == 0
        and row["checkpoint_hash_matches_test_invariant"]
        and row["initialization_paired_all_hashes"]
        and row["test_invariant_pass"]
        and row["checkpoint_retrained_for_candidate"]
        for row in audit
    )
    candidate_health = [
        row for row in health if row["arm"] == "iscf_bsca_v1"
    ]
    reference_health = [row for row in health if row["arm"] == "iscf_equal"]
    candidate_entropy = float(
        np.mean(
            [row["policy_normalized_entropy"] for row in candidate_health]
        )
    )
    candidate_pairwise = float(
        np.mean([row["pairwise_arm_l1"] for row in candidate_health])
    )
    reference_pairwise = float(
        np.mean([row["pairwise_arm_l1"] for row in reference_health])
    )
    candidate_oracle = float(
        np.mean([row["oracle_headroom_percent"] for row in candidate_health])
    )
    internal_gate = config["internal_health"]
    internal_pass = bool(
        all(row["all_finite"] for row in health)
        and candidate_entropy
        >= internal_gate["candidate_policy_normalized_entropy_min"]
        and candidate_pairwise / reference_pairwise
        >= internal_gate["candidate_to_equal_pairwise_arm_l1_ratio_min"]
        and candidate_oracle
        >= internal_gate["candidate_oracle_headroom_percent_min"]
    )

    direction = config["confirmation_gates"]["direction_robustness"]
    direction_pass = bool(
        macro_mse
        > direction["three_seed_macro_mse_gain_percent_min_exclusive"]
        and macro_mae
        > direction["three_seed_macro_mae_gain_percent_min_exclusive"]
        and seed_wins >= direction["seed_mse_wins_min"]
        and dataset_wins >= direction["dataset_mse_wins_min"]
        and horizon_wins >= direction["horizon_mse_wins_min"]
    )
    promotion = config["confirmation_gates"]["paper_core_promotion"]
    promotion_pass = bool(
        macro_mse >= promotion["three_seed_macro_mse_gain_percent_min"]
        and macro_mae
        > promotion["three_seed_macro_mae_gain_percent_min_exclusive"]
        and seed_wins >= promotion["seed_mse_wins_min"]
        and dataset_wins >= promotion["dataset_mse_wins_min"]
        and horizon_wins >= promotion["horizon_mse_wins_min"]
        and min(dataset_means.values())
        > promotion["minimum_dataset_mean_mse_gain_percent_exclusive"]
        and dataset_means["ETTm2"]
        >= promotion["ettm2_mean_mse_gain_percent_min"]
    )
    if promotion_pass and artifact_pass and internal_pass:
        decision = "passed_core_candidate_ready_for_paper_consolidation"
    elif direction_pass and artifact_pass and internal_pass:
        decision = (
            "three_seed_direction_supported_effect_size_or_heterogeneity_"
            "blocks_core_promotion"
        )
    else:
        decision = "three_seed_confirmation_not_supported_or_invalid"

    summary = {
        "candidate_version": config["candidate_version"],
        "test_access_date": "2026-07-22",
        "test_informed": True,
        "matrix_complete": len(test_rows) == 60 and len(audit) == 15,
        "macro_test_mse_gain_percent": macro_mse,
        "macro_test_mae_gain_percent": macro_mae,
        "test_cell_wins": int(
            sum(row["mse_gain_percent"] > 0.0 for row in test_rows)
        ),
        "seed_mse_means": seed_means,
        "dataset_mse_means": dataset_means,
        "horizon_mse_means": horizon_means,
        "seed_mse_wins": seed_wins,
        "dataset_mse_wins": dataset_wins,
        "horizon_mse_wins": horizon_wins,
        "artifact_protocol_nonmutation_pass": artifact_pass,
        "internal_health_pass": internal_pass,
        "candidate_policy_normalized_entropy": candidate_entropy,
        "candidate_to_equal_pairwise_arm_l1_ratio": (
            candidate_pairwise / reference_pairwise
        ),
        "candidate_oracle_headroom_percent": candidate_oracle,
        "direction_robustness_pass": direction_pass,
        "paper_core_promotion_pass": promotion_pass,
        "decision": decision,
    }
    write_rows(args.output_dir / "run_audit.csv", audit)
    write_rows(args.output_dir / "comparison_cells.csv", cells)
    write_rows(args.output_dir / "internal_health.csv", health)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
