#!/usr/bin/env python3
"""Analyze the validation-only ISCF-SPS Step 7B matched matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_iscf_sps_step7b.json"),
    )
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
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


def finite_numeric_arrays(arrays: Any) -> bool:
    return all(
        np.isfinite(value).all()
        for name in arrays.files
        if np.issubdtype((value := arrays[name]).dtype, np.number)
    )


def expected_projection_contract(
    config: dict[str, Any],
    projection_mode: str,
    mode_rank: int,
) -> tuple[list[int], list[int]]:
    contract = config["projection_contract"]
    if projection_mode == "identity":
        return contract["identity_ranks"], contract["identity_degrees"]
    matched = contract["expected_by_mode_rank"][str(mode_rank)]
    return matched[f"{projection_mode}_ranks"], matched[
        f"{projection_mode}_degrees"
    ]


def load_run(
    root: Path,
    arm: dict[str, Any],
    dataset: str,
    seed: int,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
    directory = run_dir(root, arm["id"], dataset, seed)
    required = {
        name: directory / name for name in config["artifact_schema"]["required"]
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return [], {
            "dataset": dataset,
            "arm": arm["id"],
            "status": "missing",
            "missing": ",".join(missing),
            "run_dir": str(directory),
        }, None, []

    effective = json.loads(required["effective_config.json"].read_text())
    diagnostics = json.loads(required["model_diagnostics.json"].read_text())
    invariants = json.loads(required["trained_invariants.json"].read_text())
    adapter = effective["adapter"]
    expected_rank = config["matched_ranks"][dataset][arm["rank_rule"]]
    protocol_pass = bool(
        adapter["dataset"] == dataset
        and adapter["seed"] == seed
        and adapter["readout_mode"] == arm["readout_mode"]
        and adapter["sps_projection_mode"] == arm["projection_mode"]
        and adapter["pcsd_partition"] == arm["partition"]
        and adapter["pcsd_mode_rank"] == expected_rank
        and adapter["validation_horizons"] == config["training"]["validation_horizons"]
        and adapter["final_evaluation_split"] == "val"
        and adapter["pcc_objective_mode"] == "equal_skill"
        and invariants.get("pass") is True
        and invariants.get("evaluation_split") == "val"
        and invariants.get("uses_test_split") is False
        and invariants.get("sps_diagnostics_present") is True
    )

    lookup = {
        int(row["target_horizon"]): row
        for row in read_csv(required["metrics_by_target_horizon.csv"])
    }
    metrics = []
    for horizon in config["matrix"]["horizons"]:
        row = lookup[horizon]
        mse, mae = float(row["mse"]), float(row["mae"])
        if not math.isfinite(mse) or not math.isfinite(mae):
            raise ValueError(f"non-finite validation metric: {directory} H{horizon}")
        metrics.append(
            {
                "dataset": dataset,
                "arm": arm["id"],
                "horizon": horizon,
                "mse": mse,
                "mae": mae,
                "seed": seed,
                "split": "validation",
            }
        )

    arrays = np.load(required["pcsd_validation_diagnostics.npz"])
    needed = {
        "probe_arms",
        "probe_fused",
        "probe_targets",
        "probe_direct_policy",
        "arm_row_bin_mse",
        "probe_sps_raw_arms",
        "probe_sps_projected_arms",
        "probe_sps_removed_arms",
    }
    missing_arrays = sorted(needed - set(arrays.files))
    if missing_arrays:
        raise ValueError(f"missing SPS diagnostics {missing_arrays}: {directory}")

    arms = arrays["probe_arms"].astype(np.float64)
    fused = arrays["probe_fused"].astype(np.float64)
    target = arrays["probe_targets"].astype(np.float64)
    policy = arrays["probe_direct_policy"].astype(np.float64)
    raw = arrays["probe_sps_raw_arms"].astype(np.float64)
    projected = arrays["probe_sps_projected_arms"].astype(np.float64)
    removed = arrays["probe_sps_removed_arms"].astype(np.float64)

    target_rms = max(float(np.sqrt(np.mean(target**2))), 1e-12)
    pairwise_values = [
        float(np.sqrt(np.mean((arms[:, left] - arms[:, right]) ** 2)))
        for left, right in combinations(range(arms.shape[1]), 2)
    ]
    arm_mse = np.mean((arms - target[:, None, :]) ** 2, axis=-1)
    fused_mse = np.mean((fused - target) ** 2, axis=-1)
    oracle = np.min(arm_mse, axis=1)
    oracle_headroom = float(
        100.0 * np.mean(1.0 - oracle / np.maximum(fused_mse, 1e-12))
    )
    entropy = -np.sum(policy * np.log(np.maximum(policy, 1e-12)), axis=-1)
    entropy = float(np.mean(entropy) / math.log(policy.shape[-1]))
    bin_arm_mse = arrays["arm_row_bin_mse"].mean(axis=0)
    winners = sorted(set(np.argmin(bin_arm_mse, axis=-1).tolist()))
    raw_rms = np.sqrt(np.mean(raw**2, axis=(0, 2)))
    projected_rms = np.sqrt(np.mean(projected**2, axis=(0, 2)))
    removed_rms = np.sqrt(np.mean(removed**2, axis=(0, 2)))
    retained = projected_rms / np.maximum(raw_rms, 1e-12)
    removed_ratio = float(
        np.sqrt(np.mean(removed**2)) / max(np.sqrt(np.mean(raw**2)), 1e-12)
    )

    expected_ranks, expected_degrees = expected_projection_contract(
        config,
        arm["projection_mode"],
        expected_rank,
    )
    health = {
        "dataset": dataset,
        "arm": arm["id"],
        "all_finite": finite_numeric_arrays(arrays),
        "pairwise_normalized_rms": float(mean(pairwise_values) / target_rms),
        "min_pairwise_normalized_rms": float(min(pairwise_values) / target_rms),
        "oracle_headroom_percent": oracle_headroom,
        "policy_normalized_entropy": entropy,
        "scope_winner_count": len(winners),
        "winning_scope_indices": ",".join(map(str, winners)),
        "removed_to_raw_rms": removed_ratio,
        "projection_ranks_match": diagnostics.get("sps_projection_ranks") == expected_ranks,
        "projected_degrees_match": diagnostics.get("sps_projected_degrees") == expected_degrees,
    }
    retention = [
        {
            "dataset": dataset,
            "arm": arm["id"],
            "scope": scope,
            "raw_rms": float(raw_rms[index]),
            "projected_rms": float(projected_rms[index]),
            "removed_rms": float(removed_rms[index]),
            "retained_rms_ratio": float(retained[index]),
        }
        for index, scope in enumerate(config["scales"])
    ]
    audit = {
        "dataset": dataset,
        "arm": arm["id"],
        "status": "ok" if protocol_pass else "audit_fail",
        "protocol_pass": protocol_pass,
        "run_dir": str(directory),
    }
    return metrics, audit, health, retention


def comparison_rows(
    metrics: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {
        (row["dataset"], row["arm"], row["horizon"]): row for row in metrics
    }
    cells, summaries = [], []
    for comparison in config["comparisons"]:
        for metric in config["matrix"]["metrics"]:
            gains = []
            by_dataset: dict[str, list[float]] = {}
            by_horizon: dict[int, list[float]] = {}
            for dataset in config["datasets"]:
                for horizon in config["matrix"]["horizons"]:
                    candidate = lookup[(dataset, comparison["candidate"], horizon)]
                    reference = lookup[(dataset, comparison["reference"], horizon)]
                    gain = 100.0 * (
                        1.0 - float(candidate[metric]) / float(reference[metric])
                    )
                    gains.append(gain)
                    by_dataset.setdefault(dataset, []).append(gain)
                    by_horizon.setdefault(horizon, []).append(gain)
                    cells.append(
                        {
                            "comparison": comparison["id"],
                            "layer": comparison["layer"],
                            "metric": metric,
                            "dataset": dataset,
                            "horizon": horizon,
                            "gain_percent": gain,
                            "candidate_value": candidate[metric],
                            "reference_value": reference[metric],
                        }
                    )
            summaries.append(
                {
                    "comparison": comparison["id"],
                    "layer": comparison["layer"],
                    "metric": metric,
                    "macro_gain_percent": mean(gains),
                    "cell_wins": sum(value > 0 for value in gains),
                    "dataset_wins": sum(mean(values) > 0 for values in by_dataset.values()),
                    "horizon_wins": sum(mean(values) > 0 for values in by_horizon.values()),
                    "max_dataset_degradation_percent": max(
                        max(0.0, -mean(values)) for values in by_dataset.values()
                    ),
                }
            )
    return cells, summaries


def aggregate_health(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    if not selected:
        raise ValueError(f"missing health rows for {arm}")
    winner_indices = set()
    for row in selected:
        winner_indices.update(
            int(value) for value in row["winning_scope_indices"].split(",") if value
        )
    return {
        "arm": arm,
        "all_finite": all(row["all_finite"] for row in selected),
        "pairwise_normalized_rms": mean(
            row["pairwise_normalized_rms"] for row in selected
        ),
        "min_pairwise_normalized_rms": min(
            row["min_pairwise_normalized_rms"] for row in selected
        ),
        "oracle_headroom_percent": mean(
            row["oracle_headroom_percent"] for row in selected
        ),
        "policy_normalized_entropy": mean(
            row["policy_normalized_entropy"] for row in selected
        ),
        "scope_winner_count": len(winner_indices),
        "removed_to_raw_rms": mean(row["removed_to_raw_rms"] for row in selected),
        "projection_contract_pass": all(
            row["projection_ranks_match"] and row["projected_degrees_match"]
            for row in selected
        ),
    }


def decide(
    summaries: list[dict[str, Any]],
    aggregate: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    summary = {(row["comparison"], row["metric"]): row for row in summaries}
    gates = config["validation_gates"]
    health_gate = config["internal_health"]
    primary_mse = summary[(gates["primary_comparison"], "mse")]
    primary_mae = summary[(gates["primary_comparison"], "mae")]
    global_mse = summary[("scope_specificity_vs_global", "mse")]
    random_mse = summary[("canonical_binding_vs_random", "mse")]
    candidate = aggregate["sps_scope_canonical"]
    identity = aggregate["sps_identity_canonical"]
    performance_pass = bool(
        primary_mse["macro_gain_percent"] >= gates["macro_mse_gain_percent_min"]
        and primary_mse["dataset_wins"] >= gates["dataset_wins_min"]
        and primary_mse["horizon_wins"] >= gates["horizon_wins_min"]
        and primary_mae["macro_gain_percent"] >= gates["macro_mae_gain_percent_min"]
    )
    health_pass = bool(
        candidate["all_finite"]
        and candidate["projection_contract_pass"]
        and candidate["removed_to_raw_rms"]
        >= health_gate["candidate_removed_to_raw_rms_min"]
        and candidate["min_pairwise_normalized_rms"]
        >= health_gate["candidate_pairwise_normalized_rms_min"]
        and candidate["oracle_headroom_percent"]
        >= health_gate["candidate_oracle_headroom_percent_min"]
        and candidate["policy_normalized_entropy"]
        >= health_gate["candidate_policy_normalized_entropy_min"]
        and candidate["scope_winner_count"]
        >= health_gate["candidate_scope_winner_count_min"]
        and candidate["pairwise_normalized_rms"]
        >= health_gate["candidate_pairwise_distance_vs_identity_ratio_min"]
        * identity["pairwise_normalized_rms"]
        and candidate["oracle_headroom_percent"]
        >= identity["oracle_headroom_percent"]
        - health_gate["candidate_oracle_headroom_vs_identity_tolerance_percent"]
    )
    global_attribution_pass = global_mse["macro_gain_percent"] > 0.0
    random_binding_supported = random_mse["macro_gain_percent"] > 0.0
    if performance_pass and health_pass and global_attribution_pass:
        decision = "validation_supported_pending_formal_test_design"
        if not random_binding_supported:
            decision = "performance_partial_pass_scope_binding_unresolved"
    elif performance_pass:
        decision = "performance_partial_pass_specialization_or_attribution_unresolved"
    else:
        decision = "exact_SPS_v0_validation_not_supported_rollback_step4"
    return {
        "decision": decision,
        "performance_pass": performance_pass,
        "internal_health_pass": health_pass,
        "global_attribution_pass": global_attribution_pass,
        "random_binding_supported": random_binding_supported,
        "random_partition_role": "attribution_only_not_direction_rejection",
        "test_accessed": False,
    }


def synthetic_smoke(config: dict[str, Any]) -> None:
    summaries = []
    for comparison in config["comparisons"]:
        for metric in config["matrix"]["metrics"]:
            summaries.append(
                {
                    "comparison": comparison["id"],
                    "metric": metric,
                    "macro_gain_percent": 1.0,
                    "dataset_wins": 5,
                    "horizon_wins": 4,
                }
            )
    base = {
        "all_finite": True,
        "projection_contract_pass": True,
        "removed_to_raw_rms": 0.2,
        "min_pairwise_normalized_rms": 0.1,
        "pairwise_normalized_rms": 0.2,
        "oracle_headroom_percent": 5.0,
        "policy_normalized_entropy": 0.8,
        "scope_winner_count": 4,
    }
    aggregate = {
        arm["id"]: {**base, "arm": arm["id"]} for arm in config["arms"]
    }
    result = decide(summaries, aggregate, config)
    if result["decision"] != "validation_supported_pending_formal_test_design":
        raise RuntimeError(result)
    print("iscf_sps_analyzer_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.synthetic_smoke:
        synthetic_smoke(config)
        return
    if args.root is None or args.output_dir is None:
        raise ValueError("--root and --output-dir are required")

    metrics: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    health: list[dict[str, Any]] = []
    retention: list[dict[str, Any]] = []
    for arm in config["arms"]:
        for dataset in config["datasets"]:
            run_metrics, audit, run_health, run_retention = load_run(
                args.root, arm, dataset, args.seed, config
            )
            metrics.extend(run_metrics)
            audits.append(audit)
            if run_health is not None:
                health.append(run_health)
            retention.extend(run_retention)

    if any(row["status"] != "ok" for row in audits):
        missing = [row for row in audits if row["status"] != "ok"]
        raise RuntimeError(f"incomplete or invalid SPS matrix: {missing}")
    if len(metrics) != config["matrix"]["validation_cells"]:
        raise RuntimeError(f"unexpected validation cells: {len(metrics)}")

    cells, summaries = comparison_rows(metrics, config)
    aggregate_rows = [
        aggregate_health(health, arm["id"]) for arm in config["arms"]
    ]
    aggregate = {row["arm"]: row for row in aggregate_rows}
    decision = decide(summaries, aggregate, config)
    decision.update(
        {
            "candidate_version": config["candidate_version"],
            "matrix_complete": True,
            "runs": len(audits),
            "validation_cells": len(metrics),
            "formal_test_authorized": False,
            "failure_attribution": config["failure_attribution"],
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "validation_metrics.csv", metrics)
    write_csv(args.output_dir / "comparison_cells.csv", cells)
    write_csv(args.output_dir / "comparison_summary.csv", summaries)
    write_csv(args.output_dir / "run_audit.csv", audits)
    write_csv(args.output_dir / "specialization_health.csv", health)
    write_csv(args.output_dir / "specialization_aggregate.csv", aggregate_rows)
    write_csv(args.output_dir / "projection_retention.csv", retention)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"iscf_sps_validation_analysis={decision['decision']} "
        f"runs={len(audits)} test_accessed=false"
    )


if __name__ == "__main__":
    main()
