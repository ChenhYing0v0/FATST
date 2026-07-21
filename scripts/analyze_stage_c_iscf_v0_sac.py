#!/usr/bin/env python3
"""Analyze the frozen three-seed ISCF-v0 Scope Attribution Confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_iscf_v0_scope_attribution_confirmation.json"),
    )
    parser.add_argument("--new-root", type=Path)
    parser.add_argument("--seed2021-root", type=Path)
    parser.add_argument("--fcc-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
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


def source_for(
    arm: str,
    seed: int,
    new_root: Path,
    seed2021_root: Path,
    fcc_root: Path,
) -> tuple[Path, str, str]:
    if arm == "iscf_random_partition":
        return new_root, arm, "new"
    if arm == "iscf_q1_wide" and seed in {2022, 2023}:
        return new_root, arm, "new"
    if seed == 2021:
        aliases = {
            "iscf_v0": "siff_independent_equal",
            "iscf_q1_wide": "siff_q1_wide_equal",
            "a6_full": "a6_full",
        }
        return seed2021_root, aliases[arm], "historical_seed2021"
    aliases = {"iscf_v0": "siff_independent_equal", "a6_full": "a6_full"}
    return fcc_root, aliases[arm], "historical_fcc"


def run_dir(root: Path, arm: str, dataset: str, seed: int) -> Path:
    return root / arm / dataset / "h720_full" / f"seed{seed}"


def reference_audit_lookup(config: dict[str, Any]) -> dict[tuple[str, str, int], str]:
    lookup: dict[tuple[str, str, int], str] = {}
    for source in config["artifact_sources"]:
        if source["kind"] != "historical":
            continue
        aliases = source["arm_aliases"]
        for row in read_csv(Path(source["run_audit"])):
            for arm in source["arms"]:
                if (
                    row["arm"] == aliases[arm]
                    and int(row.get("seed", source["seeds"][0]))
                    in source["seeds"]
                    and row["status"] == "ok"
                    and row["protocol_pass"] == "True"
                ):
                    lookup[(arm, row["dataset"], int(row.get("seed", source["seeds"][0])))] = row[
                        "checkpoint_sha256"
                    ]
    return lookup


def load_run(
    arm: dict[str, Any],
    dataset: str,
    seed: int,
    roots: tuple[Path, Path, Path],
    config: dict[str, Any],
    reference_hashes: dict[tuple[str, str, int], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root, source_arm, source = source_for(arm["id"], seed, *roots)
    directory = run_dir(root, source_arm, dataset, seed)
    required = {
        "validation": directory / "metrics_by_target_horizon.csv",
        "test": directory / "test_audit_metrics_by_target_horizon.csv",
        "invariants": directory / "test_audit_invariants.json",
        "effective": directory / "effective_config.json",
        "initialization": directory / "initialization_contract.json",
        "diagnostics": directory / "model_diagnostics.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return [], [], {
            "dataset": dataset,
            "arm": arm["id"],
            "seed": seed,
            "source": source,
            "status": "missing",
            "missing": ",".join(missing),
            "run_dir": str(directory),
        }

    invariants = json.loads(required["invariants"].read_text(encoding="utf-8"))
    effective = json.loads(required["effective"].read_text(encoding="utf-8"))["adapter"]
    initialization = json.loads(
        required["initialization"].read_text(encoding="utf-8")
    )
    expected_hash = reference_hashes.get((arm["id"], dataset, seed))
    protocol_pass = bool(
        effective["dataset"] == dataset
        and int(effective["seed"]) == seed
        and effective["readout_mode"] == arm["readout_mode"]
        and effective["pcc_objective_mode"] == arm["objective_mode"]
        and effective["validation_horizons"]
        == config["training"]["validation_horizons"]
        and effective["checkpoint_policy"] == "best-val"
        and effective["final_evaluation_split"] == "val"
        and effective.get("pcsd_partition", "control") == arm["partition"]
        and invariants.get("pass") is True
        and invariants.get("evaluation_split") == "test"
        and invariants.get("uses_test_split") is True
        and invariants.get("test_access_authorized") is True
        and (expected_hash is None or invariants.get("checkpoint_sha256") == expected_hash)
    )

    def metrics_from(path: Path, split: str) -> list[dict[str, Any]]:
        lookup = {int(row["target_horizon"]): row for row in read_csv(path)}
        rows = []
        for horizon in config["matrix"]["horizons"]:
            mse = float(lookup[horizon]["mse"])
            mae = float(lookup[horizon]["mae"])
            if not math.isfinite(mse) or not math.isfinite(mae):
                raise ValueError(f"non-finite {split} metric: {directory} H{horizon}")
            rows.append(
                {
                    "split": split,
                    "dataset": dataset,
                    "arm": arm["id"],
                    "seed": seed,
                    "horizon": horizon,
                    "mse": mse,
                    "mae": mae,
                    "source": source,
                }
            )
        return rows

    return (
        metrics_from(required["test"], "test"),
        metrics_from(required["validation"], "validation"),
        {
            "dataset": dataset,
            "arm": arm["id"],
            "seed": seed,
            "source": source,
            "status": "ok" if protocol_pass else "audit_fail",
            "protocol_pass": protocol_pass,
            "checkpoint_sha256": invariants.get("checkpoint_sha256", ""),
            "pcsd_initialization_hash": initialization.get(
                "pcsd_initialization_hash", ""
            ),
            "encoder_initialization_hash": initialization.get(
                "encoder_initialization_hash", ""
            ),
            "pcsd_partition_hash": initialization.get("pcsd_partition_hash", ""),
            "active_forward_parameters": json.loads(
                required["diagnostics"].read_text(encoding="utf-8")
            ).get("active_forward_parameters", ""),
            "run_dir": str(directory),
        },
    )


def comparison_rows(
    metrics: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {
        (row["dataset"], row["arm"], row["seed"], row["horizon"]): row
        for row in metrics
    }
    cells: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    comparisons = [
        *config["primary_comparisons"],
        {
            "id": "iscf_over_a6_full_context",
            "candidate": "iscf_v0",
            "reference": "a6_full",
            "tests": "historical package effectiveness context",
            "mse_macro_gain_percent_min": 0.0,
        },
    ]
    for comparison in comparisons:
        for metric in config["matrix"]["metrics"]:
            gains: list[float] = []
            by_dataset: dict[str, list[float]] = {}
            by_horizon: dict[int, list[float]] = {}
            by_seed: dict[int, list[float]] = {}
            for dataset in config["datasets"]:
                for seed in config["seeds"]:
                    for horizon in config["matrix"]["horizons"]:
                        candidate = lookup[
                            (dataset, comparison["candidate"], seed, horizon)
                        ]
                        reference = lookup[
                            (dataset, comparison["reference"], seed, horizon)
                        ]
                        gain = 100.0 * (
                            1.0 - float(candidate[metric]) / float(reference[metric])
                        )
                        gains.append(gain)
                        by_dataset.setdefault(dataset, []).append(gain)
                        by_horizon.setdefault(horizon, []).append(gain)
                        by_seed.setdefault(seed, []).append(gain)
                        cells.append(
                            {
                                "comparison": comparison["id"],
                                "metric": metric,
                                "dataset": dataset,
                                "seed": seed,
                                "horizon": horizon,
                                "gain_percent": gain,
                                "candidate_value": candidate[metric],
                                "reference_value": reference[metric],
                            }
                        )
            summaries.append(
                {
                    "comparison": comparison["id"],
                    "metric": metric,
                    "candidate": comparison["candidate"],
                    "reference": comparison["reference"],
                    "macro_gain_percent": mean(gains),
                    "cell_wins": sum(value > 0.0 for value in gains),
                    "dataset_wins": sum(
                        mean(values) > 0.0 for values in by_dataset.values()
                    ),
                    "horizon_wins": sum(
                        mean(values) > 0.0 for values in by_horizon.values()
                    ),
                    "positive_seed_macros": sum(
                        mean(values) > 0.0 for values in by_seed.values()
                    ),
                }
            )
    return cells, summaries


def internal_health_rows(
    audits: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    lookup = {(row["dataset"], row["arm"], row["seed"]): row for row in audits}
    rows = []
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            canonical = lookup[(dataset, "iscf_v0", seed)]
            random = lookup[(dataset, "iscf_random_partition", seed)]
            q1 = lookup[(dataset, "iscf_q1_wide", seed)]
            iscf_parameters = int(canonical["active_forward_parameters"])
            q1_parameters = int(q1["active_forward_parameters"])
            gap = 100.0 * (iscf_parameters - q1_parameters) / iscf_parameters
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "canonical_random_encoder_init_match": canonical[
                        "encoder_initialization_hash"
                    ]
                    == random["encoder_initialization_hash"],
                    "canonical_random_readout_init_match": canonical[
                        "pcsd_initialization_hash"
                    ]
                    == random["pcsd_initialization_hash"],
                    "canonical_random_partition_hash_differs": canonical[
                        "pcsd_partition_hash"
                    ]
                    != random["pcsd_partition_hash"],
                    "canonical_random_parameter_match": iscf_parameters
                    == int(random["active_forward_parameters"]),
                    "q1_active_parameter_gap_percent": gap,
                    "q1_gap_matches_preregistered": abs(
                        gap - config["q1_active_parameter_audit"][dataset]
                    )
                    <= 1e-10,
                }
            )
    return rows


def decide(
    summaries: list[dict[str, Any]],
    health: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    lookup = {(row["comparison"], row["metric"]): row for row in summaries}
    gate = config["gates"]
    comparison_results: dict[str, dict[str, Any]] = {}
    for comparison in config["primary_comparisons"]:
        mse = lookup[(comparison["id"], "mse")]
        mae = lookup[(comparison["id"], "mae")]
        passed = bool(
            mse["macro_gain_percent"]
            >= comparison["mse_macro_gain_percent_min"]
            and mse["dataset_wins"] >= gate["mse_dataset_wins_min"]
            and mse["horizon_wins"] >= gate["mse_horizon_wins_min"]
            and mse["positive_seed_macros"]
            >= gate["mse_positive_seed_macros_min"]
            and mae["macro_gain_percent"]
            > gate["mae_macro_gain_percent_min_exclusive"]
        )
        comparison_results[comparison["id"]] = {
            "pass": passed,
            "mse_macro_gain_percent": mse["macro_gain_percent"],
            "mae_macro_gain_percent": mae["macro_gain_percent"],
            "dataset_wins": mse["dataset_wins"],
            "horizon_wins": mse["horizon_wins"],
            "positive_seed_macros": mse["positive_seed_macros"],
        }
    health_pass = bool(
        health
        and all(
            row["canonical_random_encoder_init_match"]
            and row["canonical_random_readout_init_match"]
            and row["canonical_random_partition_hash_differs"]
            and row["canonical_random_parameter_match"]
            and row["q1_gap_matches_preregistered"]
            for row in health
        )
    )
    q1_pass = comparison_results["independent_over_q1_wide"]["pass"]
    random_pass = comparison_results["canonical_over_random_partition"]["pass"]
    if not health_pass:
        decision = "diagnostic_invalid_for_direction_rejection_repair_exact_protocol"
    elif q1_pass and random_pass:
        decision = config["failure_attribution"]["both_pass"]
    elif not q1_pass and not random_pass:
        decision = "capacity_and_temporal_scope_attribution_both_not_supported"
    elif not q1_pass:
        decision = config["failure_attribution"]["q1_wide_fails"]
    else:
        decision = config["failure_attribution"]["random_partition_fails"]
    return {
        "decision": decision,
        "comparison_results": comparison_results,
        "internal_health_pass": health_pass,
        "paper_core_promoted": False,
        "modern_baselines_authorized": False,
    }


def synthetic_smoke(config: dict[str, Any]) -> None:
    metrics = []
    factors = {
        "iscf_v0": 0.98,
        "iscf_q1_wide": 1.0,
        "iscf_random_partition": 0.995,
        "a6_full": 1.01,
    }
    for arm in config["arms"]:
        for dataset in config["datasets"]:
            for seed in config["seeds"]:
                for horizon in config["matrix"]["horizons"]:
                    metrics.append(
                        {
                            "dataset": dataset,
                            "arm": arm["id"],
                            "seed": seed,
                            "horizon": horizon,
                            "mse": factors[arm["id"]],
                            "mae": factors[arm["id"]],
                        }
                    )
    cells, summaries = comparison_rows(metrics, config)
    health = [
        {
            "canonical_random_encoder_init_match": True,
            "canonical_random_readout_init_match": True,
            "canonical_random_partition_hash_differs": True,
            "canonical_random_parameter_match": True,
            "q1_gap_matches_preregistered": True,
        }
    ]
    result = decide(summaries, health, config)
    if (
        len(metrics) != 240
        or len(cells) != 360
        or len(summaries) != 6
        or result["decision"]
        != "iscf_scope_architecture_supported_pending_modern_baselines"
    ):
        raise RuntimeError("SAC analyzer synthetic smoke failed")
    print("iscf_v0_sac_analyzer_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.synthetic_smoke:
        synthetic_smoke(config)
        return
    roots = (args.new_root, args.seed2021_root, args.fcc_root)
    if any(root is None for root in roots) or args.output_dir is None:
        raise ValueError(
            "new-root, seed2021-root, fcc-root, and output-dir are required"
        )
    concrete_roots = (roots[0], roots[1], roots[2])
    reference_hashes = reference_audit_lookup(config)
    test_metrics: list[dict[str, Any]] = []
    validation_metrics: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for arm in config["arms"]:
        for dataset in config["datasets"]:
            for seed in config["seeds"]:
                test, validation, audit = load_run(
                    arm,
                    dataset,
                    seed,
                    concrete_roots,
                    config,
                    reference_hashes,
                )
                test_metrics.extend(test)
                validation_metrics.extend(validation)
                audits.append(audit)
    if len(test_metrics) != config["matrix"]["effective_official_test_cells"]:
        raise RuntimeError("SAC formal matrix is incomplete")
    if any(row["status"] != "ok" for row in audits):
        raise RuntimeError("one or more SAC run audits failed")
    cells, summaries = comparison_rows(test_metrics, config)
    validation_cells, validation_summaries = comparison_rows(
        validation_metrics, config
    )
    health = internal_health_rows(audits, config)
    decision = decide(summaries, health, config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "test_metrics.csv", test_metrics)
    write_csv(output_dir / "validation_metrics.csv", validation_metrics)
    write_csv(output_dir / "run_audit.csv", audits)
    write_csv(output_dir / "comparison_cells.csv", cells)
    write_csv(output_dir / "comparison_summary.csv", summaries)
    write_csv(output_dir / "validation_comparison_cells.csv", validation_cells)
    write_csv(
        output_dir / "validation_comparison_summary.csv", validation_summaries
    )
    write_csv(output_dir / "internal_health.csv", health)
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"iscf_v0_sac_analysis={decision['decision']}")


if __name__ == "__main__":
    main()
