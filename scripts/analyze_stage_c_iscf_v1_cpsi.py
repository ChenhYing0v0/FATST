#!/usr/bin/env python3
"""Analyze the frozen ISCF-v1-CPSI official-test matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_iscf_v1_cpsi_step7b.json"),
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


def load_run(
    arm: dict[str, Any],
    dataset: str,
    seed: int,
    new_root: Path,
    reference_root: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    root = reference_root if arm["source"] == "reused_reference" else new_root
    directory = run_dir(
        root,
        arm.get("source_arm", arm["id"]),
        dataset,
        seed,
    )
    required = {
        "metrics": directory / "test_audit_metrics_by_target_horizon.csv",
        "invariants": directory / "test_audit_invariants.json",
        "effective": directory / "effective_config.json",
        "training": directory / "training_log.csv",
    }
    if arm["source"] == "new_training":
        required["diagnostics"] = directory / "pcsd_test_audit_diagnostics.npz"
        required["model_diagnostics"] = directory / "model_diagnostics.json"
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return [], {
            "dataset": dataset,
            "arm": arm["id"],
            "status": "missing",
            "missing": ",".join(missing),
            "run_dir": str(directory),
        }, None
    invariants = json.loads(required["invariants"].read_text(encoding="utf-8"))
    effective = json.loads(required["effective"].read_text(encoding="utf-8"))
    adapter = effective["adapter"]
    expected_hash = None
    if arm["source"] == "reused_reference":
        expected_hash = config["reference_contract"]["checkpoint_sha256"][
            arm["id"]
        ][dataset]
    metric_lookup = {
        int(row["target_horizon"]): row for row in read_csv(required["metrics"])
    }
    metrics = []
    for horizon in config["matrix"]["horizons"]:
        row = metric_lookup[horizon]
        mse, mae = float(row["mse"]), float(row["mae"])
        if not math.isfinite(mse) or not math.isfinite(mae):
            raise ValueError(f"non-finite metric: {directory} H{horizon}")
        metrics.append(
            {
                "dataset": dataset,
                "arm": arm["id"],
                "horizon": horizon,
                "mse": mse,
                "mae": mae,
                "seed": seed,
                "source": arm["source"],
            }
        )
    protocol_pass = bool(
        adapter["dataset"] == dataset
        and adapter["seed"] == seed
        and adapter["readout_mode"] == arm["readout_mode"]
        and adapter["pcc_objective_mode"] == arm["objective_mode"]
        and adapter["validation_horizons"]
        == config["training"]["validation_horizons"]
        and adapter["checkpoint_policy"] == "best-val"
        and adapter["final_evaluation_split"] == "val"
        and invariants.get("pass") is True
        and invariants.get("evaluation_split") == "test"
        and invariants.get("uses_test_split") is True
        and invariants.get("test_access_authorized") is True
        and (
            expected_hash is None
            or invariants.get("checkpoint_sha256") == expected_hash
        )
    )
    health = None
    if arm["source"] == "new_training":
        arrays = np.load(required["diagnostics"])
        model_health = json.loads(
            required["model_diagnostics"].read_text(encoding="utf-8")
        )
        health = {
            "dataset": dataset,
            "arm": arm["id"],
            "message_rms": float(np.mean(arrays["probe_cpsi_message_rms"])),
            "latent_rms": float(np.mean(arrays["probe_cpsi_latent_rms"])),
            "common_rms": float(np.mean(arrays["probe_cpsi_common_rms"])),
            "private_rms": float(np.mean(arrays["probe_cpsi_private_rms"])),
            "output_projection_norm": float(
                model_health["cpsi_output_projection_norm"]
            ),
            "all_finite": bool(
                all(np.isfinite(arrays[name]).all() for name in arrays.files)
            ),
        }
    return metrics, {
        "dataset": dataset,
        "arm": arm["id"],
        "source": arm["source"],
        "status": "ok" if protocol_pass else "audit_fail",
        "protocol_pass": protocol_pass,
        "checkpoint_sha256": invariants.get("checkpoint_sha256", ""),
        "run_dir": str(directory),
    }, health


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
            dataset_gains: dict[str, list[float]] = {}
            for dataset in config["datasets"]:
                for horizon in config["matrix"]["horizons"]:
                    candidate = lookup[(dataset, comparison["candidate"], horizon)]
                    reference = lookup[(dataset, comparison["reference"], horizon)]
                    gain = 100.0 * (
                        1.0 - float(candidate[metric]) / float(reference[metric])
                    )
                    gains.append(gain)
                    dataset_gains.setdefault(dataset, []).append(gain)
                    cells.append(
                        {
                            "comparison": comparison["id"],
                            "layer": comparison["layer"],
                            "metric": metric,
                            "candidate": comparison["candidate"],
                            "reference": comparison["reference"],
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
                    "candidate": comparison["candidate"],
                    "reference": comparison["reference"],
                    "macro_gain_percent": mean(gains),
                    "cell_wins": sum(value > 0.0 for value in gains),
                    "dataset_wins": sum(
                        mean(values) > 0.0 for values in dataset_gains.values()
                    ),
                    "max_dataset_degradation_percent": max(
                        max(0.0, -mean(values))
                        for values in dataset_gains.values()
                    ),
                }
            )
    return cells, summaries


def decide(
    summaries: list[dict[str, Any]],
    health_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    lookup = {(row["comparison"], row["metric"]): row for row in summaries}
    severity = config["severity"]
    primary = severity["initial_test_supported"]
    mse = lookup[(primary["comparison_id"], "mse")]
    mae = lookup[(primary["comparison_id"], "mae")]
    supported = bool(
        mse["macro_gain_percent"] >= primary["macro_mse_gain_percent_min"]
        and mse["dataset_wins"] >= primary["dataset_wins_min"]
        and mse["cell_wins"] >= primary["cell_wins_min"]
        and mae["macro_gain_percent"] >= primary["macro_mae_gain_percent_min"]
    )
    material = severity["material_test_negative"]
    material_negative = bool(
        mse["macro_gain_percent"] <= material["macro_mse_gain_percent_max"]
        and 5 - mse["dataset_wins"] >= material["dataset_losses_min"]
        or mse["max_dataset_degradation_percent"]
        >= material["single_dataset_degradation_percent_catastrophic"]
    )
    control_band = severity["control_attribution_band_percent"]
    control_mse = [
        row
        for row in summaries
        if row["layer"] == "matched_mechanism_attribution"
        and row["metric"] == "mse"
    ]
    attribution_supported = all(
        row["macro_gain_percent"] > control_band for row in control_mse
    )
    health_ok = bool(
        health_rows
        and all(
            row["all_finite"]
            and row["message_rms"] >= config["internal_health"]["message_rms_min"]
            and row["latent_rms"] >= config["internal_health"]["latent_rms_min"]
            and row["output_projection_norm"]
            >= config["internal_health"]["output_projection_norm_min"]
            for row in health_rows
        )
    )
    if not health_ok:
        state = "diagnostic_invalid_for_direction_rejection_repair_design"
    elif supported and attribution_supported:
        state = "initial_core_candidate_pass_confirmation_pending"
    elif supported:
        state = "performance_partial_pass_claim_blocked"
    elif material_negative:
        state = "cpsi_v1_exact_performance_fail_return_step4_5"
    else:
        state = "test_inconclusive_keep_candidate_no_claim"
    return {
        "decision": state,
        "effectiveness_supported": supported,
        "material_test_negative": material_negative,
        "matched_attribution_supported": attribution_supported,
        "internal_health_pass": health_ok,
        "controls_role": "intermediate_diagnostics_not_validation_eliminators",
        "confirmation_seeds_authorized": False,
    }


def synthetic_smoke(config: dict[str, Any]) -> None:
    metrics = []
    for arm in config["effective_arms"]:
        factor = 0.99 if arm["id"] == "iscf_v1_cpsi" else 1.0
        for dataset in config["datasets"]:
            for horizon in config["matrix"]["horizons"]:
                metrics.append(
                    {
                        "dataset": dataset,
                        "arm": arm["id"],
                        "horizon": horizon,
                        "mse": factor,
                        "mae": factor,
                    }
                )
    cells, summaries = comparison_rows(metrics, config)
    health = [
        {
            "all_finite": True,
            "message_rms": 1.0,
            "latent_rms": 1.0,
            "output_projection_norm": 1.0,
        }
    ]
    result = decide(summaries, health, config)
    if len(cells) != 240 or not result["effectiveness_supported"]:
        raise RuntimeError("CPSI analyzer synthetic smoke failed")
    print("cpsi_analyzer_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.synthetic_smoke:
        synthetic_smoke(config)
        return
    if args.new_root is None or args.reference_root is None or args.output_dir is None:
        raise ValueError("new-root, reference-root, and output-dir are required")
    metrics, audits, health_rows = [], [], []
    for arm in config["effective_arms"]:
        for dataset in config["datasets"]:
            run_metrics, audit, health = load_run(
                arm,
                dataset,
                args.seed,
                args.new_root,
                args.reference_root,
                config,
            )
            metrics.extend(run_metrics)
            audits.append(audit)
            if health is not None:
                health_rows.append(health)
    if len(metrics) != config["matrix"]["effective_official_test_cells"]:
        raise RuntimeError("formal matrix is incomplete")
    if any(row["status"] != "ok" for row in audits):
        raise RuntimeError("one or more run audits failed")
    cells, summaries = comparison_rows(metrics, config)
    decision = decide(summaries, health_rows, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metrics.csv", metrics)
    write_csv(args.output_dir / "run_audit.csv", audits)
    write_csv(args.output_dir / "comparison_cells.csv", cells)
    write_csv(args.output_dir / "comparison_summary.csv", summaries)
    write_csv(args.output_dir / "internal_health.csv", health_rows)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"cpsi_analysis={decision['decision']}")


if __name__ == "__main__":
    main()
