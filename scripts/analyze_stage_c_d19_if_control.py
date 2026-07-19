#!/usr/bin/env python3
"""Analyze the frozen D19 implicit-forecast control matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--control-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_d19_if_control_step7b.json"),
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


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_dir(root: Path, arm: str, dataset: str, seed: int) -> Path:
    return root / arm / dataset / "h720_full" / f"seed{seed}"


def nrmse(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.sqrt(np.mean(np.square(right))))
    if denominator == 0.0:
        denominator = 1.0
    return float(np.sqrt(np.mean(np.square(left - right))) / denominator)


def load_run(
    root: Path,
    arm: dict[str, Any],
    dataset: str,
    seed: int,
    horizons: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = run_dir(root, arm["id"], dataset, seed)
    required = {
        "metrics": directory / "test_audit_metrics_by_target_horizon.csv",
        "invariant": directory / "test_audit_invariants.json",
        "effective": directory / "effective_config.json",
        "initialization": directory / "initialization_contract.json",
        "checkpoint": directory / "checkpoint.pt",
    }
    if arm["training_new"]:
        required["diagnostics"] = (
            directory / "pcsd_test_audit_diagnostics.npz"
        )
        required["training"] = directory / "training_log.csv"
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return [], {
            "dataset": dataset,
            "arm": arm["id"],
            "status": "missing",
            "missing": ",".join(missing),
            "run_dir": str(directory),
        }

    effective = json.loads(required["effective"].read_text(encoding="utf-8"))
    invariant = json.loads(
        required["invariant"].read_text(encoding="utf-8")
    )
    initialization = json.loads(
        required["initialization"].read_text(encoding="utf-8")
    )
    adapter = effective["adapter"]
    lookup = {
        int(row["target_horizon"]): row for row in read_csv(required["metrics"])
    }
    selected = []
    for horizon in horizons:
        row = lookup[horizon]
        mse, mae = float(row["mse"]), float(row["mae"])
        if not math.isfinite(mse) or not math.isfinite(mae):
            raise ValueError(f"non-finite metric: {directory} H{horizon}")
        selected.append(
            {
                "dataset": dataset,
                "arm": arm["id"],
                "horizon": horizon,
                "mse": mse,
                "mae": mae,
                "seed": seed,
            }
        )
    protocol_pass = bool(
        adapter["dataset"] == dataset
        and adapter["seed"] == seed
        and adapter["readout_mode"] == arm["readout_mode"]
        and adapter["checkpoint_policy"] == "best-val"
        and adapter["validation_horizons"] == [96, 192, 336, 720]
        and invariant.get("pass") is True
        and invariant.get("evaluation_split") == "test"
        and invariant.get("checkpoint_sha256") == file_hash(required["checkpoint"])
    )
    health: dict[str, Any] = {}
    if arm["training_new"]:
        diagnostics = np.load(required["diagnostics"])
        fused = diagnostics["probe_fused"]
        health["probe_fused"] = fused
        if "probe_if_amplitude" in diagnostics:
            amplitude = diagnostics["probe_if_amplitude"]
            sine = diagnostics["probe_if_phase_sine"]
            cosine = diagnostics["probe_if_phase_cosine"]
            health.update(
                {
                    "amplitude_std": float(np.std(amplitude)),
                    "phase_radius_mean": float(
                        np.mean(np.sqrt(np.square(sine) + np.square(cosine)))
                    ),
                    "if_internal_finite": bool(
                        np.isfinite(amplitude).all()
                        and np.isfinite(sine).all()
                        and np.isfinite(cosine).all()
                    ),
                }
            )
    return selected, {
        "dataset": dataset,
        "arm": arm["id"],
        "status": "ok" if protocol_pass else "audit_fail",
        "protocol_pass": protocol_pass,
        "checkpoint_sha256": invariant["checkpoint_sha256"],
        "full_prefix_max_abs": float(invariant["full_prefix_max_abs"]),
        "all_finite": bool(invariant["all_finite"]),
        "encoder_initialization_hash": initialization.get(
            "encoder_initialization_hash",
            "",
        ),
        "decoder_initialization_hash": initialization.get(
            "implicit_frequency_initialization_hash",
            initialization.get("implicit_direct_initialization_hash", ""),
        ),
        "run_dir": str(directory),
        **health,
    }


def compare(
    metrics: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {
        (row["dataset"], row["arm"], row["horizon"]): row for row in metrics
    }
    cells: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for comparison in config["comparisons"]:
        for metric in config["matrix"]["metrics"]:
            gains: list[float] = []
            by_dataset: dict[str, list[float]] = {}
            by_horizon: dict[int, list[float]] = {}
            for dataset in config["datasets"]:
                for horizon in config["matrix"]["horizons"]:
                    candidate = lookup[
                        (dataset, comparison["candidate"], horizon)
                    ]
                    reference = lookup[
                        (dataset, comparison["reference"], horizon)
                    ]
                    gain = 100.0 * (
                        1.0 - float(candidate[metric]) / float(reference[metric])
                    )
                    gains.append(gain)
                    by_dataset.setdefault(dataset, []).append(gain)
                    by_horizon.setdefault(horizon, []).append(gain)
                    cells.append(
                        {
                            "comparison": comparison["id"],
                            "role": comparison["role"],
                            "metric": metric,
                            "dataset": dataset,
                            "horizon": horizon,
                            "candidate_value": candidate[metric],
                            "reference_value": reference[metric],
                            "gain_percent": gain,
                        }
                    )
            summaries.append(
                {
                    "comparison": comparison["id"],
                    "role": comparison["role"],
                    "metric": metric,
                    "macro_gain_percent": mean(gains),
                    "cell_wins": sum(value > 0.0 for value in gains),
                    "dataset_wins": sum(
                        mean(values) > 0.0 for values in by_dataset.values()
                    ),
                    "horizon_wins": sum(
                        mean(values) > 0.0 for values in by_horizon.values()
                    ),
                }
            )
    return cells, summaries


def effectiveness(
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, bool]:
    lookup = {
        (row["comparison"], row["metric"]): row for row in summaries
    }
    results: dict[str, bool] = {}
    for comparison, threshold in config["effectiveness_gates"].items():
        mse = lookup[(comparison, "mse")]
        mae = lookup[(comparison, "mae")]
        results[comparison] = bool(
            mse["macro_gain_percent"]
            >= threshold["mse_macro_gain_percent_min"]
            and mse["cell_wins"] >= threshold["mse_cell_wins_min"]
            and mse["dataset_wins"] >= threshold["mse_dataset_wins_min"]
            and mse["horizon_wins"] >= threshold["mse_horizon_wins_min"]
            and mae["macro_gain_percent"]
            >= threshold["mae_macro_gain_percent_min"]
        )
    return results


def internal_health(
    audits: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    by_run = {
        (row["dataset"], row["arm"]): row
        for row in audits
        if row["status"] == "ok"
    }
    rows = []
    paired_encoder = True
    paired_if_decoder = True
    for dataset in config["datasets"]:
        arms = [by_run[(dataset, arm["id"])] for arm in config["arms"]]
        encoder_hashes = {row["encoder_initialization_hash"] for row in arms}
        if_row = by_run[(dataset, "if_measure")]
        no_skip = by_run[(dataset, "if_noskip_measure")]
        direct = by_run[(dataset, "direct_nonlinear_matched_measure")]
        if_noskip_nrmse = nrmse(
            if_row["probe_fused"],
            no_skip["probe_fused"],
        )
        if_direct_nrmse = nrmse(
            if_row["probe_fused"],
            direct["probe_fused"],
        )
        paired_encoder = paired_encoder and len(encoder_hashes) == 1
        paired_if_decoder = paired_if_decoder and bool(
            if_row["decoder_initialization_hash"]
            and if_row["decoder_initialization_hash"]
            == no_skip["decoder_initialization_hash"]
        )
        rows.append(
            {
                "dataset": dataset,
                "paired_encoder_initialization": len(encoder_hashes) == 1,
                "if_no_skip_identical_decoder_initialization": (
                    if_row["decoder_initialization_hash"]
                    == no_skip["decoder_initialization_hash"]
                ),
                "if_noskip_prediction_nrmse": if_noskip_nrmse,
                "if_direct_prediction_nrmse": if_direct_nrmse,
                "if_amplitude_std": if_row["amplitude_std"],
                "if_phase_radius_mean": if_row["phase_radius_mean"],
                "if_internal_finite": if_row["if_internal_finite"],
                "max_prefix_abs": max(
                    float(row["full_prefix_max_abs"]) for row in arms
                ),
                "all_finite": all(row["all_finite"] for row in arms),
            }
        )
    gate = config["internal_health_gates"]
    results = {
        "all_protocols": all(row["status"] == "ok" for row in audits),
        "all_finite": all(row["all_finite"] for row in rows),
        "prefix_projectivity": max(row["max_prefix_abs"] for row in rows)
        <= gate["prefix_max_abs_max"],
        "paired_encoder_initialization": paired_encoder,
        "if_no_skip_identical_decoder_initialization": paired_if_decoder,
        "prediction_deformation": min(
            min(
                row["if_noskip_prediction_nrmse"],
                row["if_direct_prediction_nrmse"],
            )
            for row in rows
        )
        >= gate["prediction_deformation_nrmse_min"],
        "amplitude_noncollapsed": min(
            row["if_amplitude_std"] for row in rows
        )
        >= gate["amplitude_standard_deviation_min"],
        "phase_noncollapsed": min(
            row["if_phase_radius_mean"] for row in rows
        )
        >= gate["phase_radius_mean_min"],
    }
    return rows, results


def synthetic_smoke(config: dict[str, Any]) -> None:
    metrics = []
    for dataset_index, dataset in enumerate(config["datasets"]):
        for arm in config["arms"]:
            arm_offset = {
                "a6_measure": 0.03,
                "if_measure": 0.0,
                "if_noskip_measure": 0.02,
                "direct_nonlinear_matched_measure": 0.025,
            }[arm["id"]]
            for horizon in config["matrix"]["horizons"]:
                base = 0.2 + dataset_index * 0.01 + horizon / 10000
                metrics.append(
                    {
                        "dataset": dataset,
                        "arm": arm["id"],
                        "horizon": horizon,
                        "mse": base + arm_offset,
                        "mae": math.sqrt(base + arm_offset),
                    }
                )
    cells, summaries = compare(metrics, config)
    gates = effectiveness(summaries, config)
    if len(cells) != 120 or len(summaries) != 6 or not all(gates.values()):
        raise RuntimeError("D19 analyzer synthetic smoke failed")
    print("d19_analyzer_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.synthetic_smoke:
        synthetic_smoke(config)
        return
    if args.raw_root is None or args.output_dir is None:
        raise ValueError("raw-root and output-dir are required")
    control_root = args.control_root or Path(
        config["control_source"]["remote_root"]
    )
    metrics: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        for arm in config["arms"]:
            root = args.raw_root if arm["training_new"] else control_root
            selected, audit = load_run(
                root,
                arm,
                dataset,
                args.seed,
                config["matrix"]["horizons"],
            )
            metrics.extend(selected)
            audits.append(audit)
    missing = [row for row in audits if row["status"] != "ok"]
    if missing:
        raise RuntimeError(
            "D19 matrix incomplete or invalid: "
            + json.dumps(missing, sort_keys=True)
        )
    cells, summaries = compare(metrics, config)
    effectiveness_results = effectiveness(summaries, config)
    health_rows, health_results = internal_health(audits, config)
    paper_facing = effectiveness_results["if_vs_a6"]
    attribution = bool(
        effectiveness_results["if_vs_direct"]
        and effectiveness_results["if_vs_noskip"]
    )
    health = all(health_results.values())
    if not health:
        decision = "diagnostic_invalid_for_direction_rejection"
        failure = "optimization_or_numeric_pathology"
        rollback = "Step 6/7A"
    elif not paper_facing:
        decision = "control_negative"
        failure = "hypothesis_false_or_readout_design_wrong"
        rollback = "Step 2/4"
    elif not attribution:
        decision = "performance_partial_pass_without_attribution"
        failure = "capacity_control_explains_or_skip_not_necessary"
        rollback = "Step 4"
    else:
        decision = "control_positive_not_paper_core"
        failure = "none"
        rollback = "Step 2/4 for a native contribution"
    payload = {
        "candidate_version": config["candidate_version"],
        "role": "control_only",
        "matrix_complete": True,
        "artifact_runs": len(audits),
        "official_test_cells": len(metrics),
        "paper_facing_effectiveness": {
            "if_vs_a6_pass": paper_facing,
            "gates": effectiveness_results,
        },
        "matched_mechanism_attribution": {
            "pass": attribution,
            "if_vs_direct_pass": effectiveness_results["if_vs_direct"],
            "if_vs_noskip_pass": effectiveness_results["if_vs_noskip"],
        },
        "internal_mechanism_health": {
            "pass": health,
            "gates": health_results,
        },
        "failure_attribution": {
            "category": failure,
            "decision": decision,
            "rollback_step": rollback,
            "paper_method_promotion_forbidden": True,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "standard_metrics.csv", metrics)
    write_csv(args.output_dir / "comparison_cells.csv", cells)
    write_csv(args.output_dir / "comparison_summary.csv", summaries)
    audit_rows = [
        {
            key: value
            for key, value in row.items()
            if key != "probe_fused"
        }
        for row in audits
    ]
    write_csv(args.output_dir / "artifact_audit.csv", audit_rows)
    write_csv(args.output_dir / "internal_health.csv", health_rows)
    (args.output_dir / "four_layer_decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# D19 IF control Step 9 四层诊断",
        "",
        f"- `decision`: `{decision}`",
        f"- `paper_facing_effectiveness`: `{paper_facing}`",
        f"- `matched_mechanism_attribution`: `{attribution}`",
        f"- `internal_mechanism_health`: `{health}`",
        f"- `rollback_step`: `{rollback}`",
        "",
        "D19 始终是 `control_only`。即使全部 gate 通过，也只能证明隐式频域"
        "生成值得作为后续原生 multi-horizon 方法的设计证据，不能直接晋升为"
        "论文 Contribution。",
    ]
    (args.output_dir / "step9_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(
        f"d19_analysis=pass decision={decision} "
        f"runs={len(audits)} cells={len(metrics)}"
    )


if __name__ == "__main__":
    main()
