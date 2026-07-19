#!/usr/bin/env python3
"""Analyze the SC-D18-SPC horizon-specialization problem diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_d18_soft_projectivity_cost.json"),
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arm_dir(root: Path, arm: str, dataset: str, seed: int) -> Path:
    return root / arm / dataset / "h720_full" / f"seed{seed}"


def source_dir(directory: Path) -> Path:
    manifest = directory / "source_manifest.json"
    if not manifest.is_file():
        return directory
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return Path(payload["source_run_dir"])


def expected_arm(config: dict[str, Any], arm_id: str) -> dict[str, Any]:
    return next(arm for arm in config["arms"] if arm["id"] == arm_id)


def load_run(
    root: Path,
    arm: dict[str, Any],
    dataset: str,
    seed: int,
) -> tuple[dict[int, dict[str, float]], dict[str, Any], np.ndarray]:
    directory = arm_dir(root, arm["id"], dataset, seed)
    source = source_dir(directory)
    artifact_paths = {
        "metrics": directory / "test_audit_metrics_by_target_horizon.csv",
        "invariant": directory / "test_audit_invariants.json",
        "probes": directory / "pcsd_test_audit_diagnostics.npz",
    }
    source_paths = {
        "effective": source / "effective_config.json",
        "initialization": source / "initialization_contract.json",
        "diagnostics": source / "model_diagnostics.json",
        "training": source / "training_log.csv",
        "checkpoint": source / "checkpoint.pt",
    }
    missing = [
        name
        for name, path in {**artifact_paths, **source_paths}.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{arm['id']} {dataset} missing {','.join(missing)}"
        )
    effective = json.loads(
        source_paths["effective"].read_text(encoding="utf-8")
    )
    initialization = json.loads(
        source_paths["initialization"].read_text(encoding="utf-8")
    )
    diagnostics = json.loads(
        source_paths["diagnostics"].read_text(encoding="utf-8")
    )
    invariant = json.loads(
        artifact_paths["invariant"].read_text(encoding="utf-8")
    )
    adapter = effective["adapter"]
    checkpoint_sha256 = file_hash(source_paths["checkpoint"])
    protocol_pass = bool(
        invariant.get("pass") is True
        and invariant.get("evaluation_split") == "test"
        and invariant.get("test_access_authorized") is True
        and invariant.get("checkpoint_sha256") == checkpoint_sha256
        and adapter["dataset"] == dataset
        and int(adapter["seed"]) == seed
        and adapter["readout_mode"] == arm["readout_mode"]
        and adapter["pcc_objective_mode"] == arm["pcc_objective_mode"]
        and adapter["pred_loss_mode"] == arm["pred_loss_mode"]
        and adapter["target_horizons"] == arm["target_horizons"]
        and adapter["validation_horizons"] == arm["validation_horizons"]
        and adapter["checkpoint_policy"] == "best-val"
        and adapter["evaluation_prefix_mode"] == "full-crop"
        and adapter["final_evaluation_split"] == "val"
        and int(diagnostics["frozen_parameter_tensors"]) == 0
    )
    metric_rows = read_csv(artifact_paths["metrics"])
    metrics = {
        int(row["target_horizon"]): {
            "mse": float(row["mse"]),
            "mae": float(row["mae"]),
        }
        for row in metric_rows
    }
    with np.load(artifact_paths["probes"]) as probe_data:
        if "probe_fused" not in probe_data:
            raise KeyError(f"{artifact_paths['probes']} lacks probe_fused")
        probes = np.asarray(probe_data["probe_fused"], dtype=np.float64)
    audit = {
        "dataset": dataset,
        "arm": arm["id"],
        "training_new": arm["training_new"],
        "source_run_dir": str(source),
        "artifact_dir": str(directory),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_retrained": invariant.get("checkpoint_retrained"),
        "protocol_pass": protocol_pass,
        "all_finite": bool(
            all(
                math.isfinite(value)
                for row in metrics.values()
                for value in row.values()
            )
            and np.isfinite(probes).all()
        ),
        "probe_rows": int(probes.shape[0]),
        "total_parameters": int(diagnostics["total_parameters"]),
        "active_forward_parameters": int(
            diagnostics["active_forward_parameters"]
        ),
        "encoder_initialization_hash": initialization[
            "encoder_initialization_hash"
        ],
        "operator_initialization_hash": initialization[
            "operator_initialization_hash"
        ],
    }
    return metrics, audit, probes


def summarize_gains(
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    gains = [float(row["gain_over_a6_measure_percent"]) for row in cells]
    by_dataset: dict[str, list[float]] = {}
    by_horizon: dict[int, list[float]] = {}
    for row in cells:
        by_dataset.setdefault(row["dataset"], []).append(
            float(row["gain_over_a6_measure_percent"])
        )
        by_horizon.setdefault(int(row["horizon"]), []).append(
            float(row["gain_over_a6_measure_percent"])
        )
    return {
        "macro_gain_percent": mean(gains),
        "positive_cells": sum(value > 0.0 for value in gains),
        "dataset_gains": {
            dataset: mean(values)
            for dataset, values in by_dataset.items()
        },
        "horizon_gains": {
            str(horizon): mean(values)
            for horizon, values in by_horizon.items()
        },
        "positive_datasets": sum(
            mean(values) > 0.0 for values in by_dataset.values()
        ),
        "positive_horizons": sum(
            mean(values) > 0.0 for values in by_horizon.values()
        ),
        "minimum_horizon_gain_percent": min(
            mean(values) for values in by_horizon.values()
        ),
    }


def prediction_nrmse(
    candidate: np.ndarray,
    reference: np.ndarray,
    horizon: int,
) -> float:
    candidate_prefix = candidate[:, :horizon]
    reference_prefix = reference[:, :horizon]
    numerator = float(
        np.sqrt(np.mean(np.square(candidate_prefix - reference_prefix)))
    )
    denominator = max(
        float(np.sqrt(np.mean(np.square(reference_prefix)))),
        1e-12,
    )
    return numerator / denominator


def gate_result(
    config: dict[str, Any],
    cells: list[dict[str, Any]],
    nrmse_rows: list[dict[str, Any]],
    run_audits: list[dict[str, Any]],
) -> dict[str, Any]:
    gates = config["gates"]
    summary = summarize_gains(cells)
    full_macro = mean(
        float(row["gain_over_a6_full_percent"]) for row in cells
    )
    pathology = any(
        abs(float(row["gain_over_a6_measure_percent"]))
        > gates["maximum_absolute_cell_degradation_percent"]
        for row in cells
    )
    invariants = bool(
        all(
            row["protocol_pass"]
            and row["all_finite"]
            and row["probe_rows"]
            == config["diagnostic_protocol"]["probe_rows"]
            for row in run_audits
        )
    )
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in run_audits:
        by_dataset.setdefault(row["dataset"], []).append(row)
    matched_initialization = all(
        len({row["encoder_initialization_hash"] for row in rows}) == 1
        and len({row["operator_initialization_hash"] for row in rows}) == 1
        and len({row["total_parameters"] for row in rows}) == 1
        and len({row["active_forward_parameters"] for row in rows}) == 1
        for rows in by_dataset.values()
    )
    categories = {
        "macro_gain": summary["macro_gain_percent"]
        >= gates[
            "macro_own_horizon_mse_gain_over_a6_measure_percent_min"
        ],
        "horizon_support": summary["positive_horizons"]
        >= gates["positive_horizons_min"],
        "dataset_support": summary["positive_datasets"]
        >= gates["positive_datasets_min"],
        "cell_support": summary["positive_cells"]
        >= gates["positive_cells_min"],
        "no_horizon_regression": summary["minimum_horizon_gain_percent"]
        >= gates["minimum_horizon_macro_gain_percent"],
        "prediction_deformation": min(
            float(row["prediction_nrmse"]) for row in nrmse_rows
        )
        >= gates["shared_prefix_prediction_nrmse_min"],
        "protocol_numeric_and_matched_initialization": bool(
            invariants and matched_initialization and not pathology
        ),
    }
    overall_pass = all(categories.values())
    if pathology or not invariants or not matched_initialization:
        decision = config["decision_map"]["protocol_or_numeric_pathology"]
    elif overall_pass:
        decision = config["decision_map"]["all_problem_gates_pass"]
    elif (
        full_macro
        >= gates[
            "macro_own_horizon_mse_gain_over_a6_measure_percent_min"
        ]
        and summary["macro_gain_percent"]
        < gates[
            "macro_own_horizon_mse_gain_over_a6_measure_percent_min"
        ]
    ):
        decision = config["decision_map"]["specialists_beat_a6_full_only"]
    else:
        decision = config["decision_map"][
            "specialists_do_not_beat_a6_measure"
        ]
    return {
        **summary,
        "macro_gain_over_a6_full_percent": full_macro,
        "prediction_nrmse_min": min(
            float(row["prediction_nrmse"]) for row in nrmse_rows
        ),
        "pathology": pathology,
        "matched_initialization_and_parameter_count": matched_initialization,
        "categories": categories,
        "categories_passed": sum(categories.values()),
        "categories_total": len(categories),
        "overall_pass": overall_pass,
        "decision": decision,
    }


def synthetic_smoke(config: dict[str, Any]) -> None:
    cells = []
    nrmse_rows = []
    audits = []
    for dataset in config["datasets"]:
        for arm in config["arms"]:
            audits.append(
                {
                    "dataset": dataset,
                    "arm": arm["id"],
                    "protocol_pass": True,
                    "all_finite": True,
                    "probe_rows": 256,
                    "encoder_initialization_hash": f"enc-{dataset}",
                    "operator_initialization_hash": f"op-{dataset}",
                    "total_parameters": 100,
                    "active_forward_parameters": 90,
                }
            )
        for horizon in config["matrix"]["own_horizons"]:
            cells.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "gain_over_a6_measure_percent": 1.0,
                    "gain_over_a6_full_percent": 2.0,
                }
            )
            nrmse_rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "prediction_nrmse": 0.01,
                }
            )
    result = gate_result(config, cells, nrmse_rows, audits)
    if not result["overall_pass"]:
        raise RuntimeError(json.dumps(result, sort_keys=True))
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "synthetic.json"
        output.write_text(json.dumps(result) + "\n", encoding="utf-8")
        if not output.is_file():
            raise RuntimeError("synthetic output was not written")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.synthetic_smoke:
        synthetic_smoke(config)
        print("d18_analyzer_synthetic_smoke=pass")
        return
    if args.raw_root is None or args.output_dir is None:
        raise ValueError("raw-root and output-dir are required")

    metrics: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    probes: dict[tuple[str, str], np.ndarray] = {}
    run_audits = []
    for dataset in config["datasets"]:
        for arm in config["arms"]:
            values, audit, probe = load_run(
                args.raw_root,
                arm,
                dataset,
                args.seed,
            )
            metrics[(dataset, arm["id"])] = values
            probes[(dataset, arm["id"])] = probe
            run_audits.append(audit)

    cells = []
    nrmse_rows = []
    for dataset in config["datasets"]:
        for horizon in config["matrix"]["own_horizons"]:
            specialist = f"a6_spec{horizon}"
            candidate = metrics[(dataset, specialist)][horizon]
            measure = metrics[(dataset, "a6_measure")][horizon]
            full = metrics[(dataset, "a6_full")][horizon]
            cells.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "specialist": specialist,
                    "specialist_mse": candidate["mse"],
                    "a6_measure_mse": measure["mse"],
                    "a6_full_mse": full["mse"],
                    "gain_over_a6_measure_percent": 100.0
                    * (1.0 - candidate["mse"] / measure["mse"]),
                    "gain_over_a6_full_percent": 100.0
                    * (1.0 - candidate["mse"] / full["mse"]),
                    "specialist_mae": candidate["mae"],
                    "a6_measure_mae": measure["mae"],
                    "a6_full_mae": full["mae"],
                }
            )
            nrmse_rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "specialist": specialist,
                    "prediction_nrmse": prediction_nrmse(
                        probes[(dataset, specialist)],
                        probes[(dataset, "a6_measure")],
                        horizon,
                    ),
                }
            )
    result = gate_result(config, cells, nrmse_rows, run_audits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "own_horizon_cells.csv", cells)
    write_csv(args.output_dir / "prediction_divergence.csv", nrmse_rows)
    write_csv(args.output_dir / "run_audit.csv", run_audits)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# SC-D18-SPC Step 9 Problem Diagnostic",
        "",
        f"- matrix: `{len(run_audits)}/25 runs`, `{len(cells)}/15 own-H cells`;",
        f"- specialist vs A6_MEASURE macro MSE gain: `{result['macro_gain_percent']:+.4f}%`;",
        f"- specialist vs A6_FULL macro MSE gain: `{result['macro_gain_over_a6_full_percent']:+.4f}%`;",
        f"- dataset/horizon/cell support: `{result['positive_datasets']}/5`, `{result['positive_horizons']}/3`, `{result['positive_cells']}/15`;",
        f"- minimum shared-prefix prediction NRMSE: `{result['prediction_nrmse_min']:.6e}`;",
        f"- gates: `{result['categories_passed']}/{result['categories_total']}`; overall=`{result['overall_pass']}`;",
        f"- decision: `{result['decision']}`.",
        "",
        "该诊断只判断 exact projectivity 是否存在可测 accuracy cost；即使通过，也不把 horizon-specific models 视为论文贡献。",
    ]
    (args.output_dir / "d18_result.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
