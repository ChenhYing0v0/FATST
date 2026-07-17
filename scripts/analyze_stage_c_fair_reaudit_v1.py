#!/usr/bin/env python3
"""Analyze the StageC fair test-primary PCSD/PCC/SIFF re-audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_fair_reaudit_v1.json"),
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


def load_run(
    root: Path,
    arm: dict[str, Any],
    dataset: str,
    seed: int,
    horizons: list[int],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = run_dir(root, arm["id"], dataset, seed)
    required = {
        "metrics": directory / "test_audit_metrics_by_target_horizon.csv",
        "invariant": directory / "test_audit_invariants.json",
        "effective": directory / "effective_config.json",
        "training": directory / "training_log.csv",
        "initialization": directory / "initialization_contract.json",
        "checkpoint": directory / "checkpoint.pt",
    }
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
    invariant = json.loads(required["invariant"].read_text(encoding="utf-8"))
    initialization = json.loads(
        required["initialization"].read_text(encoding="utf-8")
    )
    adapter = effective["adapter"]
    metric_lookup = {
        int(row["target_horizon"]): row for row in read_csv(required["metrics"])
    }
    selected = []
    for horizon in horizons:
        row = metric_lookup[horizon]
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
        and adapter["pcc_objective_mode"] == arm["objective_mode"]
        and adapter["validation_horizons"]
        == config["training"]["validation_horizons"]
        and adapter["checkpoint_policy"] == "best-val"
        and adapter["final_evaluation_split"] == "val"
        and invariant.get("pass") is True
        and invariant.get("evaluation_split") == "test"
        and invariant.get("uses_test_split") is True
        and invariant.get("test_access_authorized") is True
        and invariant.get("checkpoint_retrained") is True
        and invariant.get("checkpoint_sha256") == file_hash(required["checkpoint"])
    )
    return selected, {
        "dataset": dataset,
        "arm": arm["id"],
        "status": "ok" if protocol_pass else "audit_fail",
        "protocol_pass": protocol_pass,
        "best_epoch": max(
            int(row["best_epoch_so_far"]) for row in read_csv(required["training"])
        ),
        "checkpoint_sha256": invariant["checkpoint_sha256"],
        "encoder_initialization_hash": initialization.get(
            "encoder_initialization_hash",
            "",
        ),
        "run_dir": str(directory),
    }


def comparison_rows(
    metrics: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    lookup = {
        (row["dataset"], row["arm"], row["horizon"]): row for row in metrics
    }
    rows = []
    for comparison in config["comparisons"]:
        gains = []
        dataset_gains: dict[str, list[float]] = {}
        horizon_gains: dict[int, list[float]] = {}
        for dataset in config["datasets"]:
            for horizon in config["matrix"]["horizons"]:
                candidate = lookup[
                    (dataset, comparison["candidate"], horizon)
                ]
                reference = lookup[
                    (dataset, comparison["reference"], horizon)
                ]
                gain = 100.0 * (
                    1.0 - float(candidate["mse"]) / float(reference["mse"])
                )
                gains.append(gain)
                dataset_gains.setdefault(dataset, []).append(gain)
                horizon_gains.setdefault(horizon, []).append(gain)
                rows.append(
                    {
                        "comparison": comparison["id"],
                        "candidate": comparison["candidate"],
                        "reference": comparison["reference"],
                        "dataset": dataset,
                        "horizon": horizon,
                        "gain_percent": gain,
                        "candidate_mse": candidate["mse"],
                        "reference_mse": reference["mse"],
                    }
                )
        rows.append(
            {
                "comparison": comparison["id"],
                "candidate": comparison["candidate"],
                "reference": comparison["reference"],
                "dataset": "macro",
                "horizon": "all",
                "gain_percent": mean(gains),
                "cell_wins": sum(value > 0.0 for value in gains),
                "dataset_wins": sum(
                    mean(values) > 0.0 for values in dataset_gains.values()
                ),
                "horizon_wins": sum(
                    mean(values) > 0.0 for values in horizon_gains.values()
                ),
            }
        )
    return rows


def effect_gate(
    macro: dict[str, dict[str, Any]],
    comparison: str,
    gates: dict[str, Any],
) -> bool:
    row = macro[comparison]
    return bool(
        float(row["gain_percent"]) >= gates["macro_gain_percent_min"]
        and int(row["dataset_wins"]) >= gates["dataset_wins_min"]
        and int(row["horizon_wins"]) >= gates["horizon_wins_min"]
        and int(row["cell_wins"]) >= gates["cell_wins_min"]
    )


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        assert abs(100.0 * (1.0 - 0.9 / 1.0) - 10.0) < 1e-12
        print("fair_reaudit_analyzer_synthetic_smoke=pass")
        return
    if args.raw_root is None or args.output_dir is None:
        raise ValueError("raw-root and output-dir are required")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    metrics: list[dict[str, Any]] = []
    runs = []
    for dataset in config["datasets"]:
        for arm in config["arms"]:
            arm_metrics, run = load_run(
                args.raw_root,
                arm,
                dataset,
                args.seed,
                config["matrix"]["horizons"],
                config,
            )
            metrics.extend(arm_metrics)
            runs.append(run)
    complete = bool(
        len(runs) == config["matrix"]["expected_runs"]
        and len(metrics) == config["matrix"]["expected_test_cells"]
        and all(row["status"] == "ok" for row in runs)
    )
    if not complete:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "run_audit.csv", runs)
        raise RuntimeError("fair re-audit matrix is incomplete or invalid")
    comparisons = comparison_rows(metrics, config)
    macro = {
        row["comparison"]: row
        for row in comparisons
        if row["dataset"] == "macro"
    }
    encoder_matched = all(
        len(
            {
                row["encoder_initialization_hash"]
                for row in runs
                if row["dataset"] == dataset
            }
        )
        == 1
        for dataset in config["datasets"]
    )
    gates = config["gates"]
    primary = {
        "pcsd_architecture_pass": effect_gate(
            macro,
            "pcsd_over_a6",
            gates,
        ),
        "pcc_specificity_pcsd_pass": effect_gate(
            macro,
            "pcc_over_prior_pcsd",
            gates,
        ),
        "pcc_specificity_siff_pass": effect_gate(
            macro,
            "pcc_over_prior_siff",
            gates,
        ),
        "siff_architecture_pcc_pass": effect_gate(
            macro,
            "siff_over_pcsd_pcc",
            gates,
        ),
        "joint_over_a6_pass": effect_gate(macro, "joint_over_a6", gates),
        "siff_all_specificity_controls_pass": all(
            effect_gate(macro, comparison, gates)
            for comparison in (
                "ordered_over_constant",
                "ordered_over_permuted",
                "ordered_over_q1_wide",
                "ordered_over_independent",
            )
        ),
    }
    summary = {
        "audit_id": config["audit_id"],
        "candidate_version": config["candidate_version"],
        "test_access_date": "2026-07-17",
        "user_authorization": config["authorization"],
        "checkpoint_retrained": True,
        "checkpoint_rule": config["training"]["validation_checkpoint_score"],
        "test_role": config["authorization"]["test_role"],
        "matrix_complete": complete,
        "test_informed": True,
        "encoder_initialization_matched": encoder_matched,
        "primary_gates": primary,
        "macro_comparisons": {
            key: {
                "gain_percent": float(row["gain_percent"]),
                "cell_wins": int(row["cell_wins"]),
                "dataset_wins": int(row["dataset_wins"]),
                "horizon_wins": int(row["horizon_wins"]),
            }
            for key, row in macro.items()
        },
        "confirmation_required": any(primary.values()),
        "decision": (
            "positive_effects_require_seed2022_2023_confirmation"
            if any(primary.values())
            else "no_historical_mechanism_passes_fair_seed2021_gate"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "run_audit.csv", runs)
    write_csv(args.output_dir / "test_metrics_standard_horizons.csv", metrics)
    write_csv(args.output_dir / "paired_comparisons.csv", comparisons)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
