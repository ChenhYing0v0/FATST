#!/usr/bin/env python3
"""Analyze the formal CCSF tau0.25 Phase-A four-layer audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from itertools import combinations
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
        default=Path(
            "configs/stage_c_siff_ccsf_v1_tau25_formal_candidate.json"
        ),
    )
    parser.add_argument(
        "--step6-config",
        type=Path,
        default=Path("configs/stage_c_siff_ccsf_step6.json"),
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
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
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
        "diagnostics": directory / "pcsd_test_audit_diagnostics.npz",
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
        and adapter["final_evaluation_split"]
        == config["training"]["training_final_evaluation_split"]
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
        "checkpoint_sha256": invariant.get("checkpoint_sha256", ""),
        "test_access_date": invariant.get("test_access_date"),
        "encoder_initialization_hash": initialization.get(
            "encoder_initialization_hash", ""
        ),
        "run_dir": str(directory),
    }


def comparison_rows(
    metrics: list[dict[str, Any]],
    config: dict[str, Any],
    step6: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {
        (row["dataset"], row["arm"], row["horizon"]): row for row in metrics
    }
    cells: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for comparison in step6["comparisons"]:
        for metric in config["matrix"]["metrics"]:
            gains = []
            dataset_gains: dict[str, list[float]] = {}
            horizon_gains: dict[int, list[float]] = {}
            for dataset in config["datasets"]:
                for horizon in config["matrix"]["horizons"]:
                    candidate = lookup[(dataset, comparison["candidate"], horizon)]
                    reference = lookup[(dataset, comparison["reference"], horizon)]
                    candidate_value = float(candidate[metric])
                    reference_value = float(reference[metric])
                    gain = 100.0 * (1.0 - candidate_value / reference_value)
                    gains.append(gain)
                    dataset_gains.setdefault(dataset, []).append(gain)
                    horizon_gains.setdefault(horizon, []).append(gain)
                    cells.append(
                        {
                            "comparison": comparison["id"],
                            "role": comparison["role"],
                            "metric": metric,
                            "candidate": comparison["candidate"],
                            "reference": comparison["reference"],
                            "dataset": dataset,
                            "horizon": horizon,
                            "gain_percent": gain,
                            "candidate_value": candidate_value,
                            "reference_value": reference_value,
                        }
                    )
            summaries.append(
                {
                    "comparison": comparison["id"],
                    "role": comparison["role"],
                    "metric": metric,
                    "candidate": comparison["candidate"],
                    "reference": comparison["reference"],
                    "macro_gain_percent": mean(gains),
                    "cell_wins": sum(value > 0.0 for value in gains),
                    "dataset_wins": sum(
                        mean(values) > 0.0 for values in dataset_gains.values()
                    ),
                    "horizon_wins": sum(
                        mean(values) > 0.0 for values in horizon_gains.values()
                    ),
                }
            )
    return cells, summaries


def hard_gate_results(
    summaries: list[dict[str, Any]], step6: dict[str, Any]
) -> tuple[dict[str, bool], float, bool]:
    lookup = {(row["comparison"], row["metric"]): row for row in summaries}
    gates = step6["hard_gates"]
    threshold = gates["per_comparison"]
    results = {}
    for comparison in gates["comparison_ids"]:
        row = lookup[(comparison, "mse")]
        passed = bool(
            float(row["macro_gain_percent"])
            >= threshold["mse_macro_gain_percent_min"]
            and int(row["dataset_wins"])
            >= threshold["mse_dataset_wins_min"]
            and int(row["horizon_wins"])
            >= threshold["mse_horizon_wins_min"]
            and int(row["cell_wins"]) >= threshold["mse_cell_wins_min"]
        )
        if comparison in gates["main_mae_comparison_ids"]:
            passed = passed and bool(
                float(lookup[(comparison, "mae")]["macro_gain_percent"])
                >= gates["main_mae_macro_gain_percent_min"]
            )
        results[comparison] = passed
    interaction = float(
        lookup[("calibration_on_ccsf", "mse")]["macro_gain_percent"]
        - lookup[("loss_effect_without_contrast", "mse")]["macro_gain_percent"]
    )
    interaction_pass = bool(
        interaction >= gates["interaction"]["macro_gain_percent_min"]
    )
    return results, interaction, interaction_pass


def normalized_pairwise_nrmse(arms: np.ndarray) -> float:
    denominator = max(float(np.sqrt(np.mean(np.square(arms)))), 1e-12)
    values = [
        float(np.sqrt(np.mean(np.square(arms[:, left] - arms[:, right]))))
        / denominator
        for left, right in combinations(range(arms.shape[1]), 2)
    ]
    return mean(values)


def normalized_entropy(policy: np.ndarray) -> float:
    clipped = np.clip(policy, 1e-12, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=-1) / math.log(
        policy.shape[-1]
    )
    return float(entropy.mean())


def centered_alignment(policy: np.ndarray, skill: np.ndarray) -> float:
    policy_centered = policy - policy.mean(axis=-1, keepdims=True)
    skill_centered = skill - skill.mean(axis=-1, keepdims=True)
    numerator = np.sum(policy_centered * skill_centered, axis=-1)
    denominator = np.sqrt(
        np.sum(np.square(policy_centered), axis=-1)
        * np.sum(np.square(skill_centered), axis=-1)
    )
    valid = denominator > 1e-12
    return float(np.mean(numerator[valid] / denominator[valid])) if np.any(valid) else 0.0


def internal_health_rows(
    raw_root: Path, config: dict[str, Any], seed: int
) -> list[dict[str, Any]]:
    rows = []
    for dataset in config["datasets"]:
        directory = run_dir(raw_root, "ccsf_relcal", dataset, seed)
        path = directory / "pcsd_test_audit_diagnostics.npz"
        invariant = json.loads(
            (directory / "test_audit_invariants.json").read_text(encoding="utf-8")
        )
        with np.load(path) as payload:
            required = {
                "arm_row_bin_mse",
                "fused_row_bin_mse",
                "probe_arms",
                "probe_targets",
                "probe_policy",
                "probe_base_policy",
                "probe_base_logits",
                "probe_correction_logits",
                "probe_contrast_descriptor",
            }
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(f"{dataset} CCSF diagnostics missing {missing}")
            arm_loss = payload["arm_row_bin_mse"].astype(np.float64)
            fused_loss = payload["fused_row_bin_mse"].astype(np.float64)
            arms = payload["probe_arms"].astype(np.float64)
            target = payload["probe_targets"].astype(np.float64)
            policy = payload["probe_policy"].astype(np.float64)
            base_policy = payload["probe_base_policy"].astype(np.float64)
            base_logits = payload["probe_base_logits"].astype(np.float64)
            correction = payload["probe_correction_logits"].astype(np.float64)
            contrast = payload["probe_contrast_descriptor"].astype(np.float64)
        point_error = np.abs(arms.transpose(0, 2, 1) - target[..., None])
        best_arm = np.argmin(point_error, axis=-1)
        final_accuracy = float(np.mean(np.argmax(policy, axis=-1) == best_arm))
        base_accuracy = float(
            np.mean(np.argmax(base_policy, axis=-1) == best_arm)
        )
        skill = -point_error
        final_forecast = np.sum(arms.transpose(0, 2, 1) * policy, axis=-1)
        uniform_forecast = np.mean(arms, axis=1)
        final_mse = float(np.mean(np.square(final_forecast - target)))
        uniform_mse = float(np.mean(np.square(uniform_forecast - target)))
        correction_ratio = float(
            np.sqrt(np.mean(np.square(correction)))
            / max(np.sqrt(np.mean(np.square(base_logits))), 1e-12)
        )
        finite = all(
            np.isfinite(value).all()
            for value in (
                arm_loss,
                fused_loss,
                arms,
                target,
                policy,
                base_policy,
                base_logits,
                correction,
                contrast,
            )
        )
        rows.append(
            {
                "dataset": dataset,
                "oracle_gain_percent": 100.0
                * (
                    1.0
                    - float(np.mean(np.min(arm_loss, axis=-1)))
                    / float(np.mean(fused_loss))
                ),
                "pairwise_arm_nrmse": normalized_pairwise_nrmse(arms),
                "policy_normalized_entropy": normalized_entropy(policy),
                "best_arm_accuracy": final_accuracy,
                "base_best_arm_accuracy": base_accuracy,
                "best_arm_accuracy_gain_points": final_accuracy - base_accuracy,
                "policy_skill_centered_alignment": centered_alignment(
                    policy, skill
                ),
                "allocation_gain_over_uniform_percent": 100.0
                * (1.0 - final_mse / uniform_mse),
                "contrast_correction_rms_ratio": correction_ratio,
                "prefix_projectivity_gap": float(
                    invariant["full_prefix_max_abs"]
                ),
                "all_finite": bool(finite),
            }
        )
    return rows


def internal_health_gate(
    rows: list[dict[str, Any]], step6: dict[str, Any]
) -> dict[str, bool]:
    gates = step6["internal_mechanism_health"]
    macro = lambda field: mean(float(row[field]) for row in rows)
    return {
        "finite": all(bool(row["all_finite"]) for row in rows),
        "prefix_projectivity": max(
            float(row["prefix_projectivity_gap"]) for row in rows
        )
        <= gates["prefix_projectivity_gap_max"],
        "oracle_headroom": sum(
            float(row["oracle_gain_percent"]) > 0.0 for row in rows
        )
        >= gates["oracle_positive_datasets_min"],
        "arm_diversity": macro("pairwise_arm_nrmse")
        >= gates["mean_pairwise_probe_nrmse_min"],
        "policy_entropy": gates["policy_entropy_min"]
        <= macro("policy_normalized_entropy")
        <= gates["policy_entropy_max"],
        "best_arm_accuracy_gain": macro("best_arm_accuracy_gain_points")
        >= gates["best_arm_accuracy_gain_over_v1_points_min"],
        "policy_skill_alignment": macro("policy_skill_centered_alignment")
        >= gates["policy_skill_centered_alignment_min"],
        "allocation_gain": macro("allocation_gain_over_uniform_percent")
        >= gates["policy_allocation_gain_over_uniform_percent_min"],
        "contrast_correction_use": macro("contrast_correction_rms_ratio")
        >= gates["contrast_correction_rms_ratio_min"],
    }


def decision_from_layers(
    hard: dict[str, bool],
    interaction_pass: bool,
    internal: dict[str, bool],
) -> dict[str, Any]:
    main = all(hard[key] for key in ("full_over_a6_measure", "full_over_v1"))
    attribution = all(
        value
        for key, value in hard.items()
        if key not in {"full_over_a6_measure", "full_over_v1"}
    ) and interaction_pass
    internal_pass = all(internal.values())
    if not main:
        decision = "close_exact_candidate_effectiveness_fail"
        failure = "hypothesis_false_or_readout_design_wrong"
    elif not attribution:
        decision = "performance_partial_pass_attribution_blocked"
        failure = "capacity_control_explains_or_specificity_fail"
    elif not internal_pass:
        decision = "design_fault_suspected_return_step4_or_step7a"
        failure = "intervention_point_wrong_or_internal_path_inactive"
    else:
        decision = "phase_a_pass_confirmation_required"
        failure = "none"
    return {
        "paper_facing_effectiveness": main,
        "matched_mechanism_attribution": attribution,
        "internal_mechanism_health": internal_pass,
        "failure_attribution": failure,
        "decision": decision,
        "confirmation_authorized": False,
    }


def synthetic_smoke(config: dict[str, Any], step6: dict[str, Any]) -> None:
    values = {
        "a6_measure": 1.00,
        "siff_v1_equal": 1.00,
        "siff_v1_relcal": 0.99,
        "ccsf_equal": 0.98,
        "ccsf_relcal": 0.94,
        "ccsf_stdcal": 0.96,
        "ccsf_no_contrast_equal": 1.00,
        "ccsf_no_contrast_relcal": 0.98,
        "ccsf_permuted_contrast_relcal": 0.98,
        "ccsf_independent_relcal": 0.98,
    }
    metrics = [
        {
            "dataset": dataset,
            "arm": arm["id"],
            "horizon": horizon,
            "mse": values[arm["id"]],
            "mae": values[arm["id"]],
        }
        for dataset in config["datasets"]
        for arm in config["arms"]
        for horizon in config["matrix"]["horizons"]
    ]
    _cells, summaries = comparison_rows(metrics, config, step6)
    hard, _interaction, interaction_pass = hard_gate_results(summaries, step6)
    internal = {
        "finite": True,
        "prefix_projectivity": True,
        "oracle_headroom": True,
        "arm_diversity": True,
        "policy_entropy": True,
        "best_arm_accuracy_gain": True,
        "policy_skill_alignment": True,
        "allocation_gain": True,
        "contrast_correction_use": True,
    }
    decision = decision_from_layers(hard, interaction_pass, internal)
    if not all(hard.values()) or decision["decision"] != (
        "phase_a_pass_confirmation_required"
    ):
        raise RuntimeError("CCSF four-layer analyzer synthetic smoke failed")
    print("ccsf_tau25_phase_a_analyzer_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    step6 = json.loads(args.step6_config.read_text(encoding="utf-8"))
    if args.synthetic_smoke:
        synthetic_smoke(config, step6)
        return
    if args.raw_root is None or args.output_dir is None:
        raise ValueError("raw-root and output-dir are required")
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
        len(runs) == config["matrix"]["phase_a_expected_runs"]
        and len(metrics) == config["matrix"]["phase_a_expected_test_cells"]
        and all(row["status"] == "ok" for row in runs)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "run_audit.csv", runs)
    if not complete:
        raise RuntimeError("CCSF tau25 Phase-A matrix is incomplete")
    cells, summaries = comparison_rows(metrics, config, step6)
    hard, interaction, interaction_pass = hard_gate_results(summaries, step6)
    health_rows = internal_health_rows(args.raw_root, config, args.seed)
    health = internal_health_gate(health_rows, step6)
    layers = decision_from_layers(hard, interaction_pass, health)
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
    summary = {
        "candidate_version": config["candidate_version"],
        "test_access_dates": sorted(
            {
                row["test_access_date"]
                for row in runs
                if row.get("test_access_date")
            }
        ),
        "user_authorization": config["authorization"],
        "checkpoint_retrained": True,
        "test_role": config["authorization"]["test_role"],
        "matrix_complete": complete,
        "test_informed": True,
        "encoder_initialization_matched": encoder_matched,
        "hard_comparisons": hard,
        "interaction_macro_gain_percent": interaction,
        "interaction_pass": interaction_pass,
        "internal_health": health,
        "evaluation_layers": layers,
    }
    write_csv(args.output_dir / "test_metrics_standard_horizons.csv", metrics)
    write_csv(args.output_dir / "comparison_cells.csv", cells)
    write_csv(args.output_dir / "comparison_summary.csv", summaries)
    write_csv(args.output_dir / "mechanism_health.csv", health_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
