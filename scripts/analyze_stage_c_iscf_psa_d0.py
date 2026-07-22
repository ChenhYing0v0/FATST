#!/usr/bin/env python3
"""Audit frozen ISCF policy shrinkage with leave-one-dataset-out selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_stage_c_iscf_scc_d0 import gain_percent, write_csv


REQUIRED_ARRAYS = {
    "probe_arms",
    "probe_targets",
    "probe_direct_policy",
    "scales",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_iscf_psa_d0.json"),
    )
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def policy_entropy(policy: np.ndarray) -> float:
    clipped = np.clip(policy, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=-1)
    return float(np.mean(entropy / np.log(policy.shape[-1])))


def fuse(arms: np.ndarray, policy: np.ndarray) -> np.ndarray:
    return np.sum(arms.transpose(0, 2, 1) * policy, axis=-1)


def errors(
    arms: np.ndarray,
    policy: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, float]:
    prediction = fuse(arms, policy)
    residual = prediction - targets
    return float(np.abs(residual).mean()), float(np.square(residual).mean())


def shrink_policy(
    policy: np.ndarray,
    family: str,
    value: float,
    scope_prior: np.ndarray,
) -> np.ndarray:
    if family == "convex_uniform":
        target = np.full(policy.shape[-1], 1.0 / policy.shape[-1])
        return (1.0 - value) * policy + value * target
    if family == "convex_scope_marginal":
        return (1.0 - value) * policy + value * scope_prior
    if family == "temperature":
        logits = np.log(np.clip(policy, 1e-12, 1.0)) / value
        logits -= logits.max(axis=-1, keepdims=True)
        weights = np.exp(logits)
        return weights / weights.sum(axis=-1, keepdims=True)
    raise ValueError(f"unknown policy family: {family}")


def split_rows(
    row_count: int,
    channel_count: int,
    config: dict[str, Any],
) -> int:
    nominal = int(np.floor(row_count * config["split"]["fit_row_fraction"]))
    fit_rows = nominal // channel_count * channel_count
    if fit_rows != int(config["split"]["expected_fit_rows"]):
        raise ValueError(f"unexpected fit rows: {fit_rows}")
    if row_count - fit_rows != int(
        config["split"]["expected_evaluation_rows"]
    ):
        raise ValueError("unexpected evaluation rows")
    return fit_rows


def validate_payload(
    payload: dict[str, np.ndarray],
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    missing = REQUIRED_ARRAYS.difference(payload)
    if missing:
        raise KeyError(f"missing diagnostic arrays: {sorted(missing)}")
    arms = payload["probe_arms"].astype(np.float64)
    targets = payload["probe_targets"].astype(np.float64)
    policy = payload["probe_direct_policy"].astype(np.float64)
    scales = tuple(int(value) for value in payload["scales"])
    if scales != tuple(config["coupling_scales"]):
        raise ValueError(f"unexpected scales: {scales}")
    if arms.shape != policy.transpose(0, 2, 1).shape:
        raise ValueError("arms and policy shapes are inconsistent")
    if targets.shape != arms.shape[::2]:
        raise ValueError("targets and arms shapes are inconsistent")
    if not all(np.isfinite(array).all() for array in (arms, targets, policy)):
        raise ValueError("non-finite diagnostic tensor")
    if not np.allclose(policy.sum(axis=-1), 1.0, atol=1e-5):
        raise ValueError("policy is not normalized")
    return {"arms": arms, "targets": targets, "policy": policy}


def gain_metrics(
    payload: dict[str, np.ndarray],
    rows: slice,
    family: str,
    value: float,
    scope_prior: np.ndarray,
) -> dict[str, float]:
    arms = payload["arms"][rows]
    targets = payload["targets"][rows]
    source_policy = payload["policy"][rows]
    candidate_policy = shrink_policy(
        source_policy,
        family,
        value,
        scope_prior,
    )
    baseline_l1, baseline_mse = errors(arms, source_policy, targets)
    candidate_l1, candidate_mse = errors(arms, candidate_policy, targets)
    return {
        "baseline_l1": baseline_l1,
        "candidate_l1": candidate_l1,
        "gain_l1_percent": gain_percent(baseline_l1, candidate_l1),
        "baseline_mse": baseline_mse,
        "candidate_mse": candidate_mse,
        "gain_mse_percent": gain_percent(baseline_mse, candidate_mse),
        "policy_entropy_before": policy_entropy(source_policy),
        "policy_entropy_after": policy_entropy(candidate_policy),
        "policy_mean_l1_movement": float(
            np.abs(candidate_policy - source_policy).mean()
        ),
    }


def scope_marginal_prior(
    payloads: dict[tuple[str, int], dict[str, np.ndarray]],
    source_datasets: list[str],
    config: dict[str, Any],
) -> np.ndarray:
    sums = np.zeros(len(config["coupling_scales"]), dtype=np.float64)
    count = 0
    for dataset in source_datasets:
        fit_rows = split_rows(
            256,
            int(config["channel_counts"][dataset]),
            config,
        )
        for seed in config["seeds"]:
            policy = payloads[(dataset, seed)]["policy"][:fit_rows]
            sums += policy.sum(axis=(0, 1))
            count += policy.shape[0] * policy.shape[1]
    prior = sums / count
    return prior / prior.sum()


def choose_value(
    family: str,
    values: list[float],
    payloads: dict[tuple[str, int], dict[str, np.ndarray]],
    source_datasets: list[str],
    scope_prior: np.ndarray,
    config: dict[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for value in values:
        gains = []
        mse_gains = []
        for dataset in source_datasets:
            fit_rows = split_rows(
                256,
                int(config["channel_counts"][dataset]),
                config,
            )
            for seed in config["seeds"]:
                metrics = gain_metrics(
                    payloads[(dataset, seed)],
                    slice(0, fit_rows),
                    family,
                    value,
                    scope_prior,
                )
                gains.append(metrics["gain_l1_percent"])
                mse_gains.append(metrics["gain_mse_percent"])
        rows.append(
            {
                "family": family,
                "value": value,
                "source_run_count": len(gains),
                "fit_macro_gain_l1_percent": float(np.mean(gains)),
                "fit_macro_gain_mse_percent": float(np.mean(mse_gains)),
            }
        )
    best_index = max(
        range(len(rows)),
        key=lambda index: (rows[index]["fit_macro_gain_l1_percent"], -index),
    )
    return float(rows[best_index]["value"]), rows


def rank_correlation(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) < 2:
        return float("nan")
    ranks_a = np.argsort(np.argsort(np.asarray(values_a, dtype=np.float64)))
    ranks_b = np.argsort(np.argsort(np.asarray(values_b, dtype=np.float64)))
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def analyze(
    payloads: dict[tuple[str, int], dict[str, np.ndarray]],
    config: dict[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    selection_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    datasets = list(config["datasets"])
    for heldout in datasets:
        sources = [dataset for dataset in datasets if dataset != heldout]
        prior = scope_marginal_prior(payloads, sources, config)
        for family, family_config in config["families"].items():
            values = [float(value) for value in family_config["values"]]
            selected, fit_rows = choose_value(
                family,
                values,
                payloads,
                sources,
                prior,
                config,
            )
            for row in fit_rows:
                row["heldout_dataset"] = heldout
                row["selected"] = bool(row["value"] == selected)
                selection_rows.append(row)
            evaluation_start = split_rows(
                256,
                int(config["channel_counts"][heldout]),
                config,
            )
            for seed in config["seeds"]:
                payload = payloads[(heldout, seed)]
                for value in values:
                    metrics = gain_metrics(
                        payload,
                        slice(evaluation_start, None),
                        family,
                        value,
                        prior,
                    )
                    curve_rows.append(
                        {
                            "heldout_dataset": heldout,
                            "seed": seed,
                            "family": family,
                            "value": value,
                            "selected": bool(value == selected),
                            **metrics,
                        }
                    )
                selected_metrics = gain_metrics(
                    payload,
                    slice(evaluation_start, None),
                    family,
                    selected,
                    prior,
                )
                run_rows.append(
                    {
                        "heldout_dataset": heldout,
                        "seed": seed,
                        "family": family,
                        "selected_value": selected,
                        **selected_metrics,
                    }
                )
                for start, end in config["position_bins"]:
                    subset = {
                        "arms": payload["arms"][
                            evaluation_start:, :, int(start) : int(end)
                        ],
                        "targets": payload["targets"][
                            evaluation_start:, int(start) : int(end)
                        ],
                        "policy": payload["policy"][
                            evaluation_start:, int(start) : int(end), :
                        ],
                    }
                    metrics = gain_metrics(
                        subset,
                        slice(None),
                        family,
                        selected,
                        prior,
                    )
                    position_rows.append(
                        {
                            "heldout_dataset": heldout,
                            "seed": seed,
                            "family": family,
                            "selected_value": selected,
                            "position_start": int(start),
                            "position_end": int(end),
                            **metrics,
                        }
                    )

    convex_rows = [
        row for row in run_rows if row["family"] == "convex_uniform"
    ]
    dataset_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        selected = [
            row for row in convex_rows if row["heldout_dataset"] == dataset
        ]
        dataset_rows.append(
            {
                "dataset": dataset,
                "selected_alpha": selected[0]["selected_value"],
                "run_count": len(selected),
                "macro_gain_l1_percent": float(
                    np.mean([row["gain_l1_percent"] for row in selected])
                ),
                "macro_gain_mse_percent": float(
                    np.mean([row["gain_mse_percent"] for row in selected])
                ),
                "joint_positive": bool(
                    np.mean([row["gain_l1_percent"] for row in selected]) > 0.0
                    and np.mean([row["gain_mse_percent"] for row in selected])
                    > 0.0
                ),
            }
        )
    macro_l1 = float(np.mean([row["gain_l1_percent"] for row in convex_rows]))
    macro_mse = float(
        np.mean([row["gain_mse_percent"] for row in convex_rows])
    )
    positive_runs = sum(
        row["gain_l1_percent"] > 0.0 and row["gain_mse_percent"] > 0.0
        for row in convex_rows
    )
    positive_datasets = sum(row["joint_positive"] for row in dataset_rows)
    selected_values = [float(row["selected_alpha"]) for row in dataset_rows]
    positive_folds = sum(value > 0.0 for value in selected_values)
    gates = config["gates"]
    checks = {
        "matrix_complete": len(convex_rows)
        == len(config["datasets"]) * len(config["seeds"]),
        "macro_joint_positive": macro_l1 > 0.0 and macro_mse > 0.0,
        "dataset_stability": positive_datasets
        >= int(gates["positive_dataset_count_min"]),
        "run_stability": positive_runs
        >= int(gates["positive_run_count_min"]),
        "nonzero_selection": positive_folds
        >= int(gates["positive_selected_fold_count_min"]),
    }
    full_pass = all(checks.values())
    uniform_endpoint = full_pass and all(value == 1.0 for value in selected_values)
    if uniform_endpoint:
        decision = config["decision_map"]["uniform_endpoint"]
    elif full_pass:
        decision = config["decision_map"]["supported"]
    elif checks["macro_joint_positive"]:
        decision = config["decision_map"]["unresolved"]
    else:
        decision = config["decision_map"]["not_supported"]
    summary = {
        "diagnostic_id": config["diagnostic_id"],
        "runs": len(convex_rows),
        "macro_gain_l1_percent": macro_l1,
        "macro_gain_mse_percent": macro_mse,
        "positive_run_count": positive_runs,
        "positive_dataset_count": positive_datasets,
        "positive_selected_fold_count": positive_folds,
        "selected_alphas": selected_values,
        "baseline_entropy_gain_l1_spearman": rank_correlation(
            [row["policy_entropy_before"] for row in convex_rows],
            [row["gain_l1_percent"] for row in convex_rows],
        ),
        "checks": checks,
        "decision": decision,
        "failure_attribution": (
            "none_problem_supported"
            if full_pass
            else "frozen_probe_negative_or_unstable_joint_training_unresolved"
        ),
        "forecast_model_training_authorized": False,
        "formal_test_access_authorized": False,
        "method_implementation_authorized": False,
    }
    return (
        summary,
        selection_rows,
        run_rows,
        dataset_rows,
        curve_rows,
        position_rows,
    )


def synthetic_payload(config: dict[str, Any], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = 256
    time = 720
    scopes = len(config["coupling_scales"])
    targets = rng.normal(size=(rows, time))
    common = targets[:, None, :] + rng.normal(
        scale=0.2,
        size=(rows, 1, time),
    )
    arms = common + rng.normal(scale=0.1, size=(rows, scopes, time))
    logits = 2.0 * rng.normal(size=(rows, time, scopes))
    logits -= logits.max(axis=-1, keepdims=True)
    policy = np.exp(logits)
    policy /= policy.sum(axis=-1, keepdims=True)
    return {
        "probe_arms": arms.astype(np.float32),
        "probe_targets": targets.astype(np.float32),
        "probe_direct_policy": policy.astype(np.float32),
        "scales": np.asarray(config["coupling_scales"]),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not config["authorization"][
        "existing_validation_artifact_analysis_authorized"
    ]:
        raise RuntimeError("existing validation analysis is not authorized")
    if config["authorization"]["formal_test_access_authorized"]:
        raise RuntimeError("PSA D0 must not access formal test")

    payloads: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    if args.synthetic_smoke:
        for dataset_index, dataset in enumerate(config["datasets"]):
            for seed in config["seeds"]:
                raw = synthetic_payload(config, int(seed) + dataset_index)
                payloads[(dataset, seed)] = validate_payload(raw, config)
    else:
        if args.validation_root is None or args.output_dir is None:
            raise ValueError("validation-root and output-dir are required")
        for dataset in config["datasets"]:
            for seed in config["seeds"]:
                path = (
                    args.validation_root
                    / dataset
                    / f"seed{seed}"
                    / "pcsd_validation_diagnostics.npz"
                )
                if not path.is_file():
                    raise FileNotFoundError(path)
                with np.load(path) as archive:
                    raw = {name: archive[name] for name in archive.files}
                payloads[(dataset, seed)] = validate_payload(raw, config)

    (
        summary,
        selection_rows,
        run_rows,
        dataset_rows,
        curve_rows,
        position_rows,
    ) = analyze(payloads, config)
    if args.synthetic_smoke:
        if summary["runs"] != 15 or len(dataset_rows) != 5:
            raise RuntimeError("synthetic PSA D0 smoke failed")
        print(
            "iscf_psa_d0_synthetic_smoke=pass "
            f"decision={summary['decision']}"
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "selection_curves.csv", selection_rows)
    write_csv(args.output_dir / "selected_run_metrics.csv", run_rows)
    write_csv(args.output_dir / "dataset_summary.csv", dataset_rows)
    write_csv(args.output_dir / "evaluation_curves.csv", curve_rows)
    write_csv(args.output_dir / "position_bin_metrics.csv", position_rows)
    (args.output_dir / "decision.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"iscf_psa_d0=complete runs={summary['runs']} "
        f"decision={summary['decision']}"
    )


if __name__ == "__main__":
    main()
