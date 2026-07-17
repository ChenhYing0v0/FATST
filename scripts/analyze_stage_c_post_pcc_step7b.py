#!/usr/bin/env python3
"""Analyze the SIFF/MCCA validation-only Step 7B matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
PCSD_REFERENCES = ("a6", "pcsd_direct", "dense_matched")
PCC_REFERENCES = ("equal_skill", "pcc_transport_full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--pcsd-reference-root", type=Path)
    parser.add_argument("--pcc-reference-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_post_pcc_step7b.json"),
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


def dense_metrics(path: Path) -> tuple[float, float, float, float]:
    rows = read_csv(path)
    by_horizon = {int(row["target_horizon"]): row for row in rows}
    if sorted(by_horizon) != list(range(1, 721)):
        raise ValueError(f"incomplete dense horizons: {path}")
    mse = [float(by_horizon[horizon]["mse"]) for horizon in range(1, 721)]
    mae = [float(by_horizon[horizon]["mae"]) for horizon in range(1, 721)]
    if not all(math.isfinite(value) for value in (*mse, *mae)):
        raise ValueError(f"non-finite metrics: {path}")
    return mean(mse), mean(mae), mse[-1], mae[-1]


def mechanism_statistics(path: Path) -> dict[str, float]:
    with np.load(path, allow_pickle=False) as payload:
        arms = payload["probe_arms"].astype(np.float64)
        usage = payload["policy_row_bin_usage"].astype(np.float64)
    denominator = max(float(np.sqrt(np.mean(arms**2))), 1e-12)
    pairwise = []
    for left in range(arms.shape[1]):
        for right in range(left + 1, arms.shape[1]):
            pairwise.append(
                float(
                    np.sqrt(np.mean((arms[:, left] - arms[:, right]) ** 2))
                    / denominator
                )
            )
    entropy = -(
        usage * np.log(np.clip(usage, 1e-12, None))
    ).sum(axis=-1) / math.log(usage.shape[-1])
    scope_usage = usage.mean(axis=(0, 1))
    return {
        "minimum_pairwise_probe_nrmse": min(pairwise),
        "mean_pairwise_probe_nrmse": mean(pairwise),
        "policy_normalized_entropy": float(entropy.mean()),
        "policy_usage_max": float(scope_usage.max()),
    }


def load_run(
    root: Path,
    arm: str,
    dataset: str,
    seed: int,
    *,
    expected_readout: str | None = None,
    expected_objective: str | None = None,
    coupling: bool = False,
) -> dict[str, Any]:
    directory = run_dir(root, arm, dataset, seed)
    required = {
        "metrics": directory / "metrics_by_target_horizon.csv",
        "config": directory / "effective_config.json",
        "invariants": directory / "trained_invariants.json",
        "initialization": directory / "initialization_contract.json",
    }
    if coupling:
        required["diagnostics"] = directory / "pcsd_validation_diagnostics.npz"
        required["gradient"] = directory / "pcc_shared_gradient_diagnostics.json"
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {
            "dataset": dataset,
            "arm": arm,
            "status": "missing",
            "missing": ",".join(missing),
            "run_dir": str(directory),
        }
    effective = json.loads(required["config"].read_text(encoding="utf-8"))
    adapter = effective["adapter"]
    invariant = json.loads(required["invariants"].read_text(encoding="utf-8"))
    initialization = json.loads(
        required["initialization"].read_text(encoding="utf-8")
    )
    protocol_pass = bool(
        adapter["dataset"] == dataset
        and adapter["seed"] == seed
        and adapter["mode"] == "unified"
        and adapter["pred_len"] == 720
        and adapter["checkpoint_policy"] == "best-val"
        and adapter["evaluation_prefix_mode"] == "full-crop"
        and adapter["final_evaluation_split"] == "val"
        and invariant.get("uses_test_split") is False
        and invariant.get("pass") is True
        and (expected_readout is None or adapter["readout_mode"] == expected_readout)
        and (
            expected_objective is None
            or adapter["pcc_objective_mode"] == expected_objective
        )
    )
    if coupling:
        gradient = json.loads(required["gradient"].read_text(encoding="utf-8"))
        protocol_pass = protocol_pass and bool(
            gradient.get("pass") is True
            and gradient.get("gradient_surgery_applied") is False
        )
    mse_auc, mae_auc, h720_mse, h720_mae = dense_metrics(required["metrics"])
    row: dict[str, Any] = {
        "dataset": dataset,
        "arm": arm,
        "status": "ok" if protocol_pass else "audit_fail",
        "protocol_pass": protocol_pass,
        "dense_mse_auc": mse_auc,
        "dense_mae_auc": mae_auc,
        "h720_mse": h720_mse,
        "h720_mae": h720_mae,
        "readout_mode": adapter["readout_mode"],
        "objective_mode": adapter.get("pcc_objective_mode", "not_applicable"),
        "mode_rank": adapter.get("pcsd_mode_rank", "not_applicable"),
        "encoder_initialization_hash": initialization.get(
            "encoder_initialization_hash", ""
        ),
        "decoder_initialization_hash": initialization.get(
            "pcsd_initialization_hash",
            initialization.get("pcsd_dense_initialization_hash", ""),
        ),
        "run_dir": str(directory),
    }
    if coupling:
        row.update(mechanism_statistics(required["diagnostics"]))
    return row


def relative_gain(candidate: float, reference: float) -> float:
    return 1.0 - candidate / reference


def effect_rows(
    lookup: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    comparisons = {
        "architecture_equal": ("siff_equal", "equal_skill"),
        "architecture_pcc": ("siff_pcc", "pcc_transport_full"),
        "architecture_mcca": ("siff_mcca", "pcsd_mcca"),
        "mcca_pcsd": ("pcsd_mcca", "pcc_transport_full"),
        "mcca_siff": ("siff_mcca", "siff_pcc"),
        "joint_over_a6": ("siff_mcca", "a6"),
        "ordered_over_constant": ("siff_mcca", "siff_const_mcca"),
        "ordered_over_permuted": ("siff_mcca", "siff_permuted_mcca"),
        "ordered_over_q1_width": ("siff_mcca", "pcsd_q1_wide_mcca"),
        "ordered_over_independent": (
            "siff_mcca",
            "independent_scope_matched_mcca",
        ),
        "ordered_over_dense": ("siff_mcca", "dense_siff_matched"),
        "transport_over_pointwise": ("pcsd_mcca", "pcsd_pointwise_mcca"),
        "capability_over_uniform_ot": (
            "pcsd_mcca",
            "pcsd_uniform_balanced_ot",
        ),
    }
    rows = []
    for effect, (candidate, reference) in comparisons.items():
        gains = []
        for dataset in DATASETS:
            gain = relative_gain(
                float(lookup[(dataset, candidate)]["dense_mse_auc"]),
                float(lookup[(dataset, reference)]["dense_mse_auc"]),
            )
            gains.append(gain)
            rows.append(
                {
                    "effect": effect,
                    "candidate": candidate,
                    "reference": reference,
                    "dataset": dataset,
                    "gain_fraction": gain,
                    "gain_percent": 100.0 * gain,
                }
            )
        rows.append(
            {
                "effect": effect,
                "candidate": candidate,
                "reference": reference,
                "dataset": "macro",
                "gain_fraction": mean(gains),
                "gain_percent": 100.0 * mean(gains),
                "dataset_wins": sum(value > 0.0 for value in gains),
            }
        )
    return rows


def gate_result(
    rows: list[dict[str, Any]],
    lookup: dict[tuple[str, str], dict[str, Any]],
    gates: dict[str, Any],
) -> dict[str, Any]:
    macro = {
        row["effect"]: row for row in rows if row["dataset"] == "macro"
    }
    architecture_gains = [
        relative_gain(
            float(lookup[(dataset, candidate)]["dense_mse_auc"]),
            float(lookup[(dataset, reference)]["dense_mse_auc"]),
        )
        for dataset in DATASETS
        for candidate, reference in (
            ("siff_equal", "equal_skill"),
            ("siff_pcc", "pcc_transport_full"),
            ("siff_mcca", "pcsd_mcca"),
        )
    ]
    architecture_by_dataset = [
        mean(architecture_gains[index * 3 : (index + 1) * 3])
        for index in range(len(DATASETS))
    ]
    mcca_by_dataset = [
        mean(
            (
                relative_gain(
                    float(lookup[(dataset, "pcsd_mcca")]["dense_mse_auc"]),
                    float(
                        lookup[(dataset, "pcc_transport_full")]["dense_mse_auc"]
                    ),
                ),
                relative_gain(
                    float(lookup[(dataset, "siff_mcca")]["dense_mse_auc"]),
                    float(lookup[(dataset, "siff_pcc")]["dense_mse_auc"]),
                ),
            )
        )
        for dataset in DATASETS
    ]
    architecture_pass = bool(
        sum(value > 0.0 for value in architecture_by_dataset)
        >= gates["architecture_main_effect_dataset_wins_min"]
        and mean(architecture_by_dataset)
        >= gates["architecture_main_effect_macro_gain_min"]
    )
    mcca_pass = bool(
        sum(value > 0.0 for value in mcca_by_dataset)
        >= gates["mcca_over_pcc_dataset_wins_min"]
        and mean(mcca_by_dataset) >= gates["mcca_over_pcc_macro_gain_min"]
    )
    joint = macro["joint_over_a6"]
    joint_pass = bool(
        int(joint["dataset_wins"]) >= gates["joint_over_a6_dataset_wins_min"]
        and float(joint["gain_fraction"])
        >= gates["joint_over_a6_macro_gain_min"]
    )
    retention = [
        float(lookup[(dataset, "siff_mcca")]["minimum_pairwise_probe_nrmse"])
        / max(
            float(lookup[(dataset, "pcsd_mcca")]["minimum_pairwise_probe_nrmse"]),
            1e-12,
        )
        for dataset in DATASETS
    ]
    entropy_min = min(
        float(lookup[(dataset, "siff_mcca")]["policy_normalized_entropy"])
        for dataset in DATASETS
    )
    usage_max = max(
        float(lookup[(dataset, "siff_mcca")]["policy_usage_max"])
        for dataset in DATASETS
    )
    mechanism_pass = bool(
        min(retention) >= gates["pairwise_diversity_retention_fraction_min"]
        and entropy_min >= gates["policy_normalized_entropy_min"]
        and usage_max <= gates["policy_usage_max"]
    )
    named = {
        "architecture_main_effect": architecture_pass,
        "mcca_over_same_mass_pcc": mcca_pass,
        "joint_over_a6": joint_pass,
        "diversity_and_policy": mechanism_pass,
        "ordered_over_constant": float(
            macro["ordered_over_constant"]["gain_fraction"]
        )
        > 0.0,
        "ordered_over_permuted": float(
            macro["ordered_over_permuted"]["gain_fraction"]
        )
        > 0.0,
        "mcca_over_uniform_ot": float(
            macro["capability_over_uniform_ot"]["gain_fraction"]
        )
        > 0.0,
    }
    method_pass = all(named.values())
    return {
        "gates": named,
        "method_pass": method_pass,
        "architecture_main_effect_macro_gain": mean(architecture_by_dataset),
        "architecture_main_effect_dataset_wins": sum(
            value > 0.0 for value in architecture_by_dataset
        ),
        "mcca_main_effect_macro_gain": mean(mcca_by_dataset),
        "mcca_main_effect_dataset_wins": sum(value > 0.0 for value in mcca_by_dataset),
        "diversity_retention_min": min(retention),
        "policy_normalized_entropy_min": entropy_min,
        "policy_usage_max": usage_max,
        "decision": (
            "phase_a_pass_step9_analysis_authorized"
            if method_pass
            else "phase_a_gate_fail_step9_failure_attribution_required"
        ),
    }


def synthetic_smoke(config: dict[str, Any]) -> None:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    arms = [entry["id"] for entry in config["arms"]]
    arms += [*PCSD_REFERENCES, *PCC_REFERENCES]
    for dataset in DATASETS:
        for arm in arms:
            value = 1.0
            if arm == "siff_mcca":
                value = 0.96
            elif arm in {"siff_equal", "siff_pcc", "pcsd_mcca"}:
                value = 0.98
            elif arm in {
                "siff_const_mcca",
                "siff_permuted_mcca",
                "pcsd_q1_wide_mcca",
                "independent_scope_matched_mcca",
                "dense_siff_matched",
                "pcsd_pointwise_mcca",
                "pcsd_uniform_balanced_ot",
            }:
                value = 0.99
            lookup[(dataset, arm)] = {
                "dense_mse_auc": value,
                "minimum_pairwise_probe_nrmse": 0.1,
                "policy_normalized_entropy": 0.8,
                "policy_usage_max": 0.3,
            }
    rows = effect_rows(lookup)
    result = gate_result(rows, lookup, config["gates"])
    if not result["method_pass"]:
        raise RuntimeError(f"post-PCC analyzer synthetic smoke failed: {result}")


def report_text(result: dict[str, Any]) -> str:
    gate_rows = "\n".join(
        f"| `{name}` | `{str(value).lower()}` |"
        for name, value in result["gates"].items()
    )
    return f"""# SIFF/MCCA Step7B Validation Result

| Field | Value |
| --- | --- |
| `expected_new_runs` | `{result['expected_new_runs']}` |
| `valid_new_runs` | `{result['valid_new_runs']}` |
| `valid_reference_runs` | `{result['valid_reference_runs']}` |
| `complete` | `{str(result['complete']).lower()}` |
| `method_pass` | `{str(result['method_pass']).lower()}` |
| `decision` | `{result['decision']}` |
| `test_used` | `false` |

## Frozen Gates

| Gate | Pass |
| --- | --- |
{gate_rows}

该文件只分析validation artifacts。任何test audit、confirmation seed或方法重设计，均须在
Step 9 failure attribution完成后另行授权。
"""


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.synthetic_smoke:
        synthetic_smoke(config)
        print("post_pcc_step7b_analyzer_synthetic_smoke=pass")
        return
    required_args = (
        args.raw_root,
        args.pcsd_reference_root,
        args.pcc_reference_root,
        args.output_dir,
    )
    if any(value is None for value in required_args):
        raise ValueError("all result roots and output-dir are required")
    arms = {entry["id"]: entry for entry in config["arms"]}
    new_rows = [
        load_run(
            args.raw_root,
            arm_id,
            dataset,
            args.seed,
            expected_readout=arm["readout_mode"],
            expected_objective=arm["objective_mode"],
            coupling=arm["readout_mode"] != "siff-dense-nonlinear-matched",
        )
        for dataset in DATASETS
        for arm_id, arm in arms.items()
    ]
    reference_rows = [
        load_run(
            args.pcsd_reference_root,
            arm,
            dataset,
            args.seed,
            coupling=arm == "pcsd_direct",
        )
        for dataset in DATASETS
        for arm in PCSD_REFERENCES
    ]
    reference_rows += [
        load_run(
            args.pcc_reference_root,
            arm,
            dataset,
            args.seed,
            expected_readout="pcsd-coupling-field",
            expected_objective=arm,
            coupling=True,
        )
        for dataset in DATASETS
        for arm in PCC_REFERENCES
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "run_summary.csv", [*new_rows, *reference_rows])
    valid_new = [row for row in new_rows if row.get("status") == "ok"]
    valid_reference = [
        row for row in reference_rows if row.get("status") == "ok"
    ]
    complete = bool(
        len(valid_new) == config["expected_runs"]
        and len(valid_reference)
        == len(DATASETS) * (len(PCSD_REFERENCES) + len(PCC_REFERENCES))
    )
    encoder_paired = all(
        len(
            {
                row["encoder_initialization_hash"]
                for row in valid_new
                if row["dataset"] == dataset
            }
        )
        == 1
        for dataset in DATASETS
    )
    if complete and encoder_paired:
        lookup = {
            (row["dataset"], row["arm"]): row
            for row in (*valid_new, *valid_reference)
        }
        effects = effect_rows(lookup)
        write_csv(args.output_dir / "factorial_and_control_effects.csv", effects)
        gate = gate_result(effects, lookup, config["gates"])
    else:
        gate = {
            "gates": {},
            "method_pass": False,
            "decision": "artifact_or_protocol_audit_fail",
        }
    result = {
        "candidate": config["candidate"],
        "seed": args.seed,
        "expected_new_runs": config["expected_runs"],
        "valid_new_runs": len(valid_new),
        "valid_reference_runs": len(valid_reference),
        "complete": complete,
        "paired_encoder_initialization": encoder_paired,
        "test_used": False,
        **gate,
    }
    (args.output_dir / "gate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "step7b_result_report.md").write_text(
        report_text(result), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not complete or not encoder_paired:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
