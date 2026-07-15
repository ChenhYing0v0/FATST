#!/usr/bin/env python3
"""Analyze the validation-only SC1-JAPO end-to-end screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path
from statistics import mean, median
from typing import Any


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
ARMS = (
    "a6",
    "joint_geo",
    "uniform",
    "history",
    "atom",
    "joint_perm",
    "joint_random",
)
READOUTS = {
    "a6": "learned-basis-forecast-operator",
    "joint_geo": "japo-joint-geo",
    "uniform": "japo-uniform",
    "history": "japo-history",
    "atom": "japo-atom",
    "joint_perm": "japo-joint-perm",
    "joint_random": "japo-joint-random",
}
CONTROLS = ARMS[2:]
PRIMARY = "joint_geo"
SEGMENTS = (
    ("short_h1_96", 1, 96),
    ("middle_h97_336", 97, 336),
    ("long_h337_720", 337, 720),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--seeds",
        type=str,
        help="Comma-separated seeds for a frozen multi-seed mean gate.",
    )
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
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


def finite_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value: {value}")
    return number


def improvement(candidate: float, reference: float) -> float:
    return (1.0 - candidate / reference) * 100.0


def load_run(
    root: Path,
    arm: str,
    dataset: str,
    seed: int,
) -> dict[str, Any]:
    directory = run_dir(root, arm, dataset, seed)
    required = {
        "metrics": directory / "metrics_by_target_horizon.csv",
        "training": directory / "training_log.csv",
        "config": directory / "effective_config.json",
        "diagnostics": directory / "model_diagnostics.json",
        "initialization": directory / "initialization_contract.json",
        "invariants": directory / "trained_invariants.json",
        "patch": directory / "patch_diagnostics.json",
        "patch_rows": directory / "patch_diagnostics_by_patch.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {
            "seed": seed,
            "dataset": dataset,
            "arm": arm,
            "status": "missing",
            "missing": ",".join(missing),
            "run_dir": str(directory),
        }
    metric_rows = read_csv(required["metrics"])
    by_horizon = {int(row["target_horizon"]): row for row in metric_rows}
    if sorted(by_horizon) != list(range(1, 721)):
        return {
            "seed": seed,
            "dataset": dataset,
            "arm": arm,
            "status": "incomplete_dense_horizons",
            "dense_horizon_count": len(by_horizon),
            "run_dir": str(directory),
        }
    mse = [finite_float(by_horizon[h]["mse"]) for h in range(1, 721)]
    mae = [finite_float(by_horizon[h]["mae"]) for h in range(1, 721)]
    training = read_csv(required["training"])
    config = json.loads(required["config"].read_text(encoding="utf-8"))
    diagnostics = json.loads(
        required["diagnostics"].read_text(encoding="utf-8")
    )
    initialization = json.loads(
        required["initialization"].read_text(encoding="utf-8")
    )
    invariants = json.loads(
        required["invariants"].read_text(encoding="utf-8")
    )
    patch = json.loads(required["patch"].read_text(encoding="utf-8"))
    patch_rows = read_csv(required["patch_rows"])
    adapter = config["adapter"]
    contract = config.get("training_contract", {})
    protocol_ok = (
        adapter["dataset"] == dataset
        and adapter["mode"] == "unified"
        and adapter["pred_len"] == 720
        and adapter["target_horizons"] == [720]
        and adapter["validation_horizons"] == [720]
        and adapter["checkpoint_policy"] == "best-val"
        and adapter["pred_loss_mode"] == "full"
        and adapter["protocol_class"] == "method_screening"
        and adapter["readout_mode"] == READOUTS[arm]
        and adapter["evaluation_prefix_mode"] == "full-crop"
        and adapter["final_evaluation_split"] == "val"
        and contract.get("initialization") == "from_scratch"
        and contract.get("checkpoint_input") is None
        and diagnostics.get("frozen_parameter_tensors") == 0
    )
    patch_ok = (
        patch.get("finite") is True
        and len(patch_rows) == int(patch["patch_num"])
        and finite_float(patch["flatten_block_sum_max_abs"]) <= 2e-5
        and 0.0
        <= finite_float(patch["patch_contribution_entropy"])
        <= 1.0 + 1e-6
    )
    status = (
        "ok"
        if protocol_ok and patch_ok and invariants.get("pass") is True
        else "audit_fail"
    )
    row: dict[str, Any] = {
        "seed": seed,
        "dataset": dataset,
        "arm": arm,
        "status": status,
        "protocol_pass": protocol_ok,
        "invariant_pass": invariants.get("pass") is True,
        "patch_audit_pass": patch_ok,
        "readout_mode": adapter["readout_mode"],
        "profile_hash": adapter["profile_hash"],
        "encoder_initialization_hash": initialization.get(
            "encoder_initialization_hash", ""
        ),
        "expert_bank_initialization_hash": initialization.get(
            "expert_bank_initialization_hash", ""
        ),
        "basis_hash": initialization.get("basis_hash", ""),
        "descriptor_hash": initialization.get("descriptor_hash", ""),
        "dense_mse_auc": mean(mse),
        "dense_mae_auc": mean(mae),
        "h720_mse": mse[-1],
        "h720_mae": mae[-1],
        "epochs_ran": len(training),
        "best_val_mse": min(
            finite_float(item["val_mean_mse"]) for item in training
        ),
        "total_parameters": diagnostics.get("total_parameters", ""),
        "decoder_parameters": diagnostics.get(
            "japo_decoder_parameters",
            diagnostics.get("active_forward_parameters", ""),
        ),
        "trained_gate_entropy": patch.get("japo_gate_entropy", ""),
        "trained_expert_usage": json.dumps(
            patch.get("japo_expert_usage", []),
            separators=(",", ":"),
        ),
        "run_dir": str(directory),
    }
    for label, start, end in SEGMENTS:
        row[f"{label}_mse"] = mean(mse[start - 1 : end])
    return row


def initialization_gate(summary: list[dict[str, Any]]) -> dict[str, bool]:
    valid = [row for row in summary if row.get("status") == "ok"]
    encoder_paired = all(
        len(
            {
                row["encoder_initialization_hash"]
                for row in valid
                if row["dataset"] == dataset and row["seed"] == seed
            }
        )
        == 1
        for seed in sorted({int(row["seed"]) for row in valid})
        for dataset in DATASETS
    )
    expert_paired = all(
        len(
            {
                row["expert_bank_initialization_hash"]
                for row in valid
                if row["dataset"] == dataset
                and row["seed"] == seed
                and row["arm"] != "a6"
            }
        )
        == 1
        for seed in sorted({int(row["seed"]) for row in valid})
        for dataset in DATASETS
    )
    basis_paired = all(
        len(
            {
                row["basis_hash"]
                for row in valid
                if row["dataset"] == dataset
                and row["seed"] == seed
                and row["arm"] != "a6"
            }
        )
        == 1
        for seed in sorted({int(row["seed"]) for row in valid})
        for dataset in DATASETS
    )
    return {
        "paired_encoder_initialization": encoder_paired,
        "paired_expert_bank_initialization": expert_paired,
        "paired_basis": basis_paired,
    }


def aggregate_seed_means(
    summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for arm in ARMS:
            group = [
                row
                for row in summary
                if row.get("status") == "ok"
                and row["dataset"] == dataset
                and row["arm"] == arm
            ]
            rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "dense_mse_auc": mean(
                        float(row["dense_mse_auc"]) for row in group
                    ),
                    "dense_mae_auc": mean(
                        float(row["dense_mae_auc"]) for row in group
                    ),
                    **{
                        f"{label}_mse": mean(
                            float(row[f"{label}_mse"]) for row in group
                        )
                        for label, _start, _end in SEGMENTS
                    },
                }
            )
    return rows


def comparison_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {(row["dataset"], row["arm"]): row for row in rows}
    versus_a6 = {
        dataset: improvement(
            float(lookup[(dataset, PRIMARY)]["dense_mse_auc"]),
            float(lookup[(dataset, "a6")]["dense_mse_auc"]),
        )
        for dataset in DATASETS
    }
    versus_controls = {
        control: {
            dataset: improvement(
                float(lookup[(dataset, PRIMARY)]["dense_mse_auc"]),
                float(lookup[(dataset, control)]["dense_mse_auc"]),
            )
            for dataset in DATASETS
        }
        for control in CONTROLS
    }
    versus_median = {
        dataset: improvement(
            float(lookup[(dataset, PRIMARY)]["dense_mse_auc"]),
            median(
                float(lookup[(dataset, control)]["dense_mse_auc"])
                for control in CONTROLS
            ),
        )
        for dataset in DATASETS
    }
    return {
        "joint_vs_a6_by_dataset_pct": versus_a6,
        "joint_vs_a6_macro_pct": mean(versus_a6.values()),
        "joint_vs_a6_positive_datasets": sum(
            value > 0.0 for value in versus_a6.values()
        ),
        "joint_vs_each_control_macro_pct": {
            control: mean(values.values())
            for control, values in versus_controls.items()
        },
        "joint_vs_each_control_positive_datasets": {
            control: sum(value > 0.0 for value in values.values())
            for control, values in versus_controls.items()
        },
        "joint_vs_same_bank_median_by_dataset_pct": versus_median,
        "joint_vs_same_bank_median_macro_pct": mean(versus_median.values()),
        "joint_vs_same_bank_median_positive_datasets": sum(
            value > 0.0 for value in versus_median.values()
        ),
    }


def decide_gate(
    summary: list[dict[str, Any]],
    seeds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    selected_seeds = seeds or tuple(
        sorted({int(row.get("seed", 2021)) for row in summary})
    )
    valid = [row for row in summary if row.get("status") == "ok"]
    expected = len(DATASETS) * len(ARMS) * len(selected_seeds)
    if len(valid) != expected:
        return {
            "candidate": "SC1-JAPO",
            "decision": "incomplete_or_invalid",
            "valid_runs": len(valid),
            "expected_runs": expected,
            "pass": False,
        }
    init_checks = initialization_gate(valid)
    if not all(init_checks.values()):
        return {
            "candidate": "SC1-JAPO",
            "decision": "initialization_pairing_invalid",
            "valid_runs": len(valid),
            "expected_runs": expected,
            "initialization_checks": init_checks,
            "pass": False,
        }
    gate_rows = valid if len(selected_seeds) == 1 else aggregate_seed_means(valid)
    statistics = comparison_statistics(gate_rows)
    a6_macro = statistics["joint_vs_a6_macro_pct"]
    a6_positive = statistics["joint_vs_a6_positive_datasets"]
    median_macro = statistics["joint_vs_same_bank_median_macro_pct"]
    median_positive = statistics[
        "joint_vs_same_bank_median_positive_datasets"
    ]
    control_macros = statistics["joint_vs_each_control_macro_pct"]
    control_positive = statistics[
        "joint_vs_each_control_positive_datasets"
    ]
    immediate_fail = (
        (a6_macro <= -10.0 and a6_positive <= 1)
        or (median_macro <= 0.0 and median_positive <= 1)
    )
    provisional_pass = (
        a6_macro > 0.0
        and a6_positive >= 4
        and all(value > 0.0 for value in control_macros.values())
        and all(value >= 3 for value in control_positive.values())
        and median_macro >= 1.0
        and median_positive >= 4
    )
    if len(selected_seeds) == 1:
        decision = (
            "seed2021_immediate_fail_attribute_and_stop"
            if immediate_fail
            else (
                "seed2021_provisional_pass_run_seeds2022_2023"
                if provisional_pass
                else "seed2021_inconclusive_run_seed2022_only"
            )
        )
    else:
        decision = (
            "two_seed_mean_pass_run_seed2023"
            if provisional_pass
            else "two_seed_mean_fail_stop_and_attribute"
        )
    return {
        "candidate": "SC1-JAPO",
        "decision": decision,
        "valid_runs": len(valid),
        "expected_runs": expected,
        "seeds": list(selected_seeds),
        "initialization_checks": init_checks,
        **statistics,
        "individual_seed_joint_vs_a6_macro_pct": {
            str(seed): comparison_statistics(
                [row for row in valid if int(row["seed"]) == seed]
            )["joint_vs_a6_macro_pct"]
            for seed in selected_seeds
        },
        "immediate_fail": immediate_fail,
        "provisional_pass": provisional_pass,
        "pass": provisional_pass,
    }


def comparison_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in summary if row.get("status") == "ok"]
    lookup = {
        (int(row["seed"]), row["dataset"], row["arm"]): row
        for row in valid
    }
    rows = []
    for seed in sorted({int(row["seed"]) for row in valid}):
        for dataset in DATASETS:
            required = [
                (seed, dataset, arm)
                for arm in (PRIMARY, "a6", *CONTROLS)
            ]
            if not all(key in lookup for key in required):
                continue
            candidate = lookup[(seed, dataset, PRIMARY)]
            for reference in ("a6", *CONTROLS):
                control = lookup[(seed, dataset, reference)]
                rows.append(
                    {
                        "seed": seed,
                        "dataset": dataset,
                        "candidate": PRIMARY,
                        "reference": reference,
                        "dense_mse_improvement_pct": improvement(
                            float(candidate["dense_mse_auc"]),
                            float(control["dense_mse_auc"]),
                        ),
                        "dense_mae_improvement_pct": improvement(
                            float(candidate["dense_mae_auc"]),
                            float(control["dense_mae_auc"]),
                        ),
                        **{
                            f"{label}_mse_improvement_pct": improvement(
                                float(candidate[f"{label}_mse"]),
                                float(control[f"{label}_mse"]),
                            )
                            for label, _start, _end in SEGMENTS
                        },
                    }
                )
    return rows


def failure_attribution(
    summary: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    valid = [row for row in summary if row.get("status") == "ok"]
    joint = [row for row in valid if row["arm"] == PRIMARY]
    entropies = [
        float(row["trained_gate_entropy"])
        for row in joint
        if row.get("trained_gate_entropy", "") != ""
    ]
    complete = all(
        key in gate
        for key in (
            "joint_vs_same_bank_median_macro_pct",
            "joint_vs_same_bank_median_positive_datasets",
            "joint_vs_a6_macro_pct",
            "joint_vs_a6_positive_datasets",
        )
    )
    return {
        "candidate": "SC1-JAPO",
        "protocol_or_artifact_pathology": len(valid) != gate["expected_runs"],
        "numeric_pathology": False,
        "capacity_control_explains": complete
        and (
            gate["joint_vs_same_bank_median_macro_pct"] <= 0.0
            and gate["joint_vs_same_bank_median_positive_datasets"] <= 1
        ),
        "router_under_specialization_suspected": bool(entropies)
        and min(entropies) >= 0.99,
        "minimum_joint_router_entropy": min(entropies) if entropies else None,
        "a6_effectiveness_supported": complete
        and (
            gate["joint_vs_a6_macro_pct"] > 0.0
            and gate["joint_vs_a6_positive_datasets"] >= 4
        ),
        "direction_level_rejection_authorized": False,
        "interpretation": (
            "Artifacts or protocol are incomplete; method attribution is "
            "invalid until the matrix passes the audit."
            if not complete
            else (
                "The completed matrix is stable; apply only the frozen gate. "
                "Near-uniform routing, when present, is a design/optimization "
                "suspicion rather than a direction-level rejection."
            )
        ),
        "next_action": (
            "run_seed2022_without_design_change"
            if gate["decision"] == "seed2021_inconclusive_run_seed2022_only"
            else "follow_frozen_gate"
        ),
    }


def synthetic_smoke() -> None:
    rows = []
    mse = {
        "a6": 1.00,
        "joint_geo": 0.95,
        "uniform": 0.98,
        "history": 0.975,
        "atom": 0.985,
        "joint_perm": 0.98,
        "joint_random": 0.99,
    }
    for dataset in DATASETS:
        for arm in ARMS:
            rows.append(
                {
                    "seed": 2021,
                    "dataset": dataset,
                    "arm": arm,
                    "status": "ok",
                    "dense_mse_auc": mse[arm],
                    "dense_mae_auc": mse[arm],
                    "short_h1_96_mse": mse[arm],
                    "middle_h97_336_mse": mse[arm],
                    "long_h337_720_mse": mse[arm],
                    "encoder_initialization_hash": f"encoder-{dataset}",
                    "expert_bank_initialization_hash": (
                        "" if arm == "a6" else f"expert-{dataset}"
                    ),
                    "basis_hash": "" if arm == "a6" else "basis",
                }
            )
    gate = decide_gate(rows)
    if gate["decision"] != "seed2021_provisional_pass_run_seeds2022_2023":
        raise AssertionError(gate)
    two_seed_rows = rows + [
        row | {"seed": 2022}
        for row in rows
    ]
    two_seed_gate = decide_gate(two_seed_rows, (2021, 2022))
    if two_seed_gate["decision"] != "two_seed_mean_pass_run_seed2023":
        raise AssertionError(two_seed_gate)
    incomplete_rows = [row.copy() for row in rows]
    incomplete_rows[0]["status"] = "missing"
    incomplete_gate = decide_gate(incomplete_rows, (2021,))
    incomplete_attribution = failure_attribution(
        incomplete_rows,
        incomplete_gate,
    )
    if (
        incomplete_gate["decision"] != "incomplete_or_invalid"
        or incomplete_attribution["protocol_or_artifact_pathology"] is not True
    ):
        raise AssertionError(incomplete_attribution)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "gate.json"
        path.write_text(json.dumps(gate), encoding="utf-8")
        json.loads(path.read_text(encoding="utf-8"))
    print("stage_c_sc1_japo_analyzer_synthetic=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    if args.raw_root is None or args.output_dir is None:
        raise ValueError("--raw-root and --output-dir are required")
    seeds = (
        tuple(int(value) for value in args.seeds.split(",") if value)
        if args.seeds
        else (args.seed,)
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique integer seeds")
    summary = [
        load_run(args.raw_root, arm, dataset, seed)
        for seed in seeds
        for dataset in DATASETS
        for arm in ARMS
    ]
    gate = decide_gate(summary, seeds)
    comparisons = comparison_rows(summary)
    attribution = failure_attribution(summary, gate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "run_summary.csv", summary)
    if comparisons:
        write_csv(args.output_dir / "control_comparisons.csv", comparisons)
    (args.output_dir / "gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "failure_attribution.json").write_text(
        json.dumps(attribution, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "research_interpretation.md").write_text(
        "\n".join(
            [
                "# SC1-JAPO validation screen",
                "",
                f"- decision: `{gate['decision']}`",
                f"- valid runs: `{gate['valid_runs']}/{gate['expected_runs']}`",
                f"- seeds: `{list(seeds)}`",
                "- evaluation split: `validation` only",
                "- test used: `false`",
                "",
                "该自动报告只执行冻结gate；方向级失败归因必须另行审计。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(gate, sort_keys=True))


if __name__ == "__main__":
    main()
