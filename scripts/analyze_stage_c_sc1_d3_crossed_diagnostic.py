#!/usr/bin/env python3
"""Analyze the paired 2x2 basis-group factorial diagnostic for SC1-D3."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
CHECKPOINT_SEEDS = (2021, 2022, 2023)
STRUCTURE_SEEDS = (3101, 3102, 3103)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d2-root", type=Path)
    parser.add_argument("--d3-root", type=Path)
    parser.add_argument("--d3-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if args.synthetic_smoke:
        return args
    required = ("d2_root", "d3_root", "d3_config", "output_dir")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def standard_deviation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def relative_reduction(log_effect: float) -> float:
    """Convert log(control/candidate) to candidate error reduction."""
    return 1.0 - math.exp(-log_effect)


def factorial_effects(tt: float, tr: float, rt: float, rr: float) -> dict[str, float]:
    if min(tt, tr, rt, rr) <= 0.0:
        raise ValueError("factorial error cells must be positive")
    basis_true_group = math.log(rt / tt)
    basis_random_group = math.log(rr / tr)
    basis_main = 0.5 * (basis_true_group + basis_random_group)
    group_true_basis = math.log(tr / tt)
    group_random_basis = math.log(rr / rt)
    group_main = 0.5 * (group_true_basis + group_random_basis)
    interaction = basis_true_group - basis_random_group
    return {
        "basis_true_group_log_effect": basis_true_group,
        "basis_random_group_log_effect": basis_random_group,
        "basis_main_log_effect": basis_main,
        "group_true_basis_log_effect": group_true_basis,
        "group_random_basis_log_effect": group_random_basis,
        "group_main_log_effect": group_main,
        "interaction_log_effect": interaction,
        "basis_true_group_reduction": relative_reduction(basis_true_group),
        "basis_random_group_reduction": relative_reduction(basis_random_group),
        "basis_main_reduction": relative_reduction(basis_main),
        "group_main_reduction": relative_reduction(group_main),
    }


def index_metrics(rows: list[dict[str, str]]) -> dict[tuple[str, int, str], dict[str, str]]:
    indexed: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in rows:
        key = (row["dataset"], int(row["checkpoint_seed"]), row["arm"])
        if key in indexed:
            raise ValueError(f"duplicate metric row: {key}")
        indexed[key] = row
    return indexed


def build_factorial_blocks(
    d2_rows: list[dict[str, str]], d3_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    d2 = index_metrics(d2_rows)
    d3 = index_metrics(d3_rows)
    outputs: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for checkpoint_seed in CHECKPOINT_SEEDS:
            for structure_seed in STRUCTURE_SEEDS:
                names = {
                    "tt": "true_scale_grouped",
                    "tr": f"random_group_s{structure_seed}",
                    "rt": f"random_basis_s{structure_seed}",
                    "rr": f"random_basis_random_group_s{structure_seed}",
                }
                sources = {"tt": d2, "tr": d2, "rt": d2, "rr": d3}
                cells: dict[str, dict[str, str]] = {}
                for cell, arm in names.items():
                    key = (dataset, checkpoint_seed, arm)
                    if key not in sources[cell]:
                        raise ValueError(f"missing factorial cell: {key}")
                    cells[cell] = sources[cell][key]
                mse = {
                    cell: float(row["val_mse_eval"]) for cell, row in cells.items()
                }
                mae = {
                    cell: float(row["val_mae_eval"]) for cell, row in cells.items()
                }
                mse_effects = factorial_effects(mse["tt"], mse["tr"], mse["rt"], mse["rr"])
                mae_effects = factorial_effects(mae["tt"], mae["tr"], mae["rt"], mae["rr"])
                outputs.append(
                    {
                        "dataset": dataset,
                        "checkpoint_seed": checkpoint_seed,
                        "structure_seed": structure_seed,
                        "tt_mse": mse["tt"],
                        "tr_mse": mse["tr"],
                        "rt_mse": mse["rt"],
                        "rr_mse": mse["rr"],
                        "tt_mae": mae["tt"],
                        "tr_mae": mae["tr"],
                        "rt_mae": mae["rt"],
                        "rr_mae": mae["rr"],
                        **{f"mse_{key}": value for key, value in mse_effects.items()},
                        **{f"mae_{key}": value for key, value in mae_effects.items()},
                    }
                )
    return outputs


def checkpoint_effects(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in blocks:
        grouped[(str(row["dataset"]), int(row["checkpoint_seed"]))].append(row)
    fields = [
        key
        for key in blocks[0]
        if key.startswith(("mse_", "mae_"))
        and (key.endswith("_log_effect") or key.endswith("_reduction"))
    ]
    outputs: list[dict[str, Any]] = []
    for (dataset, checkpoint_seed), rows in sorted(grouped.items()):
        if len(rows) != len(STRUCTURE_SEEDS):
            raise ValueError(f"incomplete structure blocks for {dataset} seed{checkpoint_seed}")
        output: dict[str, Any] = {
            "dataset": dataset,
            "checkpoint_seed": checkpoint_seed,
            "structure_seed_count": len(rows),
        }
        for field in fields:
            if field.endswith("_log_effect"):
                log_value = mean([float(row[field]) for row in rows])
                output[field] = log_value
                reduction_field = field.removesuffix("_log_effect") + "_reduction"
                output[reduction_field] = relative_reduction(log_value)
        outputs.append(output)
    return outputs


def dataset_summaries(checkpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in checkpoints:
        grouped[str(row["dataset"])].append(row)
    effect_fields = (
        "mse_basis_main_log_effect",
        "mse_basis_true_group_log_effect",
        "mse_basis_random_group_log_effect",
        "mse_group_main_log_effect",
        "mse_interaction_log_effect",
        "mae_basis_main_log_effect",
    )
    outputs: list[dict[str, Any]] = []
    for dataset in DATASETS:
        rows = grouped[dataset]
        if len(rows) != len(CHECKPOINT_SEEDS):
            raise ValueError(f"incomplete checkpoint units for {dataset}")
        output: dict[str, Any] = {"dataset": dataset, "checkpoint_count": len(rows)}
        for field in effect_fields:
            values = [float(row[field]) for row in rows]
            log_mean = mean(values)
            stem = field.removesuffix("_log_effect")
            output[f"mean_{field}"] = log_mean
            output[f"std_{field}"] = standard_deviation(values)
            output[f"mean_{stem}_reduction"] = relative_reduction(log_mean)
            output[f"positive_checkpoints_{field}"] = sum(value > 0.0 for value in values)
        main_effect = float(output["mean_mse_basis_main_log_effect"])
        interaction = float(output["mean_mse_interaction_log_effect"])
        output["interaction_not_dominant"] = abs(interaction) <= abs(main_effect)
        outputs.append(output)
    return outputs


def effect_gate(
    checkpoints: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    field: str,
    minimum_reduction: float,
    minimum_datasets: int,
) -> dict[str, Any]:
    macro_log = mean([float(row[field]) for row in checkpoints])
    passing_datasets = sum(
        int(row[f"positive_checkpoints_{field}"]) >= 2 for row in datasets
    )
    reduction = relative_reduction(macro_log)
    return {
        "macro_log_effect": macro_log,
        "macro_relative_reduction": reduction,
        "datasets_with_at_least_2_positive_checkpoints": passing_datasets,
        "pass": reduction >= minimum_reduction and passing_datasets >= minimum_datasets,
    }


def metadata_invariants(
    d2_metadata: list[dict[str, Any]], d3_metadata: list[dict[str, Any]]
) -> tuple[bool, dict[str, Any]]:
    expected_count = len(DATASETS) * len(CHECKPOINT_SEEDS)
    d2_hashes = {item["contract_hash"] for item in d2_metadata}
    d3_hashes = {item["contract_hash"] for item in d3_metadata}
    d3_config_hashes = {item["d3_config_hash"] for item in d3_metadata}
    d2_ok = all(
        not item["uses_test_split"]
        and not item["forecast_model_updated"]
        and not item["official_validation_used_for_early_stopping"]
        and float(item["basis_orthogonality_max_abs"]) <= 1e-5
        and float(item["parseval_relative_gap"]) <= 1e-5
        for item in d2_metadata
    )
    d3_ok = all(
        not item["uses_test_split"]
        and not item["forecast_model_updated"]
        and not item["official_validation_used_for_early_stopping"]
        and float(item["basis_orthogonality_max_abs"]) <= 1e-5
        and max(float(value) for value in item["random_basis_orthogonality_max_abs"].values())
        <= 1e-5
        for item in d3_metadata
    )
    pass_gate = (
        len(d2_metadata) == expected_count
        and len(d3_metadata) == expected_count
        and len(d2_hashes) == 1
        and d2_hashes == d3_hashes
        and len(d3_config_hashes) == 1
        and d2_ok
        and d3_ok
    )
    return pass_gate, {
        "d2_metadata_count": len(d2_metadata),
        "d3_metadata_count": len(d3_metadata),
        "contract_hashes_match": d2_hashes == d3_hashes and len(d2_hashes) == 1,
        "d3_config_hash_count": len(d3_config_hashes),
        "d2_invariants_pass": d2_ok,
        "d3_invariants_pass": d3_ok,
    }


def build_summary(
    blocks: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    d2_metadata: list[dict[str, Any]],
    d3_metadata: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    gate_config = config["gate"]
    minimum = float(gate_config["minimum_macro_mse_reduction"])
    minimum_datasets = int(
        gate_config["minimum_datasets_with_two_of_three_positive_checkpoints"]
    )
    complete = len(blocks) == 45 and len(checkpoints) == 15 and len(datasets) == 5
    invariants, invariant_details = metadata_invariants(d2_metadata, d3_metadata)
    main = effect_gate(
        checkpoints, datasets, "mse_basis_main_log_effect", minimum, minimum_datasets
    )
    true_group = effect_gate(
        checkpoints,
        datasets,
        "mse_basis_true_group_log_effect",
        minimum,
        minimum_datasets,
    )
    random_group = effect_gate(
        checkpoints,
        datasets,
        "mse_basis_random_group_log_effect",
        minimum,
        minimum_datasets,
    )
    mae_log = mean([float(row["mae_basis_main_log_effect"]) for row in checkpoints])
    mae_reduction = relative_reduction(mae_log)
    mae_pass = mae_reduction >= float(gate_config["minimum_macro_mae_reduction"])
    interaction_not_dominant = sum(bool(row["interaction_not_dominant"]) for row in datasets)
    interaction_pass = interaction_not_dominant >= int(
        gate_config["minimum_interaction_not_dominant_datasets"]
    )
    gate_pass = (
        complete
        and invariants
        and main["pass"]
        and true_group["pass"]
        and random_group["pass"]
        and mae_pass
        and interaction_pass
    )
    if not complete or not invariants:
        decision = "diagnostic_invalid_for_direction_rejection"
    elif gate_pass:
        decision = "basis_main_effect_supported_return_step4"
    elif main["pass"] and true_group["pass"] and not random_group["pass"]:
        decision = "basis_advantage_group_dependent_reformulate_step2"
    elif main["pass"] and not interaction_pass:
        decision = "basis_signal_interaction_dominated_refine_step2"
    else:
        decision = "basis_main_effect_not_supported_reformulate_step2"
    return {
        "candidate": "SC1-D3",
        "role": "diagnostic_only",
        "current_step": "Step 2/3",
        "complete": complete,
        "factorial_block_count": len(blocks),
        "primary_unit_count": len(checkpoints),
        "dataset_count": len(datasets),
        "invariant_gate": {"pass": invariants, **invariant_details},
        "basis_main_gate": main,
        "basis_true_group_gate": true_group,
        "basis_random_group_gate": random_group,
        "mae_guard": {"macro_relative_reduction": mae_reduction, "pass": mae_pass},
        "interaction_guard": {
            "datasets_where_abs_interaction_le_abs_main": interaction_not_dominant,
            "pass": interaction_pass,
        },
        "gate_pass": gate_pass,
        "decision": decision,
        "method_training_authorized": False,
        "authorization_if_passed": "return_to_step4_only",
        "limitations": [
            "The three paired structure seeds are averaged before gating and are not treated as independent checkpoints.",
            "The diagonal seed pairing does not estimate every random-basis x random-group combination.",
            "A pass identifies an effect only within the frozen-memory grouped nonlinear probe family; it does not establish method novelty.",
        ],
    }


def render_report(
    summary: dict[str, Any], datasets: list[dict[str, Any]], config_path: Path
) -> str:
    lines = [
        "# SC1-D3 Crossed Basis-Group Diagnostic Report",
        "",
        "## Decision",
        "",
        f"- `decision`: `{summary['decision']}`",
        f"- `gate_pass`: `{str(summary['gate_pass']).lower()}`",
        "- `method_training_authorized`: `false`",
        f"- preregistration: `{config_path}`",
        "",
        "## What Was Tested",
        "",
        "补齐D2缺失的`random basis × random group` cell，形成paired 2×2：",
        "`TT=true basis/true group`、`TR=true basis/random group`、",
        "`RT=random basis/true group`、`RR=random basis/random group`。",
        "每个dataset/checkpoint先平均3个structure-seed blocks，因此primary units为15而不是45。",
        "",
        "## Dataset Effects",
        "",
        "| Dataset | Basis main MSE reduction | True-group conditional | Random-group conditional | Interaction log effect | Interaction not dominant |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in datasets:
        lines.append(
            "| {dataset} | {main:.4%} | {true:.4%} | {random:.4%} | {interaction:+.6f} | {guard} |".format(
                dataset=row["dataset"],
                main=row["mean_mse_basis_main_reduction"],
                true=row["mean_mse_basis_true_group_reduction"],
                random=row["mean_mse_basis_random_group_reduction"],
                interaction=row["mean_mse_interaction_log_effect"],
                guard="pass" if row["interaction_not_dominant"] else "fail",
            )
        )
    lines.extend(
        [
            "",
            "## Gates",
            "",
            f"- basis main: `{summary['basis_main_gate']}`",
            f"- true-group conditional: `{summary['basis_true_group_gate']}`",
            f"- random-group conditional: `{summary['basis_random_group_gate']}`",
            f"- MAE guard: `{summary['mae_guard']}`",
            f"- interaction guard: `{summary['interaction_guard']}`",
            f"- invariant gate: `{summary['invariant_gate']}`",
            "",
            "## Failure Attribution Boundary",
            "",
            "若random-group conditional失败，只能说明D2的basis优势依赖当前group context；",
            "若interaction guard失败，只能说明main-effect解释不足。non-finite、orthogonality failure、",
            "artifact不完整或split污染均使diagnostic失效，不能否定更广义future-aware architecture。",
            "即便全部通过，也只允许返回Step 4设计新的paper-core idea。",
            "",
        ]
    )
    return "\n".join(lines)


def load_metadata(root: Path, filename: str) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for dataset in DATASETS:
        outputs.extend(json.loads((root / dataset / filename).read_text(encoding="utf-8")))
    return outputs


def synthetic_smoke() -> None:
    effects = factorial_effects(1.0, 1.02, 1.05, 1.071)
    if abs(effects["basis_true_group_log_effect"] - effects["basis_random_group_log_effect"]) > 1e-10:
        raise RuntimeError("zero-interaction synthetic case failed")
    if effects["basis_main_reduction"] <= 0.0:
        raise RuntimeError("basis direction convention changed")
    dominated = factorial_effects(1.0, 1.0, 1.10, 1.0)
    if abs(dominated["interaction_log_effect"]) <= abs(dominated["basis_main_log_effect"]):
        raise RuntimeError("interaction dominance check changed")
    d2_rows: list[dict[str, str]] = []
    d3_rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        for checkpoint_seed in CHECKPOINT_SEEDS:
            d2_rows.append(
                {
                    "dataset": dataset,
                    "checkpoint_seed": str(checkpoint_seed),
                    "arm": "true_scale_grouped",
                    "val_mse_eval": "1.0",
                    "val_mae_eval": "1.0",
                }
            )
            for structure_seed in STRUCTURE_SEEDS:
                d2_rows.extend(
                    [
                        {
                            "dataset": dataset,
                            "checkpoint_seed": str(checkpoint_seed),
                            "arm": f"random_group_s{structure_seed}",
                            "val_mse_eval": "1.02",
                            "val_mae_eval": "1.02",
                        },
                        {
                            "dataset": dataset,
                            "checkpoint_seed": str(checkpoint_seed),
                            "arm": f"random_basis_s{structure_seed}",
                            "val_mse_eval": "1.05",
                            "val_mae_eval": "1.05",
                        },
                    ]
                )
                d3_rows.append(
                    {
                        "dataset": dataset,
                        "checkpoint_seed": str(checkpoint_seed),
                        "arm": f"random_basis_random_group_s{structure_seed}",
                        "val_mse_eval": "1.071",
                        "val_mae_eval": "1.071",
                    }
                )
    blocks = build_factorial_blocks(d2_rows, d3_rows)
    checkpoints = checkpoint_effects(blocks)
    datasets = dataset_summaries(checkpoints)
    if len(blocks) != 45 or len(checkpoints) != 15 or len(datasets) != 5:
        raise RuntimeError("factorial aggregation counts changed")
    if any(not row["interaction_not_dominant"] for row in datasets):
        raise RuntimeError("zero-interaction aggregation changed")
    print("stage_c_sc1_d3_analyzer_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    config = json.loads(args.d3_config.read_text(encoding="utf-8"))
    d2_rows: list[dict[str, str]] = []
    d3_rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        d2_rows.extend(read_csv(args.d2_root / dataset / "d2_probe_metrics.csv"))
        d3_rows.extend(read_csv(args.d3_root / dataset / "d3_probe_metrics.csv"))
    blocks = build_factorial_blocks(d2_rows, d3_rows)
    checkpoints = checkpoint_effects(blocks)
    datasets = dataset_summaries(checkpoints)
    d2_metadata = load_metadata(args.d2_root, "d2_metadata.json")
    d3_metadata = load_metadata(args.d3_root, "d3_metadata.json")
    summary = build_summary(
        blocks, checkpoints, datasets, d2_metadata, d3_metadata, config
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "d3_factorial_blocks.csv", blocks)
    write_csv(args.output_dir / "d3_checkpoint_effects.csv", checkpoints)
    write_csv(args.output_dir / "d3_dataset_summary.csv", datasets)
    (args.output_dir / "d3_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "d3_diagnostic_report.md").write_text(
        render_report(summary, datasets, args.d3_config), encoding="utf-8"
    )
    print(
        f"stage_c_sc1_d3_analysis_done decision={summary['decision']} "
        f"output_dir={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
