#!/usr/bin/env python3
"""Aggregate SC1-D11 artifacts and apply the frozen responsibility gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


TOTAL_METRICS = (
    "short_norm",
    "long_norm",
    "norm_ratio",
    "cosine",
    "dot",
)
COMPONENT_METRICS = (
    "responsibility_js",
    "short_pair_cosine",
    "long_pair_cosine",
    "short_negative_pair_fraction",
    "long_negative_pair_fraction",
    "same_component_cosine",
    "same_component_negative_fraction",
    "short_alignment_efficiency",
    "long_alignment_efficiency",
    "short_cancellation",
    "long_cancellation",
    "short_additivity_relative_gap",
    "long_additivity_relative_gap",
)
PRIMARY_TARGETS = ("coeff_tensor", "coeff_params", "encoder_params")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if args.synthetic_smoke:
        return args
    missing = [name for name in ("input_root", "design", "output_dir") if getattr(args, name) is None]
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


def finite(value: str | float | int) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def aggregate_total(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, int, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["dataset"],
                int(row["seed"]),
                row["split"],
                row["loss"],
                row["target"],
            )
        ].append(row)
    outputs = []
    for key, values in sorted(grouped.items()):
        output: dict[str, Any] = {
            "dataset": key[0],
            "seed": key[1],
            "split": key[2],
            "loss": key[3],
            "target": key[4],
            "batch_count": len(values),
            "negative_fraction": statistics.mean(
                1.0 if row["negative"].lower() == "true" else 0.0 for row in values
            ),
        }
        for metric in TOTAL_METRICS:
            output[f"mean_{metric}"] = statistics.mean(float(row[metric]) for row in values)
        outputs.append(output)
    return outputs


def aggregate_components(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, int, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["dataset"],
                int(row["seed"]),
                row["split"],
                row["loss"],
                row["basis"],
            )
        ].append(row)
    outputs = []
    for key, values in sorted(grouped.items()):
        output: dict[str, Any] = {
            "dataset": key[0],
            "seed": key[1],
            "split": key[2],
            "loss": key[3],
            "basis": key[4],
            "batch_count": len(values),
        }
        for metric in COMPONENT_METRICS:
            output[f"mean_{metric}"] = statistics.mean(float(row[metric]) for row in values)
        output["component_negative_fraction"] = max(
            output["mean_short_negative_pair_fraction"],
            output["mean_long_negative_pair_fraction"],
            output["mean_same_component_negative_fraction"],
        )
        output["maximum_cancellation"] = max(
            output["mean_short_cancellation"], output["mean_long_cancellation"]
        )
        outputs.append(output)
    return outputs


def index_rows(
    rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> dict[tuple[Any, ...], dict[str, Any]]:
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"duplicate aggregate key: {key}")
        result[key] = row
    return result


def apply_gate(
    total_seed: list[dict[str, Any]],
    component_seed: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    design: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gates = design["gates"]
    total_index = index_rows(total_seed, ("dataset", "seed", "split", "loss", "target"))
    component_index = index_rows(
        component_seed, ("dataset", "seed", "split", "loss", "basis")
    )
    random_bases = [name for name in design["basis_families"] if name.startswith("random_s")]
    dataset_rows = []
    directional_dataset_count = 0
    component_dataset_count = 0
    generic_dataset_count = 0
    magnitude_dataset_count = 0

    for dataset in design["datasets"]:
        directional_targets: dict[str, int] = {}
        magnitude_targets: dict[str, int] = {}
        for target in PRIMARY_TARGETS:
            directional_seed_count = 0
            magnitude_seed_count = 0
            for seed in design["checkpoint_seeds"]:
                validation = total_index[(dataset, seed, "validation", "mse", target)]
                train = total_index[(dataset, seed, "train", "mse", target)]
                l1 = total_index[(dataset, seed, "validation", "l1", target)]
                if (
                    validation["mean_cosine"] < 0.0
                    and validation["negative_fraction"]
                    >= float(gates["validation_negative_batch_fraction_min"])
                    and train["negative_fraction"]
                    >= float(gates["train_negative_batch_fraction_min"])
                    and l1["mean_cosine"] < 0.0
                ):
                    directional_seed_count += 1
                if validation["mean_norm_ratio"] >= float(gates["magnitude_norm_ratio_min"]):
                    magnitude_seed_count += 1
            directional_targets[target] = directional_seed_count
            magnitude_targets[target] = magnitude_seed_count
        directional_support = max(directional_targets.values()) >= int(gates["seed_required"])
        magnitude_support = max(magnitude_targets.values()) >= int(gates["seed_required"])

        component_seed_count = 0
        generic_seed_count = 0
        for seed in design["checkpoint_seeds"]:
            rgnb_val = component_index[(dataset, seed, "validation", "mse", "rgnb")]
            rgnb_train = component_index[(dataset, seed, "train", "mse", "rgnb")]
            rgnb_l1 = component_index[(dataset, seed, "validation", "l1", "rgnb")]
            dct_val = component_index[(dataset, seed, "validation", "mse", "dct")]
            random_values = [
                component_index[(dataset, seed, "validation", "mse", basis)]
                for basis in random_bases
            ]
            random_cancellation = statistics.median(
                row["maximum_cancellation"] for row in random_values
            )
            random_js = statistics.median(
                row["mean_responsibility_js"] for row in random_values
            )
            primary_pressure = (
                rgnb_val["mean_responsibility_js"]
                >= float(gates["responsibility_js_min"])
                and rgnb_val["component_negative_fraction"]
                >= float(gates["component_negative_fraction_min"])
            )
            random_specific = (
                rgnb_val["maximum_cancellation"] - random_cancellation
                >= float(gates["random_cancellation_gap_min"])
                or rgnb_val["mean_responsibility_js"] - random_js
                >= float(gates["random_js_gap_min"])
            )
            dct_specific = (
                rgnb_val["maximum_cancellation"] - dct_val["maximum_cancellation"]
                >= float(gates["dct_specificity_gap_min"])
                or rgnb_val["mean_responsibility_js"] - dct_val["mean_responsibility_js"]
                >= float(gates["dct_specificity_gap_min"])
            )
            train_replication = rgnb_train["component_negative_fraction"] >= float(
                gates["component_negative_fraction_min"]
            )
            l1_replication = rgnb_l1["component_negative_fraction"] >= float(
                gates["component_negative_fraction_min"]
            )
            if primary_pressure:
                generic_seed_count += 1
            if (
                primary_pressure
                and random_specific
                and dct_specific
                and train_replication
                and l1_replication
            ):
                component_seed_count += 1
        component_support = component_seed_count >= int(gates["seed_required"])
        generic_support = generic_seed_count >= int(gates["seed_required"])
        directional_dataset_count += int(directional_support)
        component_dataset_count += int(component_support)
        generic_dataset_count += int(generic_support)
        magnitude_dataset_count += int(magnitude_support)
        dataset_rows.append(
            {
                "dataset": dataset,
                "directional_support": directional_support,
                "best_directional_target": max(directional_targets, key=directional_targets.get),
                "directional_seed_count": max(directional_targets.values()),
                "component_support": component_support,
                "component_seed_count": component_seed_count,
                "generic_component_pressure": generic_support,
                "generic_seed_count": generic_seed_count,
                "magnitude_support": magnitude_support,
                "magnitude_seed_count": max(magnitude_targets.values()),
            }
        )

    expected_metadata = len(design["datasets"]) * len(design["checkpoint_seeds"])
    invariants = {
        "metadata_count": len(metadata) == expected_metadata,
        "no_test": all(not bool(row["uses_test_split"]) for row in metadata),
        "no_training": all(not bool(row["trains_forecast_model"]) for row in metadata),
        "no_update": all(not bool(row["updates_forecast_model"]) for row in metadata),
        "forward_reconstruction": all(
            float(row["forward_reconstruction_max_abs"])
            <= float(gates["forward_reconstruction_max_abs"])
            for row in metadata
        ),
        "gradient_additivity": all(
            float(row["gradient_additivity_relative_max"])
            <= float(gates["gradient_additivity_relative_max"])
            for row in metadata
        ),
        "orthogonality": all(
            float(row["orthogonality_max_abs"])
            <= float(gates["orthogonality_max_abs"])
            for row in metadata
        ),
    }
    invariant_pass = all(invariants.values())
    required = int(gates["dataset_required"])
    directional_pass = directional_dataset_count >= required
    component_pass = component_dataset_count >= required
    generic_pass = generic_dataset_count >= required
    magnitude_pass = magnitude_dataset_count >= required
    if not invariant_pass:
        decision = "diagnostic_invalid_for_direction_rejection"
    elif directional_pass and component_pass:
        decision = "directional_and_support_specific_conflict_supported_return_step4"
    elif directional_pass:
        decision = "directional_conflict_supported_return_step4"
    elif component_pass:
        decision = "support_specific_component_conflict_supported_return_step4"
    elif generic_pass:
        decision = "transform_generic_pressure_sc2_only"
    elif magnitude_pass:
        decision = "magnitude_imbalance_only_simple_balancing_control"
    else:
        decision = "future_component_conflict_not_supported_rollback_step2"
    gate = {
        "diagnostic_id": "SC1-D11",
        "invariant_checks": invariants,
        "invariant_pass": invariant_pass,
        "counts": {
            "directional_datasets": directional_dataset_count,
            "support_specific_component_datasets": component_dataset_count,
            "generic_component_pressure_datasets": generic_dataset_count,
            "magnitude_imbalance_datasets": magnitude_dataset_count,
        },
        "directional_conflict_pass": directional_pass,
        "support_specific_component_conflict_pass": component_pass,
        "generic_component_pressure_pass": generic_pass,
        "magnitude_imbalance_pass": magnitude_pass,
        "decision": decision,
        "method_implementation_authorized": False,
        "sc2_authorized": False,
    }
    return gate, dataset_rows


def render_report(
    gate: dict[str, Any], dataset_rows: list[dict[str, Any]], component_seed: list[dict[str, Any]]
) -> str:
    component_index = index_rows(
        component_seed, ("dataset", "seed", "split", "loss", "basis")
    )
    lines = [
        "# SC1-D11 Future-Component Responsibility Result",
        "",
        "## Decision",
        "",
        f"- `decision`: `{gate['decision']}`；",
        f"- strict directional conflict datasets: `{gate['counts']['directional_datasets']}/5`；",
        f"- support-specific component conflict datasets: `{gate['counts']['support_specific_component_datasets']}/5`；",
        f"- generic component pressure datasets: `{gate['counts']['generic_component_pressure_datasets']}/5`；",
        f"- magnitude imbalance datasets: `{gate['counts']['magnitude_imbalance_datasets']}/5`；",
        f"- invariant pass: `{str(gate['invariant_pass']).lower()}`；method/test/SC2 remain false。",
        "",
        "## Dataset Gates",
        "",
        "| Dataset | directional | target/seeds | support-specific | seeds | generic | magnitude |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in dataset_rows:
        lines.append(
            f"| {row['dataset']} | {row['directional_support']} | "
            f"{row['best_directional_target']}/{row['directional_seed_count']} | "
            f"{row['component_support']} | {row['component_seed_count']} | "
            f"{row['generic_component_pressure']} | {row['magnitude_support']} |"
        )
    lines.extend(
        [
            "",
            "## Canonical Validation MSE Means",
            "",
            "| Dataset | RGNB JS | RGNB comp-neg | RGNB cancel | DCT JS | random median JS |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    random_names = ("random_s20260715", "random_s20260716", "random_s20260717")
    for dataset in [row["dataset"] for row in dataset_rows]:
        rgnb = [
            component_index[(dataset, seed, "validation", "mse", "rgnb")]
            for seed in (2021, 2022, 2023)
        ]
        dct = [
            component_index[(dataset, seed, "validation", "mse", "dct")]
            for seed in (2021, 2022, 2023)
        ]
        random_js = [
            component_index[(dataset, seed, "validation", "mse", name)][
                "mean_responsibility_js"
            ]
            for seed in (2021, 2022, 2023)
            for name in random_names
        ]
        lines.append(
            f"| {dataset} | {statistics.mean(row['mean_responsibility_js'] for row in rgnb):.6f} | "
            f"{statistics.mean(row['component_negative_fraction'] for row in rgnb):.6f} | "
            f"{statistics.mean(row['maximum_cancellation'] for row in rgnb):.6f} | "
            f"{statistics.mean(row['mean_responsibility_js'] for row in dct):.6f} | "
            f"{statistics.median(random_js):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "D11是checkpoint-local、validation-primary diagnostic。它区分strict negative-gradient conflict、"
            "component cancellation、generic transform pressure与magnitude imbalance；任何positive decision只返回"
            "Step4，不直接授权decoder、loss、optimizer或SC2。",
            "",
        ]
    )
    return "\n".join(lines)


def synthetic_smoke() -> None:
    design = {
        "datasets": ["A", "B", "C", "D", "E"],
        "checkpoint_seeds": [2021, 2022, 2023],
        "basis_families": [
            "rgnb",
            "dct",
            "random_s20260715",
            "random_s20260716",
            "random_s20260717",
        ],
        "gates": {
            "dataset_required": 3,
            "seed_required": 2,
            "validation_negative_batch_fraction_min": 0.25,
            "train_negative_batch_fraction_min": 0.2,
            "responsibility_js_min": 0.05,
            "component_negative_fraction_min": 0.2,
            "random_cancellation_gap_min": 0.05,
            "random_js_gap_min": 0.02,
            "dct_specificity_gap_min": 0.02,
            "magnitude_norm_ratio_min": 2.0,
            "orthogonality_max_abs": 1e-8,
            "gradient_additivity_relative_max": 1e-5,
            "forward_reconstruction_max_abs": 1e-5,
        },
    }
    total = []
    components = []
    for dataset in design["datasets"]:
        positive = dataset in {"A", "B", "C"}
        for seed in design["checkpoint_seeds"]:
            for split in ("train", "validation"):
                for loss in ("mse", "l1"):
                    for target in ("coeff_tensor", "coeff_params", "encoder_params"):
                        total.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "split": split,
                                "loss": loss,
                                "target": target,
                                "batch_count": 4,
                                "negative_fraction": 0.5 if positive else 0.0,
                                "mean_cosine": -0.1 if positive else 0.5,
                                "mean_norm_ratio": 1.2,
                            }
                        )
                    for basis in design["basis_families"]:
                        is_rgnb = basis == "rgnb"
                        components.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "split": split,
                                "loss": loss,
                                "basis": basis,
                                "mean_responsibility_js": 0.12 if positive and is_rgnb else 0.01,
                                "component_negative_fraction": 0.35 if positive and is_rgnb else 0.05,
                                "maximum_cancellation": 0.3 if positive and is_rgnb else 0.1,
                            }
                        )
    metadata = [
        {
            "uses_test_split": False,
            "trains_forecast_model": False,
            "updates_forecast_model": False,
            "forward_reconstruction_max_abs": 0.0,
            "gradient_additivity_relative_max": 0.0,
            "orthogonality_max_abs": 0.0,
        }
        for _ in range(15)
    ]
    gate, _rows = apply_gate(total, components, metadata, design)
    if gate["decision"] != "directional_and_support_specific_conflict_supported_return_step4":
        raise RuntimeError(f"synthetic gate failed: {gate['decision']}")
    print("stage_c_sc1_d11_analyzer_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    design = json.loads(args.design.read_text(encoding="utf-8"))
    total_rows = []
    component_rows = []
    group_rows = []
    reachability_rows = []
    metadata = []
    for dataset in design["datasets"]:
        root = args.input_root / dataset
        total_rows.extend(read_csv(root / "total_gradient_metrics.csv"))
        component_rows.extend(read_csv(root / "component_metrics.csv"))
        group_rows.extend(read_csv(root / "component_group_metrics.csv"))
        reachability_rows.extend(read_csv(root / "reachability_metrics.csv"))
        metadata.extend(json.loads((root / "metadata.json").read_text(encoding="utf-8")))

    expected = {
        "total": 1200,
        "component": 1200,
        "group": 8400,
        "reachability": 120,
    }
    observed = {
        "total": len(total_rows),
        "component": len(component_rows),
        "group": len(group_rows),
        "reachability": len(reachability_rows),
    }
    if observed != expected:
        raise RuntimeError(f"artifact count mismatch: {observed} != {expected}")
    for row in total_rows:
        if not all(finite(row[metric]) for metric in TOTAL_METRICS):
            raise RuntimeError("non-finite total-gradient metric")
    for row in component_rows:
        if not all(finite(row[metric]) for metric in COMPONENT_METRICS):
            raise RuntimeError("non-finite component metric")

    total_seed = aggregate_total(total_rows)
    component_seed = aggregate_components(component_rows)
    gate, dataset_rows = apply_gate(total_seed, component_seed, metadata, design)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "total_seed_summary.csv", total_seed)
    write_csv(args.output_dir / "component_seed_summary.csv", component_seed)
    write_csv(args.output_dir / "dataset_gate_summary.csv", dataset_rows)
    write_csv(args.output_dir / "reachability_metrics.csv", reachability_rows)
    (args.output_dir / "gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "research_interpretation.md").write_text(
        render_report(gate, dataset_rows, component_seed), encoding="utf-8"
    )
    print(f"stage_c_sc1_d11_analysis_done decision={gate['decision']}")


if __name__ == "__main__":
    main()
