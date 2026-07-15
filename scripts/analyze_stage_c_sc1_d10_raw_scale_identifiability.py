#!/usr/bin/env python3
"""Aggregate SC1-D10 raw scale-identifiability artifacts and apply frozen gates."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


FAMILIES = ("canonical", "history_perm", "future_perm")
SPLITS = ("holdout", "validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_smoke:
        required = ("input_root", "design", "output_dir")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"missing required arguments: {', '.join(missing)}")
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def matrix_metrics(matrix: torch.Tensor) -> dict[str, float | int]:
    if matrix.shape != (7, 7):
        raise ValueError(f"unexpected matrix shape: {matrix.shape}")
    gains = []
    best_count = 0
    for index in range(1, 7):
        alternatives = torch.cat((matrix[index, 1:index], matrix[index, index + 1 :]))
        gains.append(float(matrix[index, index] - alternatives.median()))
        best_count += int(
            int(torch.argmax(matrix[index, 1:]).item()) == index - 1
        )
    observed = float(torch.diagonal(matrix[1:, 1:]).mean())
    permutation_values = []
    for permutation in itertools.permutations(range(6)):
        permutation_values.append(
            sum(float(matrix[row + 1, permutation[row] + 1]) for row in range(6))
            / 6.0
        )
    permutation_p = (1 + sum(value >= observed for value in permutation_values)) / (
        len(permutation_values) + 1
    )
    return {
        "detail_monotone_gain": sum(gains) / len(gains),
        "detail_canonical_best_count": best_count,
        "detail_permutation_p": permutation_p,
    }


def binary_metrics(matrix: torch.Tensor) -> dict[str, float]:
    if matrix.shape != (2, 2):
        raise ValueError(f"unexpected binary matrix shape: {matrix.shape}")
    global_selectivity = float(matrix[0, 0] - matrix[0, 1])
    detail_selectivity = float(matrix[1, 1] - matrix[1, 0])
    return {
        "global_selectivity": global_selectivity,
        "detail_selectivity": detail_selectivity,
        "binary_interaction": 0.5 * (global_selectivity + detail_selectivity),
    }


def key_fields(row: dict[str, str]) -> tuple[str, str, int, float, str]:
    return (
        row["dataset"],
        row["family"],
        int(row["sketch_seed"]),
        float(row["ridge_lambda"]),
        row["split"],
    )


def build_replicate_metrics(
    matrix_rows: list[dict[str, str]],
    binary_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    matrix_groups: dict[tuple[str, str, int, float, str], torch.Tensor] = {}
    binary_groups: dict[tuple[str, str, int, float, str], torch.Tensor] = {}
    for row in matrix_rows:
        key = key_fields(row)
        matrix = matrix_groups.setdefault(
            key, torch.full((7, 7), float("nan"), dtype=torch.float64)
        )
        matrix[int(row["future_group_index"]), int(row["history_group_index"])] = float(
            row["r2"]
        )
    for row in binary_rows:
        key = key_fields(row)
        matrix = binary_groups.setdefault(
            key, torch.full((2, 2), float("nan"), dtype=torch.float64)
        )
        matrix[int(row["future_binary_index"]), int(row["history_binary_index"])] = float(
            row["r2"]
        )
    if set(matrix_groups) != set(binary_groups):
        raise RuntimeError("matrix/binary replicate keys differ")
    rows = []
    for key in sorted(matrix_groups):
        matrix = matrix_groups[key]
        binary = binary_groups[key]
        if not torch.isfinite(matrix).all() or not torch.isfinite(binary).all():
            raise RuntimeError(f"incomplete/non-finite replicate: {key}")
        dataset, family, sketch_seed, ridge_lambda, split = key
        rows.append(
            {
                "dataset": dataset,
                "family": family,
                "sketch_seed": sketch_seed,
                "ridge_lambda": ridge_lambda,
                "split": split,
                **binary_metrics(binary),
                **matrix_metrics(matrix),
            }
        )
    return rows


def build_dataset_metrics(
    matrix_rows: list[dict[str, str]],
    binary_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    matrix_sums: dict[tuple[str, str, str], torch.Tensor] = {}
    matrix_counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    binary_sums: dict[tuple[str, str, str], torch.Tensor] = {}
    binary_counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    replicate_keys: defaultdict[tuple[str, str, str], set[tuple[int, float]]] = defaultdict(set)
    for row in matrix_rows:
        key = (row["dataset"], row["family"], row["split"])
        matrix_sums.setdefault(key, torch.zeros(7, 7, dtype=torch.float64))[
            int(row["future_group_index"]), int(row["history_group_index"])
        ] += float(row["r2"])
        matrix_counts[key] += 1
        replicate_keys[key].add((int(row["sketch_seed"]), float(row["ridge_lambda"])))
    for row in binary_rows:
        key = (row["dataset"], row["family"], row["split"])
        binary_sums.setdefault(key, torch.zeros(2, 2, dtype=torch.float64))[
            int(row["future_binary_index"]), int(row["history_binary_index"])
        ] += float(row["r2"])
        binary_counts[key] += 1
    rows = []
    for key in sorted(matrix_sums):
        replicate_count = len(replicate_keys[key])
        expected_matrix_cells = replicate_count * 49
        expected_binary_cells = replicate_count * 4
        if matrix_counts[key] != expected_matrix_cells or binary_counts[key] != expected_binary_cells:
            raise RuntimeError(f"dataset aggregate cell count mismatch: {key}")
        matrix = matrix_sums[key] / replicate_count
        binary = binary_sums[key] / replicate_count
        dataset, family, split = key
        rows.append(
            {
                "dataset": dataset,
                "family": family,
                "split": split,
                "replicate_count": replicate_count,
                **binary_metrics(binary),
                **matrix_metrics(matrix),
            }
        )
    return rows


def aggregate_cell_rows(
    rows: list[dict[str, str]],
    index_fields: tuple[str, str],
) -> list[dict[str, Any]]:
    values: defaultdict[tuple[str, str, str, int, int], list[float]] = defaultdict(list)
    left_field, right_field = index_fields
    for row in rows:
        key = (
            row["dataset"],
            row["family"],
            row["split"],
            int(row[left_field]),
            int(row[right_field]),
        )
        values[key].append(float(row["r2"]))
    return [
        {
            "dataset": key[0],
            "family": key[1],
            "split": key[2],
            left_field: key[3],
            right_field: key[4],
            "replicate_count": len(cell_values),
            "mean_r2": sum(cell_values) / len(cell_values),
        }
        for key, cell_values in sorted(values.items())
    ]


def apply_gate(
    replicate_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    design: dict[str, Any],
) -> dict[str, Any]:
    gates = design["gates"]
    datasets = [str(value) for value in design["datasets"]]
    lookup = {
        (row["dataset"], row["family"], row["split"]): row
        for row in dataset_rows
    }
    validation = [lookup[(dataset, "canonical", "validation")] for dataset in datasets]
    holdout = [lookup[(dataset, "canonical", "holdout")] for dataset in datasets]
    binary_effect = sum(
        float(row["binary_interaction"]) >= float(gates["binary_interaction_min"])
        for row in validation
    )
    binary_directions = sum(
        float(row["global_selectivity"]) > 0.0
        and float(row["detail_selectivity"]) > 0.0
        for row in validation
    )
    binary_control = 0
    monotone_control = 0
    for dataset, row in zip(datasets, validation, strict=True):
        strongest_binary_control = max(
            float(lookup[(dataset, family, "validation")]["binary_interaction"])
            for family in ("history_perm", "future_perm")
        )
        strongest_monotone_control = max(
            float(lookup[(dataset, family, "validation")]["detail_monotone_gain"])
            for family in ("history_perm", "future_perm")
        )
        binary_control += int(
            float(row["binary_interaction"]) - strongest_binary_control
            >= float(gates["binary_control_gain_min"])
        )
        monotone_control += int(
            float(row["detail_monotone_gain"]) - strongest_monotone_control
            >= float(gates["monotone_control_gain_min"])
        )
    canonical_validation_replicates = [
        row
        for row in replicate_rows
        if row["family"] == "canonical" and row["split"] == "validation"
    ]
    binary_replicate_positive = sum(
        float(row["binary_interaction"]) > 0.0
        for row in canonical_validation_replicates
    )
    binary_split_stability = sum(
        float(validation[index]["binary_interaction"]) > 0.0
        and float(holdout[index]["binary_interaction"]) > 0.0
        for index in range(len(datasets))
    )
    monotone_effect = sum(
        float(row["detail_monotone_gain"]) >= float(gates["monotone_gain_min"])
        for row in validation
    )
    monotone_best = sum(
        int(row["detail_canonical_best_count"])
        >= int(gates["monotone_best_count_min"])
        for row in validation
    )
    monotone_permutation = sum(
        float(row["detail_permutation_p"])
        <= float(gates["monotone_permutation_p_max"])
        for row in validation
    )
    monotone_replicate_positive = sum(
        float(row["detail_monotone_gain"]) > 0.0
        for row in canonical_validation_replicates
    )
    required = int(gates["dataset_required"])
    replicate_required = int(gates["replicate_positive_required"])
    binary_checks = {
        "dataset_effect": binary_effect >= required,
        "directional_selectivity": binary_directions >= required,
        "paired_controls": binary_control >= required,
        "replicate_sign": binary_replicate_positive >= replicate_required,
        "holdout_validation_sign": binary_split_stability >= required,
    }
    monotone_checks = {
        "dataset_effect": monotone_effect >= required,
        "canonical_best": monotone_best >= required,
        "mapping_permutation": monotone_permutation >= required,
        "paired_controls": monotone_control >= required,
        "replicate_sign": monotone_replicate_positive >= replicate_required,
    }
    tolerance = float(gates["orthogonality_max_abs"])
    invariant_checks = {
        "metadata_count": len(metadata) == len(datasets),
        "no_test": all(not bool(row["uses_test_split"]) for row in metadata),
        "no_forecast_update": all(
            not bool(row["forecast_model_updated"])
            and not bool(row["forecast_model_trained"])
            for row in metadata
        ),
        "fit_holdout_disjoint": all(
            not bool(row["fit_holdout_observation_overlap"])
            and int(row["fit_holdout_index_gap"]) >= 2 * int(design["series_length"])
            for row in metadata
        ),
        "orthogonality": all(
            max(
                float(row["dct_orthogonality_max_abs"]),
                float(row["rgnb_orthogonality_max_abs"]),
                float(row["sketch_orthogonality_max_abs"]),
            )
            <= tolerance
            for row in metadata
        ),
        "row_widths": True,
    }
    binary_pass = all(binary_checks.values())
    monotone_pass = all(monotone_checks.values())
    invariant_pass = all(invariant_checks.values())
    if not invariant_pass:
        decision = "diagnostic_invalid_for_direction_rejection"
    elif monotone_pass:
        decision = "raw_detail_monotone_supported_return_step4"
    elif binary_pass:
        decision = "raw_binary_global_detail_supported_return_step4"
    else:
        decision = "raw_aligned_scale_not_supported_rollback_step2"
    return {
        "diagnostic_id": "SC1-D10",
        "decision": decision,
        "invariant_pass": invariant_pass,
        "binary_pass": binary_pass,
        "detail_monotone_pass": monotone_pass,
        "method_implementation_authorized": False,
        "sc2_authorized": False,
        "counts": {
            "binary_effect_datasets": binary_effect,
            "binary_direction_datasets": binary_directions,
            "binary_control_datasets": binary_control,
            "binary_positive_replicates": binary_replicate_positive,
            "binary_split_stability_datasets": binary_split_stability,
            "monotone_effect_datasets": monotone_effect,
            "monotone_best_datasets": monotone_best,
            "monotone_permutation_datasets": monotone_permutation,
            "monotone_control_datasets": monotone_control,
            "monotone_positive_replicates": monotone_replicate_positive,
        },
        "binary_checks": binary_checks,
        "monotone_checks": monotone_checks,
        "invariant_checks": invariant_checks,
    }


def write_report(
    path: Path,
    gate: dict[str, Any],
    dataset_rows: list[dict[str, Any]],
    datasets: list[str],
) -> None:
    lookup = {
        (row["dataset"], row["family"], row["split"]): row
        for row in dataset_rows
    }
    lines = [
        "# SC1-D10 Raw Scale Identifiability Result",
        "",
        "## Decision",
        "",
        f"- `decision`: `{gate['decision']}`；",
        f"- `binary_pass`: `{str(gate['binary_pass']).lower()}`；",
        f"- `detail_monotone_pass`: `{str(gate['detail_monotone_pass']).lower()}`；",
        f"- `invariant_pass`: `{str(gate['invariant_pass']).lower()}`；",
        "- method/test/SC2 authorization: `false`。",
        "",
        "## Canonical Validation Aggregates",
        "",
        "| Dataset | binary interaction | global selectivity | detail selectivity | detail monotone gain | best count | permutation p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in datasets:
        row = lookup[(dataset, "canonical", "validation")]
        lines.append(
            f"| {dataset} | {row['binary_interaction']:.6f} | "
            f"{row['global_selectivity']:.6f} | {row['detail_selectivity']:.6f} | "
            f"{row['detail_monotone_gain']:.6f} | "
            f"{int(row['detail_canonical_best_count'])}/6 | "
            f"{row['detail_permutation_p']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "D10是raw-data、capacity-matched、validation-only diagnostic。任何positive decision只返回Step4 "
            "candidate design，不证明architecture effectiveness；negative decision只关闭aligned-scale routing "
            "problem，不否定future-side RGNB/projectivity。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def synthetic_smoke() -> None:
    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]
    matrix_rows = []
    binary_rows = []
    for dataset in datasets:
        for family in FAMILIES:
            for sketch_seed in (1, 2, 3):
                for ridge_lambda in (0.001, 0.01, 0.1):
                    for split in SPLITS:
                        for future_index in range(7):
                            for history_index in range(7):
                                canonical = family == "canonical" and future_index == history_index
                                matrix_rows.append(
                                    {
                                        "dataset": dataset,
                                        "family": family,
                                        "sketch_seed": str(sketch_seed),
                                        "ridge_lambda": str(ridge_lambda),
                                        "split": split,
                                        "future_group_index": str(future_index),
                                        "history_group_index": str(history_index),
                                        "r2": "0.25" if canonical else "0.05",
                                    }
                                )
                        for future_index in range(2):
                            for history_index in range(2):
                                canonical = family == "canonical" and future_index == history_index
                                binary_rows.append(
                                    {
                                        "dataset": dataset,
                                        "family": family,
                                        "sketch_seed": str(sketch_seed),
                                        "ridge_lambda": str(ridge_lambda),
                                        "split": split,
                                        "future_binary_index": str(future_index),
                                        "history_binary_index": str(history_index),
                                        "r2": "0.30" if canonical else "0.10",
                                    }
                                )
    replicate_rows = build_replicate_metrics(matrix_rows, binary_rows)
    dataset_rows = build_dataset_metrics(matrix_rows, binary_rows)
    dataset_matrix_cells = aggregate_cell_rows(
        matrix_rows, ("future_group_index", "history_group_index")
    )
    dataset_binary_cells = aggregate_cell_rows(
        binary_rows, ("future_binary_index", "history_binary_index")
    )
    metadata = [
        {
            "uses_test_split": False,
            "forecast_model_updated": False,
            "forecast_model_trained": False,
            "fit_holdout_observation_overlap": False,
            "fit_holdout_index_gap": 1440,
            "dct_orthogonality_max_abs": 0.0,
            "rgnb_orthogonality_max_abs": 0.0,
            "sketch_orthogonality_max_abs": 0.0,
        }
        for _dataset in datasets
    ]
    design = {
        "datasets": datasets,
        "series_length": 720,
        "sketch_width": 16,
        "gates": {
            "dataset_required": 4,
            "replicate_positive_required": 36,
            "binary_interaction_min": 0.01,
            "binary_control_gain_min": 0.005,
            "monotone_gain_min": 0.01,
            "monotone_best_count_min": 4,
            "monotone_permutation_p_max": 0.05,
            "monotone_control_gain_min": 0.005,
            "orthogonality_max_abs": 1e-8,
        },
    }
    gate = apply_gate(replicate_rows, dataset_rows, metadata, design)
    if gate["decision"] != "raw_detail_monotone_supported_return_step4":
        raise RuntimeError(f"synthetic analyzer smoke failed: {gate}")
    print("stage_c_sc1_d10_analyzer_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    design = json.loads(args.design.read_text(encoding="utf-8"))
    matrix_rows: list[dict[str, str]] = []
    binary_rows: list[dict[str, str]] = []
    metadata = []
    for dataset in [str(value) for value in design["datasets"]]:
        dataset_dir = args.input_root / dataset
        matrix_rows.extend(read_csv(dataset_dir / "matrix_cell_metrics.csv"))
        binary_rows.extend(read_csv(dataset_dir / "binary_cell_metrics.csv"))
        metadata.append(
            json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
        )
    expected_matrix_rows = len(design["datasets"]) * 2646
    expected_binary_rows = len(design["datasets"]) * 216
    if len(matrix_rows) != expected_matrix_rows or len(binary_rows) != expected_binary_rows:
        raise RuntimeError(
            f"artifact completeness failed: matrix={len(matrix_rows)}/{expected_matrix_rows}, "
            f"binary={len(binary_rows)}/{expected_binary_rows}"
        )
    if any(
        int(row["input_width"]) != int(design["sketch_width"])
        or int(row["output_width"]) != int(design["sketch_width"])
        for row in matrix_rows + binary_rows
    ):
        raise RuntimeError("capacity-matched width invariant failed")
    replicate_rows = build_replicate_metrics(matrix_rows, binary_rows)
    dataset_rows = build_dataset_metrics(matrix_rows, binary_rows)
    dataset_matrix_cells = aggregate_cell_rows(
        matrix_rows, ("future_group_index", "history_group_index")
    )
    dataset_binary_cells = aggregate_cell_rows(
        binary_rows, ("future_binary_index", "history_binary_index")
    )
    gate = apply_gate(replicate_rows, dataset_rows, metadata, design)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "replicate_metrics.csv", replicate_rows)
    write_csv(args.output_dir / "dataset_metrics.csv", dataset_rows)
    write_csv(args.output_dir / "dataset_matrix_cells.csv", dataset_matrix_cells)
    write_csv(args.output_dir / "dataset_binary_cells.csv", dataset_binary_cells)
    (args.output_dir / "gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(
        args.output_dir / "research_interpretation.md",
        gate,
        dataset_rows,
        [str(value) for value in design["datasets"]],
    )
    print(f"stage_c_sc1_d10_analysis_done decision={gate['decision']}")


if __name__ == "__main__":
    main()
