#!/usr/bin/env python3
"""Aggregate StageC D1 PMFO/PIR offline diagnostic artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DATASETS = ("Weather", "ETTm1", "ETTh2")
NONUNIFORM_MEASURES = ("uniform_h", "log_uniform_h", "benchmark_h")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_smoke and (args.input_root is None or args.output_dir is None):
        parser.error("--input-root and --output-dir are required")
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def selected_mean(
    rows: list[dict[str, str]],
    value_key: str,
    **conditions: str,
) -> float:
    values = [
        float(row[value_key])
        for row in rows
        if all(row[key] == value for key, value in conditions.items())
    ]
    return mean(values)


def probe_level_mean(
    rows: list[dict[str, str]],
    dataset: str,
    feature: str,
) -> float:
    values = [
        float(row["r2"])
        for row in rows
        if row["dataset"] == dataset
        and row["feature"] == feature
        and row["source"] == "label"
        and row["family"] == "dct"
        and int(row["level_end_rank"]) <= 72
    ]
    return mean(values)


def analyze_dataset(
    dataset: str,
    structure: list[dict[str, str]],
    probes: list[dict[str, str]],
    geometry: list[dict[str, str]],
    gradients: list[dict[str, str]],
    frozen_decoder: list[dict[str, str]],
) -> dict[str, Any]:
    structure_dataset = [row for row in structure if row["dataset"] == dataset]
    probe_dataset = [row for row in probes if row["dataset"] == dataset]
    geometry_dataset = [row for row in geometry if row["dataset"] == dataset]
    gradient_dataset = [row for row in gradients if row["dataset"] == dataset]
    frozen_dataset = [row for row in frozen_decoder if row["dataset"] == dataset]

    captures: dict[tuple[str, str], float] = {}
    for source in ("label", "residual"):
        for family in ("dct", "random", "block"):
            captures[(source, family)] = selected_mean(
                structure_dataset,
                "cumulative_energy_share",
                source=source,
                family=family,
                cumulative_rank="144",
            )
    label_advantage = max(captures[("label", "dct")], captures[("label", "block")]) - captures[("label", "random")]
    residual_advantage = max(
        captures[("residual", "dct")],
        captures[("residual", "block")],
    ) - captures[("residual", "random")]
    label_structure_pass = label_advantage >= 0.10
    residual_structure_pass = residual_advantage >= 0.02

    full_probe = probe_level_mean(probe_dataset, dataset, "full_hidden")
    shuffled_probe = probe_level_mean(probe_dataset, dataset, "patch_shuffled")
    raw_probe = probe_level_mean(probe_dataset, dataset, "raw_history")
    probe_gain = full_probe - shuffled_probe
    retention = full_probe / raw_probe if raw_probe > 0.0 else float("nan")
    linear_probe_pass = full_probe >= 0.05 and probe_gain >= 0.01

    model_r2 = mean([float(row["model_r2_vs_zero_deviation"]) for row in frozen_dataset])
    shuffle_increase = mean(
        [float(row["shuffle_relative_sse_increase"]) for row in frozen_dataset]
    )
    collapse_increase = mean(
        [float(row["collapse_relative_sse_increase"]) for row in frozen_dataset]
    )
    reconstruction_max_abs = max(
        (float(row["forward_reconstruction_max_abs"]) for row in frozen_dataset),
        default=float("inf"),
    )
    ordered_memory_effect = max(shuffle_increase, collapse_increase)
    frozen_decoder_pass = (
        model_r2 > 0.0
        and ordered_memory_effect >= 0.01
        and reconstruction_max_abs <= 1e-5
    )

    learned_label_capture = selected_mean(
        structure_dataset,
        "cumulative_energy_share",
        source="label",
        family="learned_basis_subspace",
        cumulative_rank="256",
    )
    dct_label_capture = selected_mean(
        structure_dataset,
        "cumulative_energy_share",
        source="label",
        family="dct",
        cumulative_rank="256",
    )
    learned_residual_capture = selected_mean(
        structure_dataset,
        "cumulative_energy_share",
        source="residual",
        family="learned_basis_subspace",
        cumulative_rank="256",
    )
    basis_entropy = selected_mean(
        geometry_dataset,
        "value",
        metric="column_entropy_mean",
        scale="all",
    )
    basis_support = selected_mean(
        geometry_dataset,
        "value",
        metric="support_90_fraction_mean",
        scale="all",
    )
    basis_effective_rank = selected_mean(
        geometry_dataset,
        "value",
        metric="effective_rank",
        scale="all",
    )

    parseval_rows = [
        row
        for row in gradient_dataset
        if row["form"] == "projected"
        and row["measure"] == "delta_720"
        and row["module"] == "all"
    ]
    parseval_max_gap = max(
        (float(row["loss_relative_gap_to_same_measure_raw"]) for row in parseval_rows),
        default=float("inf"),
    )
    parseval_min_cosine = min(
        (float(row["cosine_to_same_measure_raw"]) for row in parseval_rows),
        default=float("-inf"),
    )
    parseval_pass = parseval_max_gap <= 1e-4 and parseval_min_cosine >= 0.9999

    raw_measure_rows = [
        row
        for row in gradient_dataset
        if row["form"] == "raw"
        and row["measure"] in NONUNIFORM_MEASURES
        and row["module"] == "all"
    ]
    measure_gradient_separation = mean(
        [1.0 - float(row["cosine_to_raw_delta720"]) for row in raw_measure_rows]
    )
    raw_separation_by_measure = {
        measure: mean(
            [
                1.0 - float(row["cosine_to_raw_delta720"])
                for row in raw_measure_rows
                if row["measure"] == measure
            ]
        )
        for measure in NONUNIFORM_MEASURES
    }
    measure_gradient_pass = measure_gradient_separation >= 0.005
    projected_rows = [
        row
        for row in gradient_dataset
        if row["form"] == "projected"
        and row["measure"] in NONUNIFORM_MEASURES
        and row["module"] == "all"
    ]
    projected_excess_separation = mean(
        [1.0 - float(row["cosine_to_same_measure_raw"]) for row in projected_rows]
    )
    projected_separation_by_measure = {
        measure: mean(
            [
                1.0 - float(row["cosine_to_same_measure_raw"])
                for row in projected_rows
                if row["measure"] == measure
            ]
        )
        for measure in NONUNIFORM_MEASURES
    }
    projected_separation_pass = projected_excess_separation >= 0.005

    return {
        "dataset": dataset,
        "label_capture_dct_rank144": captures[("label", "dct")],
        "label_capture_block_rank144": captures[("label", "block")],
        "label_capture_random_rank144": captures[("label", "random")],
        "label_structured_advantage": label_advantage,
        "label_structure_pass": label_structure_pass,
        "residual_capture_dct_rank144": captures[("residual", "dct")],
        "residual_capture_block_rank144": captures[("residual", "block")],
        "residual_capture_random_rank144": captures[("residual", "random")],
        "residual_structured_advantage": residual_advantage,
        "residual_structure_pass": residual_structure_pass,
        "full_hidden_coarse_mid_r2": full_probe,
        "shuffled_coarse_mid_r2": shuffled_probe,
        "raw_history_coarse_mid_r2": raw_probe,
        "full_vs_shuffled_r2_gain": probe_gain,
        "full_vs_raw_r2_retention": retention,
        "linear_probe_pass": linear_probe_pass,
        "frozen_decoder_r2_vs_zero_deviation": model_r2,
        "frozen_decoder_shuffle_relative_increase": shuffle_increase,
        "frozen_decoder_collapse_relative_increase": collapse_increase,
        "frozen_decoder_ordered_memory_effect": ordered_memory_effect,
        "forward_reconstruction_max_abs": reconstruction_max_abs,
        "encoder_sufficiency_pass": frozen_decoder_pass,
        "learned_basis_label_capture_rank256": learned_label_capture,
        "dct_label_capture_rank256": dct_label_capture,
        "learned_basis_residual_capture_rank256": learned_residual_capture,
        "basis_effective_rank": basis_effective_rank,
        "basis_column_entropy": basis_entropy,
        "basis_support_90_fraction": basis_support,
        "parseval_max_relative_gap": parseval_max_gap,
        "parseval_min_gradient_cosine": parseval_min_cosine,
        "parseval_pass": parseval_pass,
        "raw_measure_gradient_separation": measure_gradient_separation,
        "raw_uniform_h_gradient_separation": raw_separation_by_measure["uniform_h"],
        "raw_log_uniform_h_gradient_separation": raw_separation_by_measure["log_uniform_h"],
        "raw_benchmark_h_gradient_separation": raw_separation_by_measure["benchmark_h"],
        "measure_gradient_pass": measure_gradient_pass,
        "projected_excess_gradient_separation": projected_excess_separation,
        "projected_uniform_h_gradient_separation": projected_separation_by_measure["uniform_h"],
        "projected_log_uniform_h_gradient_separation": projected_separation_by_measure["log_uniform_h"],
        "projected_benchmark_h_gradient_separation": projected_separation_by_measure["benchmark_h"],
        "projected_separation_pass": projected_separation_pass,
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    header = "| " + " | ".join(label for label, _key in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for _label, key in columns:
            value = row[key]
            if isinstance(value, bool):
                values.append("pass" if value else "fail")
            elif isinstance(value, float):
                values.append("nan" if math.isnan(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return [header, separator, *body]


def synthetic_smoke() -> None:
    dataset = "Weather"
    structure = []
    for source, advantage in (("label", 0.20), ("residual", 0.05)):
        structure.extend(
            [
                {
                    "dataset": dataset,
                    "source": source,
                    "family": "dct",
                    "cumulative_rank": "144",
                    "cumulative_energy_share": str(0.5 + advantage),
                },
                {
                    "dataset": dataset,
                    "source": source,
                    "family": "block",
                    "cumulative_rank": "144",
                    "cumulative_energy_share": str(0.45 + advantage),
                },
                {
                    "dataset": dataset,
                    "source": source,
                    "family": "random",
                    "cumulative_rank": "144",
                    "cumulative_energy_share": "0.5",
                },
                {
                    "dataset": dataset,
                    "source": source,
                    "family": "learned_basis_subspace",
                    "cumulative_rank": "256",
                    "cumulative_energy_share": "0.8",
                },
                {
                    "dataset": dataset,
                    "source": source,
                    "family": "dct",
                    "cumulative_rank": "256",
                    "cumulative_energy_share": "0.85",
                },
            ]
        )
    probes = []
    for feature, r2 in (("full_hidden", 0.40), ("patch_shuffled", 0.20), ("raw_history", 0.45)):
        probes.append(
            {
                "dataset": dataset,
                "feature": feature,
                "source": "label",
                "family": "dct",
                "level_end_rank": "72",
                "r2": str(r2),
            }
        )
    geometry = [
        {"dataset": dataset, "metric": "column_entropy_mean", "scale": "all", "value": "0.9"},
        {"dataset": dataset, "metric": "support_90_fraction_mean", "scale": "all", "value": "0.8"},
        {"dataset": dataset, "metric": "effective_rank", "scale": "all", "value": "200"},
    ]
    gradients = [
        {
            "dataset": dataset,
            "form": "projected",
            "measure": "delta_720",
            "module": "all",
            "loss_relative_gap_to_same_measure_raw": "0",
            "cosine_to_same_measure_raw": "1",
            "cosine_to_raw_delta720": "1",
        }
    ]
    for measure in NONUNIFORM_MEASURES:
        gradients.extend(
            [
                {
                    "dataset": dataset,
                    "form": "raw",
                    "measure": measure,
                    "module": "all",
                    "loss_relative_gap_to_same_measure_raw": "0",
                    "cosine_to_same_measure_raw": "1",
                    "cosine_to_raw_delta720": "0.99",
                },
                {
                    "dataset": dataset,
                    "form": "projected",
                    "measure": measure,
                    "module": "all",
                    "loss_relative_gap_to_same_measure_raw": "0.01",
                    "cosine_to_same_measure_raw": "0.99",
                    "cosine_to_raw_delta720": "0.98",
                },
            ]
        )
    frozen_decoder = [
        {
            "dataset": dataset,
            "model_r2_vs_zero_deviation": "0.2",
            "shuffle_relative_sse_increase": "0.05",
            "collapse_relative_sse_increase": "0.10",
            "forward_reconstruction_max_abs": "0.0",
        }
    ]
    row = analyze_dataset(
        dataset,
        structure,
        probes,
        geometry,
        gradients,
        frozen_decoder,
    )
    if not row["label_structure_pass"] or not row["encoder_sufficiency_pass"] or not row["parseval_pass"]:
        raise RuntimeError("analyzer synthetic gate invariant failed")
    print("stage_c_d1_analyzer_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    inputs: dict[str, list[dict[str, str]]] = defaultdict(list)
    metadata: list[dict[str, Any]] = []
    complete = True
    for dataset in DATASETS:
        dataset_root = args.input_root / dataset
        required = {
            "structure": dataset_root / "d1_structure_metrics.csv",
            "probes": dataset_root / "d1_probe_metrics.csv",
            "geometry": dataset_root / "d1_basis_geometry.csv",
            "gradients": dataset_root / "d1_gradient_metrics.csv",
            "frozen_decoder": dataset_root / "d1_frozen_decoder_metrics.csv",
        }
        for name, path in required.items():
            if not path.exists():
                complete = False
                continue
            inputs[name].extend(read_csv(path))
        metadata_path = dataset_root / "d1_metadata.json"
        if metadata_path.exists():
            metadata.extend(json.loads(metadata_path.read_text(encoding="utf-8")))
        else:
            complete = False
    if not complete:
        raise SystemExit("D1 artifacts incomplete; all three dataset directories are required")
    if any(row.get("uses_test_split") for row in metadata):
        raise ValueError("D1 diagnostic must not use the test split")
    if any(row.get("trains_forecast_model") for row in metadata):
        raise ValueError("D1 diagnostic must not train a forecast model")
    expected_source_space = "evaluation_space_future_deviation_and_residual"
    if any(row.get("source_space") != expected_source_space for row in metadata):
        raise ValueError("D1 v2 requires evaluation-space future deviation and residual")

    dataset_rows = [
        analyze_dataset(
            dataset,
            inputs["structure"],
            inputs["probes"],
            inputs["geometry"],
            inputs["gradients"],
            inputs["frozen_decoder"],
        )
        for dataset in DATASETS
    ]
    label_structure_passes = sum(row["label_structure_pass"] for row in dataset_rows)
    residual_structure_passes = sum(row["residual_structure_pass"] for row in dataset_rows)
    encoder_passes = sum(row["encoder_sufficiency_pass"] for row in dataset_rows)
    measure_passes = sum(row["measure_gradient_pass"] for row in dataset_rows)
    projected_passes = sum(row["projected_separation_pass"] for row in dataset_rows)
    parseval_pass = all(row["parseval_pass"] for row in dataset_rows)
    pmfo_pass = label_structure_passes >= 2 and residual_structure_passes >= 2 and encoder_passes >= 2
    pir_pass = parseval_pass and measure_passes >= 2 and projected_passes >= 2
    if pmfo_pass and pir_pass:
        decision = "pmfo_pir_problem_gate_passed"
    elif pmfo_pass:
        decision = "pmfo_passed_pir_rollback_required"
    elif pir_pass:
        decision = "pir_passed_pmfo_rollback_required"
    else:
        decision = "pmfo_pir_problem_gate_not_passed"

    summary = {
        "candidate": "SC1/SC2-D1",
        "complete": complete,
        "datasets": list(DATASETS),
        "seed_rows": len(metadata),
        "uses_test_split": False,
        "trains_forecast_model": False,
        "source_space": expected_source_space,
        "gates": {
            "label_structure_dataset_passes": label_structure_passes,
            "residual_structure_dataset_passes": residual_structure_passes,
            "encoder_sufficiency_dataset_passes": encoder_passes,
            "parseval_pass": parseval_pass,
            "measure_gradient_dataset_passes": measure_passes,
            "projected_separation_dataset_passes": projected_passes,
            "pmfo_problem_gate": pmfo_pass,
            "pir_problem_gate": pir_pass,
        },
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "d1_dataset_gate.csv", dataset_rows)
    (args.output_dir / "d1_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    report = [
        "# StageC D1 PMFO/PIR Offline Diagnostic Report",
        "",
        "## Protocol Audit",
        "",
        f"- datasets: `{', '.join(DATASETS)}`；seed/profile instances: `{len(metadata)}`；",
        "- test split: `false`；new forecast-model training: `false`；",
        "- primary space: evaluation-space future deviation / frozen-A6 residual；",
        "- diagnostics: nested energy、frozen-encoder ridge probes、frozen-decoder "
        "counterfactual、learned-basis geometry、measure/projected gradients。",
        "",
        "## PMFO Evidence",
        "",
        *markdown_table(
            dataset_rows,
            [
                ("Dataset", "dataset"),
                ("Label adv.", "label_structured_advantage"),
                ("Residual adv.", "residual_structured_advantage"),
                ("Full R2", "full_hidden_coarse_mid_r2"),
                ("Shuffle gain", "full_vs_shuffled_r2_gain"),
                ("Linear", "linear_probe_pass"),
                ("Frozen R2", "frozen_decoder_r2_vs_zero_deviation"),
                ("Order effect", "frozen_decoder_ordered_memory_effect"),
                ("Encoder", "encoder_sufficiency_pass"),
            ],
        ),
        "",
        f"[Decision] PMFO problem gate: `{'pass' if pmfo_pass else 'fail'}`。Label structure passes="
        f"`{label_structure_passes}/3`，residual structure passes=`{residual_structure_passes}/3`，"
        f"encoder sufficiency passes=`{encoder_passes}/3`。",
        "[Scope] Encoder gate只证明当前frozen decoder确实利用有序patch memory，"
        "不等价于Encoder已经提供了完备的multiresolution sufficient statistics。",
        "",
        "## Basis Geometry",
        "",
        *markdown_table(
            dataset_rows,
            [
                ("Dataset", "dataset"),
                ("Learned label@256", "learned_basis_label_capture_rank256"),
                ("DCT label@256", "dct_label_capture_rank256"),
                ("Learned residual@256", "learned_basis_residual_capture_rank256"),
                ("Effective rank", "basis_effective_rank"),
                ("Entropy", "basis_column_entropy"),
                ("Support90", "basis_support_90_fraction"),
            ],
        ),
        "",
        "[Fact] 当前 learned basis 无 nested/refinement constraint；本表只判断其容量、"
        "条件与局部化几何，"
        "不能把高 capture 解释为 PMFO 已存在。",
        "",
        "## PIR Evidence",
        "",
        *markdown_table(
            dataset_rows,
            [
                ("Dataset", "dataset"),
                ("Parseval gap", "parseval_max_relative_gap"),
                ("Parseval cosine", "parseval_min_gradient_cosine"),
                ("Raw measure sep.", "raw_measure_gradient_separation"),
                ("Projected excess", "projected_excess_gradient_separation"),
                ("Measure", "measure_gradient_pass"),
                ("Projected", "projected_separation_pass"),
            ],
        ),
        "",
        f"[Decision] PIR problem gate: `{'pass' if pir_pass else 'fail'}`。Parseval invariant="
        f"`{'pass' if parseval_pass else 'fail'}`，measure passes=`{measure_passes}/3`，"
        f"projected-excess passes=`{projected_passes}/3`。",
        "",
        "### Per-Measure Separation Audit",
        "",
        *markdown_table(
            dataset_rows,
            [
                ("Dataset", "dataset"),
                ("Raw uniform", "raw_uniform_h_gradient_separation"),
                ("Raw log", "raw_log_uniform_h_gradient_separation"),
                ("Raw benchmark", "raw_benchmark_h_gradient_separation"),
                ("PIR uniform", "projected_uniform_h_gradient_separation"),
                ("PIR log", "projected_log_uniform_h_gradient_separation"),
                ("PIR benchmark", "projected_benchmark_h_gradient_separation"),
            ],
        ),
        "",
        "[Scope] Aggregate PIR gate必须结合per-measure列解释，不能用log-uniform的强差异替代"
        "benchmark measure下的独立证据。",
        "",
        "## Overall Decision And Failure Attribution",
        "",
        f"[Decision] `{decision}`。",
        "",
        "若 Parseval invariant失败，本诊断只能标记 "
        "`diagnostic_invalid_for_direction_rejection`，因为projection或gradient实现存在问题。"
        "若invariant通过但structure/probe/gradient gate失败，结论只针对当前"
        "PMFO/PIR problem formulation；"
        "不得扩大为所有 multiresolution decoder 或 training strategy 的方向级否决。",
    ]
    (args.output_dir / "d1_diagnostic_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(f"stage_c_d1_analysis_done decision={decision} datasets={len(dataset_rows)}")


if __name__ == "__main__":
    main()
