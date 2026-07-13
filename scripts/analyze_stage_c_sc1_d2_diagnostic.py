#!/usr/bin/env python3
"""Aggregate SC1-D2 frozen-memory rank/nonlinearity/scale diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


CORE3 = ("Weather", "ETTm1", "ETTh2")
FORMAL5 = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
DENSE_ARMS = ("dense_nonlinear_param_h197", "dense_nonlinear_units_h352")
RANDOM_PREFIXES = ("random_group_", "random_basis_")
EXPECTED_SEEDS = (2021, 2022, 2023)
MSE_MARGIN = 0.005
MAE_GUARD = -0.0025


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--expected-suite",
        choices=["core3", "formal5"],
        default="core3",
    )
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if args.synthetic_smoke:
        return args
    if args.input_root is None or args.output_dir is None:
        parser.error("--input-root and --output-dir are required")
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


def relative_improvement(control: float, candidate: float) -> float:
    return (control - candidate) / max(abs(control), 1e-12)


def mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def standard_deviation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def is_random_arm(arm: str) -> bool:
    return arm.startswith(RANDOM_PREFIXES)


def pairwise_rows(metrics: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in metrics:
        key = (row["dataset"], int(row["checkpoint_seed"]))
        grouped[key][row["arm"]] = row
    outputs: list[dict[str, Any]] = []
    for (dataset, seed), arms in sorted(grouped.items()):
        required = {
            "rank256_linear",
            "full_affine",
            "true_scale_grouped",
            *DENSE_ARMS,
        }
        missing = sorted(required - set(arms))
        random_rows = [row for arm, row in arms.items() if is_random_arm(arm)]
        if missing or len(random_rows) != 6:
            raise ValueError(
                f"incomplete D2 arms for {dataset} seed{seed}: "
                f"missing={missing}, random={len(random_rows)}"
            )
        rank = float(arms["rank256_linear"]["val_mse_eval"])
        full = float(arms["full_affine"]["val_mse_eval"])
        dense_arm = min(DENSE_ARMS, key=lambda name: float(arms[name]["val_mse_eval"]))
        dense = float(arms[dense_arm]["val_mse_eval"])
        dense_mae = float(arms[dense_arm]["val_mae_eval"])
        true = float(arms["true_scale_grouped"]["val_mse_eval"])
        true_mae = float(arms["true_scale_grouped"]["val_mae_eval"])
        random_values = sorted(float(row["val_mse_eval"]) for row in random_rows)
        random_median = 0.5 * (random_values[2] + random_values[3])
        outputs.append(
            {
                "dataset": dataset,
                "checkpoint_seed": seed,
                "best_dense_arm": dense_arm,
                "rank256_mse": rank,
                "full_affine_mse": full,
                "best_dense_mse": dense,
                "true_scale_mse": true,
                "random_median_mse": random_median,
                "full_vs_rank256_improvement": relative_improvement(rank, full),
                "dense_vs_full_improvement": relative_improvement(full, dense),
                "true_vs_dense_improvement": relative_improvement(dense, true),
                "true_vs_random_median_improvement": relative_improvement(random_median, true),
                "true_vs_dense_mae_improvement": relative_improvement(dense_mae, true_mae),
                "random_controls_beaten": sum(true < value for value in random_values),
            }
        )
    return outputs


def dataset_summary(pairwise: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairwise:
        grouped[str(row["dataset"])].append(row)
    fields = (
        "full_vs_rank256_improvement",
        "dense_vs_full_improvement",
        "true_vs_dense_improvement",
        "true_vs_random_median_improvement",
        "true_vs_dense_mae_improvement",
    )
    outputs = []
    for dataset, rows in sorted(grouped.items()):
        output: dict[str, Any] = {
            "dataset": dataset,
            "seed_count": len(rows),
        }
        for field in fields:
            values = [float(row[field]) for row in rows]
            output[f"mean_{field}"] = mean(values)
            output[f"std_{field}"] = standard_deviation(values)
            output[f"positive_seeds_{field}"] = sum(value > 0.0 for value in values)
        output["mean_random_controls_beaten"] = mean(
            [float(row["random_controls_beaten"]) for row in rows]
        )
        outputs.append(output)
    return outputs


def comparison_gate(
    pairwise: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    macro = mean([float(row[field]) for row in pairwise])
    passing_datasets = 0
    for row in summaries:
        if int(row[f"positive_seeds_{field}"]) >= 2:
            passing_datasets += 1
    return {
        "macro_improvement": macro,
        "datasets_with_at_least_2_positive_seeds": passing_datasets,
        "pass": macro >= MSE_MARGIN and passing_datasets >= 3,
    }


def build_summary(
    metrics: list[dict[str, str]],
    metadata: list[dict[str, Any]],
    expected_suite: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pairwise = pairwise_rows(metrics)
    summaries = dataset_summary(pairwise)
    expected = CORE3 if expected_suite == "core3" else FORMAL5
    observed = tuple(sorted({row["dataset"] for row in pairwise}))
    expected_sorted = tuple(sorted(expected))
    complete = observed == expected_sorted and all(
        sum(row["dataset"] == dataset for row in pairwise) == len(EXPECTED_SEEDS)
        for dataset in expected
    )
    invariant_gate = all(
        not item["uses_test_split"]
        and not item["forecast_model_updated"]
        and not item["official_validation_used_for_early_stopping"]
        and float(item["basis_orthogonality_max_abs"]) <= 1e-5
        and float(item["parseval_relative_gap"]) <= 1e-5
        for item in metadata
    )
    rank_gate = comparison_gate(pairwise, summaries, "full_vs_rank256_improvement")
    nonlinear_gate = comparison_gate(pairwise, summaries, "dense_vs_full_improvement")
    scale_dense_gate = comparison_gate(pairwise, summaries, "true_vs_dense_improvement")
    scale_random_gate = comparison_gate(
        pairwise, summaries, "true_vs_random_median_improvement"
    )
    mae_macro = mean(
        [float(row["true_vs_dense_mae_improvement"]) for row in pairwise]
    )
    random_beaten = mean([float(row["random_controls_beaten"]) for row in pairwise])
    formal_authorized = expected_suite == "formal5" and complete and invariant_gate
    scale_alignment_pass = (
        formal_authorized
        and scale_dense_gate["pass"]
        and scale_random_gate["pass"]
        and mae_macro >= MAE_GUARD
        and random_beaten >= 5.0
    )
    if expected_suite == "core3":
        decision = "core3_precheck_only_formal5_pending"
    elif not complete or not invariant_gate:
        decision = "diagnostic_incomplete_or_invalid"
    elif scale_alignment_pass:
        decision = "scale_alignment_problem_supported_return_step4"
    else:
        decision = "scale_alignment_not_supported_reformulate_step2"
    summary = {
        "candidate": "SC1-D2",
        "role": "diagnostic_only",
        "expected_suite": expected_suite,
        "expected_datasets": list(expected),
        "observed_datasets": list(observed),
        "complete": complete,
        "invariant_gate": invariant_gate,
        "rank_expansion_gate": rank_gate,
        "generic_nonlinearity_gate": nonlinear_gate,
        "scale_vs_dense_gate": scale_dense_gate,
        "scale_vs_random_gate": scale_random_gate,
        "scale_vs_dense_mae_macro_improvement": mae_macro,
        "mean_random_controls_beaten": random_beaten,
        "scale_alignment_pass": scale_alignment_pass,
        "decision": decision,
        "method_implementation_authorized": False,
        "uses_test_split": False,
        "thresholds": {
            "macro_mse_improvement": MSE_MARGIN,
            "datasets_with_2_of_3_positive_seeds": 3,
            "mae_macro_guard": MAE_GUARD,
            "mean_random_controls_beaten": 5.0,
        },
    }
    return pairwise, summaries, summary


def report_markdown(
    summaries: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    lines = [
        "# SC1-D2 Frozen-Memory Diagnostic Report",
        "",
        "## Decision Summary",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| `role` | `{summary['role']}` |",
        f"| `suite` | `{summary['expected_suite']}` |",
        f"| `complete` | `{str(summary['complete']).lower()}` |",
        f"| `invariant_gate` | `{str(summary['invariant_gate']).lower()}` |",
        f"| `decision` | `{summary['decision']}` |",
        f"| `method_implementation_authorized` | `false` |",
        "",
        "## Per-Dataset Attribution",
        "",
        "所有数值均为相对control的validation evaluation-space MSE improvement；正值表示后者更好。",
        "",
        "| Dataset | Full vs rank256 | Dense nonlinear vs full | True scale vs best dense | True scale vs random median | Random controls beaten |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['dataset']} | "
            f"{row['mean_full_vs_rank256_improvement']:.4%} | "
            f"{row['mean_dense_vs_full_improvement']:.4%} | "
            f"{row['mean_true_vs_dense_improvement']:.4%} | "
            f"{row['mean_true_vs_random_median_improvement']:.4%} | "
            f"{row['mean_random_controls_beaten']:.2f}/6 |"
        )
    lines.extend(
        [
            "",
            "## Gate Reading",
            "",
            f"- rank expansion macro：`{summary['rank_expansion_gate']['macro_improvement']:.4%}`；",
            f"- generic nonlinearity macro：`{summary['generic_nonlinearity_gate']['macro_improvement']:.4%}`；",
            f"- true scale vs strongest dense macro：`{summary['scale_vs_dense_gate']['macro_improvement']:.4%}`；",
            f"- true scale vs random median macro：`{summary['scale_vs_random_gate']['macro_improvement']:.4%}`；",
            f"- true scale vs strongest dense MAE macro：`{summary['scale_vs_dense_mae_macro_improvement']:.4%}`。",
            "",
        ]
    )
    if summary["expected_suite"] == "core3":
        lines.extend(
            [
                "[Boundary] 本轮是三套已冻结dataset的`core3_precheck`。它可暴露明显机制或数值问题，",
                "但不能形成formal pass，也不能以失败否定方向；ETTh1/ETTm2 profile冻结后必须运行formal5。",
            ]
        )
    elif summary["scale_alignment_pass"]:
        lines.append(
            "[Decision] scale-aligned conditional structure通过problem gate；只允许返回Step 4设计新idea。"
        )
    else:
        lines.append(
            "[Decision] 当前scale-alignment problem未获支持；回到Step 2重定义Contribution 1问题。"
        )
    lines.extend(
        [
            "",
            "## Failure Attribution Boundary",
            "",
            "non-finite loss、basis/Parseval invariant失败、缺arm/seed或official validation参与early stopping时，",
            "结果必须标记`diagnostic_invalid_for_direction_rejection`。若full-affine或dense controls的fit/holdout",
            "明显未收敛，只能怀疑optimization protocol，不能据此否定rank/nonlinearity/scale方向。",
            "",
        ]
    )
    return "\n".join(lines)


def synthetic_smoke() -> None:
    metrics: list[dict[str, str]] = []
    arms = [
        "rank256_linear",
        "full_affine",
        *DENSE_ARMS,
        "true_scale_grouped",
        "random_group_s3101",
        "random_group_s3102",
        "random_group_s3103",
        "random_basis_s3101",
        "random_basis_s3102",
        "random_basis_s3103",
    ]
    for dataset in CORE3:
        for seed in EXPECTED_SEEDS:
            for index, arm in enumerate(arms):
                value = 1.0 - 0.01 * index
                metrics.append(
                    {
                        "dataset": dataset,
                        "checkpoint_seed": str(seed),
                        "arm": arm,
                        "val_mse_eval": str(value),
                        "val_mae_eval": str(math.sqrt(value)),
                    }
                )
    metadata = [
        {
            "uses_test_split": False,
            "forecast_model_updated": False,
            "official_validation_used_for_early_stopping": False,
            "basis_orthogonality_max_abs": 1e-7,
            "parseval_relative_gap": 1e-7,
        }
        for _ in range(9)
    ]
    pairwise, summaries, summary = build_summary(metrics, metadata, "core3")
    if len(pairwise) != 9 or len(summaries) != 3:
        raise RuntimeError("synthetic aggregation completeness failed")
    if summary["decision"] != "core3_precheck_only_formal5_pending":
        raise RuntimeError("core3 decision boundary changed")
    print("stage_c_sc1_d2_analyzer_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    expected = CORE3 if args.expected_suite == "core3" else FORMAL5
    metrics: list[dict[str, str]] = []
    metadata: list[dict[str, Any]] = []
    for dataset in expected:
        dataset_root = args.input_root / dataset
        metrics.extend(read_csv(dataset_root / "d2_probe_metrics.csv"))
        metadata.extend(
            json.loads((dataset_root / "d2_metadata.json").read_text(encoding="utf-8"))
        )
    pairwise, summaries, summary = build_summary(metrics, metadata, args.expected_suite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "d2_pairwise_metrics.csv", pairwise)
    write_csv(args.output_dir / "d2_dataset_summary.csv", summaries)
    (args.output_dir / "d2_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "d2_diagnostic_report.md").write_text(
        report_markdown(summaries, summary),
        encoding="utf-8",
    )
    print(
        f"stage_c_sc1_d2_analysis_done decision={summary['decision']} "
        f"output_dir={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
