#!/usr/bin/env python3
"""Analyze C1 shared-scale and validation-selected carrier feasibility."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = ("ETTh2", "ETTm1", "Weather")
HORIZONS = (96, 192, 336, 720)
SCALES = ("gamp_p16s8", "gamp_p48s24")
SELECTORS = ("last", "best_val")
MIXED_LABEL = "mixed_h96_h192_h336_h720"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--fixed-reference",
        type=Path,
        default=repo_root
        / "analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706"
        / "clean_a6_vs_fixed.csv",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_dir(root: Path, arm: str, dataset: str, seed: int) -> Path:
    return (
        root
        / f"A6_C1_{arm}_balanced_dual"
        / dataset
        / MIXED_LABEL
        / f"seed{seed}"
    )


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def metric_map(
    root: Path,
    arm: str,
    dataset: str,
    selector: str,
    seed: int,
) -> dict[int, dict[str, float]]:
    path = run_dir(root, arm, dataset, seed) / f"metrics_{selector}_by_target_horizon.csv"
    return {
        int(row["target_horizon"]): {
            "mse": float(row["mse"]),
            "mae": float(row["mae"]),
        }
        for row in read_csv(path)
    }


def fixed_map(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    return {
        (row["dataset"], int(row["target_horizon"])): {
            "mse": float(row["reference_mse"]),
            "mae": float(row["reference_mae"]),
        }
        for row in read_csv(path)
        if row["reference"] == "fixed_horizon_timealign" and row["status"] == "ok"
    }


def collect_rows(
    root: Path,
    fixed: dict[tuple[str, int], dict[str, float]],
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in SCALES:
        for selector in SELECTORS:
            for dataset in DATASETS:
                candidate = metric_map(root, arm, dataset, selector, seed)
                baseline = metric_map(root, "a6_clean", dataset, selector, seed)
                for horizon in HORIZONS:
                    fixed_row = fixed[(dataset, horizon)]
                    rows.append(
                        {
                            "arm": arm,
                            "selector": selector,
                            "dataset": dataset,
                            "target_horizon": horizon,
                            "candidate_mse": candidate[horizon]["mse"],
                            "a6_mse": baseline[horizon]["mse"],
                            "relative_mse_vs_a6_pct": pct(
                                candidate[horizon]["mse"], baseline[horizon]["mse"]
                            ),
                            "candidate_mae": candidate[horizon]["mae"],
                            "a6_mae": baseline[horizon]["mae"],
                            "relative_mae_vs_a6_pct": pct(
                                candidate[horizon]["mae"], baseline[horizon]["mae"]
                            ),
                            "fixed_mse": fixed_row["mse"],
                            "relative_mse_vs_fixed_pct": pct(
                                candidate[horizon]["mse"], fixed_row["mse"]
                            ),
                            "wins_vs_fixed": int(
                                candidate[horizon]["mse"] < fixed_row["mse"]
                            ),
                        }
                    )
    return rows


def gate_summary(rows: list[dict[str, Any]], arm: str, selector: str) -> dict[str, Any]:
    selected = [
        row for row in rows if row["arm"] == arm and row["selector"] == selector
    ]
    dataset_means = {
        dataset: mean(
            float(row["relative_mse_vs_a6_pct"])
            for row in selected
            if row["dataset"] == dataset
        )
        for dataset in DATASETS
    }
    dataset_fixed_wins = {
        dataset: sum(
            int(row["wins_vs_fixed"])
            for row in selected
            if row["dataset"] == dataset
        )
        for dataset in DATASETS
    }
    overall_a6 = mean(float(row["relative_mse_vs_a6_pct"]) for row in selected)
    max_a6 = max(float(row["relative_mse_vs_a6_pct"]) for row in selected)
    overall_fixed = mean(float(row["relative_mse_vs_fixed_pct"]) for row in selected)
    fixed_wins = sum(int(row["wins_vs_fixed"]) for row in selected)
    passed = (
        overall_a6 <= 1.0
        and max(dataset_means.values()) <= 1.5
        and max_a6 <= 3.0
        and overall_fixed <= -3.0
        and fixed_wins >= 8
        and min(dataset_fixed_wins.values()) >= 2
    )
    return {
        "arm": arm,
        "selector": selector,
        "overall_mean_mse_vs_a6_pct": overall_a6,
        "max_horizon_mse_vs_a6_pct": max_a6,
        **{
            f"{dataset}_mean_mse_vs_a6_pct": dataset_means[dataset]
            for dataset in DATASETS
        },
        "overall_mean_mse_vs_fixed_pct": overall_fixed,
        "wins_vs_fixed": fixed_wins,
        **{
            f"{dataset}_wins_vs_fixed": dataset_fixed_wins[dataset]
            for dataset in DATASETS
        },
        "gate_pass": int(passed),
    }


def best_validation(root: Path, arm: str, dataset: str, seed: int) -> float:
    rows = read_csv(run_dir(root, arm, dataset, seed) / "training_log.csv")
    return min(float(row["val_mean_mse"]) for row in rows)


def validation_selected_rows(
    rows: list[dict[str, Any]],
    root: Path,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selection: list[dict[str, Any]] = []
    selected_arm: dict[str, str] = {}
    for dataset in DATASETS:
        values = {
            arm: best_validation(root, arm, dataset, seed) for arm in SCALES
        }
        chosen = min(values, key=values.get)
        selected_arm[dataset] = chosen
        selection.append(
            {
                "dataset": dataset,
                "selected_arm": chosen,
                "p16s8_best_val_mean_mse": values["gamp_p16s8"],
                "p48s24_best_val_mean_mse": values["gamp_p48s24"],
            }
        )
    selected_rows = [
        row | {"arm": "validation_selected"}
        for row in rows
        if row["arm"] == selected_arm[row["dataset"]]
    ]
    return selection, selected_rows


def diagnostics_rows(root: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ("a6_clean", *SCALES):
        for dataset in DATASETS:
            path = run_dir(root, arm, dataset, seed) / "model_diagnostics.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "arm": arm,
                    "dataset": dataset,
                    "encoder_mode": payload["encoder_mode"],
                    "active_forward_parameters": payload.get(
                        "active_forward_parameters", ""
                    ),
                    "history_patch_len": payload.get("history_patch_len", ""),
                    "history_patch_stride": payload.get("history_patch_stride", ""),
                    "history_local_patch_num": payload.get(
                        "history_local_patch_num", ""
                    ),
                    "history_token_dropout_p": payload.get(
                        "history_token_dropout_p", ""
                    ),
                    "history_attn_dropout_p": payload.get(
                        "history_attn_dropout_p", ""
                    ),
                    "history_attn_residual_dropout_p": payload.get(
                        "history_attn_residual_dropout_p", ""
                    ),
                    "history_ffn_dropout_p": payload.get(
                        "history_ffn_dropout_p", ""
                    ),
                    "history_ffn_residual_dropout_p": payload.get(
                        "history_ffn_residual_dropout_p", ""
                    ),
                }
            )
    return rows


def protocol_audit_rows(root: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ("a6_clean", *SCALES):
        for dataset in DATASETS:
            path = run_dir(root, arm, dataset, seed) / "effective_config.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            effective_lr = float(payload["official_args"]["learning_rate"])
            source_lr = float(payload["official_preset"]["learning_rate"])
            rows.append(
                {
                    "arm": arm,
                    "dataset": dataset,
                    "effective_learning_rate": effective_lr,
                    "source_preset_learning_rate": source_lr,
                    "source_learning_rate_match": int(effective_lr == source_lr),
                }
            )
    return rows


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values: list[str] = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    protocol_audit: list[dict[str, Any]],
) -> None:
    shared_pass = any(
        all(
            row["gate_pass"] == 1
            for row in summaries
            if row["arm"] == arm
        )
        for arm in SCALES
    )
    selected_pass = all(
        row["gate_pass"] == 1
        for row in summaries
        if row["arm"] == "validation_selected"
    )
    if shared_pass:
        decision = "shared_scale_small_gate_pass"
    elif selected_pass:
        decision = "validation_selected_scale_small_gate_pass"
    else:
        decision = "c1_carrier_normalization_gate_failed"
    lines = [
        "# Phase5 C1 Global-Anchored Multi-Patch Gate Report",
        "",
        "## Decision",
        "",
        f"`{decision}`",
        "",
        "C1 只评估统一 carrier/interface，不构成 StageB 创新点。",
        "",
        "## Protocol audit",
        "",
        *markdown_table(
            protocol_audit,
            [
                "arm",
                "dataset",
                "effective_learning_rate",
                "source_preset_learning_rate",
                "source_learning_rate_match",
            ],
        ),
        "",
        "Runner 对全部 arms 显式使用 `learning_rate=1e-4`。ETTh2 source preset 为 "
        "`5e-4`，所以 ETTh2 A6 不是 source-faithful reproduction；同一 dataset 内的 "
        "C1/A6 controlled comparison仍使用相同 learning rate。ETTm1 与 Weather 无此偏差。",
        "",
        "## Gate summary",
        "",
        *markdown_table(
            summaries,
            [
                "arm",
                "selector",
                "overall_mean_mse_vs_a6_pct",
                "max_horizon_mse_vs_a6_pct",
                "ETTh2_mean_mse_vs_a6_pct",
                "ETTm1_mean_mse_vs_a6_pct",
                "Weather_mean_mse_vs_a6_pct",
                "overall_mean_mse_vs_fixed_pct",
                "wins_vs_fixed",
                "gate_pass",
            ],
        ),
        "",
        "## Validation-only scale selection",
        "",
        *markdown_table(
            selection,
            [
                "dataset",
                "selected_arm",
                "p16s8_best_val_mean_mse",
                "p48s24_best_val_mean_mse",
            ],
        ),
        "",
        "Scale selection只读取 training log 的 minimum validation mean MSE，不读取 test metrics。",
        "",
        "## Next action",
        "",
        "- shared/selected gate通过：追加 seeds，并执行 local-token use control；",
        "- near miss且 training-validation gap支持 dropout问题：只追加一个 dropout policy；",
        "- 其余情况：恢复 accepted A6 + exact HPM，关闭 C1。",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "c1_global_anchored_multipatch_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    fixed = fixed_map(args.fixed_reference)
    rows = collect_rows(args.raw_root, fixed, args.seed)
    summaries = [
        gate_summary(rows, arm, selector)
        for arm in SCALES
        for selector in SELECTORS
    ]
    selection, selected = validation_selected_rows(rows, args.raw_root, args.seed)
    combined = rows + selected
    summaries.extend(
        gate_summary(combined, "validation_selected", selector)
        for selector in SELECTORS
    )
    diagnostics = diagnostics_rows(args.raw_root, args.seed)
    protocol_audit = protocol_audit_rows(args.raw_root, args.seed)
    write_csv(args.output_dir / "c1_comparisons.csv", combined)
    write_csv(args.output_dir / "c1_gate_summary.csv", summaries)
    write_csv(args.output_dir / "c1_validation_scale_selection.csv", selection)
    write_csv(args.output_dir / "c1_model_diagnostics.csv", diagnostics)
    write_csv(args.output_dir / "c1_protocol_audit.csv", protocol_audit)
    write_report(args.output_dir, summaries, selection, protocol_audit)


if __name__ == "__main__":
    main()
