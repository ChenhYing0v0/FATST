#!/usr/bin/env python3
"""Analyze the StageB C0 ETTm1 Encoder control without promoting it to a method."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


HORIZONS = (96, 192, 336, 720)
SELECTORS = {"last": "official-last", "best_val": "best-val"}
ARMS = {
    "p1_d256_f256_d09": (1, 256, 256, 0.9, 699_600),
    "p1_d384_f96_d09": (1, 384, 96, 0.9, 710_416),
    "p5_d52_f256_d09": (5, 52, 256, 0.9, 313_468),
    "p5_d52_f2048_d09": (5, 52, 2048, 0.9, 689_788),
    "p1_d256_f256_d02": (1, 256, 256, 0.2, 699_600),
    "p5_d52_f2048_d02": (5, 52, 2048, 0.2, 689_788),
}
COMPARISONS = (
    ("global_width_d09", "p1_d384_f96_d09", "p1_d256_f256_d09"),
    ("patch_low_capacity_d09", "p5_d52_f256_d09", "p1_d256_f256_d09"),
    ("patch_matched_d09", "p5_d52_f2048_d09", "p1_d256_f256_d09"),
    ("patch_matched_d02", "p5_d52_f2048_d02", "p1_d256_f256_d02"),
)
MIXED_LABEL = "mixed_h96_h192_h336_h720"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2021)
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


def run_dir(root: Path, arm: str, seed: int) -> Path:
    return (
        root
        / f"A6_C0_{arm}_dual"
        / "ETTm1"
        / MIXED_LABEL
        / f"seed{seed}"
    )


def load_metrics(
    root: Path,
    arm: str,
    selector_file: str,
    seed: int,
) -> dict[int, dict[str, float]] | None:
    path = run_dir(root, arm, seed) / f"metrics_{selector_file}_by_target_horizon.csv"
    if not path.exists():
        return None
    return {
        int(row["target_horizon"]): {
            "mse": float(row["mse"]),
            "mae": float(row["mae"]),
        }
        for row in read_csv(path)
    }


def relative_pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def collect_comparisons(root: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison, candidate, baseline in COMPARISONS:
        for selector_file, selector_label in SELECTORS.items():
            candidate_metrics = load_metrics(root, candidate, selector_file, seed)
            baseline_metrics = load_metrics(root, baseline, selector_file, seed)
            for horizon in HORIZONS:
                if (
                    candidate_metrics is None
                    or baseline_metrics is None
                    or horizon not in candidate_metrics
                    or horizon not in baseline_metrics
                ):
                    rows.append(
                        {
                            "comparison": comparison,
                            "candidate": candidate,
                            "baseline": baseline,
                            "selector": selector_label,
                            "target_horizon": horizon,
                            "status": "missing",
                        }
                    )
                    continue
                candidate_row = candidate_metrics[horizon]
                baseline_row = baseline_metrics[horizon]
                rows.append(
                    {
                        "comparison": comparison,
                        "candidate": candidate,
                        "baseline": baseline,
                        "selector": selector_label,
                        "target_horizon": horizon,
                        "status": "ok",
                        "candidate_mse": candidate_row["mse"],
                        "baseline_mse": baseline_row["mse"],
                        "relative_mse_pct": relative_pct(
                            candidate_row["mse"], baseline_row["mse"]
                        ),
                        "candidate_mae": candidate_row["mae"],
                        "baseline_mae": baseline_row["mae"],
                        "relative_mae_pct": relative_pct(
                            candidate_row["mae"], baseline_row["mae"]
                        ),
                        "mse_win": int(candidate_row["mse"] < baseline_row["mse"]),
                    }
                )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for comparison, _candidate, _baseline in COMPARISONS:
        for selector_label in SELECTORS.values():
            selected = [
                row
                for row in rows
                if row["comparison"] == comparison
                and row["selector"] == selector_label
                and row["status"] == "ok"
            ]
            if len(selected) != len(HORIZONS):
                summaries.append(
                    {
                        "comparison": comparison,
                        "selector": selector_label,
                        "status": "incomplete",
                        "settings": len(selected),
                    }
                )
                continue
            deltas = [float(row["relative_mse_pct"]) for row in selected]
            summaries.append(
                {
                    "comparison": comparison,
                    "selector": selector_label,
                    "status": "ok",
                    "settings": len(selected),
                    "mse_wins": sum(int(row["mse_win"]) for row in selected),
                    "mean_relative_mse_pct": mean(deltas),
                    "max_relative_mse_pct": max(deltas),
                    "mean_relative_mae_pct": mean(
                        float(row["relative_mae_pct"]) for row in selected
                    ),
                    "preliminary_gate_pass": int(
                        mean(deltas) <= -0.5
                        and sum(int(row["mse_win"]) for row in selected) >= 3
                        and max(deltas) <= 1.0
                    ),
                }
            )
    return summaries


def collect_diagnostics(root: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, expected in ARMS.items():
        patch_num, d_model, d_ff, dropout, expected_active = expected
        directory = run_dir(root, arm, seed)
        diagnostics_path = directory / "model_diagnostics.json"
        config_path = directory / "effective_config.json"
        if not diagnostics_path.exists() or not config_path.exists():
            rows.append({"arm": arm, "status": "missing"})
            continue
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        effective = config["official_args"]
        observed_active = int(diagnostics["active_forward_parameters"])
        config_ok = (
            int(effective["patch_num"]) == patch_num
            and int(effective["d_model"]) == d_model
            and int(effective["d_ff"]) == d_ff
            and abs(float(effective["dropout"]) - dropout) < 1e-12
        )
        rows.append(
            {
                "arm": arm,
                "status": "ok" if config_ok and observed_active == expected_active else "mismatch",
                "patch_num": effective["patch_num"],
                "d_model": effective["d_model"],
                "d_ff": effective["d_ff"],
                "dropout": effective["dropout"],
                "active_forward_parameters": observed_active,
                "expected_active_forward_parameters": expected_active,
                "unused_proj_x_parameters": diagnostics["unused_proj_x_parameters"],
                "dual_metrics_present": int(
                    (directory / "metrics_last_by_target_horizon.csv").exists()
                    and (directory / "metrics_best_val_by_target_horizon.csv").exists()
                ),
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
    comparison_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> None:
    complete = all(row["status"] == "ok" for row in summary_rows) and all(
        row["status"] == "ok" and row.get("dual_metrics_present") == 1
        for row in diagnostics
    )
    matched = [
        row
        for row in summary_rows
        if row["comparison"] in {"patch_matched_d09", "patch_matched_d02"}
    ]
    patch_gate = complete and len(matched) == 4 and all(
        row.get("preliminary_gate_pass") == 1 for row in matched
    )
    if not complete:
        decision = "incomplete_wait_for_all_six_arms"
    elif patch_gate:
        decision = "small_gate_pass_requires_multiseed_confirmation"
    else:
        directions = [float(row["mean_relative_mse_pct"]) < 0.0 for row in matched]
        decision = (
            "patch_effect_confounded_by_regularization_or_selector"
            if any(directions) and not all(directions)
            else "patch_num_performance_defect_not_supported"
        )

    lines = [
        "# Phase5 StageB C0 ETTm1 Encoder Control Report",
        "",
        "## Scope",
        "",
        "本报告只审计 Encoder/carrier 与 training protocol，不构成 StageB 创新点。",
        "",
        "## Summary",
        "",
        *markdown_table(
            summary_rows,
            [
                "comparison",
                "selector",
                "status",
                "mse_wins",
                "mean_relative_mse_pct",
                "max_relative_mse_pct",
                "preliminary_gate_pass",
            ],
        ),
        "",
        "## Gate decision",
        "",
        f"`{decision}`",
        "",
        "Gate pass 只授权 multi-seed confirmation；不授权将 patching 或 mixer 写为 StageB method。",
        "",
        "## Statistic definitions",
        "",
        "- `relative_mse_pct = (candidate_mse / baseline_mse - 1) * 100`；负值更好。",
        "- `mse_wins` 是四个 target horizons 中 candidate MSE 更低的数量。",
        "- `preliminary_gate_pass` 要求 mean delta <= -0.5%、至少 3/4 wins、最大 regression <= +1.0%。",
        "- `selector` 的 last 与 best-val 来自同一次 training trajectory。",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "c0_ettm1_encoder_control_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    comparison_rows = collect_comparisons(args.raw_root, args.seed)
    summary_rows = summarize(comparison_rows)
    diagnostics = collect_diagnostics(args.raw_root, args.seed)
    write_csv(args.output_dir / "c0_comparisons.csv", comparison_rows)
    write_csv(args.output_dir / "c0_summary.csv", summary_rows)
    write_csv(args.output_dir / "c0_model_diagnostics.csv", diagnostics)
    write_report(args.output_dir, comparison_rows, summary_rows, diagnostics)


if __name__ == "__main__":
    main()
