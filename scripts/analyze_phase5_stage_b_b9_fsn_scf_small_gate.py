from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = ["ETTh2", "ETTm1", "Weather"]
HORIZONS = [96, 192, 336, 720]
SEED = 2021
MIXED_LABEL = "mixed_h96_h192_h336_h720"
ARMS = ["a6_clean", "b9_fsn_scf", "b9_no_stage"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def fmt_pct(value: float) -> str:
    if abs(value) < 0.01:
        return f"{value:+.4f}%"
    return f"{value:+.2f}%"


def run_dir(root: Path, arm: str, dataset: str) -> Path:
    return (
        root
        / f"TimeAlignOfficialUnified720_{arm}_official-last"
        / dataset
        / MIXED_LABEL
        / f"seed{SEED}"
    )


def metric_row(root: Path, arm: str, dataset: str, horizon: int) -> dict[str, float] | None:
    path = run_dir(root, arm, dataset) / "metrics_by_target_horizon.csv"
    if not path.exists():
        return None
    for row in read_csv(path):
        if int(row["target_horizon"]) == horizon:
            return {"mse": float(row["mse"]), "mae": float(row["mae"])}
    return None


def load_diagnostics(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for dataset in DATASETS:
            path = run_dir(root, arm, dataset) / "model_diagnostics.json"
            if not path.exists():
                rows.append({"arm": arm, "dataset": dataset, "status": "missing"})
                continue
            payload = json.loads(path.read_text())
            rows.append(
                {
                    "arm": arm,
                    "dataset": dataset,
                    "status": "ok",
                    "readout_mode": payload.get("readout_mode", ""),
                    "total_parameters": payload.get("total_parameters", ""),
                    "trainable_parameters": payload.get("trainable_parameters", ""),
                    "stage_count": payload.get("stage_count", ""),
                    "stage_gate_mean": payload.get("stage_gate_mean", ""),
                    "stage_gate_min": payload.get("stage_gate_min", ""),
                    "stage_gate_max": payload.get("stage_gate_max", ""),
                    "stage_coeff_up_l2": payload.get("stage_coeff_up_l2", ""),
                }
            )
    return rows


def collect_comparison(root: Path, candidate_arm: str, baseline_arm: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            candidate = metric_row(root, candidate_arm, dataset, horizon)
            baseline = metric_row(root, baseline_arm, dataset, horizon)
            if candidate is None or baseline is None:
                rows.append(
                    {
                        "comparison": f"{candidate_arm}_vs_{baseline_arm}",
                        "candidate_arm": candidate_arm,
                        "baseline_arm": baseline_arm,
                        "dataset": dataset,
                        "target_horizon": horizon,
                        "status": "missing",
                    }
                )
                continue
            rows.append(
                {
                    "comparison": f"{candidate_arm}_vs_{baseline_arm}",
                    "candidate_arm": candidate_arm,
                    "baseline_arm": baseline_arm,
                    "dataset": dataset,
                    "target_horizon": horizon,
                    "status": "ok",
                    "candidate_mse": candidate["mse"],
                    "baseline_mse": baseline["mse"],
                    "relative_mse_pct": pct(candidate["mse"], baseline["mse"]),
                    "candidate_mae": candidate["mae"],
                    "baseline_mae": baseline["mae"],
                    "relative_mae_pct": pct(candidate["mae"], baseline["mae"]),
                    "candidate_wins_mse": candidate["mse"] < baseline["mse"],
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    summary: list[dict[str, Any]] = []
    comparisons = sorted({row["comparison"] for row in ok_rows})
    for comparison in comparisons:
        comp_rows = [row for row in ok_rows if row["comparison"] == comparison]
        for dataset in DATASETS:
            subset = [row for row in comp_rows if row["dataset"] == dataset]
            if not subset:
                continue
            summary.append(
                {
                    "comparison": comparison,
                    "dataset": dataset,
                    "settings": len(subset),
                    "mse_wins": sum(1 for row in subset if row["candidate_wins_mse"]),
                    "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in subset),
                    "mean_relative_mae_pct": mean(row["relative_mae_pct"] for row in subset),
                }
            )
        if comp_rows:
            summary.append(
                {
                    "comparison": comparison,
                    "dataset": "ALL",
                    "settings": len(comp_rows),
                    "mse_wins": sum(1 for row in comp_rows if row["candidate_wins_mse"]),
                    "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in comp_rows),
                    "mean_relative_mae_pct": mean(row["relative_mae_pct"] for row in comp_rows),
                }
            )
    return summary


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float) and field.startswith("mean_relative"):
                value = fmt_pct(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(output_dir: Path, comparison_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    missing = [row for row in comparison_rows if row.get("status") != "ok"]
    b9_vs_a6 = [
        row for row in summary_rows
        if row["comparison"] == "b9_fsn_scf_vs_a6_clean" and row["dataset"] == "ALL"
    ]
    b9_vs_no_stage = [
        row for row in summary_rows
        if row["comparison"] == "b9_fsn_scf_vs_b9_no_stage" and row["dataset"] == "ALL"
    ]
    lines = [
        "# Phase5 StageB B9-FSN-SCF Small Gate Report",
        "",
        "## Scope",
        "",
        "Arms: `a6_clean`, `b9_fsn_scf`, `b9_no_stage`.",
        "Datasets: ETTh2, ETTm1, Weather. Horizons: 96, 192, 336, 720.",
        "",
        "## Summary",
        "",
        *markdown_table(
            summary_rows,
            ["comparison", "dataset", "settings", "mse_wins", "mean_relative_mse_pct", "mean_relative_mae_pct"],
        ),
        "",
        "## Gate Reading",
        "",
    ]
    if missing:
        lines.append(f"[Decision] `incomplete`: {len(missing)} comparison rows are missing.")
    elif not b9_vs_a6 or not b9_vs_no_stage:
        lines.append("[Decision] `incomplete`: required comparison summaries are missing.")
    else:
        rel_a6 = float(b9_vs_a6[0]["mean_relative_mse_pct"])
        rel_no_stage = float(b9_vs_no_stage[0]["mean_relative_mse_pct"])
        wins_a6 = int(b9_vs_a6[0]["mse_wins"])
        wins_no_stage = int(b9_vs_no_stage[0]["mse_wins"])
        if rel_a6 <= 0.20 and rel_no_stage < 0.0 and wins_no_stage >= 7:
            lines.append(
                "[Decision] `small_gate_pass_candidate`: B9 is within the clean-A6 tolerance and beats no-stage overall."
            )
        elif rel_no_stage >= 0.0:
            lines.append(
                "[Decision] `blocked_by_no_stage_control`: B9 does not beat the no-stage capacity control overall."
            )
        else:
            lines.append("[Decision] `small_gate_fail_or_mixed`: B9 does not satisfy all pass criteria.")
        lines.extend(
            [
                "",
                f"- B9 vs A6 mean relative MSE: `{fmt_pct(rel_a6)}`, MSE wins `{wins_a6}/12`.",
                f"- B9 vs no-stage mean relative MSE: `{fmt_pct(rel_no_stage)}`, MSE wins `{wins_no_stage}/12`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `b9_fsn_scf_small_gate_comparison.csv`",
            "- `b9_fsn_scf_small_gate_summary.csv`",
            "- `b9_fsn_scf_model_diagnostics.csv`",
            "",
        ]
    )
    (output_dir / "b9_fsn_scf_small_gate_report.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase5 StageB B9-FSN-SCF small gate.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707/raw/official-last"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows = []
    comparison_rows.extend(collect_comparison(args.raw_root, "b9_fsn_scf", "a6_clean"))
    comparison_rows.extend(collect_comparison(args.raw_root, "b9_no_stage", "a6_clean"))
    comparison_rows.extend(collect_comparison(args.raw_root, "b9_fsn_scf", "b9_no_stage"))
    summary_rows = summarize(comparison_rows)
    diagnostics_rows = load_diagnostics(args.raw_root)
    write_csv(args.output_dir / "b9_fsn_scf_small_gate_comparison.csv", comparison_rows)
    write_csv(args.output_dir / "b9_fsn_scf_small_gate_summary.csv", summary_rows)
    write_csv(args.output_dir / "b9_fsn_scf_model_diagnostics.csv", diagnostics_rows)
    write_report(args.output_dir, comparison_rows, summary_rows)


if __name__ == "__main__":
    main()
