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
ARMS = ["a6_clean", "b11_bcf", "b11_no_basis", "b11_constant_slot", "b11_shuffled_basis"]
REQUIRED_ARMS = ["a6_clean", "b11_bcf", "b11_no_basis", "b11_constant_slot"]


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
                    "basis_field_window_count": payload.get("basis_field_window_count", ""),
                    "basis_field_rank": payload.get("basis_field_rank", ""),
                    "basis_field_tau": payload.get("basis_field_tau", ""),
                    "basis_field_gate_sigmoid": payload.get("basis_field_gate_sigmoid", ""),
                    "basis_field_delta_l2": payload.get("basis_field_delta_l2", ""),
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


def summary_lookup(summary_rows: list[dict[str, Any]], comparison: str) -> dict[str, Any] | None:
    for row in summary_rows:
        if row["comparison"] == comparison and row["dataset"] == "ALL":
            return row
    return None


def write_report(output_dir: Path, comparison_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    missing = [row for row in comparison_rows if row.get("status") != "ok"]
    b11_vs_a6 = summary_lookup(summary_rows, "b11_bcf_vs_a6_clean")
    b11_vs_no_basis = summary_lookup(summary_rows, "b11_bcf_vs_b11_no_basis")
    b11_vs_constant = summary_lookup(summary_rows, "b11_bcf_vs_b11_constant_slot")
    lines = [
        "# Phase5 StageB B11-BCF Small Gate Report",
        "",
        "## Scope",
        "",
        "Required arms: `a6_clean`, `b11_bcf`, `b11_no_basis`, `b11_constant_slot`.",
        "Optional arm: `b11_shuffled_basis`.",
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
    elif b11_vs_a6 is None or b11_vs_no_basis is None or b11_vs_constant is None:
        lines.append("[Decision] `incomplete`: required comparison summaries are missing.")
    else:
        rel_a6 = float(b11_vs_a6["mean_relative_mse_pct"])
        rel_no_basis = float(b11_vs_no_basis["mean_relative_mse_pct"])
        rel_constant = float(b11_vs_constant["mean_relative_mse_pct"])
        wins_a6 = int(b11_vs_a6["mse_wins"])
        wins_no_basis = int(b11_vs_no_basis["mse_wins"])
        wins_constant = int(b11_vs_constant["mse_wins"])
        if rel_a6 <= 0.0 and rel_no_basis < 0.0 and rel_constant < 0.0 and wins_no_basis >= 7 and wins_constant >= 7:
            lines.append(
                "[Decision] `small_gate_pass_candidate`: B11-BCF improves over A6 and beats both required mechanism controls."
            )
        elif rel_a6 <= 0.0 and (rel_no_basis >= 0.0 or rel_constant >= 0.0):
            lines.append(
                "[Decision] `capacity_or_head_effect_suspected`: B11-BCF improves over A6 but does not beat required controls."
            )
        else:
            lines.append("[Decision] `small_gate_failed`: B11-BCF does not improve over the A6 clean anchor.")
        lines.extend(
            [
                "",
                f"- B11 vs A6: mean MSE {fmt_pct(rel_a6)}, wins {wins_a6}/12.",
                f"- B11 vs no-basis: mean MSE {fmt_pct(rel_no_basis)}, wins {wins_no_basis}/12.",
                f"- B11 vs constant-slot: mean MSE {fmt_pct(rel_constant)}, wins {wins_constant}/12.",
            ]
        )
    lines.extend(
        [
            "",
            "## Failure Attribution Rule",
            "",
            "This report may reject only the tested B11-BCF implementation unless the required controls show that the broader basis-conditioned direction is false.",
            "If `no_basis` or `constant_slot` explains the gain, classify the result as capacity/head effect, not basis-conditioned mechanism evidence.",
            "",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "b11_bcf_small_gate_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase5 StageB B11-BCF small gate artifacts.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    comparisons = [
        ("b11_bcf", "a6_clean"),
        ("b11_no_basis", "a6_clean"),
        ("b11_constant_slot", "a6_clean"),
        ("b11_bcf", "b11_no_basis"),
        ("b11_bcf", "b11_constant_slot"),
    ]
    if any((run_dir(args.raw_root, "b11_shuffled_basis", dataset) / "metrics_by_target_horizon.csv").exists() for dataset in DATASETS):
        comparisons.extend(
            [
                ("b11_shuffled_basis", "a6_clean"),
                ("b11_bcf", "b11_shuffled_basis"),
            ]
        )

    comparison_rows: list[dict[str, Any]] = []
    for candidate_arm, baseline_arm in comparisons:
        comparison_rows.extend(collect_comparison(args.raw_root, candidate_arm, baseline_arm))
    summary_rows = summarize(comparison_rows)
    diagnostics_rows = load_diagnostics(args.raw_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "b11_bcf_small_gate_comparisons.csv", comparison_rows)
    write_csv(args.output_dir / "b11_bcf_small_gate_summary.csv", summary_rows)
    write_csv(args.output_dir / "b11_bcf_small_gate_model_diagnostics.csv", diagnostics_rows)
    write_report(args.output_dir, comparison_rows, summary_rows)


if __name__ == "__main__":
    main()
