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
ARMS = ["a6_clean", "stbo_shared", "stbo_bank4", "stbo_dct", "stbo_independent"]
REQUIRED_ARMS = ["a6_clean", "stbo_shared", "stbo_bank4", "stbo_dct", "stbo_independent"]


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
                    "basis_rank": payload.get("basis_rank", ""),
                    "learned_temporal_basis_l2": payload.get("learned_temporal_basis_l2", ""),
                    "learned_basis_coeff_l2": payload.get("learned_basis_coeff_l2", ""),
                    "stbo_tile_len": payload.get("stbo_tile_len", ""),
                    "stbo_tile_count": payload.get("stbo_tile_count", ""),
                    "stbo_rank": payload.get("stbo_rank", ""),
                    "stbo_coeff_l2": payload.get("stbo_coeff_l2", ""),
                    "stbo_shared_basis_l2": payload.get("stbo_shared_basis_l2", ""),
                    "stbo_bank_count": payload.get("stbo_bank_count", ""),
                    "stbo_basis_bank_l2": payload.get("stbo_basis_bank_l2", ""),
                    "stbo_tile_bank_entropy_mean": payload.get("stbo_tile_bank_entropy_mean", ""),
                    "stbo_tile_basis_l2": payload.get("stbo_tile_basis_l2", ""),
                    "stbo_dct_basis_l2": payload.get("stbo_dct_basis_l2", ""),
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


def arm_result(summary_rows: list[dict[str, Any]], candidate: str, baseline: str) -> tuple[float, int] | None:
    row = summary_lookup(summary_rows, f"{candidate}_vs_{baseline}")
    if row is None:
        return None
    return float(row["mean_relative_mse_pct"]), int(row["mse_wins"])


def write_report(output_dir: Path, comparison_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    missing = [row for row in comparison_rows if row.get("status") != "ok"]
    shared_vs_a6 = arm_result(summary_rows, "stbo_shared", "a6_clean")
    bank4_vs_a6 = arm_result(summary_rows, "stbo_bank4", "a6_clean")
    dct_vs_a6 = arm_result(summary_rows, "stbo_dct", "a6_clean")
    independent_vs_a6 = arm_result(summary_rows, "stbo_independent", "a6_clean")
    shared_vs_dct = arm_result(summary_rows, "stbo_shared", "stbo_dct")
    bank4_vs_dct = arm_result(summary_rows, "stbo_bank4", "stbo_dct")
    shared_vs_independent = arm_result(summary_rows, "stbo_shared", "stbo_independent")
    bank4_vs_independent = arm_result(summary_rows, "stbo_bank4", "stbo_independent")

    lines = [
        "# Phase5 StageB B12-STBO Small Gate Report",
        "",
        "## Scope",
        "",
        "Required arms: `a6_clean`, `stbo_shared`, `stbo_bank4`, `stbo_dct`, `stbo_independent`.",
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
    elif any(
        result is None
        for result in [
            shared_vs_a6,
            bank4_vs_a6,
            dct_vs_a6,
            independent_vs_a6,
            shared_vs_dct,
            bank4_vs_dct,
            shared_vs_independent,
            bank4_vs_independent,
        ]
    ):
        lines.append("[Decision] `incomplete`: required comparison summaries are missing.")
    else:
        assert shared_vs_a6 is not None
        assert bank4_vs_a6 is not None
        assert dct_vs_a6 is not None
        assert independent_vs_a6 is not None
        assert shared_vs_dct is not None
        assert bank4_vs_dct is not None
        assert shared_vs_independent is not None
        assert bank4_vs_independent is not None
        shared_a6_mse, shared_a6_wins = shared_vs_a6
        bank4_a6_mse, bank4_a6_wins = bank4_vs_a6
        dct_a6_mse, dct_a6_wins = dct_vs_a6
        independent_a6_mse, independent_a6_wins = independent_vs_a6
        shared_dct_mse, shared_dct_wins = shared_vs_dct
        bank4_dct_mse, bank4_dct_wins = bank4_vs_dct
        shared_ind_mse, shared_ind_wins = shared_vs_independent
        bank4_ind_mse, bank4_ind_wins = bank4_vs_independent

        learned_beats_dct = (shared_dct_mse < 0.0 and shared_dct_wins >= 7) or (
            bank4_dct_mse < 0.0 and bank4_dct_wins >= 7
        )
        learned_not_degraded = (shared_a6_mse <= 0.5 and shared_a6_wins >= 5) or (
            bank4_a6_mse <= 0.5 and bank4_a6_wins >= 5
        )
        independent_only = (
            independent_a6_mse < min(shared_a6_mse, bank4_a6_mse, dct_a6_mse)
            and shared_ind_mse > 0.0
            and bank4_ind_mse > 0.0
        )

        if learned_beats_dct and learned_not_degraded and not independent_only:
            lines.append(
                "[Decision] `small_gate_pass_candidate`: learned STBO beats the DCT control and is not explained only by independent-tile capacity."
            )
        elif not learned_beats_dct:
            lines.append(
                "[Decision] `generic_local_basis_control_explains`: learned STBO does not beat the fixed local DCT control."
            )
        elif independent_only:
            lines.append(
                "[Decision] `independent_tile_capacity_explains`: the independent-tile control explains the positive result better than shared/bank STBO."
            )
        else:
            lines.append("[Decision] `small_gate_failed`: learned STBO does not improve enough over the A6 clean anchor.")

        lines.extend(
            [
                "",
                f"- STBO-shared vs A6: mean MSE {fmt_pct(shared_a6_mse)}, wins {shared_a6_wins}/12.",
                f"- STBO-bank4 vs A6: mean MSE {fmt_pct(bank4_a6_mse)}, wins {bank4_a6_wins}/12.",
                f"- STBO-DCT vs A6: mean MSE {fmt_pct(dct_a6_mse)}, wins {dct_a6_wins}/12.",
                f"- STBO-independent vs A6: mean MSE {fmt_pct(independent_a6_mse)}, wins {independent_a6_wins}/12.",
                f"- STBO-shared vs DCT: mean MSE {fmt_pct(shared_dct_mse)}, wins {shared_dct_wins}/12.",
                f"- STBO-bank4 vs DCT: mean MSE {fmt_pct(bank4_dct_mse)}, wins {bank4_dct_wins}/12.",
                f"- STBO-shared vs independent: mean MSE {fmt_pct(shared_ind_mse)}, wins {shared_ind_wins}/12.",
                f"- STBO-bank4 vs independent: mean MSE {fmt_pct(bank4_ind_mse)}, wins {bank4_ind_wins}/12.",
            ]
        )

    lines.extend(
        [
            "",
            "## Failure Attribution Rule",
            "",
            "This report may reject only the tested B12-STBO implementation unless learned shared/bank STBO is stable and still fails the required DCT and independent controls.",
            "If DCT matches learned STBO, classify the result as `generic_local_basis_control_explains`, not as a rejection of all native multi-horizon operators.",
            "If only independent tile wins, classify the result as `independent_tile_capacity_explains`, not as a shared subspace method.",
            "",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "b12_stbo_small_gate_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase5 StageB B12-STBO small gate artifacts.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    comparisons = [
        ("stbo_shared", "a6_clean"),
        ("stbo_bank4", "a6_clean"),
        ("stbo_dct", "a6_clean"),
        ("stbo_independent", "a6_clean"),
        ("stbo_shared", "stbo_dct"),
        ("stbo_bank4", "stbo_dct"),
        ("stbo_shared", "stbo_independent"),
        ("stbo_bank4", "stbo_independent"),
        ("stbo_bank4", "stbo_shared"),
    ]

    comparison_rows: list[dict[str, Any]] = []
    for candidate_arm, baseline_arm in comparisons:
        comparison_rows.extend(collect_comparison(args.raw_root, candidate_arm, baseline_arm))
    summary_rows = summarize(comparison_rows)
    diagnostics_rows = load_diagnostics(args.raw_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "b12_stbo_small_gate_comparisons.csv", comparison_rows)
    write_csv(args.output_dir / "b12_stbo_small_gate_summary.csv", summary_rows)
    write_csv(args.output_dir / "b12_stbo_small_gate_model_diagnostics.csv", diagnostics_rows)
    write_report(args.output_dir, comparison_rows, summary_rows)


if __name__ == "__main__":
    main()
