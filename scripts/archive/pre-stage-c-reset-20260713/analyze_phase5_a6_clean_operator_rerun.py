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
FIXED_LABEL = {
    96: "mixed_h96",
    192: "mixed_h192",
    336: "mixed_h336",
    720: "mixed_h720",
}


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
    return f"{value:+.2f}%"


def metric_from_csv(path: Path, horizon: int) -> dict[str, float] | None:
    if not path.exists():
        return None
    for row in read_csv(path):
        if int(row["target_horizon"]) == horizon:
            return {"mse": float(row["mse"]), "mae": float(row["mae"])}
    return None


def clean_run_dir(root: Path, dataset: str) -> Path:
    return (
        root
        / "official-last"
        / "TimeAlignOfficialUnified720_A6LBF_r256_main_official-last"
        / dataset
        / MIXED_LABEL
        / f"seed{SEED}"
    )


def historical_a6_dir(root: Path, dataset: str) -> Path:
    return (
        root
        / "official-last"
        / "TimeAlignOfficialUnified720_A6_a6_lbf_r256_official-last"
        / dataset
        / MIXED_LABEL
        / f"seed{SEED}"
    )


def official_unified_dir(root: Path, dataset: str) -> Path:
    return (
        root
        / "official"
        / "official-last"
        / "TimeAlignOfficialUnified720_official-last"
        / dataset
        / MIXED_LABEL
        / f"seed{SEED}"
    )


def official_fixed_dir(root: Path, dataset: str, horizon: int) -> Path:
    return (
        root
        / "official"
        / "official-last"
        / f"TimeAlignOfficialFixedH{horizon}_official-last"
        / dataset
        / FIXED_LABEL[horizon]
        / f"seed{SEED}"
    )


def collect_comparison(
    *,
    clean_root: Path,
    reference_root: Path,
    reference_name: str,
    reference_path_fn: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        clean_metrics = clean_run_dir(clean_root, dataset) / "metrics_by_target_horizon.csv"
        for horizon in HORIZONS:
            clean = metric_from_csv(clean_metrics, horizon)
            reference = metric_from_csv(reference_path_fn(reference_root, dataset, horizon), horizon)
            if clean is None or reference is None:
                rows.append(
                    {
                        "dataset": dataset,
                        "target_horizon": horizon,
                        "reference": reference_name,
                        "status": "missing",
                        "clean_mse": "" if clean is None else clean["mse"],
                        "reference_mse": "" if reference is None else reference["mse"],
                    }
                )
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "target_horizon": horizon,
                    "reference": reference_name,
                    "status": "ok",
                    "clean_mse": clean["mse"],
                    "reference_mse": reference["mse"],
                    "relative_mse_pct": pct(clean["mse"], reference["mse"]),
                    "clean_mae": clean["mae"],
                    "reference_mae": reference["mae"],
                    "relative_mae_pct": pct(clean["mae"], reference["mae"]),
                    "clean_wins_mse": clean["mse"] < reference["mse"],
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    summary: list[dict[str, Any]] = []
    for dataset in DATASETS:
        subset = [row for row in ok_rows if row["dataset"] == dataset]
        if not subset:
            continue
        summary.append(
            {
                "dataset": dataset,
                "settings": len(subset),
                "mse_wins": sum(1 for row in subset if row["clean_wins_mse"]),
                "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in subset),
                "mean_relative_mae_pct": mean(row["relative_mae_pct"] for row in subset),
            }
        )
    if ok_rows:
        summary.append(
            {
                "dataset": "ALL",
                "settings": len(ok_rows),
                "mse_wins": sum(1 for row in ok_rows if row["clean_wins_mse"]),
                "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in ok_rows),
                "mean_relative_mae_pct": mean(row["relative_mae_pct"] for row in ok_rows),
            }
        )
    return summary


def load_training_rows(clean_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        run_dir = clean_run_dir(clean_root, dataset)
        log_path = run_dir / "training_log.csv"
        config_path = run_dir / "effective_config.json"
        if not log_path.exists():
            rows.append({"dataset": dataset, "status": "missing_training_log"})
            continue
        logs = read_csv(log_path)
        config: dict[str, Any] = {}
        if config_path.exists():
            config = json.loads(config_path.read_text())
        adapter_config = config.get("adapter", {})
        official_args = config.get("official_args", {})
        best = min(logs, key=lambda row: float(row["val_mean_mse"]))
        rows.append(
            {
                "dataset": dataset,
                "status": "ok",
                "epochs_ran": len(logs),
                "best_epoch": int(best["epoch"]),
                "best_val_mean_mse": float(best["val_mean_mse"]),
                "last_val_mean_mse": float(logs[-1]["val_mean_mse"]),
                "effective_w_recon": official_args.get("w_recon", ""),
                "effective_w_align": official_args.get("w_align", ""),
                "readout_mode": official_args.get("readout_mode", ""),
                "basis_rank": official_args.get("basis_rank", ""),
                "pred_loss_mode": adapter_config.get("pred_loss_mode", ""),
            }
        )
    return rows


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
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def summary_line(summary: list[dict[str, Any]]) -> str:
    all_row = next((row for row in summary if row["dataset"] == "ALL"), None)
    if all_row is None:
        return "missing"
    return f"{fmt_pct(all_row['mean_relative_mse_pct'])}, {all_row['mse_wins']}/{all_row['settings']} MSE wins"


def decide(
    fixed_summary: list[dict[str, Any]],
    unified_summary: list[dict[str, Any]],
    historical_summary: list[dict[str, Any]],
) -> str:
    fixed_all = next((row for row in fixed_summary if row["dataset"] == "ALL"), None)
    unified_all = next((row for row in unified_summary if row["dataset"] == "ALL"), None)
    historical_all = next((row for row in historical_summary if row["dataset"] == "ALL"), None)
    if fixed_all is None or unified_all is None or historical_all is None:
        return "incomplete"
    stable_vs_historical = abs(historical_all["mean_relative_mse_pct"]) <= 0.5
    passes_fixed = fixed_all["mean_relative_mse_pct"] < 0 and fixed_all["mse_wins"] >= 7
    passes_unified = unified_all["mean_relative_mse_pct"] < 0 and unified_all["mse_wins"] >= 7
    if stable_vs_historical and passes_fixed and passes_unified:
        return "clean_a6_validated"
    if passes_fixed and passes_unified:
        return "clean_a6_effective_but_not_identical_to_historical"
    return "clean_a6_needs_recheck_or_rollback"


def write_report(
    *,
    output_dir: Path,
    fixed_summary: list[dict[str, Any]],
    unified_summary: list[dict[str, Any]],
    historical_summary: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
) -> None:
    decision = decide(fixed_summary, unified_summary, historical_summary)
    lines = [
        "# Phase5 Clean A6-LBF-r256 Rerun Report",
        "",
        "## Decision",
        "",
        f"[Decision] `{decision}`.",
        "[Current Step] StageA clean validation after removing the A6 future reconstruction/alignment branch.",
        "[Gate] This report only decides whether the clean A6-LBF-r256 operator remains a valid Contribution 1 evidence base.",
        "[Rollback] If this gate fails, roll back to StageA Step 9/10 and inspect the code cut or retrain variance before opening StageB methods.",
        "",
        "## Summary",
        "",
        f"- vs fixed-horizon TimeAlign: `{summary_line(fixed_summary)}`.",
        f"- vs official unified TimeAlign: `{summary_line(unified_summary)}`.",
        f"- vs historical A6-LBF-r256: `{summary_line(historical_summary)}`.",
        "",
        "## Clean Run Training Check",
        "",
        *markdown_table(
            training_rows,
            [
                "dataset",
                "status",
                "epochs_ran",
                "best_epoch",
                "best_val_mean_mse",
                "last_val_mean_mse",
                "effective_w_recon",
                "effective_w_align",
                "readout_mode",
                "basis_rank",
                "pred_loss_mode",
            ],
        ),
        "",
        "## vs Fixed-Horizon TimeAlign",
        "",
        *markdown_table(
            fixed_summary,
            ["dataset", "settings", "mse_wins", "mean_relative_mse_pct", "mean_relative_mae_pct"],
        ),
        "",
        "## vs Official Unified TimeAlign",
        "",
        *markdown_table(
            unified_summary,
            ["dataset", "settings", "mse_wins", "mean_relative_mse_pct", "mean_relative_mae_pct"],
        ),
        "",
        "## vs Historical A6-LBF-r256",
        "",
        *markdown_table(
            historical_summary,
            ["dataset", "settings", "mse_wins", "mean_relative_mse_pct", "mean_relative_mae_pct"],
        ),
        "",
        "## Reading",
        "",
        "- [Fact] A clean pass validates the pure learned-basis forecast operator as the current paper-core method.",
        "- [Fact] This gate does not revive B6-PLO or any StageB objective candidate.",
        "- [Inference] If clean A6 remains close to historical A6 while preserving fixed/unified wins, the future-recon branch removal improves narrative clarity without weakening the empirical base.",
    ]
    (output_dir / "clean_a6_rerun_report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze clean A6-LBF-r256 rerun artifacts.")
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=Path("analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/raw"),
    )
    parser.add_argument(
        "--historical-a6-root",
        type=Path,
        default=Path("analysis/phase5_timealign_hss_a6_capacity_native_gate_20260703/raw"),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path("analysis/phase5_timealign_hss_a3e_ettm1_replacement_gate_20260701/raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vs_fixed = collect_comparison(
        clean_root=args.clean_root,
        reference_root=args.official_root,
        reference_name="fixed_horizon_timealign",
        reference_path_fn=lambda root, dataset, horizon: official_fixed_dir(root, dataset, horizon)
        / "metrics_by_target_horizon.csv",
    )
    vs_unified = collect_comparison(
        clean_root=args.clean_root,
        reference_root=args.official_root,
        reference_name="official_unified_timealign",
        reference_path_fn=lambda root, dataset, horizon: official_unified_dir(root, dataset)
        / "metrics_by_target_horizon.csv",
    )
    vs_historical = collect_comparison(
        clean_root=args.clean_root,
        reference_root=args.historical_a6_root,
        reference_name="historical_a6_lbf_r256",
        reference_path_fn=lambda root, dataset, horizon: historical_a6_dir(root, dataset)
        / "metrics_by_target_horizon.csv",
    )
    fixed_summary = summarize(vs_fixed)
    unified_summary = summarize(vs_unified)
    historical_summary = summarize(vs_historical)
    training_rows = load_training_rows(args.clean_root)

    write_csv(args.output_dir / "clean_a6_vs_fixed.csv", vs_fixed)
    write_csv(args.output_dir / "clean_a6_vs_official_unified.csv", vs_unified)
    write_csv(args.output_dir / "clean_a6_vs_historical_a6.csv", vs_historical)
    write_csv(args.output_dir / "clean_a6_vs_fixed_summary.csv", fixed_summary)
    write_csv(args.output_dir / "clean_a6_vs_official_unified_summary.csv", unified_summary)
    write_csv(args.output_dir / "clean_a6_vs_historical_a6_summary.csv", historical_summary)
    write_csv(args.output_dir / "clean_a6_training_summary.csv", training_rows)
    write_report(
        output_dir=args.output_dir,
        fixed_summary=fixed_summary,
        unified_summary=unified_summary,
        historical_summary=historical_summary,
        training_rows=training_rows,
    )


if __name__ == "__main__":
    main()
