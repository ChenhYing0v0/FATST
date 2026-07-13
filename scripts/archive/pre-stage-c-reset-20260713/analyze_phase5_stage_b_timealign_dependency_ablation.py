from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any

DATASETS = ("ETTh2", "ETTm1", "Weather")
HORIZONS = (96, 192, 336, 720)
ARMS = (
    "current_align_recon",
    "no_align_recon",
    "align_no_recon",
    "no_align_no_recon",
)
RUN_PREFIX = "TimeAlignOfficialUnified720_A6LBF_r256_"
RUN_SUFFIX = "_official-last"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def arm_from_run_name(run_name: str) -> str:
    if not run_name.startswith(RUN_PREFIX) or not run_name.endswith(RUN_SUFFIX):
        raise ValueError(f"Unexpected run name: {run_name}")
    return run_name[len(RUN_PREFIX) : -len(RUN_SUFFIX)]


def metric_files(raw_root: Path) -> list[Path]:
    files = sorted((raw_root / "official-last").glob("*/**/metrics_by_target_horizon.csv"))
    if len(files) != len(ARMS) * len(DATASETS):
        raise FileNotFoundError(f"Expected {len(ARMS) * len(DATASETS)} metric files, found {len(files)}")
    return files


def collect_metric_rows(raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in metric_files(raw_root):
        for row in read_csv(path):
            run_name = row["run_name"]
            arm = arm_from_run_name(run_name)
            horizon = int(row["target_horizon"])
            if arm not in ARMS or horizon not in HORIZONS:
                continue
            rows.append(
                {
                    "arm": arm,
                    "dataset": row["dataset"],
                    "target_horizon": horizon,
                    "mse": float(row["mse"]),
                    "mae": float(row["mae"]),
                    "num_samples": int(row["num_samples"]),
                    "run_name": run_name,
                }
            )
    return rows


def collect_training_rows(raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((raw_root / "official-last").glob("*/**/training_log.csv")):
        run_name = path.parts[-5]
        arm = arm_from_run_name(run_name)
        dataset = path.parts[-4]
        logs = read_csv(path)
        best = min(logs, key=lambda row: float(row["val_mean_mse"]))
        last = logs[-1]
        for selector, source in (("best_val", best), ("last", last)):
            rows.append(
                {
                    "arm": arm,
                    "dataset": dataset,
                    "epoch_selector": selector,
                    "epoch": int(source["epoch"]),
                    "train_loss": float(source["train_loss"]),
                    "train_prediction_l1": float(source["train_prediction_l1"]),
                    "train_reconstruction_l1": float(source["train_reconstruction_l1"]),
                    "train_alignment_loss": float(source["train_alignment_loss"]),
                    "train_weighted_reconstruction_l1": float(source["train_weighted_reconstruction_l1"]),
                    "train_weighted_alignment_loss": float(source["train_weighted_alignment_loss"]),
                    "val_mean_mse": float(source["val_mean_mse"]),
                }
            )
    return rows


def with_current_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current = {
        (row["dataset"], row["target_horizon"]): row
        for row in rows
        if row["arm"] == "current_align_recon"
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        base = current[(row["dataset"], row["target_horizon"])]
        output.append(
            {
                "arm": row["arm"],
                "dataset": row["dataset"],
                "target_horizon": row["target_horizon"],
                "mse": f"{row['mse']:.10f}",
                "mae": f"{row['mae']:.10f}",
                "mse_vs_current_pct": f"{((row['mse'] / base['mse']) - 1.0) * 100.0:.6f}",
                "mae_vs_current_pct": f"{((row['mae'] / base['mae']) - 1.0) * 100.0:.6f}",
                "better_than_current_mse": int(row["mse"] < base["mse"]),
            }
        )
    return output


def summarize_by_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current_mean = {
        dataset: mean(row["mse"] for row in rows if row["dataset"] == dataset and row["arm"] == "current_align_recon")
        for dataset in DATASETS
    }
    output: list[dict[str, Any]] = []
    for arm in ARMS:
        for dataset in DATASETS:
            subset = [row for row in rows if row["arm"] == arm and row["dataset"] == dataset]
            mse_mean = mean(row["mse"] for row in subset)
            mae_mean = mean(row["mae"] for row in subset)
            output.append(
                {
                    "arm": arm,
                    "dataset": dataset,
                    "mean_mse": f"{mse_mean:.10f}",
                    "mean_mae": f"{mae_mean:.10f}",
                    "mean_mse_vs_current_pct": f"{((mse_mean / current_mean[dataset]) - 1.0) * 100.0:.6f}",
                    "wins_vs_current": sum(
                        row["mse"] < base["mse"]
                        for row in subset
                        for base in rows
                        if base["arm"] == "current_align_recon"
                        and base["dataset"] == row["dataset"]
                        and base["target_horizon"] == row["target_horizon"]
                    ),
                    "num_horizons": len(subset),
                }
            )
    return output


def summarize_overall(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current_by_key = {
        (row["dataset"], row["target_horizon"]): row
        for row in rows
        if row["arm"] == "current_align_recon"
    }
    output: list[dict[str, Any]] = []
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        deltas = [
            ((row["mse"] / current_by_key[(row["dataset"], row["target_horizon"])]["mse"]) - 1.0) * 100.0
            for row in subset
        ]
        output.append(
            {
                "arm": arm,
                "mean_mse": f"{mean(row['mse'] for row in subset):.10f}",
                "mean_mae": f"{mean(row['mae'] for row in subset):.10f}",
                "mean_mse_vs_current_pct": f"{mean(deltas):.6f}",
                "wins_vs_current": sum(
                    row["mse"] < current_by_key[(row["dataset"], row["target_horizon"])]["mse"]
                    for row in subset
                ),
                "num_settings": len(subset),
                "max_regression_pct": f"{max(deltas):.6f}",
                "max_gain_pct": f"{min(deltas):.6f}",
            }
        )
    return output


def format_training_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        train_loss = row["train_loss"]
        output.append(
            {
                "arm": row["arm"],
                "dataset": row["dataset"],
                "epoch_selector": row["epoch_selector"],
                "epoch": row["epoch"],
                "train_loss": f"{train_loss:.10f}",
                "prediction_share": f"{row['train_prediction_l1'] / train_loss:.6f}",
                "weighted_recon_share": f"{row['train_weighted_reconstruction_l1'] / train_loss:.6f}",
                "weighted_align_share": f"{row['train_weighted_alignment_loss'] / train_loss:.6f}",
                "val_mean_mse": f"{row['val_mean_mse']:.10f}",
            }
        )
    return output


def render_report(
    horizon_rows: list[dict[str, Any]],
    dataset_summary: list[dict[str, Any]],
    overall_summary: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
) -> str:
    overall_by_arm = {row["arm"]: row for row in overall_summary}
    dataset_by_arm = {
        (row["arm"], row["dataset"]): row
        for row in dataset_summary
    }
    no_align = overall_by_arm["no_align_recon"]
    align_no_recon = overall_by_arm["align_no_recon"]
    pure = overall_by_arm["no_align_no_recon"]
    current = overall_by_arm["current_align_recon"]

    lines = [
        "# Phase5 StageB TimeAlign Dependency Ablation Report",
        "",
        "`current_step`: StageB Step 9/10 dependency ablation result analysis.",
        "",
        "## Scope",
        "",
        "[Fact] This report analyzes the completed 12-run remote matrix under `official-last` policy.",
        "",
        "[Boundary] These runs test dependency on inherited `w_align` and `w_recon`. They do not implement basis-aware alignment.",
        "",
        "## 11-step Record",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | StageB Step 9/10 dependency ablation analysis |",
        "| `problem` | Does A6-LBF depend on inherited TimeAlign alignment/reconstruction enough to require PhaseB align innovation? |",
        "| `existence_evidence` | 4-arm no-align/no-recon matrix on ETTh2/ETTm1/Weather with horizons 96/192/336/720 |",
        "| `idea` | Compare removing `w_align` and/or `w_recon` against current A6-LBF |",
        "| `theory_check` | If no-align/no-recon is competitive, full architecture independence is less risky; if removing align collapses performance, basis-aware align is better motivated |",
        "| `design` | Remote official-last ablation; no new method trained |",
        "| `narrative_gate` | dependency_ablation_pass_for_head_contribution_but_not_for_b5 |",
        "| `effectiveness_gate` | not applicable for new method; this is diagnostic evidence |",
        "| `artifacts` | this directory |",
        "| `decision` | A6-LBF is not heavily dependent on inherited alignment; do not prioritize B5 basis-aware alignment as the next method |",
        "",
        "## Overall Summary",
        "",
        "| Arm | Mean MSE | Mean MSE vs Current | Wins vs Current | Max Regression | Max Gain |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        row = overall_by_arm[arm]
        lines.append(
            f"| `{arm}` | `{float(row['mean_mse']):.4f}` | `{float(row['mean_mse_vs_current_pct']):.2f}%` | "
            f"{row['wins_vs_current']}/{row['num_settings']} | `{float(row['max_regression_pct']):.2f}%` | "
            f"`{float(row['max_gain_pct']):.2f}%` |"
        )
    lines.extend(
        [
            "",
            "## Dataset Summary",
            "",
            "| Arm | ETTh2 vs Current | ETTm1 vs Current | Weather vs Current |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for arm in ARMS:
        lines.append(
            f"| `{arm}` | `{float(dataset_by_arm[(arm, 'ETTh2')]['mean_mse_vs_current_pct']):.2f}%` | "
            f"`{float(dataset_by_arm[(arm, 'ETTm1')]['mean_mse_vs_current_pct']):.2f}%` | "
            f"`{float(dataset_by_arm[(arm, 'Weather')]['mean_mse_vs_current_pct']):.2f}%` |"
        )
    lines.extend(
        [
            "",
            "## Horizon-Level Deltas",
            "",
            "| Arm | Dataset | H96 | H192 | H336 | H720 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    by_key = {
        (row["arm"], row["dataset"], int(row["target_horizon"])): row
        for row in horizon_rows
    }
    for arm in ARMS:
        for dataset in DATASETS:
            lines.append(
                f"| `{arm}` | {dataset} | "
                f"`{float(by_key[(arm, dataset, 96)]['mse_vs_current_pct']):.2f}%` | "
                f"`{float(by_key[(arm, dataset, 192)]['mse_vs_current_pct']):.2f}%` | "
                f"`{float(by_key[(arm, dataset, 336)]['mse_vs_current_pct']):.2f}%` | "
                f"`{float(by_key[(arm, dataset, 720)]['mse_vs_current_pct']):.2f}%` |"
            )
    lines.extend(
        [
            "",
            "## Training Component Shares",
            "",
            "| Arm | Dataset | Selector | Epoch | Pred Share | Recon Share | Align Share | Val MSE |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in training_rows:
        if row["epoch_selector"] != "last":
            continue
        lines.append(
            f"| `{row['arm']}` | {row['dataset']} | `{row['epoch_selector']}` | {row['epoch']} | "
            f"`{float(row['prediction_share']):.2f}` | `{float(row['weighted_recon_share']):.2f}` | "
            f"`{float(row['weighted_align_share']):.2f}` | `{float(row['val_mean_mse']):.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"[Fact] Removing alignment while keeping reconstruction (`no_align_recon`) changes mean MSE by `{float(no_align['mean_mse_vs_current_pct']):.2f}%` and wins `{no_align['wins_vs_current']}/12` settings against current.",
            "",
            f"[Fact] Removing reconstruction while keeping alignment (`align_no_recon`) changes mean MSE by `{float(align_no_recon['mean_mse_vs_current_pct']):.2f}%` and wins `{align_no_recon['wins_vs_current']}/12` settings against current.",
            "",
            f"[Fact] Removing both inherited losses (`no_align_no_recon`) changes mean MSE by `{float(pure['mean_mse_vs_current_pct']):.2f}%` and wins `{pure['wins_vs_current']}/12` settings against current.",
            "",
            "[Strong Evidence] A6-LBF keeps most of its performance without inherited TimeAlign alignment/reconstruction. The pure head/operator arm is not collapsing.",
            "",
            "[Strong Evidence] The full inherited-auxiliary setting is not clearly better than ablated settings. `align_no_recon` is slightly better on mean MSE, and `no_align_no_recon` wins more than half of the horizon settings despite removing both inherited losses.",
            "",
            "[Mechanism Note] `no_align_recon` and `no_align_no_recon` are effectively identical in the returned metrics. This is expected from the current code path: when `w_align=0`, the future reconstruction branch has no active path into the history-derived prediction head, so reconstruction alone mostly trains the future branch/proj_y rather than the forecast operator.",
            "",
            "## Decision",
            "",
            "[Decision] `dependency_ablation_pass_for_head_contribution_but_not_for_b5`.",
            "",
            "[Narrative Consequence] The paper can defend A6-LBF as more than a TimeAlign-alignment artifact: the learned-basis head/operator remains competitive even without `w_align` and `w_recon`. This reduces the urgency of modifying TimeAlign's align mechanism just to claim independence.",
            "",
            "[StageB Consequence] B5 basis-aware future alignment is not strongly motivated as the next paper-core mechanism. If we implement it now, it risks being a small auxiliary-loss variant rather than a necessary architectural innovation.",
            "",
            "[Next Research Direction] Prefer returning to architecture-aware objective design around prefix-native learned basis / label-autocorrelation, with `no_align_no_recon` and `current_align_recon` as controls.",
            "",
            "## Output Files",
            "",
            "- `stage_b_dependency_ablation_horizon_metrics.csv`",
            "- `stage_b_dependency_ablation_dataset_summary.csv`",
            "- `stage_b_dependency_ablation_overall_summary.csv`",
            "- `stage_b_dependency_ablation_training_summary.csv`",
            "- `stage_b_dependency_ablation_report.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze StageB TimeAlign dependency ablation results.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("analysis/phase5_stage_b_timealign_dependency_ablation_20260706/raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/phase5_stage_b_timealign_dependency_ablation_20260706"),
    )
    args = parser.parse_args()

    raw_rows = collect_metric_rows(args.raw_root)
    horizon_rows = with_current_deltas(raw_rows)
    dataset_summary = summarize_by_dataset(raw_rows)
    overall_summary = summarize_overall(raw_rows)
    training_rows = format_training_rows(collect_training_rows(args.raw_root))

    write_csv(args.output_dir / "stage_b_dependency_ablation_horizon_metrics.csv", horizon_rows)
    write_csv(args.output_dir / "stage_b_dependency_ablation_dataset_summary.csv", dataset_summary)
    write_csv(args.output_dir / "stage_b_dependency_ablation_overall_summary.csv", overall_summary)
    write_csv(args.output_dir / "stage_b_dependency_ablation_training_summary.csv", training_rows)
    (args.output_dir / "stage_b_dependency_ablation_report.md").write_text(
        render_report(horizon_rows, dataset_summary, overall_summary, training_rows)
    )
    print(args.output_dir / "stage_b_dependency_ablation_report.md")


if __name__ == "__main__":
    main()
