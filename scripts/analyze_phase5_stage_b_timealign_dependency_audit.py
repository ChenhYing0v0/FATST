from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

DATASETS = ("ETTh2", "ETTm1", "Weather")
HORIZONS = (96, 192, 336, 720)
DEFAULT_A6_ROOT = Path("analysis/phase5_timealign_hss_a6_capacity_native_gate_20260703/raw/official-last")
DEFAULT_OFFICIAL_ROOT = Path(
    "analysis/phase5_timealign_hss_a3e_ettm1_replacement_gate_20260701/raw/official/official-last"
)
A6_RUN = "TimeAlignOfficialUnified720_A6_a6_lbf_r256_official-last"
OFFICIAL_RUN = "TimeAlignOfficialUnified720_official-last"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def run_dir(root: Path, run_name: str, dataset: str) -> Path:
    return root / run_name / dataset / "mixed_h96_h192_h336_h720" / "seed2021"


def config_weights(path: Path) -> tuple[float, float]:
    config = read_json(path)
    adapter = config.get("adapter", {})
    official = config.get("official_args", {})
    w_recon = float(adapter.get("w_recon", official.get("w_recon", 1.0)))
    w_align = float(official.get("w_align", 0.0))
    return w_recon, w_align


def component_rows(root: Path, run_name: str, variant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        directory = run_dir(root, run_name, dataset)
        logs = read_csv(directory / "training_log.csv")
        w_recon, w_align = config_weights(directory / "effective_config.json")
        best_row = min(logs, key=lambda row: float(row["val_mean_mse"]))
        last_row = logs[-1]
        for label, source in (("best_val", best_row), ("last", last_row)):
            pred = float(source["train_prediction_l1"])
            recon = float(source["train_reconstruction_l1"])
            align = float(source["train_alignment_loss"])
            total = float(source["train_loss"])
            weighted_recon = w_recon * recon
            weighted_align = w_align * align
            reconstructed_total = pred + weighted_recon + weighted_align
            rows.append(
                {
                    "variant": variant,
                    "dataset": dataset,
                    "epoch_selector": label,
                    "epoch": source["epoch"],
                    "train_loss": f"{total:.10f}",
                    "reconstructed_loss": f"{reconstructed_total:.10f}",
                    "prediction_l1": f"{pred:.10f}",
                    "weighted_reconstruction_l1": f"{weighted_recon:.10f}",
                    "weighted_alignment_loss": f"{weighted_align:.10f}",
                    "prediction_share": f"{pred / total:.6f}",
                    "reconstruction_share": f"{weighted_recon / total:.6f}",
                    "alignment_share": f"{weighted_align / total:.6f}",
                    "val_mean_mse": f"{float(source['val_mean_mse']):.10f}",
                    "w_recon": f"{w_recon:.6f}",
                    "w_align": f"{w_align:.6f}",
                }
            )
    return rows


def metrics_by_dataset(root: Path, run_name: str, variant: str) -> dict[str, dict[int, float]]:
    metrics: dict[str, dict[int, float]] = {}
    for dataset in DATASETS:
        path = run_dir(root, run_name, dataset) / "metrics_by_target_horizon.csv"
        dataset_metrics: dict[int, float] = {}
        for row in read_csv(path):
            horizon = int(row.get("target_horizon", row.get("horizon", "0")))
            if horizon in HORIZONS:
                dataset_metrics[horizon] = float(row["mse"])
        metrics[dataset] = dataset_metrics
    return metrics


def comparison_rows(a6_root: Path, official_root: Path) -> list[dict[str, Any]]:
    a6 = metrics_by_dataset(a6_root, A6_RUN, "a6_lbf_r256")
    official = metrics_by_dataset(official_root, OFFICIAL_RUN, "official_unified")
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            a6_mse = a6[dataset][horizon]
            official_mse = official[dataset][horizon]
            rows.append(
                {
                    "dataset": dataset,
                    "target_horizon": horizon,
                    "a6_lbf_mse": f"{a6_mse:.10f}",
                    "official_unified_mse": f"{official_mse:.10f}",
                    "a6_vs_official_mse_pct": f"{((a6_mse / official_mse) - 1.0) * 100.0:.6f}",
                    "same_timealign_alignment": 1,
                    "causal_attribution": "readout_difference_under_same_align_recon",
                }
            )
    return rows


def render_report(component: list[dict[str, Any]], comparison: list[dict[str, Any]]) -> str:
    a6_last = [row for row in component if row["variant"] == "a6_lbf_r256" and row["epoch_selector"] == "last"]
    official_last = [row for row in component if row["variant"] == "official_unified" and row["epoch_selector"] == "last"]
    wins = [row for row in comparison if float(row["a6_vs_official_mse_pct"]) < 0.0]
    mean_delta = mean(float(row["a6_vs_official_mse_pct"]) for row in comparison)
    lines = [
        "# Phase5 StageB TimeAlign Dependency Audit",
        "",
        "`current_step`: StageB Step 2/3 dependency diagnostic before any align/encoder modification.",
        "",
        "## Scope",
        "",
        "[Fact] This audit uses existing official-last artifacts only. It does not retrain any no-align/no-recon ablation.",
        "",
        "[Boundary] The audit can show whether A6-LBF improves under the same inherited TimeAlign alignment/reconstruction setting. It cannot causally isolate the alignment mechanism without new ablation runs.",
        "",
        "## 11-step Record",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | StageB Step 2/3 TimeAlign dependency diagnostic |",
        "| `problem` | Is A6-LBF's apparent advantage merely inherited from TimeAlign's future alignment mechanism? |",
        "| `existence_evidence` | A6 vs official unified metrics under the same align/recon setting; training loss component shares |",
        "| `idea` | Separate same-backbone readout evidence from missing causal alignment ablations |",
        "| `theory_check` | Same-align improvement supports a head/operator contribution, but does not make the whole architecture independent from TimeAlign |",
        "| `design` | Artifact-only audit; no code or remote training change |",
        "| `narrative_gate` | partial_dependency_risk_confirmed |",
        "| `effectiveness_gate` | not applicable; no new method trained |",
        "| `artifacts` | this directory |",
        "| `decision` | A6-LBF has same-align readout evidence, but paper still needs no-align/no-recon and basis-aware align diagnostics |",
        "",
        "## Same-Alignment Metric Comparison",
        "",
        "| Dataset | Horizon | A6 MSE | Official Unified MSE | A6 vs Official |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison:
        lines.append(
            f"| {row['dataset']} | {row['target_horizon']} | `{float(row['a6_lbf_mse']):.4f}` | "
            f"`{float(row['official_unified_mse']):.4f}` | `{float(row['a6_vs_official_mse_pct']):.2f}%` |"
        )
    lines.extend(
        [
            "",
            f"[Evidence] A6-LBF wins `{len(wins)}/12` settings against official unified TimeAlign under the same inherited align/recon setting; mean MSE change is `{mean_delta:.2f}%`.",
            "",
            "## Last-Epoch Loss Component Shares",
            "",
            "| Variant | Dataset | Pred Share | Recon Share | Align Share | Val MSE |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in a6_last + official_last:
        lines.append(
            f"| `{row['variant']}` | {row['dataset']} | `{float(row['prediction_share']):.2f}` | "
            f"`{float(row['reconstruction_share']):.2f}` | `{float(row['alignment_share']):.2f}` | "
            f"`{float(row['val_mean_mse']):.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "[Decision] `partial_dependency_risk_confirmed`.",
            "",
            "[Interpretation] A6-LBF has real same-alignment evidence because it improves over official unified while keeping TimeAlign's align/recon mechanism. However, this is not enough to claim a fully independent architecture. The training objective still contains inherited `w_recon * recon_loss + w_align * align_loss`, and no no-align/no-recon ablation exists in the current artifact set.",
            "",
            "[Next Required Diagnostic] Run a minimal TimeAlign dependency ablation matrix before any PhaseB align innovation claim:",
            "",
            "- A6-LBF with current `w_recon=1.0,w_align=0.1`;",
            "- A6-LBF with `w_align=0.0,w_recon=1.0`;",
            "- A6-LBF with `w_align=0.1,w_recon=0.0`;",
            "- A6-LBF with `w_align=0.0,w_recon=0.0`;",
            "- official unified TimeAlign under the same protocol as control.",
            "",
            "[Basis-Align Precondition] A future PhaseB align mechanism should only proceed if a separate checkpoint-based diagnostic shows that history-derived and future-derived coefficients are alignable in A6-LBF basis space.",
            "",
            "## Output Files",
            "",
            "- `stage_b_timealign_dependency_metric_comparison.csv`",
            "- `stage_b_timealign_dependency_training_components.csv`",
            "- `stage_b_timealign_dependency_report.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit A6-LBF dependency on inherited TimeAlign alignment.")
    parser.add_argument("--a6-root", type=Path, default=DEFAULT_A6_ROOT)
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/phase5_stage_b_timealign_dependency_audit_20260706"),
    )
    args = parser.parse_args()

    component = component_rows(args.a6_root, A6_RUN, "a6_lbf_r256")
    component.extend(component_rows(args.official_root, OFFICIAL_RUN, "official_unified"))
    comparison = comparison_rows(args.a6_root, args.official_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stage_b_timealign_dependency_training_components.csv", component)
    write_csv(args.output_dir / "stage_b_timealign_dependency_metric_comparison.csv", comparison)
    (args.output_dir / "stage_b_timealign_dependency_report.md").write_text(
        render_report(component, comparison)
    )
    print(args.output_dir / "stage_b_timealign_dependency_report.md")


if __name__ == "__main__":
    main()
