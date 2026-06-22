from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from statistics import mean

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="fatst-mpl-"))

import matplotlib.pyplot as plt


DATASETS = ["ETTh2", "ETTm1", "Weather"]
SEGMENTS = [(1, 96), (97, 192), (193, 336), (337, 720)]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def format_pct(value: float) -> str:
    return f"{value:+.2f}%"


def h720_step_mse(root: Path, model: str, dataset: str, mixed: bool) -> list[float]:
    if mixed:
        path = root / model / dataset / "mixed_h96_h192_h336_h720" / "seed2021" / "h720" / "metrics_by_horizon.csv"
    else:
        path = root / model / dataset / "h720" / "seed2021" / "metrics_by_horizon.csv"
    return [float(row["mse"]) for row in read_csv(path)]


def relative_curve(target: list[float], base: list[float]) -> list[float]:
    return [(target_value / base_value - 1.0) * 100.0 for target_value, base_value in zip(target, base, strict=True)]


def segment_rows(
    fixed_root: Path,
    r3_root: Path,
    phase2_root: Path,
) -> tuple[list[dict[str, object]], dict[str, list[float]]]:
    rows: list[dict[str, object]] = []
    curves: dict[str, list[float]] = {}
    for dataset in DATASETS:
        fixed = h720_step_mse(fixed_root, "PatchEncoderFixedHead", dataset, mixed=False)
        r3 = h720_step_mse(r3_root, "PatchEncoderPrefixRiskWeighted", dataset, mixed=True)
        phase2 = h720_step_mse(phase2_root, "PatchEncoderFutureStateAlignment", dataset, mixed=True)
        curves[f"{dataset}_r3_vs_fixed"] = relative_curve(r3, fixed)
        curves[f"{dataset}_phase2_vs_r3"] = relative_curve(phase2, r3)
        for start, end in SEGMENTS:
            fixed_mean = mean(fixed[start - 1 : end])
            r3_mean = mean(r3[start - 1 : end])
            phase2_mean = mean(phase2[start - 1 : end])
            rows.append(
                {
                    "dataset": dataset,
                    "segment": f"{start}-{end}",
                    "fixed_h720_mse": fixed_mean,
                    "r3_h720_mse": r3_mean,
                    "phase2_h720_mse": phase2_mean,
                    "r3_vs_fixed_mse_pct": (r3_mean / fixed_mean - 1.0) * 100.0,
                    "phase2_vs_r3_mse_pct": (phase2_mean / r3_mean - 1.0) * 100.0,
                    "r3_improves_fixed": r3_mean < fixed_mean,
                    "phase2_improves_r3": phase2_mean < r3_mean,
                }
            )
    return rows, curves


def curve_summary(curves: dict[str, list[float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, values in curves.items():
        dataset, comparison = key.split("_", 1)
        signs = [value < 0.0 for value in values]
        sign_changes = sum(1 for left, right in zip(signs, signs[1:], strict=False) if left != right)
        rows.append(
            {
                "dataset": dataset,
                "comparison": comparison,
                "mean_relative_mse_pct": mean(values),
                "early_1_96_relative_mse_pct": mean(values[:96]),
                "late_337_720_relative_mse_pct": mean(values[336:720]),
                "min_relative_mse_pct": min(values),
                "max_relative_mse_pct": max(values),
                "improved_steps": sum(1 for value in values if value < 0.0),
                "degraded_steps": sum(1 for value in values if value > 0.0),
                "sign_changes": sign_changes,
            }
        )
    return rows


def plot_curves(curves: dict[str, list[float]], output_dir: Path) -> None:
    for dataset in DATASETS:
        fig, ax = plt.subplots(figsize=(10, 4))
        steps = list(range(1, 721))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.plot(steps, curves[f"{dataset}_r3_vs_fixed"], label="R.3 vs FixedHead", linewidth=1.3)
        ax.plot(steps, curves[f"{dataset}_phase2_vs_r3"], label="Phase2-A vs R.3", linewidth=1.3)
        for boundary in [96, 192, 336]:
            ax.axvline(boundary, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(f"{dataset} H720 Step-wise Relative MSE")
        ax.set_xlabel("Forecast step")
        ax.set_ylabel("Relative MSE (%)")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"{dataset}_h720_step_relative_mse.png", dpi=180)
        plt.close(fig)


def write_report(
    path: Path,
    segment_stats: list[dict[str, object]],
    curve_stats: list[dict[str, object]],
) -> None:
    by_dataset = {
        dataset: [row for row in segment_stats if row["dataset"] == dataset]
        for dataset in DATASETS
    }
    lines = [
        "# Output/Error-Process Decoder Problem Diagnosis",
        "",
        "## Purpose",
        "",
        "[Fact] This report uses completed H720 step-wise artifacts to decide whether a future fallback direction should target the decoder output/error process rather than latent future-state alignment.",
        "",
        "## Main Finding",
        "",
        "[Strong Evidence] The current target-set/future-state line does not only have an average-MSE problem; it has a step-region error-process problem. R.3 often improves early steps but can degrade middle or late H720 regions, and Phase2-A changes this pattern in a dataset-dependent way.",
        "",
        "## Segment-Level H720 Evidence",
        "",
        "| Dataset | Segment | R.3 vs FixedHead | Phase2-A vs R.3 | Interpretation |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for dataset in DATASETS:
        for row in by_dataset[dataset]:
            r3_delta = float(row["r3_vs_fixed_mse_pct"])
            phase2_delta = float(row["phase2_vs_r3_mse_pct"])
            if r3_delta < 0 and phase2_delta > 0:
                interpretation = "R.3 helps, Phase2-A erases gain"
            elif r3_delta > 0 and phase2_delta < 0:
                interpretation = "R.3 weak region, Phase2-A repairs"
            elif r3_delta > 0 and phase2_delta > 0:
                interpretation = "both stages worsen this region"
            else:
                interpretation = "both stages help this region"
            lines.append(
                "| {dataset} | {segment} | {r3} | {phase2} | {interp} |".format(
                    dataset=dataset,
                    segment=row["segment"],
                    r3=format_pct(r3_delta),
                    phase2=format_pct(phase2_delta),
                    interp=interpretation,
                )
            )
    lines += [
        "",
        "## Curve Summary",
        "",
        "| Dataset | Comparison | Mean | Early 1-96 | Late 337-720 | Improved steps | Degraded steps | Sign changes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in curve_stats:
        lines.append(
            "| {dataset} | {comparison} | {mean} | {early} | {late} | {improved} | {degraded} | {changes} |".format(
                dataset=row["dataset"],
                comparison=row["comparison"],
                mean=format_pct(float(row["mean_relative_mse_pct"])),
                early=format_pct(float(row["early_1_96_relative_mse_pct"])),
                late=format_pct(float(row["late_337_720_relative_mse_pct"])),
                improved=row["improved_steps"],
                degraded=row["degraded_steps"],
                changes=row["sign_changes"],
            )
        )
    lines += [
        "",
        "## Decision Impact",
        "",
        "[Inference] If Phase2-R.1 fails, the next problem should not be another stronger future teacher. A more defensible decoder problem is: how should a one-model decoder model the non-uniform error growth and residual process across forecast steps?",
        "",
        "[Candidate Problem] Current decoder states produce point segments independently after conditioning on target queries. They do not explicitly model that error is an output process with step-region structure, sign-changing gains, and late-horizon growth. A next architecture should treat the prediction trajectory or residual trajectory as the object being decoded, not just each segment's mean state.",
        "",
        "[Candidate Mechanism Direction] Output/error-process decoder: generate a base forecast plus a structured residual process over future steps, with constraints or parameterization that can express monotone error growth, covariance between adjacent steps, and segment-specific residual corrections. This should be evaluated before adding MoE if Phase2-R.1 fails.",
        "",
        "## Figures",
        "",
        "- `ETTh2_h720_step_relative_mse.png`",
        "- `ETTm1_h720_step_relative_mse.png`",
        "- `Weather_h720_step_relative_mse.png`",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose output/error-process decoder problem from H720 curves.")
    parser.add_argument(
        "--fixed-root",
        default="analysis/phase1_trajectory_basis_residual_gate_20260622/raw",
    )
    parser.add_argument(
        "--r3-root",
        default="analysis/phase1_prefix_risk_weighted_gate_20260622/raw",
    )
    parser.add_argument(
        "--phase2-root",
        default="analysis/phase2_future_state_alignment_gate_20260622/raw",
    )
    parser.add_argument(
        "--output-root",
        default="analysis/output_error_process_diagnosis_20260623",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows, curves = segment_rows(
        fixed_root=Path(args.fixed_root),
        r3_root=Path(args.r3_root),
        phase2_root=Path(args.phase2_root),
    )
    curve_stats = curve_summary(curves)
    write_csv(output_root / "h720_segment_error_process.csv", rows)
    write_csv(output_root / "h720_curve_error_process_summary.csv", curve_stats)
    (output_root / "h720_step_relative_mse_curves.json").write_text(json.dumps(curves))
    plot_curves(curves, output_root)
    write_report(output_root / "output_error_process_diagnosis_report.md", rows, curve_stats)


if __name__ == "__main__":
    main()
