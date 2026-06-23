from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from statistics import mean

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="fatst-mpl-"))

import matplotlib.pyplot as plt


DATASETS = ["ETTh2", "ETTm1", "Weather"]
HORIZONS = [96, 192, 336, 720]
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


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_den = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_den = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_den == 0.0 or right_den == 0.0:
        return float("nan")
    return numerator / (left_den * right_den)


def step_weights(max_pred_len: int, mode: str, alpha: float) -> list[float]:
    if mode == "uniform":
        return [1.0] * max_pred_len
    if mode != "prefix_risk":
        raise ValueError(f"Unsupported mode: {mode}")
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    raw = [(step / float(max_pred_len)) ** (-alpha) for step in range(1, max_pred_len + 1)]
    scale = mean(raw)
    return [value / scale for value in raw]


def expected_step_pressure(
    horizons: list[int],
    max_pred_len: int,
    mode: str,
    alpha: float,
) -> list[float]:
    weights = step_weights(max_pred_len, mode, alpha)
    pressure = []
    for step in range(1, max_pred_len + 1):
        coeff = 0.0
        for horizon in horizons:
            if step <= horizon:
                coeff += weights[step - 1] / float(horizon)
        pressure.append(coeff / float(len(horizons)))
    return pressure


def pressure_rows(max_pred_len: int, alpha: float) -> tuple[list[dict[str, object]], dict[str, list[float]]]:
    uniform = expected_step_pressure(HORIZONS, max_pred_len, "uniform", alpha)
    prefix = expected_step_pressure(HORIZONS, max_pred_len, "prefix_risk", alpha)
    uniform_total = sum(uniform)
    prefix_total = sum(prefix)
    uniform_share = [value / uniform_total for value in uniform]
    prefix_share = [value / prefix_total for value in prefix]
    rows: list[dict[str, object]] = []
    for start, end in SEGMENTS:
        uniform_region = sum(uniform_share[start - 1 : end])
        prefix_region = sum(prefix_share[start - 1 : end])
        rows.append(
            {
                "segment": f"{start}-{end}",
                "uniform_pressure_share": uniform_region,
                "prefix_pressure_share": prefix_region,
                "pressure_share_delta_pct": (prefix_region / uniform_region - 1.0) * 100.0,
                "prefix_raw_pressure": sum(prefix[start - 1 : end]),
                "uniform_raw_pressure": sum(uniform[start - 1 : end]),
                "raw_pressure_ratio": sum(prefix[start - 1 : end]) / sum(uniform[start - 1 : end]),
            }
        )
    for horizon in HORIZONS:
        weights = step_weights(max_pred_len, "prefix_risk", alpha)
        rows.append(
            {
                "segment": f"horizon_{horizon}_loss_multiplier",
                "uniform_pressure_share": 1.0,
                "prefix_pressure_share": mean(weights[:horizon]),
                "pressure_share_delta_pct": (mean(weights[:horizon]) - 1.0) * 100.0,
                "prefix_raw_pressure": mean(weights[:horizon]),
                "uniform_raw_pressure": 1.0,
                "raw_pressure_ratio": mean(weights[:horizon]),
            }
        )
    curves = {
        "uniform_pressure_share": uniform_share,
        "prefix_pressure_share": prefix_share,
        "prefix_vs_uniform_share_ratio": [
            prefix_value / uniform_value
            for prefix_value, uniform_value in zip(prefix_share, uniform_share, strict=True)
        ],
    }
    return rows, curves


def keyed(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def compare_r3_to_uniform(
    uniform_root: Path,
    r3_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    uniform_main = keyed(read_csv(uniform_root / "phase1_target_set_decoder_vs_fixed.csv"), ("dataset", "horizon"))
    r3_main = keyed(read_csv(r3_root / "phase1_prefix_risk_weighted_vs_fixed.csv"), ("dataset", "horizon"))
    main_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            key = (dataset, str(horizon))
            uniform = uniform_main[key]
            r3 = r3_main[key]
            uniform_mse = float(uniform["target_mse"])
            r3_mse = float(r3["target_mse"])
            main_rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "uniform_mse": uniform_mse,
                    "r3_mse": r3_mse,
                    "fixed_mse": float(r3["fixed_mse"]),
                    "r3_vs_uniform_mse_pct": (r3_mse / uniform_mse - 1.0) * 100.0,
                    "uniform_vs_fixed_mse_pct": float(uniform["relative_mse_pct"]),
                    "r3_vs_fixed_mse_pct": float(r3["relative_mse_pct"]),
                    "r3_improves_uniform": r3_mse < uniform_mse,
                    "r3_improves_fixed": r3_mse < float(r3["fixed_mse"]),
                }
            )

    uniform_segments = keyed(
        read_csv(uniform_root / "phase1_target_set_decoder_vs_fixed_segments.csv"),
        ("dataset", "horizon", "segment"),
    )
    r3_segments = keyed(
        read_csv(r3_root / "phase1_prefix_risk_weighted_vs_fixed_segments.csv"),
        ("dataset", "horizon", "segment"),
    )
    segment_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            for start, end in SEGMENTS:
                if start > horizon:
                    continue
                segment = f"{start}-{min(end, horizon)}"
                key = (dataset, str(horizon), segment)
                if key not in uniform_segments or key not in r3_segments:
                    continue
                uniform = uniform_segments[key]
                r3 = r3_segments[key]
                uniform_mse = float(uniform["target_mse"])
                r3_mse = float(r3["target_mse"])
                fixed_mse = float(r3["fixed_mse"])
                segment_rows.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "segment": segment,
                        "uniform_mse": uniform_mse,
                        "r3_mse": r3_mse,
                        "fixed_mse": fixed_mse,
                        "r3_vs_uniform_mse_pct": (r3_mse / uniform_mse - 1.0) * 100.0,
                        "uniform_vs_fixed_mse_pct": float(uniform["relative_mse_pct"]),
                        "r3_vs_fixed_mse_pct": float(r3["relative_mse_pct"]),
                        "r3_improves_uniform": r3_mse < uniform_mse,
                        "r3_improves_fixed": r3_mse < fixed_mse,
                    }
                )

    uniform_prefix = keyed(
        read_csv(uniform_root / "phase1_target_set_decoder_h720_prefix_reference.csv"),
        ("dataset", "prefix_horizon"),
    )
    r3_prefix = keyed(
        read_csv(r3_root / "phase1_prefix_risk_weighted_h720_prefix_reference.csv"),
        ("dataset", "prefix_horizon"),
    )
    prefix_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            key = (dataset, str(horizon))
            uniform = uniform_prefix[key]
            r3 = r3_prefix[key]
            uniform_mse = float(uniform["target_h720_prefix_mse"])
            r3_mse = float(r3["target_h720_prefix_mse"])
            fixed_mse = float(r3["fixed_h720_prefix_mse"])
            prefix_rows.append(
                {
                    "dataset": dataset,
                    "prefix_horizon": horizon,
                    "uniform_h720_prefix_mse": uniform_mse,
                    "r3_h720_prefix_mse": r3_mse,
                    "fixed_h720_prefix_mse": fixed_mse,
                    "r3_vs_uniform_mse_pct": (r3_mse / uniform_mse - 1.0) * 100.0,
                    "uniform_vs_fixed_mse_pct": float(uniform["relative_mse_pct"]),
                    "r3_vs_fixed_mse_pct": float(r3["relative_mse_pct"]),
                    "r3_improves_uniform": r3_mse < uniform_mse,
                    "r3_improves_fixed": r3_mse < fixed_mse,
                }
            )
    return main_rows, segment_rows, prefix_rows


def segment_pressure_delta_map(pressure_summary_rows: list[dict[str, object]]) -> dict[str, float]:
    return {
        str(row["segment"]): float(row["pressure_share_delta_pct"])
        for row in pressure_summary_rows
        if not str(row["segment"]).startswith("horizon_")
    }


def horizon_multiplier_map(pressure_summary_rows: list[dict[str, object]]) -> dict[int, float]:
    prefix = "horizon_"
    suffix = "_loss_multiplier"
    rows = {}
    for row in pressure_summary_rows:
        segment = str(row["segment"])
        if segment.startswith(prefix) and segment.endswith(suffix):
            horizon = int(segment[len(prefix) : -len(suffix)])
            rows[horizon] = float(row["raw_pressure_ratio"])
    return rows


def summarize(
    pressure_summary_rows: list[dict[str, object]],
    main_rows: list[dict[str, object]],
    segment_rows: list[dict[str, object]],
    prefix_rows: list[dict[str, object]],
) -> dict[str, object]:
    main_delta = [float(row["r3_vs_uniform_mse_pct"]) for row in main_rows]
    segment_delta = [float(row["r3_vs_uniform_mse_pct"]) for row in segment_rows]
    prefix_delta = [float(row["r3_vs_uniform_mse_pct"]) for row in prefix_rows]
    h96_main = [float(row["r3_vs_uniform_mse_pct"]) for row in main_rows if int(row["horizon"]) == 96]
    h720_main = [float(row["r3_vs_uniform_mse_pct"]) for row in main_rows if int(row["horizon"]) == 720]
    h720_prefix_short = [
        float(row["r3_vs_uniform_mse_pct"])
        for row in prefix_rows
        if int(row["prefix_horizon"]) in {96, 192}
    ]
    pressure_delta = segment_pressure_delta_map(pressure_summary_rows)
    horizon_multiplier = horizon_multiplier_map(pressure_summary_rows)
    segment_pressure_values = [pressure_delta[str(row["segment"])] for row in segment_rows]
    horizon_pressure_values = [horizon_multiplier[int(row["horizon"])] for row in main_rows]
    gate = {
        "r3_improves_uniform_mean_mse": mean(main_delta) < -0.5,
        "r3_improves_all_h96_vs_uniform": all(value < 0.0 for value in h96_main),
        "r3_improves_h720_prefix_short_vs_uniform": mean(h720_prefix_short) < -1.0,
        "r3_still_has_fixed_specialist_gap": any(float(row["r3_vs_fixed_mse_pct"]) > 0.0 for row in main_rows),
        "segment_effect_varies_by_pressure": not math.isnan(pearson(segment_pressure_values, segment_delta))
        and abs(pearson(segment_pressure_values, segment_delta)) >= 0.20,
    }
    gate["objective_problem_exists"] = all(
        [
            gate["r3_improves_uniform_mean_mse"],
            gate["r3_improves_all_h96_vs_uniform"],
            gate["r3_improves_h720_prefix_short_vs_uniform"],
        ]
    )
    gate["naive_prefix_risk_is_insufficient"] = gate["r3_still_has_fixed_specialist_gap"]
    return {
        "mean_r3_vs_uniform_mse_pct": mean(main_delta),
        "wins_r3_vs_uniform": sum(1 for row in main_rows if bool(row["r3_improves_uniform"])),
        "mean_segment_r3_vs_uniform_mse_pct": mean(segment_delta),
        "segment_wins_r3_vs_uniform": sum(1 for row in segment_rows if bool(row["r3_improves_uniform"])),
        "mean_h96_r3_vs_uniform_mse_pct": mean(h96_main),
        "mean_h720_r3_vs_uniform_mse_pct": mean(h720_main),
        "mean_h720_prefix_short_r3_vs_uniform_mse_pct": mean(h720_prefix_short),
        "mean_h720_prefix_all_r3_vs_uniform_mse_pct": mean(prefix_delta),
        "main_delta_vs_horizon_loss_multiplier_pearson": pearson(horizon_pressure_values, main_delta),
        "segment_delta_vs_pressure_share_delta_pearson": pearson(segment_pressure_values, segment_delta),
        "gate": gate,
    }


def plot_pressure(curves: dict[str, list[float]], output_dir: Path) -> None:
    steps = list(range(1, len(curves["uniform_pressure_share"]) + 1))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(steps, curves["uniform_pressure_share"], label="uniform", linewidth=1.2)
    axes[0].plot(steps, curves["prefix_pressure_share"], label="prefix-risk", linewidth=1.2)
    axes[0].set_ylabel("Expected pressure share")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(steps, curves["prefix_vs_uniform_share_ratio"], color="#b45309", linewidth=1.2)
    axes[1].axhline(1.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Forecast step")
    axes[1].set_ylabel("Prefix / uniform")
    axes[1].grid(alpha=0.25)
    for ax in axes:
        for boundary in [96, 192, 336]:
            ax.axvline(boundary, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    fig.suptitle("Mixed-Horizon Objective Pressure")
    fig.tight_layout()
    fig.savefig(output_dir / "objective_pressure_curve.png", dpi=180)
    plt.close(fig)


def plot_segment_effect(segment_rows: list[dict[str, object]], pressure_rows_in: list[dict[str, object]], output_dir: Path) -> None:
    pressure_delta = segment_pressure_delta_map(pressure_rows_in)
    x = [pressure_delta[str(row["segment"])] for row in segment_rows]
    y = [float(row["r3_vs_uniform_mse_pct"]) for row in segment_rows]
    colors = {"ETTh2": "#2563eb", "ETTm1": "#16a34a", "Weather": "#dc2626"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset in DATASETS:
        subset = [row for row in segment_rows if row["dataset"] == dataset]
        ax.scatter(
            [pressure_delta[str(row["segment"])] for row in subset],
            [float(row["r3_vs_uniform_mse_pct"]) for row in subset],
            label=dataset,
            s=38,
            alpha=0.8,
            color=colors[dataset],
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Pressure share delta vs uniform (%)")
    ax.set_ylabel("R.3 relative MSE vs uniform (%)")
    ax.set_title(f"Segment Effect vs Objective Pressure (r={pearson(x, y):.3f})")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "segment_effect_vs_pressure.png", dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    pressure_summary_rows: list[dict[str, object]],
    main_rows: list[dict[str, object]],
    segment_rows: list[dict[str, object]],
    prefix_rows: list[dict[str, object]],
    summary: dict[str, object],
    alpha: float,
) -> None:
    gate = summary["gate"]
    decision = (
        "`mixed-horizon objective / step-region covariance` is a real Phase2-C problem."
        if gate["objective_problem_exists"]
        else "`mixed-horizon objective / step-region covariance` is not yet proven as the next problem."
    )
    insufficiency = (
        "However, naive monotone prefix-risk weighting is insufficient as a paper-core mechanism."
        if gate["naive_prefix_risk_is_insufficient"]
        else "Naive prefix-risk weighting does not yet show a clear fixed-specialist gap."
    )
    lines = [
        "# Phase2-C Objective Pressure Diagnostic",
        "",
        "## Decision",
        "",
        f"[Decision] {decision}",
        "",
        f"[Inference] {insufficiency}",
        "",
        "## What Was Tested",
        "",
        "[Fact] This diagnostic compares R.3 `PatchEncoderPrefixRiskWeighted` against the uniform `PatchEncoderTargetSetDecoder`, not only against `PatchEncoderFixedHead`.",
        "",
        "[Fact] The expected step pressure follows the actual training loop: one horizon is sampled uniformly from `{96,192,336,720}`, and the loss is averaged over the selected horizon. For prefix-risk, the per-step weight is normalized by the full `Hmax=720` weight mean.",
        "",
        "For a step $t$, the expected pressure is:",
        "",
        "$$",
        "p_t = \\frac{1}{|\\mathcal{H}|}\\sum_{H\\in\\mathcal{H}, t\\le H}\\frac{w_t}{H}.",
        "$$",
        "",
        f"Here `alpha={alpha}` for R.3.",
        "",
        "## Main Evidence",
        "",
        f"- R.3 wins vs uniform target-set: `{summary['wins_r3_vs_uniform']}/12`.",
        f"- Mean relative MSE vs uniform: `{format_pct(float(summary['mean_r3_vs_uniform_mse_pct']))}`.",
        f"- Mean h96 relative MSE vs uniform: `{format_pct(float(summary['mean_h96_r3_vs_uniform_mse_pct']))}`.",
        f"- Mean h720 relative MSE vs uniform: `{format_pct(float(summary['mean_h720_r3_vs_uniform_mse_pct']))}`.",
        f"- H720-prefix h96/h192 mean relative MSE vs uniform: `{format_pct(float(summary['mean_h720_prefix_short_r3_vs_uniform_mse_pct']))}`.",
        f"- Segment wins vs uniform: `{summary['segment_wins_r3_vs_uniform']}/{len(segment_rows)}`.",
        f"- Segment relative MSE vs uniform: `{format_pct(float(summary['mean_segment_r3_vs_uniform_mse_pct']))}`.",
        "",
        "## Objective Pressure Shift",
        "",
        "| Region | Uniform pressure share | Prefix-risk pressure share | Share delta | Raw pressure ratio |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in pressure_summary_rows:
        segment = str(row["segment"])
        if segment.startswith("horizon_"):
            continue
        lines.append(
            "| {segment} | {uniform:.4f} | {prefix:.4f} | {delta} | {ratio:.3f} |".format(
                segment=segment,
                uniform=float(row["uniform_pressure_share"]),
                prefix=float(row["prefix_pressure_share"]),
                delta=format_pct(float(row["pressure_share_delta_pct"])),
                ratio=float(row["raw_pressure_ratio"]),
            )
        )
    lines += [
        "",
        "## Horizon Loss Multipliers",
        "",
        "| Horizon | Mean prefix-risk weight | Interpretation |",
        "| ---: | ---: | --- |",
    ]
    for row in pressure_summary_rows:
        segment = str(row["segment"])
        if not segment.startswith("horizon_"):
            continue
        horizon = segment.removeprefix("horizon_").removesuffix("_loss_multiplier")
        ratio = float(row["raw_pressure_ratio"])
        if ratio > 1.0:
            interpretation = "amplified relative to uniform"
        elif ratio < 1.0:
            interpretation = "down-weighted relative to uniform"
        else:
            interpretation = "same as uniform"
        lines.append(f"| {horizon} | {ratio:.3f} | {interpretation} |")
    lines += [
        "",
        "## R.3 vs Uniform Target-Set",
        "",
        "| Dataset | Horizon | R.3 vs uniform | Uniform vs FixedHead | R.3 vs FixedHead |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in main_rows:
        lines.append(
            "| {dataset} | {horizon} | {du} | {uf} | {rf} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                du=format_pct(float(row["r3_vs_uniform_mse_pct"])),
                uf=format_pct(float(row["uniform_vs_fixed_mse_pct"])),
                rf=format_pct(float(row["r3_vs_fixed_mse_pct"])),
            )
        )
    lines += [
        "",
        "## Alignment Between Pressure And Effect",
        "",
        f"- Pearson r between horizon loss multiplier and R.3 main-horizon delta: `{summary['main_delta_vs_horizon_loss_multiplier_pearson']:.4f}`.",
        f"- Pearson r between segment pressure-share delta and segment-level R.3 delta: `{summary['segment_delta_vs_pressure_share_delta_pearson']:.4f}`.",
        "",
        "[Inference] A useful objective-level direction should not be another hand-tuned monotone prefix emphasis. The evidence supports a more structured objective that can distinguish early prefixes, middle regions, and long-tail regions rather than assigning all steps a single decreasing curve.",
        "",
        "## Gate",
        "",
    ]
    for key, value in gate.items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "",
        "## Decision Impact",
        "",
        "[Decision] The next implementable candidate should be an objective-level mechanism, not a new target-state interaction or MoE layer. It should explicitly model step-region covariance or horizon-region balance, then be evaluated against both uniform target-set and R.3.",
        "",
        "[Candidate] Phase2-C can test a `Step-Covariance Balanced Objective`: estimate a fixed region covariance/importance prior from training targets or validation residual structure, then use it to balance loss pressure across early prefix, middle transition, and long-tail regions. The pass condition must require improvement over R.3, not merely over uniform target-set.",
        "",
        "## Artifacts",
        "",
        "- `objective_pressure_summary.csv`",
        "- `r3_vs_uniform_main.csv`",
        "- `r3_vs_uniform_segments.csv`",
        "- `r3_vs_uniform_h720_prefix.csv`",
        "- `objective_pressure_summary.json`",
        "- `objective_pressure_curve.png`",
        "- `segment_effect_vs_pressure.png`",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit mixed-horizon objective pressure and R.3 effects.")
    parser.add_argument(
        "--uniform-root",
        default="analysis/phase1_target_set_decoder_gate_20260622",
        help="Analysis directory for uniform PatchEncoderTargetSetDecoder.",
    )
    parser.add_argument(
        "--r3-root",
        default="analysis/phase1_prefix_risk_weighted_gate_20260622",
        help="Analysis directory for PatchEncoderPrefixRiskWeighted.",
    )
    parser.add_argument("--output-root", default="analysis/phase2_objective_pressure_diagnostic_20260623")
    parser.add_argument("--max-pred-len", type=int, default=720)
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pressure_summary, pressure_curves = pressure_rows(args.max_pred_len, args.alpha)
    main_rows, segment_rows, prefix_rows = compare_r3_to_uniform(Path(args.uniform_root), Path(args.r3_root))
    summary = summarize(pressure_summary, main_rows, segment_rows, prefix_rows)
    write_csv(output_root / "objective_pressure_summary.csv", pressure_summary)
    write_csv(output_root / "r3_vs_uniform_main.csv", main_rows)
    write_csv(output_root / "r3_vs_uniform_segments.csv", segment_rows)
    write_csv(output_root / "r3_vs_uniform_h720_prefix.csv", prefix_rows)
    (output_root / "objective_pressure_curves.json").write_text(json.dumps(pressure_curves))
    (output_root / "objective_pressure_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    plot_pressure(pressure_curves, output_root)
    plot_segment_effect(segment_rows, pressure_summary, output_root)
    write_report(
        output_root / "phase2_objective_pressure_diagnostic_report.md",
        pressure_summary,
        main_rows,
        segment_rows,
        prefix_rows,
        summary,
        args.alpha,
    )
    print(f"report={output_root / 'phase2_objective_pressure_diagnostic_report.md'}")


if __name__ == "__main__":
    main()
