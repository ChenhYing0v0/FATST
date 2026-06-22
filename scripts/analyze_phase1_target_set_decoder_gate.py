from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


DATASETS = ["ETTh2", "ETTm1", "Weather"]
HORIZONS = [96, 192, 336, 720]
SEED = 2021
TARGET_RUN = "mixed_h96_h192_h336_h720"


def read_json(path: Path) -> dict[str, float]:
    return json.loads(path.read_text())


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


def target_run_dir(raw_root: Path, dataset: str) -> Path:
    return raw_root / "PatchEncoderTargetSetDecoder" / dataset / TARGET_RUN / f"seed{SEED}"


def fixed_run_dir(fixed_raw_root: Path, dataset: str, horizon: int) -> Path:
    return fixed_raw_root / "PatchEncoderFixedHead" / dataset / f"h{horizon}" / f"seed{SEED}"


def collect_main_metrics(raw_root: Path, fixed_raw_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        target_metrics = read_csv(target_run_dir(raw_root, dataset) / "metrics_by_target_horizon.csv")
        target_by_horizon = {int(row["target_horizon"]): row for row in target_metrics}
        for horizon in HORIZONS:
            target = target_by_horizon[horizon]
            fixed = read_json(fixed_run_dir(fixed_raw_root, dataset, horizon) / "metrics.json")
            target_mse = float(target["mse"])
            fixed_mse = float(fixed["mse"])
            target_mae = float(target["mae"])
            fixed_mae = float(fixed["mae"])
            rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "target_mse": target_mse,
                    "target_mae": target_mae,
                    "fixed_mse": fixed_mse,
                    "fixed_mae": fixed_mae,
                    "relative_mse_pct": (target_mse / fixed_mse - 1.0) * 100.0,
                    "relative_mae_pct": (target_mae / fixed_mae - 1.0) * 100.0,
                    "target_wins_mse": target_mse < fixed_mse,
                    "target_wins_mae": target_mae < fixed_mae,
                }
            )
    return rows


def collect_segment_metrics(raw_root: Path, fixed_raw_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            target_rows = read_csv(target_run_dir(raw_root, dataset) / f"h{horizon}" / "metrics_by_segment.csv")
            fixed_rows = read_csv(fixed_run_dir(fixed_raw_root, dataset, horizon) / "metrics_by_segment.csv")
            fixed_by_segment = {row["segment"]: row for row in fixed_rows}
            for target in target_rows:
                segment = target["segment"]
                fixed = fixed_by_segment[segment]
                target_mse = float(target["mse"])
                fixed_mse = float(fixed["mse"])
                target_mae = float(target["mae"])
                fixed_mae = float(fixed["mae"])
                rows.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "segment": segment,
                        "target_mse": target_mse,
                        "target_mae": target_mae,
                        "fixed_mse": fixed_mse,
                        "fixed_mae": fixed_mae,
                        "relative_mse_pct": (target_mse / fixed_mse - 1.0) * 100.0,
                        "relative_mae_pct": (target_mae / fixed_mae - 1.0) * 100.0,
                        "target_wins_mse": target_mse < fixed_mse,
                        "target_wins_mae": target_mae < fixed_mae,
                    }
                )
    return rows


def collect_prefix_metrics(raw_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for row in read_csv(target_run_dir(raw_root, dataset) / "prefix_consistency.csv"):
            rows.append(
                {
                    "dataset": dataset,
                    "short_horizon": int(row["short_horizon"]),
                    "long_horizon": int(row["long_horizon"]),
                    "prefix_mismatch_mse": float(row["prefix_mismatch_mse"]),
                    "prefix_mismatch_mae": float(row["prefix_mismatch_mae"]),
                    "truth_alignment_mse": float(row["truth_alignment_mse"]),
                }
            )
    return rows


def collect_state_metrics(raw_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            run_dir = target_run_dir(raw_root, dataset) / f"h{horizon}"
            similarity = read_csv(run_dir / "target_state_similarity.csv")
            conditioning = read_csv(run_dir / "target_conditioning_stats.csv")
            all_conditioning = next(row for row in conditioning if row["scope"] == "all")
            cosine_values = [float(row["cosine"]) for row in similarity]
            rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "mean_target_state_cosine": mean(cosine_values) if cosine_values else 1.0,
                    "mean_abs_gamma": float(all_conditioning["mean_abs_gamma"]),
                    "mean_abs_beta": float(all_conditioning["mean_abs_beta"]),
                    "mean_target_state_norm": float(all_conditioning["mean_target_state_norm"]),
                    "std_target_state_norm": float(all_conditioning["std_target_state_norm"]),
                    "mean_history_readout_norm": float(all_conditioning["mean_history_readout_norm"]),
                }
            )
    return rows


def collect_h720_prefix_reference(raw_root: Path) -> list[dict[str, object]]:
    phase0_rows = read_csv(Path("analysis/phase0_prefix_consistency_raw_20260621.csv"))
    fixed_prefix = {
        (row["dataset"], int(row["prefix_horizon"])): row
        for row in phase0_rows
        if row["model"] == "PatchEncoderFixedHead" and int(row["max_horizon"]) == 720
    }
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        h720_steps = read_csv(target_run_dir(raw_root, dataset) / "h720" / "metrics_by_horizon.csv")
        step_mse = [float(row["mse"]) for row in h720_steps]
        step_mae = [float(row["mae"]) for row in h720_steps]
        for horizon in HORIZONS:
            fixed = fixed_prefix[(dataset, horizon)]
            target_mse = mean(step_mse[:horizon])
            target_mae = mean(step_mae[:horizon])
            fixed_mse = float(fixed["max_prefix_mse"])
            fixed_mae = float(fixed["max_prefix_mae"])
            rows.append(
                {
                    "dataset": dataset,
                    "prefix_horizon": horizon,
                    "target_h720_prefix_mse": target_mse,
                    "target_h720_prefix_mae": target_mae,
                    "fixed_h720_prefix_mse": fixed_mse,
                    "fixed_h720_prefix_mae": fixed_mae,
                    "relative_mse_pct": (target_mse / fixed_mse - 1.0) * 100.0,
                    "relative_mae_pct": (target_mae / fixed_mae - 1.0) * 100.0,
                    "target_wins_mse": target_mse < fixed_mse,
                    "target_wins_mae": target_mae < fixed_mae,
                }
            )
    return rows


def summarize(
    main_rows: list[dict[str, object]],
    segment_rows: list[dict[str, object]],
    prefix_rows: list[dict[str, object]],
    h720_prefix_rows: list[dict[str, object]],
    state_rows: list[dict[str, object]],
) -> dict[str, object]:
    rel = [float(row["relative_mse_pct"]) for row in main_rows]
    segment_rel = [float(row["relative_mse_pct"]) for row in segment_rows]
    dataset_mean = {
        dataset: mean(float(row["relative_mse_pct"]) for row in main_rows if row["dataset"] == dataset)
        for dataset in DATASETS
    }
    horizon_mean = {
        str(horizon): mean(float(row["relative_mse_pct"]) for row in main_rows if row["horizon"] == horizon)
        for horizon in HORIZONS
    }
    compatibility = {
        "mean_relative_mse_le_1pct": mean(rel) <= 1.0,
        "no_dataset_degrades_over_3pct": max(dataset_mean.values()) <= 3.0,
        "h96_h192_not_worse_than_fixed_h720_prefix": all(
            bool(row["target_wins_mse"]) or abs(float(row["relative_mse_pct"])) < 1e-12
            for row in h720_prefix_rows
            if int(row["prefix_horizon"]) in {96, 192}
        ),
        "prefix_mismatch_near_zero": max(float(row["prefix_mismatch_mse"]) for row in prefix_rows) <= 1e-10,
        "target_states_non_identical": mean(float(row["mean_target_state_cosine"]) for row in state_rows) < 0.999,
    }
    compatibility["pass"] = all(compatibility.values())
    return {
        "main_rows": len(main_rows),
        "segment_rows": len(segment_rows),
        "mse_wins": sum(1 for row in main_rows if row["target_wins_mse"]),
        "mae_wins": sum(1 for row in main_rows if row["target_wins_mae"]),
        "mean_relative_mse_pct": mean(rel),
        "min_relative_mse_pct": min(rel),
        "max_relative_mse_pct": max(rel),
        "dataset_mean_relative_mse_pct": dataset_mean,
        "horizon_mean_relative_mse_pct": horizon_mean,
        "segment_mse_wins": sum(1 for row in segment_rows if row["target_wins_mse"]),
        "segment_mean_relative_mse_pct": mean(segment_rel),
        "h720_prefix_mse_wins": sum(1 for row in h720_prefix_rows if row["target_wins_mse"]),
        "h720_prefix_h96_h192_mean_relative_mse_pct": mean(
            float(row["relative_mse_pct"])
            for row in h720_prefix_rows
            if int(row["prefix_horizon"]) in {96, 192}
        ),
        "max_prefix_mismatch_mse": max(float(row["prefix_mismatch_mse"]) for row in prefix_rows),
        "mean_target_state_cosine": mean(float(row["mean_target_state_cosine"]) for row in state_rows),
        "mean_abs_gamma": mean(float(row["mean_abs_gamma"]) for row in state_rows),
        "mean_abs_beta": mean(float(row["mean_abs_beta"]) for row in state_rows),
        "compatibility": compatibility,
    }


def format_pct(value: float) -> str:
    return f"{value:+.2f}%"


def write_report(
    path: Path,
    summary: dict[str, object],
    main_rows: list[dict[str, object]],
    h720_prefix_rows: list[dict[str, object]],
) -> None:
    dataset_mean = summary["dataset_mean_relative_mse_pct"]
    horizon_mean = summary["horizon_mean_relative_mse_pct"]
    compatibility = summary["compatibility"]
    lines = [
        "# Phase1-R Target-Set Decoder Gate Report",
        "",
        "## Decision",
        "",
    ]
    if compatibility["pass"]:
        lines.append("[Decision] `PatchEncoderTargetSetDecoder` reaches compatibility pass.")
    else:
        lines.append("[Decision] `PatchEncoderTargetSetDecoder` does not reach compatibility pass.")
    lines += [
        "",
        "It should not be treated as paper-core unless follow-up mechanisms convert its target-side state into stable forecast gains.",
        "",
        "## Main Metrics",
        "",
        f"- MSE wins vs horizon-specific `PatchEncoderFixedHead`: `{summary['mse_wins']}/12`.",
        f"- MAE wins vs horizon-specific `PatchEncoderFixedHead`: `{summary['mae_wins']}/12`.",
        f"- Mean relative MSE: `{format_pct(float(summary['mean_relative_mse_pct']))}`.",
        f"- Relative MSE range: `{format_pct(float(summary['min_relative_mse_pct']))}` to `{format_pct(float(summary['max_relative_mse_pct']))}`.",
        "",
        "| Dataset | Mean relative MSE |",
        "| --- | ---: |",
    ]
    for dataset in DATASETS:
        lines.append(f"| {dataset} | {format_pct(float(dataset_mean[dataset]))} |")
    lines += [
        "",
        "| Horizon | Mean relative MSE |",
        "| --- | ---: |",
    ]
    for horizon in HORIZONS:
        lines.append(f"| {horizon} | {format_pct(float(horizon_mean[str(horizon)]))} |")
    lines += [
        "",
        "## Compatibility Gate",
        "",
        f"- mean relative MSE <= +1.0%: `{compatibility['mean_relative_mse_le_1pct']}`",
        f"- no dataset average degradation > +3.0%: `{compatibility['no_dataset_degrades_over_3pct']}`",
        f"- h96/h192 not worse than fixed H720-prefix: `{compatibility['h96_h192_not_worse_than_fixed_h720_prefix']}`",
        f"- prefix mismatch near zero: `{compatibility['prefix_mismatch_near_zero']}`",
        f"- target states non-identical: `{compatibility['target_states_non_identical']}`",
        f"- H720-prefix MSE wins vs fixed H720-prefix: `{summary['h720_prefix_mse_wins']}/12`",
        f"- H720-prefix h96/h192 mean relative MSE: `{format_pct(float(summary['h720_prefix_h96_h192_mean_relative_mse_pct']))}`",
        f"- max prefix mismatch MSE: `{summary['max_prefix_mismatch_mse']:.6g}`",
        f"- mean target state cosine: `{float(summary['mean_target_state_cosine']):.6f}`",
        f"- mean |gamma| / |beta|: `{float(summary['mean_abs_gamma']):.6f}` / `{float(summary['mean_abs_beta']):.6f}`",
        "",
        "## Interpretation",
        "",
        "[Inference] The first target-set implementation proves the prefix-stable interface works mechanically, because short-horizon predictions match the corresponding H=720 prefixes up to numerical noise.",
        "",
        "[Inference] On the same H=720-aligned windows used by the Phase0 prefix diagnostic, the target-set decoder improves h96/h192 prefixes over the fixed H720-prefix reference on average, but it does not satisfy the strict no-degradation condition in every setting. ETTh2 h96 and ETTm1 h96 are slightly worse than the fixed H720-prefix reference.",
        "",
        "[Inference] The accuracy side is the decisive issue. If the mean relative MSE is positive, the current dense target-conditioned readout is not yet a paper-core decoder; it can only remain as a carrier if the amortization gap is within the compatibility threshold.",
        "",
        "## Per-Setting Relative MSE",
        "",
        "| Dataset | Horizon | Relative MSE | Target MSE | Fixed MSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in main_rows:
        lines.append(
            "| {dataset} | {horizon} | {rel} | {target:.6f} | {fixed:.6f} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                rel=format_pct(float(row["relative_mse_pct"])),
                target=float(row["target_mse"]),
                fixed=float(row["fixed_mse"]),
            )
        )
    lines += [
        "",
        "## H720-Aligned Prefix Reference",
        "",
        "| Dataset | Prefix horizon | Relative MSE vs fixed H720-prefix | Target prefix MSE | Fixed H720-prefix MSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in h720_prefix_rows:
        lines.append(
            "| {dataset} | {horizon} | {rel} | {target:.6f} | {fixed:.6f} |".format(
                dataset=row["dataset"],
                horizon=row["prefix_horizon"],
                rel=format_pct(float(row["relative_mse_pct"])),
                target=float(row["target_h720_prefix_mse"]),
                fixed=float(row["fixed_h720_prefix_mse"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    analysis_root = Path("analysis/phase1_target_set_decoder_gate_20260622")
    raw_root = analysis_root / "raw"
    fixed_raw_root = Path("analysis/phase1_trajectory_basis_residual_gate_20260622/raw")
    main_rows = collect_main_metrics(raw_root, fixed_raw_root)
    segment_rows = collect_segment_metrics(raw_root, fixed_raw_root)
    prefix_rows = collect_prefix_metrics(raw_root)
    h720_prefix_rows = collect_h720_prefix_reference(raw_root)
    state_rows = collect_state_metrics(raw_root)
    summary = summarize(main_rows, segment_rows, prefix_rows, h720_prefix_rows, state_rows)

    write_csv(analysis_root / "phase1_target_set_decoder_vs_fixed.csv", main_rows)
    write_csv(analysis_root / "phase1_target_set_decoder_vs_fixed_segments.csv", segment_rows)
    write_csv(analysis_root / "phase1_target_set_decoder_prefix_consistency.csv", prefix_rows)
    write_csv(analysis_root / "phase1_target_set_decoder_h720_prefix_reference.csv", h720_prefix_rows)
    write_csv(analysis_root / "phase1_target_set_decoder_state_stats.csv", state_rows)
    (analysis_root / "phase1_target_set_decoder_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(analysis_root / "phase1_target_set_decoder_gate_report.md", summary, main_rows, h720_prefix_rows)


if __name__ == "__main__":
    main()
