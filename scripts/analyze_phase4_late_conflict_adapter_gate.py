from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


MODELS = [
    "PatchEncoderLateConflictAdapterRouting",
    "PatchEncoderFullTimeMSE720",
    "PatchEncoderR3PrefixRisk",
]

LABELS = {
    "PatchEncoderLateConflictAdapterRouting": "RG-A_late_conflict_adapter",
    "PatchEncoderFullTimeMSE720": "D0_full_time_mse",
    "PatchEncoderR3PrefixRisk": "D1_r3_prefix_risk",
}

PRIMARY = "PatchEncoderLateConflictAdapterRouting"
BASELINES = ["PatchEncoderFullTimeMSE720", "PatchEncoderR3PrefixRisk"]
DATASETS = ["ETTh2", "Weather"]
HORIZONS = [96, 192, 336, 720]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def run_dir(root: Path, model: str, dataset: str) -> Path:
    return root / model / dataset / "mixed_h96_h192_h336_h720" / "seed2021"


def parse_segment(segment: str) -> tuple[int, int]:
    start, end = segment.split("-", maxsplit=1)
    return int(start), int(end)


def future_region(segment_end: int) -> str:
    if segment_end <= 96:
        return "early_1_96"
    if segment_end <= 336:
        return "middle_97_336"
    return "late_337_720"


def load_main_metrics(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            for row in read_csv(run_dir(raw_root, model, dataset) / "metrics_by_target_horizon.csv"):
                rows.append(
                    {
                        "model": model,
                        "strategy": LABELS[model],
                        "dataset": dataset,
                        "horizon": int(row["target_horizon"]),
                        "mse": float(row["mse"]),
                        "mae": float(row["mae"]),
                    }
                )
    return rows


def add_main_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["model"], row["dataset"], row["horizon"]): row for row in rows}
    output = []
    for baseline in BASELINES:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                candidate = by_key[(PRIMARY, dataset, horizon)]
                base = by_key[(baseline, dataset, horizon)]
                output.append(
                    {
                        **candidate,
                        "baseline_model": baseline,
                        "baseline_strategy": LABELS[baseline],
                        "baseline_mse": base["mse"],
                        "baseline_mae": base["mae"],
                        "relative_mse_pct": pct(candidate["mse"], base["mse"]),
                        "relative_mae_pct": pct(candidate["mae"], base["mae"]),
                        "mse_win": candidate["mse"] < base["mse"],
                        "mae_win": candidate["mae"] < base["mae"],
                    }
                )
    return output


def summarize(rows: list[dict[str, Any]], group_key: str | None = None) -> list[dict[str, Any]]:
    output = []
    for baseline in BASELINES:
        baseline_rows = [row for row in rows if row["baseline_model"] == baseline]
        values = ["all"] if group_key is None else sorted({row[group_key] for row in baseline_rows})
        for value in values:
            subset = baseline_rows if group_key is None else [row for row in baseline_rows if row[group_key] == value]
            item = {
                "strategy": LABELS[PRIMARY],
                "baseline_strategy": LABELS[baseline],
                "settings": len(subset),
                "mse_wins": sum(1 for row in subset if row["mse_win"]),
                "mae_wins": sum(1 for row in subset if row["mae_win"]),
                "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in subset),
                "mean_relative_mae_pct": mean(row["relative_mae_pct"] for row in subset),
            }
            if group_key is not None:
                item[group_key] = value
            output.append(item)
    return output


def load_segment_metrics(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                path = run_dir(raw_root, model, dataset) / f"h{horizon}" / "metrics_by_segment.csv"
                for row in read_csv(path):
                    start, end = parse_segment(row["segment"])
                    rows.append(
                        {
                            "model": model,
                            "strategy": LABELS[model],
                            "dataset": dataset,
                            "horizon": horizon,
                            "segment": row["segment"],
                            "segment_start": start,
                            "segment_end": end,
                            "future_region": future_region(end),
                            "mse": float(row["mse"]),
                            "mae": float(row["mae"]),
                        }
                    )
    return rows


def add_segment_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["model"], row["dataset"], row["horizon"], row["segment"]): row for row in rows}
    output = []
    for baseline in BASELINES:
        for row in rows:
            if row["model"] != PRIMARY:
                continue
            base = by_key[(baseline, row["dataset"], row["horizon"], row["segment"])]
            output.append(
                {
                    **row,
                    "baseline_model": baseline,
                    "baseline_strategy": LABELS[baseline],
                    "baseline_mse": base["mse"],
                    "baseline_mae": base["mae"],
                    "relative_mse_pct": pct(row["mse"], base["mse"]),
                    "relative_mae_pct": pct(row["mae"], base["mae"]),
                    "mse_win": row["mse"] < base["mse"],
                    "mae_win": row["mae"] < base["mae"],
                }
            )
    return output


def summarize_segments(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    output = []
    for baseline in BASELINES:
        baseline_rows = [row for row in rows if row["baseline_model"] == baseline]
        for value in sorted({row[group_key] for row in baseline_rows}):
            subset = [row for row in baseline_rows if row[group_key] == value]
            output.append(
                {
                    "strategy": LABELS[PRIMARY],
                    "baseline_strategy": LABELS[baseline],
                    group_key: value,
                    "segments": len(subset),
                    "mse_wins": sum(1 for row in subset if row["mse_win"]),
                    "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in subset),
                    "mean_relative_mae_pct": mean(row["relative_mae_pct"] for row in subset),
                }
            )
    return output


def load_trace_summary(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            base_dir = run_dir(raw_root, model, dataset)
            config = json.loads((base_dir / "effective_config.json").read_text())
            trace = read_csv(base_dir / "supervision_trace.csv")
            log = read_csv(base_dir / "training_log.csv")
            by_unit: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in trace:
                by_unit[row["unit_type"]].append(row)
            for unit_type, unit_rows in sorted(by_unit.items()):
                rows.append(
                    {
                        "model": model,
                        "strategy": LABELS[model],
                        "dataset": dataset,
                        "training_evaluation_decoupled": config.get("training_evaluation_decoupled", False),
                        "train_horizons_effective": ",".join(str(x) for x in config.get("train_horizons_effective", [])),
                        "step_loss_weighting": config.get("step_loss_weighting", ""),
                        "unit_type": unit_type,
                        "epochs_ran": len(log),
                        "trace_rows": len(unit_rows),
                        "mean_active_steps": mean(float(row["active_steps"]) for row in unit_rows),
                        "mean_auxiliary_weight": mean(float(row.get("auxiliary_weight", 0) or 0) for row in unit_rows),
                        "mean_adapter_start_step": mean(float(row.get("adapter_start_step", 0) or 0) for row in unit_rows),
                        "mean_adapter_active_steps": mean(float(row.get("adapter_active_steps", 0) or 0) for row in unit_rows),
                        "mean_adapter_abs_residual": mean(float(row.get("adapter_mean_abs_residual", 0) or 0) for row in unit_rows),
                        "last_adapter_abs_residual": float(unit_rows[-1].get("adapter_mean_abs_residual", 0) or 0),
                        "mean_time_loss": mean(float(row["loss_time"]) for row in unit_rows),
                        "mean_unit_loss": mean(float(row["loss_unit"]) for row in unit_rows),
                        "mean_total_loss": mean(float(row["loss_total"]) for row in unit_rows),
                    }
                )
    return rows


def load_prefix_summary(raw_root: Path) -> list[dict[str, Any]]:
    output = []
    for model in MODELS:
        for dataset in DATASETS:
            rows = read_csv(run_dir(raw_root, model, dataset) / "prefix_consistency.csv")
            output.append(
                {
                    "model": model,
                    "strategy": LABELS[model],
                    "dataset": dataset,
                    "rows": len(rows),
                    "max_prefix_mismatch_mse": max(float(row["prefix_mismatch_mse"]) for row in rows),
                    "max_prefix_mismatch_mae": max(float(row["prefix_mismatch_mae"]) for row in rows),
                }
            )
    return output


def write_report(
    path: Path,
    main_delta: list[dict[str, Any]],
    overall: list[dict[str, Any]],
    dataset_summary: list[dict[str, Any]],
    horizon_summary: list[dict[str, Any]],
    segment_delta: list[dict[str, Any]],
    segment_region_summary: list[dict[str, Any]],
    trace_summary: list[dict[str, Any]],
    prefix_summary: list[dict[str, Any]],
) -> None:
    r3_weather = [
        row for row in main_delta
        if row["baseline_model"] == "PatchEncoderR3PrefixRisk" and row["dataset"] == "Weather"
    ]
    r3_etth2 = [
        row for row in main_delta
        if row["baseline_model"] == "PatchEncoderR3PrefixRisk" and row["dataset"] == "ETTh2"
    ]
    full_rows = [row for row in overall if row["baseline_strategy"] == "D0_full_time_mse"][0]
    r3_rows = [row for row in overall if row["baseline_strategy"] == "D1_r3_prefix_risk"][0]
    weather_late = [
        row for row in segment_delta
        if row["dataset"] == "Weather"
        and row["baseline_model"] == "PatchEncoderR3PrefixRisk"
        and row["horizon"] == 720
        and row["segment"] == "337-720"
    ][0]
    etth2_late = [
        row for row in segment_delta
        if row["dataset"] == "ETTh2"
        and row["baseline_model"] == "PatchEncoderR3PrefixRisk"
        and row["horizon"] == 720
        and row["segment"] == "337-720"
    ][0]
    adapter_trace = [row for row in trace_summary if row["model"] == PRIMARY]
    max_prefix = max(row["max_prefix_mismatch_mse"] for row in prefix_summary)

    lines = [
        "# Phase4 RG-A Late-Conflict Adapter Gate Report",
        "",
        "## 11-Step Record",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | Step 9/10: evaluate returned artifacts and decide whether method passes |",
        "| `problem` | Weather late/early gradient conflict suggests loss-only HSS may contaminate shared representation |",
        "| `existence_evidence` | SRP code audit + gradient conflict diagnostic + S1/S2 failures |",
        "| `idea` | Route late conflict auxiliary pressure into a zero-init adapter residual while dense base keeps shared path |",
        "| `theory_check` | If conflict is mainly late/readout, adapter late residual should improve Weather late segment without damaging ETTh2 |",
        "| `design` | `late_conflict_adapter_routing` vs `full_time_mse` and R.3 on ETTh2/Weather |",
        "| `gate` | beat full-time, avoid Weather 0/4 vs R.3, improve Weather late segment, retain ETTh2 signal, prefix zero |",
        f"| `artifacts` | `{path.parent}` |",
        "| `decision` | Fail as paper-core candidate; partial mechanism signal vs full-time, but R.3 gap and Weather late failure remain |",
        "",
        "## Main Result",
        "",
        f"[Fact] RG-A vs `full_time_mse`: MSE wins `{full_rows['mse_wins']}/{full_rows['settings']}`, mean relative MSE `{fmt_pct(full_rows['mean_relative_mse_pct'])}`。",
        f"[Fact] RG-A vs R.3: MSE wins `{r3_rows['mse_wins']}/{r3_rows['settings']}`, mean relative MSE `{fmt_pct(r3_rows['mean_relative_mse_pct'])}`。",
        f"[Fact] Weather vs R.3 remains `{sum(1 for row in r3_weather if row['mse_win'])}/4` wins, mean relative MSE `{fmt_pct(mean(row['relative_mse_pct'] for row in r3_weather))}`。",
        f"[Fact] ETTh2 vs R.3 drops to `{sum(1 for row in r3_etth2 if row['mse_win'])}/4` wins, mean relative MSE `{fmt_pct(mean(row['relative_mse_pct'] for row in r3_etth2))}`。",
        "",
        "## Overall Summary",
        "",
        "| Baseline | Settings | MSE wins | MAE wins | Mean relative MSE | Mean relative MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall:
        lines.append(
            f"| `{row['baseline_strategy']}` | {row['settings']} | {row['mse_wins']} | {row['mae_wins']} | "
            f"{fmt_pct(row['mean_relative_mse_pct'])} | {fmt_pct(row['mean_relative_mae_pct'])} |"
        )
    lines += [
        "",
        "## Dataset Summary",
        "",
        "| Dataset | Baseline | Settings | MSE wins | Mean relative MSE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in dataset_summary:
        lines.append(
            f"| `{row['dataset']}` | `{row['baseline_strategy']}` | {row['settings']} | {row['mse_wins']} | "
            f"{fmt_pct(row['mean_relative_mse_pct'])} |"
        )
    lines += [
        "",
        "## Per-Horizon Metrics",
        "",
        "| Dataset | Horizon | RG-A MSE | Full-time MSE | R.3 MSE | RG-A vs full-time | RG-A vs R.3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_base = {(row["dataset"], row["horizon"], row["baseline_model"]): row for row in main_delta}
    for dataset in DATASETS:
        for horizon in HORIZONS:
            full = by_base[(dataset, horizon, "PatchEncoderFullTimeMSE720")]
            r3 = by_base[(dataset, horizon, "PatchEncoderR3PrefixRisk")]
            lines.append(
                f"| `{dataset}` | {horizon} | {full['mse']:.6f} | {full['baseline_mse']:.6f} | "
                f"{r3['baseline_mse']:.6f} | {fmt_pct(full['relative_mse_pct'])} | {fmt_pct(r3['relative_mse_pct'])} |"
            )
    lines += [
        "",
        "## Segment Gate",
        "",
        f"[Fact] Weather h720 late segment vs R.3 is `{fmt_pct(weather_late['relative_mse_pct'])}`; gate requires improvement, so this fails.",
        f"[Fact] ETTh2 h720 late segment vs R.3 is `{fmt_pct(etth2_late['relative_mse_pct'])}`; adapter helps the intended late segment only on ETTh2.",
        "",
        "| Future region | Baseline | Segments | MSE wins | Mean relative MSE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in segment_region_summary:
        lines.append(
            f"| `{row['future_region']}` | `{row['baseline_strategy']}` | {row['segments']} | {row['mse_wins']} | "
            f"{fmt_pct(row['mean_relative_mse_pct'])} |"
        )
    lines += [
        "",
        "## Trace And Prefix",
        "",
        "| Dataset | Unit type | Epochs | Mean adapter active steps | Mean abs adapter residual | Last abs adapter residual | Mean time loss | Mean adapter/unit loss |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in adapter_trace:
        lines.append(
            f"| `{row['dataset']}` | `{row['unit_type']}` | {row['epochs_ran']} | "
            f"{row['mean_adapter_active_steps']:.0f} | {row['mean_adapter_abs_residual']:.6f} | "
            f"{row['last_adapter_abs_residual']:.6f} | {row['mean_time_loss']:.6f} | {row['mean_unit_loss']:.6f} |"
        )
    lines += [
        "",
        f"[Fact] max prefix mismatch MSE across all runs is `{max_prefix:.3e}`, so prefix consistency remains numerical-zero.",
        "",
        "## Interpretation",
        "",
        "[Strong Evidence] The method validates one narrow mechanism: routing late auxiliary pressure into a zero-init adapter can still beat `full_time_mse` on average, and it improves ETTh2 h720 late segment vs R.3.",
        "",
        "[Counter-Evidence] It fails the paper-core gate. Weather remains `0/4` vs R.3, and Weather h720 late segment is worse than R.3 by `+4.74%`. This directly falsifies the hypothesis that a fixed late adapter route is enough to repair the Weather conflict found by gradient diagnostics.",
        "",
        "[Inference] The failure is likely not only about gradient destination. The fixed late adapter sees late residual pressure but has no state/difficulty condition and starts from a weak base that is already worse on Weather early/middle regions. It cannot decide when late signal is learnable vs noisy, so it behaves like a small late residual correction rather than a real supervision scheduler.",
        "",
        "## Decision",
        "",
        "[Decision] Do not enter full matrix. Do not sweep `aux_weight` or `adapter_start_step` yet. RG-A becomes a negative/partial evidence point: gradient routing is a viable axis, but fixed late routing is not enough.",
        "",
        "[Rollback] Return to Step 5/6. Next candidate must use a dynamic conflict/predictability router, likely residual-stability-conditioned, and should route only units whose train-side evidence indicates learnable conflict rather than all late steps.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase4 late-conflict adapter gate.")
    parser.add_argument("--raw-root", default="analysis/phase4_late_conflict_adapter_gate_20260625/raw")
    parser.add_argument("--output-root", default="analysis/phase4_late_conflict_adapter_gate_20260625")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    main_rows = load_main_metrics(raw_root)
    main_delta = add_main_deltas(main_rows)
    segment_rows = load_segment_metrics(raw_root)
    segment_delta = add_segment_deltas(segment_rows)
    overall = summarize(main_delta)
    dataset_summary = summarize(main_delta, "dataset")
    horizon_summary = summarize(main_delta, "horizon")
    segment_region_summary = summarize_segments(segment_delta, "future_region")
    trace_summary = load_trace_summary(raw_root)
    prefix_summary = load_prefix_summary(raw_root)

    write_csv(output_root / "phase4_late_conflict_main_metrics.csv", main_rows)
    write_csv(output_root / "phase4_late_conflict_main_delta.csv", main_delta)
    write_csv(output_root / "phase4_late_conflict_overall_summary.csv", overall)
    write_csv(output_root / "phase4_late_conflict_dataset_summary.csv", dataset_summary)
    write_csv(output_root / "phase4_late_conflict_horizon_summary.csv", horizon_summary)
    write_csv(output_root / "phase4_late_conflict_segment_delta.csv", segment_delta)
    write_csv(output_root / "phase4_late_conflict_segment_region_summary.csv", segment_region_summary)
    write_csv(output_root / "phase4_late_conflict_trace_summary.csv", trace_summary)
    write_csv(output_root / "phase4_late_conflict_prefix_summary.csv", prefix_summary)
    write_report(
        output_root / "phase4_late_conflict_adapter_gate_report.md",
        main_delta,
        overall,
        dataset_summary,
        horizon_summary,
        segment_delta,
        segment_region_summary,
        trace_summary,
        prefix_summary,
    )
    print(output_root / "phase4_late_conflict_adapter_gate_report.md")


if __name__ == "__main__":
    main()
