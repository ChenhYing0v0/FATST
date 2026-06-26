from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


MODELS = {
    "PatchEncoderSingle720PrefixRisk": "single_720_prefix_risk",
    "PatchEncoderR3PrefixRisk": "r3_prefix_risk",
    "PatchEncoderDynamicResidualStabilityRouting": "dynamic_residual_stability_routing",
    "PatchEncoderSCCConditionDeltaDetached": "scc_condition_delta_detached",
    "PatchEncoderSCCConditionDeltaStateOpen": "scc_condition_delta_state_open",
}
CANDIDATES = ["scc_condition_delta_detached", "scc_condition_delta_state_open"]
BASELINES = ["single_720_prefix_risk", "r3_prefix_risk", "dynamic_residual_stability_routing"]
DATASETS = ["ETTh2", "Weather"]
HORIZONS = [96, 192, 336, 720]
HORIZON_LABEL = "mixed_h96_h192_h336_h720"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def fmt_float(value: float) -> str:
    return f"{value:.6f}"


def run_dir(raw_root: Path, model: str, dataset: str) -> Path:
    return raw_root / model / dataset / HORIZON_LABEL / "seed2021"


def load_main_metrics(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model, strategy in MODELS.items():
        for dataset in DATASETS:
            path = run_dir(raw_root, model, dataset) / "metrics_by_target_horizon.csv"
            if not path.exists():
                continue
            for row in read_csv(path):
                rows.append(
                    {
                        "model": model,
                        "strategy": strategy,
                        "dataset": dataset,
                        "horizon": int(row["target_horizon"]),
                        "mse": float(row["mse"]),
                        "mae": float(row["mae"]),
                    }
                )
    return rows


def load_training_summary(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model, strategy in MODELS.items():
        for dataset in DATASETS:
            path = run_dir(raw_root, model, dataset) / "training_log.csv"
            if not path.exists():
                continue
            log_rows = read_csv(path)
            val_values = [float(row["val_mean_mse"]) for row in log_rows]
            best_index = min(range(len(val_values)), key=val_values.__getitem__)
            rows.append(
                {
                    "strategy": strategy,
                    "dataset": dataset,
                    "epochs_ran": len(log_rows),
                    "best_epoch": best_index + 1,
                    "best_val_mean_mse": val_values[best_index],
                    "last_val_mean_mse": val_values[-1],
                    "post_best_val_drift_pct": pct(val_values[-1], val_values[best_index]),
                    "last_condition_delta_grad_norm": float(
                        log_rows[-1].get("train_condition_delta_grad_norm") or 0.0
                    ),
                }
            )
    return rows


def load_trace_summary(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model, strategy in MODELS.items():
        for dataset in DATASETS:
            path = run_dir(raw_root, model, dataset) / "supervision_trace.csv"
            if not path.exists():
                continue
            trace = read_csv(path)
            if not trace:
                continue
            rows.append(
                {
                    "strategy": strategy,
                    "dataset": dataset,
                    "trace_rows": len(trace),
                    "mean_learnable_blocks": mean(
                        float(row.get("residual_stability_learnable_blocks") or 0.0) for row in trace
                    ),
                    "mean_noisy_blocks": mean(
                        float(row.get("residual_stability_noisy_blocks") or 0.0) for row in trace
                    ),
                    "mean_noisy_suppression_ratio": mean(
                        float(row.get("residual_stability_noisy_suppression_ratio") or 0.0) for row in trace
                    ),
                    "mean_condition_delta_abs_residual": mean(
                        float(row.get("condition_delta_mean_abs_residual") or 0.0) for row in trace
                    ),
                    "last_condition_delta_abs_residual": float(
                        trace[-1].get("condition_delta_mean_abs_residual") or 0.0
                    ),
                }
            )
    return rows


def load_checkpoint_diagnostics(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model, strategy in MODELS.items():
        for dataset in DATASETS:
            path = run_dir(raw_root, model, dataset) / "checkpoint_selection_diagnostics.csv"
            if not path.exists():
                continue
            for row in read_csv(path):
                rows.append(
                    {
                        "strategy": strategy,
                        "dataset": dataset,
                        "selector": row["selector"],
                        "horizons": row["horizons"],
                        "best_epoch": int(row["best_epoch"]),
                        "best_selector_mse": float(row["best_selector_mse"]),
                        "official_best_epoch": int(row["official_best_epoch"]),
                        "official_gap_to_selector_best_pct": float(row["official_gap_to_selector_best_pct"]),
                    }
                )
    return rows


def build_delta_rows(main_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["strategy"], row["dataset"], row["horizon"]): row
        for row in main_rows
    }
    rows = []
    for candidate_name in CANDIDATES:
        for baseline_name in BASELINES:
            for dataset in DATASETS:
                for horizon in HORIZONS:
                    key = (candidate_name, dataset, horizon)
                    base_key = (baseline_name, dataset, horizon)
                    if key not in by_key or base_key not in by_key:
                        continue
                    candidate = by_key[key]
                    base = by_key[base_key]
                    rows.append(
                        {
                            "candidate_strategy": candidate_name,
                            "baseline_strategy": baseline_name,
                            "dataset": dataset,
                            "horizon": horizon,
                            "candidate_mse": candidate["mse"],
                            "baseline_mse": base["mse"],
                            "relative_mse_pct": pct(candidate["mse"], base["mse"]),
                            "mse_win": candidate["mse"] < base["mse"],
                            "candidate_mae": candidate["mae"],
                            "baseline_mae": base["mae"],
                            "relative_mae_pct": pct(candidate["mae"], base["mae"]),
                            "mae_win": candidate["mae"] < base["mae"],
                        }
                    )
    return rows


def summarize_delta(delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in delta_rows:
        grouped[(row["candidate_strategy"], row["baseline_strategy"], row["dataset"])].append(row)
    output = []
    for (candidate, baseline, dataset), subset in sorted(grouped.items()):
        output.append(
            {
                "candidate_strategy": candidate,
                "baseline_strategy": baseline,
                "dataset": dataset,
                "settings": len(subset),
                "mse_wins": sum(1 for row in subset if row["mse_win"]),
                "mae_wins": sum(1 for row in subset if row["mae_win"]),
                "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in subset),
                "mean_relative_mae_pct": mean(row["relative_mae_pct"] for row in subset),
            }
        )
    return output


def rows_for_table(rows: list[dict[str, Any]], keys: list[str]) -> str:
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        values = []
        for key in keys:
            value = row[key]
            if isinstance(value, float):
                value = fmt_pct(value) if key.endswith("_pct") else fmt_float(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    output_path: Path,
    delta_summary: list[dict[str, Any]],
    training_summary: list[dict[str, Any]],
    trace_summary: list[dict[str, Any]],
    checkpoint_diagnostics: list[dict[str, Any]],
) -> None:
    candidate_vs_r3 = [
        row for row in delta_summary if row["baseline_strategy"] == "r3_prefix_risk"
    ]
    lines = [
        "# Phase4 SCC Condition Carrier Gate Report",
        "",
        "## Decision",
        "",
        "[Strong Evidence] SCC condition carrier is active and improves over the adapter carrier on Weather,",
        "but it does not pass the core gate against R.3. The state-open variant gets one Weather h720 win,",
        "yet remains worse than R.3 on mean MSE and most horizons.",
        "",
        "[Decision] SCC-E1 fails as a core-route candidate. Do not continue local sweeps over aux weight,",
        "top ratio, or condition-delta size. The next step should roll back to Step 2/3 and reassess whether",
        "Phase4 should pivot from local supervision scheduling to future-aware representation or pretraining.",
        "",
        "## Candidate vs R.3",
        "",
        rows_for_table(
            candidate_vs_r3,
            [
                "candidate_strategy",
                "dataset",
                "settings",
                "mse_wins",
                "mean_relative_mse_pct",
                "mae_wins",
                "mean_relative_mae_pct",
            ],
        ),
        "",
        "## Training Summary",
        "",
        rows_for_table(
            training_summary,
            [
                "strategy",
                "dataset",
                "epochs_ran",
                "best_epoch",
                "post_best_val_drift_pct",
                "last_condition_delta_grad_norm",
            ],
        ),
        "",
        "## Trace Summary",
        "",
        rows_for_table(
            trace_summary,
            [
                "strategy",
                "dataset",
                "trace_rows",
                "mean_learnable_blocks",
                "mean_noisy_blocks",
                "mean_noisy_suppression_ratio",
                "mean_condition_delta_abs_residual",
                "last_condition_delta_abs_residual",
            ],
        ),
        "",
        "## Checkpoint Selection Diagnostics",
        "",
        rows_for_table(
            [row for row in checkpoint_diagnostics if row["selector"] in {"long_mean", "h720"}],
            [
                "strategy",
                "dataset",
                "selector",
                "best_epoch",
                "official_best_epoch",
                "official_gap_to_selector_best_pct",
            ],
        ),
        "",
        "## Gate",
        "",
        "Pass only if a SCC candidate is within `+0.5%` mean MSE vs R.3, wins at least `2/4` Weather horizons,",
        "keeps ETTh2 h96/h192 within `+1.0%`, has drift below about `8%`, and shows non-collapsed carrier trace.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("artifacts/runs/phase4_scc_condition_carrier_gate"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/phase4_scc_condition_carrier_gate_20260626"),
    )
    args = parser.parse_args()

    main_rows = load_main_metrics(args.raw_root)
    delta_rows = build_delta_rows(main_rows)
    delta_summary = summarize_delta(delta_rows)
    training_summary = load_training_summary(args.raw_root)
    trace_summary = load_trace_summary(args.raw_root)
    checkpoint_diagnostics = load_checkpoint_diagnostics(args.raw_root)

    write_csv(args.output_dir / "phase4_scc_main_metrics.csv", main_rows)
    write_csv(args.output_dir / "phase4_scc_delta.csv", delta_rows)
    write_csv(args.output_dir / "phase4_scc_delta_summary.csv", delta_summary)
    write_csv(args.output_dir / "phase4_scc_training_summary.csv", training_summary)
    write_csv(args.output_dir / "phase4_scc_trace_summary.csv", trace_summary)
    write_csv(args.output_dir / "phase4_scc_checkpoint_diagnostics.csv", checkpoint_diagnostics)
    write_report(
        args.output_dir / "phase4_scc_condition_carrier_gate_report.md",
        delta_summary,
        training_summary,
        trace_summary,
        checkpoint_diagnostics,
    )


if __name__ == "__main__":
    main()
