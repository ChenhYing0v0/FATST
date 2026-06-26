from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = ["ETTh2", "Weather"]
HORIZONS = [96, 192, 336, 720]
HORIZON_LABEL = "mixed_h96_h192_h336_h720"
SEED = 2021

F1_MODELS = {
    "F1-C0": "PatchEncoderFSAF1SinglePrefixBase",
    "F1-C1": "PatchEncoderFSAF1R3Base",
    "F1-A0": "PatchEncoderFSAF1SinglePrefixFutureAnchor",
    "F1-A1": "PatchEncoderFSAF1R3FutureAnchor",
}
F2_MODELS = {
    "F2-A0": "PatchEncoderFSAF2SinglePrefixSelectiveAnchor",
    "F2-A1": "PatchEncoderFSAF2R3SelectiveAnchor",
}
LABELS = {
    "F1-C0": "F1-C0_single_prefix_base",
    "F1-C1": "F1-C1_r3_base",
    "F1-A0": "F1-A0_single_prefix_floor005",
    "F1-A1": "F1-A1_r3_floor005",
    "F2-A0": "F2-A0_single_prefix_floor000",
    "F2-A1": "F2-A1_r3_floor000",
}
COMPARISONS = [
    ("F2-A0", "F1-C0", "selective_anchor_vs_single_prefix_base"),
    ("F2-A0", "F1-A0", "selective_anchor_vs_floor005_single"),
    ("F2-A1", "F1-C1", "selective_anchor_vs_r3_base"),
    ("F2-A1", "F1-A1", "selective_anchor_vs_floor005_r3"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = []
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


def model_for_arm(arm: str) -> str:
    return F2_MODELS.get(arm, F1_MODELS[arm])


def root_for_arm(f2_root: Path, f1_root: Path, arm: str) -> Path:
    return f2_root if arm.startswith("F2-") else f1_root


def run_dir(f2_root: Path, f1_root: Path, arm: str, dataset: str) -> Path:
    return root_for_arm(f2_root, f1_root, arm) / model_for_arm(arm) / dataset / HORIZON_LABEL / f"seed{SEED}"


def load_main_metrics(f2_root: Path, f1_root: Path) -> list[dict[str, Any]]:
    rows = []
    for arm in {candidate for candidate, _, _ in COMPARISONS} | {baseline for _, baseline, _ in COMPARISONS}:
        for dataset in DATASETS:
            path = run_dir(f2_root, f1_root, arm, dataset) / "metrics_by_target_horizon.csv"
            if not path.exists():
                continue
            for row in read_csv(path):
                rows.append(
                    {
                        "arm": arm,
                        "label": LABELS[arm],
                        "dataset": dataset,
                        "horizon": int(row["target_horizon"]),
                        "mse": float(row["mse"]),
                        "mae": float(row["mae"]),
                    }
                )
    return rows


def load_h720_segments(f2_root: Path, f1_root: Path) -> list[dict[str, Any]]:
    rows = []
    for arm in {candidate for candidate, _, _ in COMPARISONS} | {baseline for _, baseline, _ in COMPARISONS}:
        for dataset in DATASETS:
            path = run_dir(f2_root, f1_root, arm, dataset) / "h720" / "metrics_by_segment.csv"
            if not path.exists():
                continue
            for row in read_csv(path):
                rows.append(
                    {
                        "arm": arm,
                        "label": LABELS[arm],
                        "dataset": dataset,
                        "horizon": 720,
                        "segment": row["segment"],
                        "mse": float(row["mse"]),
                        "mae": float(row["mae"]),
                    }
                )
    return rows


def build_deltas(rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    by_key = {(row["arm"], *[row[field] for field in key_fields]): row for row in rows}
    output = []
    for candidate_arm, baseline_arm, comparison in COMPARISONS:
        candidate_keys = [
            key[1:]
            for key in by_key
            if key[0] == candidate_arm
        ]
        for fields in sorted(candidate_keys):
            candidate_key = (candidate_arm, *fields)
            baseline_key = (baseline_arm, *fields)
            if baseline_key not in by_key:
                continue
            candidate = by_key[candidate_key]
            baseline = by_key[baseline_key]
            item = {
                "comparison": comparison,
                "candidate_arm": candidate_arm,
                "candidate_label": LABELS[candidate_arm],
                "baseline_arm": baseline_arm,
                "baseline_label": LABELS[baseline_arm],
                "candidate_mse": candidate["mse"],
                "baseline_mse": baseline["mse"],
                "relative_mse_pct": pct(candidate["mse"], baseline["mse"]),
                "mse_win": candidate["mse"] < baseline["mse"],
                "candidate_mae": candidate["mae"],
                "baseline_mae": baseline["mae"],
                "relative_mae_pct": pct(candidate["mae"], baseline["mae"]),
                "mae_win": candidate["mae"] < baseline["mae"],
            }
            for field, value in zip(key_fields, fields):
                item[field] = value
            output.append(item)
    return output


def summarize(rows: list[dict[str, Any]], group_fields: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["comparison"], row["candidate_arm"], row["baseline_arm"], *[row[field] for field in group_fields])
        grouped[key].append(row)
    output = []
    for key, subset in sorted(grouped.items()):
        comparison, candidate_arm, baseline_arm, *group_values = key
        item = {
            "comparison": comparison,
            "candidate_arm": candidate_arm,
            "candidate_label": LABELS[candidate_arm],
            "baseline_arm": baseline_arm,
            "baseline_label": LABELS[baseline_arm],
            "settings": len(subset),
            "mse_wins": sum(1 for row in subset if row["mse_win"]),
            "mae_wins": sum(1 for row in subset if row["mae_win"]),
            "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in subset),
            "mean_relative_mae_pct": mean(row["relative_mae_pct"] for row in subset),
        }
        for field, value in zip(group_fields, group_values):
            item[field] = value
        output.append(item)
    return output


def load_training_summary(f2_root: Path, f1_root: Path) -> list[dict[str, Any]]:
    rows = []
    for arm in F2_MODELS:
        for dataset in DATASETS:
            path = run_dir(f2_root, f1_root, arm, dataset) / "training_log.csv"
            if not path.exists():
                continue
            logs = read_csv(path)
            val_values = [float(row["val_mean_mse"]) for row in logs]
            train_values = [float(row["train_prediction_loss"]) for row in logs]
            best_index = min(range(len(val_values)), key=val_values.__getitem__)
            rows.append(
                {
                    "arm": arm,
                    "label": LABELS[arm],
                    "dataset": dataset,
                    "epochs_ran": len(logs),
                    "best_epoch": best_index + 1,
                    "best_val_mean_mse": val_values[best_index],
                    "post_best_val_drift_pct": pct(val_values[-1], val_values[best_index]),
                    "train_loss_drop_pct": pct(train_values[-1], train_values[0]),
                    "first_train_future_alignment_confidence_mean": float(
                        logs[0].get("train_future_alignment_confidence_mean") or 0.0
                    ),
                    "last_train_future_alignment_confidence_mean": float(
                        logs[-1].get("train_future_alignment_confidence_mean") or 0.0
                    ),
                }
            )
    return rows


def load_future_alignment_summary(f2_root: Path, f1_root: Path) -> list[dict[str, Any]]:
    rows = []
    for arm in F2_MODELS:
        for dataset in DATASETS:
            values = []
            for horizon in HORIZONS:
                path = run_dir(f2_root, f1_root, arm, dataset) / f"h{horizon}" / "future_alignment_stats.csv"
                if path.exists():
                    values.append(read_csv(path)[0])
            if not values:
                continue
            rows.append(
                {
                    "arm": arm,
                    "label": LABELS[arm],
                    "dataset": dataset,
                    "horizons": len(values),
                    "mean_teacher_student_cosine": mean(float(row["teacher_student_cosine"]) for row in values),
                    "mean_future_reconstruction_loss": mean(float(row["future_reconstruction_loss"]) for row in values),
                    "mean_future_alignment_confidence": mean(
                        float(row["future_alignment_confidence_mean"]) for row in values
                    ),
                    "min_future_alignment_confidence": min(
                        float(row["future_alignment_confidence_min"]) for row in values
                    ),
                    "max_prediction_leakage_abs": max(
                        float(row["prediction_leakage_max_abs"]) for row in values
                    ),
                }
            )
    return rows


def load_checkpoint_diagnostics(f2_root: Path, f1_root: Path) -> list[dict[str, Any]]:
    rows = []
    for arm in F2_MODELS:
        for dataset in DATASETS:
            path = run_dir(f2_root, f1_root, arm, dataset) / "checkpoint_selection_diagnostics.csv"
            if not path.exists():
                continue
            for row in read_csv(path):
                rows.append(
                    {
                        "arm": arm,
                        "label": LABELS[arm],
                        "dataset": dataset,
                        "selector": row["selector"],
                        "best_epoch": int(row["best_epoch"]),
                        "official_best_epoch": int(row["official_best_epoch"]),
                        "official_gap_to_selector_best_pct": float(row["official_gap_to_selector_best_pct"]),
                    }
                )
    return rows


def load_config_summary(f2_root: Path, f1_root: Path) -> list[dict[str, Any]]:
    rows = []
    for arm in F2_MODELS:
        for dataset in DATASETS:
            path = run_dir(f2_root, f1_root, arm, dataset) / "effective_config.json"
            if not path.exists():
                continue
            import json

            config = json.loads(path.read_text())
            rows.append(
                {
                    "arm": arm,
                    "label": LABELS[arm],
                    "dataset": dataset,
                    "supervision_strategy": config.get("supervision_strategy"),
                    "future_align_weight": config.get("future_align_weight"),
                    "future_recon_weight": config.get("future_recon_weight"),
                    "future_relation_weight": config.get("future_relation_weight"),
                    "future_confidence_floor": config.get("future_confidence_floor"),
                    "future_align_weighting": config.get("future_align_weighting"),
                }
            )
    return rows


def table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            if isinstance(value, float):
                value = fmt_pct(value) if field.endswith("_pct") else f"{value:.6f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def pick(rows: list[dict[str, Any]], comparison: str, dataset: str | None = None) -> dict[str, Any]:
    for row in rows:
        if row["comparison"] != comparison:
            continue
        if dataset is None or row.get("dataset") == dataset:
            return row
    raise KeyError((comparison, dataset))


def weather_late_delta(segment_deltas: list[dict[str, Any]], comparison: str) -> float:
    for row in segment_deltas:
        if row["comparison"] == comparison and row["dataset"] == "Weather" and row["segment"] == "337-720":
            return row["relative_mse_pct"]
    raise KeyError(comparison)


def write_report(
    path: Path,
    main_summary: list[dict[str, Any]],
    dataset_summary: list[dict[str, Any]],
    segment_deltas: list[dict[str, Any]],
    training_summary: list[dict[str, Any]],
    future_summary: list[dict[str, Any]],
    checkpoint_diagnostics: list[dict[str, Any]],
) -> None:
    f2a0_vs_base = pick(main_summary, "selective_anchor_vs_single_prefix_base")
    f2a1_vs_base = pick(main_summary, "selective_anchor_vs_r3_base")
    f2a1_etth2 = pick(dataset_summary, "selective_anchor_vs_r3_base", "ETTh2")
    f2a0_weather_late = weather_late_delta(segment_deltas, "selective_anchor_vs_single_prefix_base")
    f2a1_weather_late = weather_late_delta(segment_deltas, "selective_anchor_vs_r3_base")
    max_leakage = max((row["max_prediction_leakage_abs"] for row in future_summary), default=float("nan"))
    f2a0_pass = f2a0_vs_base["mean_relative_mse_pct"] < 0 and f2a0_weather_late <= 0.5
    f2a1_pass = f2a1_etth2["mean_relative_mse_pct"] <= 1.0 and f2a1_weather_late < 0
    verdict = "pass_to_anchor_hss_design" if f2a0_pass or f2a1_pass else "fail_stop_future_anchor_stacking"

    lines = [
        "# Phase4-FSA-F2 Anchor Pressure Gate Report",
        "",
        "## Decision",
        "",
        f"[Decision] `{verdict}`.",
        "",
        "## Main Summary",
        "",
        table(
            main_summary,
            [
                "comparison",
                "candidate_arm",
                "baseline_arm",
                "settings",
                "mse_wins",
                "mean_relative_mse_pct",
                "mae_wins",
                "mean_relative_mae_pct",
            ],
        ),
        "",
        "## Dataset Summary",
        "",
        table(
            dataset_summary,
            ["comparison", "candidate_arm", "baseline_arm", "dataset", "settings", "mse_wins", "mean_relative_mse_pct"],
        ),
        "",
        "## H720 Segment Deltas",
        "",
        table(
            segment_deltas,
            ["comparison", "candidate_arm", "baseline_arm", "dataset", "segment", "relative_mse_pct", "mse_win"],
        ),
        "",
        "## Focused Diagnostics",
        "",
        "### Future Alignment",
        "",
        table(
            future_summary,
            [
                "arm",
                "dataset",
                "horizons",
                "mean_teacher_student_cosine",
                "mean_future_reconstruction_loss",
                "mean_future_alignment_confidence",
                "min_future_alignment_confidence",
                "max_prediction_leakage_abs",
            ],
        ),
        "",
        "### Training Dynamics",
        "",
        table(
            training_summary,
            [
                "arm",
                "dataset",
                "epochs_ran",
                "best_epoch",
                "post_best_val_drift_pct",
                "train_loss_drop_pct",
                "first_train_future_alignment_confidence_mean",
                "last_train_future_alignment_confidence_mean",
            ],
        ),
        "",
        "### Checkpoint Selection",
        "",
        table(
            [row for row in checkpoint_diagnostics if row["selector"] in {"long_mean", "h720"}],
            ["arm", "dataset", "selector", "best_epoch", "official_best_epoch", "official_gap_to_selector_best_pct"],
        ),
        "",
        "## Gate Reading",
        "",
        f"- [Fact] `F2-A0` vs `F1-C0`: mean MSE `{fmt_pct(f2a0_vs_base['mean_relative_mse_pct'])}`; Weather h720 late `{fmt_pct(f2a0_weather_late)}`.",
        f"- [Fact] `F2-A1` vs `F1-C1`: mean MSE `{fmt_pct(f2a1_vs_base['mean_relative_mse_pct'])}`; ETTh2 mean `{fmt_pct(f2a1_etth2['mean_relative_mse_pct'])}`; Weather h720 late `{fmt_pct(f2a1_weather_late)}`.",
        f"- [Fact] Future leakage max `{max_leakage:.3e}`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("analysis/phase4_fsa_f2_anchor_pressure_gate/raw"))
    parser.add_argument("--f1-raw-root", type=Path, default=Path("analysis/phase4_future_state_anchor_gate_20260626/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/phase4_fsa_f2_anchor_pressure_gate_20260626"))
    args = parser.parse_args()

    main_rows = load_main_metrics(args.raw_root, args.f1_raw_root)
    segment_rows = load_h720_segments(args.raw_root, args.f1_raw_root)
    main_deltas = build_deltas(main_rows, ("dataset", "horizon"))
    segment_deltas = build_deltas(segment_rows, ("dataset", "horizon", "segment"))
    main_summary = summarize(main_deltas)
    dataset_summary = summarize(main_deltas, ("dataset",))
    training_summary = load_training_summary(args.raw_root, args.f1_raw_root)
    future_summary = load_future_alignment_summary(args.raw_root, args.f1_raw_root)
    checkpoint_diagnostics = load_checkpoint_diagnostics(args.raw_root, args.f1_raw_root)
    config_summary = load_config_summary(args.raw_root, args.f1_raw_root)

    write_csv(args.output_dir / "phase4_fsa_f2_main_deltas.csv", main_deltas)
    write_csv(args.output_dir / "phase4_fsa_f2_main_summary.csv", main_summary)
    write_csv(args.output_dir / "phase4_fsa_f2_dataset_summary.csv", dataset_summary)
    write_csv(args.output_dir / "phase4_fsa_f2_h720_segment_deltas.csv", segment_deltas)
    write_csv(args.output_dir / "phase4_fsa_f2_training_summary.csv", training_summary)
    write_csv(args.output_dir / "phase4_fsa_f2_future_alignment_summary.csv", future_summary)
    write_csv(args.output_dir / "phase4_fsa_f2_checkpoint_diagnostics.csv", checkpoint_diagnostics)
    write_csv(args.output_dir / "phase4_fsa_f2_config_summary.csv", config_summary)
    write_report(
        args.output_dir / "phase4_fsa_f2_anchor_pressure_gate_report.md",
        main_summary,
        dataset_summary,
        segment_deltas,
        training_summary,
        future_summary,
        checkpoint_diagnostics,
    )


if __name__ == "__main__":
    main()
