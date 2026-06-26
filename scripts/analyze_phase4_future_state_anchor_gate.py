from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = ["ETTh2", "Weather"]
HORIZONS = [96, 192, 336, 720]
HORIZON_LABEL = "mixed_h96_h192_h336_h720"
SEED = 2021

MODELS = {
    "F1-C0": {
        "model": "PatchEncoderFSAF1SinglePrefixBase",
        "strategy": "single_720_prefix_risk",
        "future_alignment": "off",
        "label": "F1-C0_single_prefix_base",
    },
    "F1-C1": {
        "model": "PatchEncoderFSAF1R3Base",
        "strategy": "r3_prefix_risk",
        "future_alignment": "off",
        "label": "F1-C1_r3_base",
    },
    "F1-A0": {
        "model": "PatchEncoderFSAF1SinglePrefixFutureAnchor",
        "strategy": "single_720_prefix_risk",
        "future_alignment": "on",
        "label": "F1-A0_single_prefix_future_anchor",
    },
    "F1-A1": {
        "model": "PatchEncoderFSAF1R3FutureAnchor",
        "strategy": "r3_prefix_risk",
        "future_alignment": "on",
        "label": "F1-A1_r3_future_anchor",
    },
    "F1-W0": {
        "model": "PatchEncoderFSAF1FullTimeFutureAnchor",
        "strategy": "full_time_mse",
        "future_alignment": "on",
        "label": "F1-W0_full_time_future_anchor",
    },
}

COMPARISONS = [
    ("F1-A0", "F1-C0", "anchor_vs_single_prefix_base"),
    ("F1-A1", "F1-C1", "anchor_vs_r3_base"),
    ("F1-A0", "F1-C1", "single_anchor_vs_r3_reference"),
    ("F1-W0", "F1-C0", "full_time_anchor_vs_single_prefix_base"),
    ("F1-W0", "F1-C1", "full_time_anchor_vs_r3_reference"),
]


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


def run_dir(raw_root: Path, arm: str, dataset: str) -> Path:
    return raw_root / MODELS[arm]["model"] / dataset / HORIZON_LABEL / f"seed{SEED}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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
    for arm, meta in MODELS.items():
        for dataset in DATASETS:
            path = run_dir(raw_root, arm, dataset) / "metrics_by_target_horizon.csv"
            if not path.exists():
                continue
            for row in read_csv(path):
                rows.append(
                    {
                        "arm": arm,
                        "model": meta["model"],
                        "label": meta["label"],
                        "strategy": meta["strategy"],
                        "future_alignment": meta["future_alignment"],
                        "dataset": dataset,
                        "horizon": int(row["target_horizon"]),
                        "mse": float(row["mse"]),
                        "mae": float(row["mae"]),
                    }
                )
    return rows


def load_segment_metrics(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for arm, meta in MODELS.items():
        for dataset in DATASETS:
            for horizon in HORIZONS:
                path = run_dir(raw_root, arm, dataset) / f"h{horizon}" / "metrics_by_segment.csv"
                if not path.exists():
                    continue
                for row in read_csv(path):
                    start, end = parse_segment(row["segment"])
                    rows.append(
                        {
                            "arm": arm,
                            "model": meta["model"],
                            "label": meta["label"],
                            "strategy": meta["strategy"],
                            "future_alignment": meta["future_alignment"],
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


def build_main_deltas(main_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["arm"], row["dataset"], row["horizon"]): row for row in main_rows}
    rows = []
    for candidate_arm, baseline_arm, comparison in COMPARISONS:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                candidate_key = (candidate_arm, dataset, horizon)
                baseline_key = (baseline_arm, dataset, horizon)
                if candidate_key not in by_key or baseline_key not in by_key:
                    continue
                candidate = by_key[candidate_key]
                baseline = by_key[baseline_key]
                rows.append(
                    {
                        "comparison": comparison,
                        "candidate_arm": candidate_arm,
                        "candidate_label": MODELS[candidate_arm]["label"],
                        "baseline_arm": baseline_arm,
                        "baseline_label": MODELS[baseline_arm]["label"],
                        "dataset": dataset,
                        "horizon": horizon,
                        "candidate_mse": candidate["mse"],
                        "baseline_mse": baseline["mse"],
                        "relative_mse_pct": pct(candidate["mse"], baseline["mse"]),
                        "mse_win": candidate["mse"] < baseline["mse"],
                        "candidate_mae": candidate["mae"],
                        "baseline_mae": baseline["mae"],
                        "relative_mae_pct": pct(candidate["mae"], baseline["mae"]),
                        "mae_win": candidate["mae"] < baseline["mae"],
                    }
                )
    return rows


def build_segment_deltas(segment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["arm"], row["dataset"], row["horizon"], row["segment"]): row
        for row in segment_rows
    }
    rows = []
    for candidate_arm, baseline_arm, comparison in COMPARISONS:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                candidate_segments = [
                    key[-1]
                    for key in by_key
                    if key[0] == candidate_arm and key[1] == dataset and key[2] == horizon
                ]
                for segment in sorted(candidate_segments, key=lambda value: parse_segment(value)):
                    candidate_key = (candidate_arm, dataset, horizon, segment)
                    baseline_key = (baseline_arm, dataset, horizon, segment)
                    if candidate_key not in by_key or baseline_key not in by_key:
                        continue
                    candidate = by_key[candidate_key]
                    baseline = by_key[baseline_key]
                    rows.append(
                        {
                            "comparison": comparison,
                            "candidate_arm": candidate_arm,
                            "candidate_label": MODELS[candidate_arm]["label"],
                            "baseline_arm": baseline_arm,
                            "baseline_label": MODELS[baseline_arm]["label"],
                            "dataset": dataset,
                            "horizon": horizon,
                            "segment": segment,
                            "future_region": candidate["future_region"],
                            "candidate_mse": candidate["mse"],
                            "baseline_mse": baseline["mse"],
                            "relative_mse_pct": pct(candidate["mse"], baseline["mse"]),
                            "mse_win": candidate["mse"] < baseline["mse"],
                            "candidate_mae": candidate["mae"],
                            "baseline_mae": baseline["mae"],
                            "relative_mae_pct": pct(candidate["mae"], baseline["mae"]),
                            "mae_win": candidate["mae"] < baseline["mae"],
                        }
                    )
    return rows


def summarize_deltas(rows: list[dict[str, Any]], group_keys: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["comparison"], row["candidate_arm"], row["baseline_arm"], *[row[key] for key in group_keys])
        grouped[key].append(row)

    output = []
    for key, subset in sorted(grouped.items()):
        comparison, candidate_arm, baseline_arm, *group_values = key
        item = {
            "comparison": comparison,
            "candidate_arm": candidate_arm,
            "candidate_label": MODELS[candidate_arm]["label"],
            "baseline_arm": baseline_arm,
            "baseline_label": MODELS[baseline_arm]["label"],
            "settings": len(subset),
            "mse_wins": sum(1 for row in subset if row["mse_win"]),
            "mae_wins": sum(1 for row in subset if row["mae_win"]),
            "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in subset),
            "mean_relative_mae_pct": mean(row["relative_mae_pct"] for row in subset),
        }
        for field, value in zip(group_keys, group_values):
            item[field] = value
        output.append(item)
    return output


def load_training_summary(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for arm, meta in MODELS.items():
        for dataset in DATASETS:
            path = run_dir(raw_root, arm, dataset) / "training_log.csv"
            if not path.exists():
                continue
            log_rows = read_csv(path)
            val_values = [float(row["val_mean_mse"]) for row in log_rows]
            train_values = [float(row["train_prediction_loss"]) for row in log_rows]
            best_index = min(range(len(val_values)), key=val_values.__getitem__)
            item = {
                "arm": arm,
                "label": meta["label"],
                "strategy": meta["strategy"],
                "future_alignment": meta["future_alignment"],
                "dataset": dataset,
                "epochs_ran": len(log_rows),
                "best_epoch": best_index + 1,
                "first_val_mean_mse": val_values[0],
                "best_val_mean_mse": val_values[best_index],
                "last_val_mean_mse": val_values[-1],
                "post_best_val_drift_pct": pct(val_values[-1], val_values[best_index]),
                "first_train_prediction_loss": train_values[0],
                "last_train_prediction_loss": train_values[-1],
                "train_loss_drop_pct": pct(train_values[-1], train_values[0]),
            }
            if "train_future_local_alignment_loss" in log_rows[0]:
                future_cols = [
                    "train_future_local_alignment_loss",
                    "train_future_relation_alignment_loss",
                    "train_future_reconstruction_loss",
                    "train_future_alignment_confidence_mean",
                ]
                for col in future_cols:
                    values = [float(row[col]) for row in log_rows if row.get(col)]
                    if values:
                        item[f"first_{col}"] = values[0]
                        item[f"last_{col}"] = values[-1]
            rows.append(item)
    return rows


def load_future_alignment_summary(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for arm, meta in MODELS.items():
        if meta["future_alignment"] != "on":
            continue
        for dataset in DATASETS:
            for horizon in HORIZONS:
                path = run_dir(raw_root, arm, dataset) / f"h{horizon}" / "future_alignment_stats.csv"
                if not path.exists():
                    continue
                stats = read_csv(path)[0]
                rows.append(
                    {
                        "arm": arm,
                        "label": meta["label"],
                        "strategy": meta["strategy"],
                        "dataset": dataset,
                        "horizon": horizon,
                        "future_local_alignment_loss": float(stats["future_local_alignment_loss"]),
                        "future_relation_alignment_loss": float(stats["future_relation_alignment_loss"]),
                        "future_reconstruction_loss": float(stats["future_reconstruction_loss"]),
                        "future_raw_reconstruction_loss": float(
                            stats.get("future_raw_reconstruction_loss", stats["future_reconstruction_loss"])
                        ),
                        "future_normalized_reconstruction_loss": float(
                            stats.get(
                                "future_normalized_reconstruction_loss",
                                stats["future_reconstruction_loss"],
                            )
                        ),
                        "future_alignment_confidence_mean": float(
                            stats.get("future_alignment_confidence_mean", "1.0")
                        ),
                        "future_alignment_confidence_min": float(
                            stats.get("future_alignment_confidence_min", "1.0")
                        ),
                        "future_alignment_confidence_max": float(
                            stats.get("future_alignment_confidence_max", "1.0")
                        ),
                        "teacher_student_cosine": float(stats["teacher_student_cosine"]),
                        "prediction_leakage_max_abs": float(stats["prediction_leakage_max_abs"]),
                    }
                )
    return rows


def summarize_future_alignment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["arm"], row["dataset"])].append(row)
    output = []
    for (arm, dataset), subset in sorted(grouped.items()):
        output.append(
            {
                "arm": arm,
                "label": MODELS[arm]["label"],
                "dataset": dataset,
                "horizons": len(subset),
                "mean_teacher_student_cosine": mean(row["teacher_student_cosine"] for row in subset),
                "mean_future_local_alignment_loss": mean(row["future_local_alignment_loss"] for row in subset),
                "mean_future_reconstruction_loss": mean(row["future_reconstruction_loss"] for row in subset),
                "mean_future_alignment_confidence": mean(
                    row["future_alignment_confidence_mean"] for row in subset
                ),
                "min_future_alignment_confidence": min(row["future_alignment_confidence_min"] for row in subset),
                "max_prediction_leakage_abs": max(row["prediction_leakage_max_abs"] for row in subset),
            }
        )
    return output


def load_checkpoint_diagnostics(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for arm, meta in MODELS.items():
        for dataset in DATASETS:
            path = run_dir(raw_root, arm, dataset) / "checkpoint_selection_diagnostics.csv"
            if not path.exists():
                continue
            for row in read_csv(path):
                rows.append(
                    {
                        "arm": arm,
                        "label": meta["label"],
                        "strategy": meta["strategy"],
                        "future_alignment": meta["future_alignment"],
                        "dataset": dataset,
                        "selector": row["selector"],
                        "horizons": row["horizons"],
                        "best_epoch": int(row["best_epoch"]),
                        "official_best_epoch": int(row["official_best_epoch"]),
                        "best_selector_mse": float(row["best_selector_mse"]),
                        "official_gap_to_selector_best_pct": float(row["official_gap_to_selector_best_pct"]),
                    }
                )
    return rows


def load_config_summary(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for arm, meta in MODELS.items():
        for dataset in DATASETS:
            path = run_dir(raw_root, arm, dataset) / "effective_config.json"
            if not path.exists():
                continue
            config = read_json(path)
            rows.append(
                {
                    "arm": arm,
                    "label": meta["label"],
                    "dataset": dataset,
                    "supervision_strategy": config.get("supervision_strategy"),
                    "step_loss_weighting": config.get("step_loss_weighting"),
                    "learning_rate": config.get("learning_rate"),
                    "future_teacher_layers": config.get("future_teacher_layers"),
                    "future_align_weight": config.get("future_align_weight"),
                    "future_relation_weight": config.get("future_relation_weight"),
                    "future_recon_weight": config.get("future_recon_weight"),
                    "future_recon_normalization": config.get("future_recon_normalization"),
                    "future_align_weighting": config.get("future_align_weighting"),
                    "future_confidence_floor": config.get("future_confidence_floor"),
                }
            )
    return rows


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


def comparison_row(rows: list[dict[str, Any]], comparison: str, dataset: str | None = None) -> dict[str, Any]:
    for row in rows:
        if row["comparison"] != comparison:
            continue
        if dataset is None or row.get("dataset") == dataset:
            return row
    raise KeyError((comparison, dataset))


def write_report(
    output_path: Path,
    main_summary: list[dict[str, Any]],
    main_dataset_summary: list[dict[str, Any]],
    segment_region_summary: list[dict[str, Any]],
    segment_deltas: list[dict[str, Any]],
    training_summary: list[dict[str, Any]],
    future_summary: list[dict[str, Any]],
    checkpoint_diagnostics: list[dict[str, Any]],
) -> None:
    a0_vs_c0 = comparison_row(main_summary, "anchor_vs_single_prefix_base")
    a1_vs_c1 = comparison_row(main_summary, "anchor_vs_r3_base")
    a1_weather = comparison_row(main_dataset_summary, "anchor_vs_r3_base", "Weather")
    a0_etth2 = comparison_row(main_dataset_summary, "anchor_vs_single_prefix_base", "ETTh2")
    a0_weather = comparison_row(main_dataset_summary, "anchor_vs_single_prefix_base", "Weather")
    weather_h720 = [
        row for row in segment_deltas
        if row["comparison"] == "anchor_vs_r3_base"
        and row["dataset"] == "Weather"
        and row["horizon"] == 720
        and row["segment"] == "337-720"
    ]
    weather_late_vs_r3 = weather_h720[0]["relative_mse_pct"] if weather_h720 else float("nan")
    max_leakage = max((row["max_prediction_leakage_abs"] for row in future_summary), default=float("nan"))
    min_confidence = min((row["min_future_alignment_confidence"] for row in future_summary), default=float("nan"))
    oracle_rows = [
        row for row in checkpoint_diagnostics
        if row["selector"] in {"long_mean", "h720"}
        and row["arm"] in {"F1-A0", "F1-A1"}
    ]
    max_oracle_gap = max((row["official_gap_to_selector_best_pct"] for row in oracle_rows), default=float("nan"))

    f1_a1_pass = a1_vs_c1["mean_relative_mse_pct"] <= 0.3 and (
        a1_weather["mean_relative_mse_pct"] < 0 or weather_late_vs_r3 < 0
    )
    f1_a0_pass = (
        a0_vs_c0["mse_wins"] >= 5
        or (
            a0_vs_c0["mean_relative_mse_pct"] < 0
            and a0_etth2["mean_relative_mse_pct"] <= 1.0
            and a0_weather["mean_relative_mse_pct"] <= 1.0
        )
    )
    diagnostics_pass = max_leakage <= 1e-7 and min_confidence > 0.05
    verdict = "pass_to_fsa_f2" if (f1_a0_pass or f1_a1_pass) and diagnostics_pass else "fail_or_partial"

    lines = [
        "# Phase4-FSA-F1 Future-State Anchor Gate Report",
        "",
        "## 11-step Record",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | Step 9/10：评估 Future-State Anchored HSS substrate diagnostic |",
        "| `problem` | Phase4 scheduling signal 有效，但当前 `target_states` 可能缺少承接 HSS pressure 的 future-structured representation |",
        "| `existence_evidence` | S1/S2、HSSG、SCC 均有 active signal 但不能稳定超过 R.3；OP-A 否定 full-time base pretrain；R3D 指向 prefix-risk stabilized base |",
        "| `idea` | 用 training-only future teacher 先锚定 `target_states`，再判断 prefix-risk/HSS pressure 是否更可泛化 |",
        "| `theory_check` | 若 future-state geometry 是瓶颈，future anchor 应改善或至少不破坏 R.3/single-prefix base，并给出非 collapse alignment diagnostics |",
        "| `design` | F1-C0/F1-C1 controls；F1-A0/F1-A1 future-anchor candidates；F1-W0 weak full-time control；ETTh2 + Weather，seed2021 |",
        "| `gate` | A1 vs R.3 不劣于 +0.3% 且改善 Weather long/late；或 A0 vs single-prefix 至少 5/8 wins；future leakage/confidence 非 collapse；oracle gap 不是唯一解释 |",
        f"| `artifacts` | `{output_path.parent}` |",
        f"| `decision` | {verdict} |",
        "",
        "## Main Comparison Summary",
        "",
        rows_for_table(
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
        "## Dataset Split",
        "",
        rows_for_table(
            main_dataset_summary,
            [
                "comparison",
                "candidate_arm",
                "baseline_arm",
                "dataset",
                "settings",
                "mse_wins",
                "mean_relative_mse_pct",
            ],
        ),
        "",
        "## Future Alignment Diagnostics",
        "",
        rows_for_table(
            future_summary,
            [
                "arm",
                "dataset",
                "horizons",
                "mean_teacher_student_cosine",
                "mean_future_local_alignment_loss",
                "mean_future_reconstruction_loss",
                "mean_future_alignment_confidence",
                "min_future_alignment_confidence",
                "max_prediction_leakage_abs",
            ],
        ),
        "",
        "## Segment Region Summary",
        "",
        rows_for_table(
            [
                row for row in segment_region_summary
                if row["comparison"] in {"anchor_vs_single_prefix_base", "anchor_vs_r3_base"}
            ],
            [
                "comparison",
                "candidate_arm",
                "baseline_arm",
                "dataset",
                "future_region",
                "settings",
                "mse_wins",
                "mean_relative_mse_pct",
            ],
        ),
        "",
        "## Training Dynamics",
        "",
        rows_for_table(
            training_summary,
            [
                "arm",
                "dataset",
                "future_alignment",
                "epochs_ran",
                "best_epoch",
                "post_best_val_drift_pct",
                "train_loss_drop_pct",
            ],
        ),
        "",
        "## Checkpoint Selection Diagnostics",
        "",
        rows_for_table(
            [
                row for row in checkpoint_diagnostics
                if row["selector"] in {"long_mean", "h720"}
            ],
            [
                "arm",
                "dataset",
                "selector",
                "best_epoch",
                "official_best_epoch",
                "official_gap_to_selector_best_pct",
            ],
        ),
        "",
        "## Gate Reading",
        "",
        f"- [Fact] `F1-A0` vs `F1-C0`: `{a0_vs_c0['mse_wins']}/8` MSE wins, mean relative MSE `{fmt_pct(a0_vs_c0['mean_relative_mse_pct'])}`.",
        f"- [Fact] `F1-A1` vs `F1-C1/R.3`: `{a1_vs_c1['mse_wins']}/8` MSE wins, mean relative MSE `{fmt_pct(a1_vs_c1['mean_relative_mse_pct'])}`.",
        f"- [Fact] `F1-A1` Weather mean relative MSE vs R.3 `{fmt_pct(a1_weather['mean_relative_mse_pct'])}`; Weather h720 late `337-720` segment vs R.3 `{fmt_pct(weather_late_vs_r3)}`.",
        f"- [Fact] Future leakage max `{max_leakage:.3e}`; min confidence `{min_confidence:.6f}`; max official-to-oracle gap among A0/A1 long/h720 selectors `{fmt_pct(max_oracle_gap)}`.",
        "",
        "## Decision Rules",
        "",
        "- 若 `F1-A0` 通过而 `F1-A1` 不通过：进入 FSA-F2，在 anchored h720-only state 上重新测试 HSS/gradient routing。",
        "- 若 `F1-A1` 通过而 `F1-A0` 不通过：future-state anchor 可能需要 mixed exposure support，后续叙事必须保留 compound exposure control。",
        "- 若二者均失败且 diagnostics 非 collapse：future teacher 与 forecasting objective 语义不一致，回 Step 2/3 重新定义 representation 问题。",
        "- 若只有 oracle checkpoint 有收益：先修 validation metric，不继续改 model。",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("artifacts/runs/phase4_future_state_anchor_gate"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/phase4_future_state_anchor_gate_20260626"),
    )
    args = parser.parse_args()

    main_rows = load_main_metrics(args.raw_root)
    segment_rows = load_segment_metrics(args.raw_root)
    main_deltas = build_main_deltas(main_rows)
    segment_deltas = build_segment_deltas(segment_rows)
    main_summary = summarize_deltas(main_deltas)
    main_dataset_summary = summarize_deltas(main_deltas, ("dataset",))
    segment_region_summary = summarize_deltas(segment_deltas, ("dataset", "future_region"))
    training_summary = load_training_summary(args.raw_root)
    future_rows = load_future_alignment_summary(args.raw_root)
    future_summary = summarize_future_alignment(future_rows)
    checkpoint_diagnostics = load_checkpoint_diagnostics(args.raw_root)
    config_summary = load_config_summary(args.raw_root)

    write_csv(args.output_dir / "phase4_fsa_f1_main_metrics.csv", main_rows)
    write_csv(args.output_dir / "phase4_fsa_f1_main_deltas.csv", main_deltas)
    write_csv(args.output_dir / "phase4_fsa_f1_main_summary.csv", main_summary)
    write_csv(args.output_dir / "phase4_fsa_f1_dataset_summary.csv", main_dataset_summary)
    write_csv(args.output_dir / "phase4_fsa_f1_segment_deltas.csv", segment_deltas)
    write_csv(args.output_dir / "phase4_fsa_f1_segment_region_summary.csv", segment_region_summary)
    write_csv(args.output_dir / "phase4_fsa_f1_training_summary.csv", training_summary)
    write_csv(args.output_dir / "phase4_fsa_f1_future_alignment.csv", future_rows)
    write_csv(args.output_dir / "phase4_fsa_f1_future_alignment_summary.csv", future_summary)
    write_csv(args.output_dir / "phase4_fsa_f1_checkpoint_diagnostics.csv", checkpoint_diagnostics)
    write_csv(args.output_dir / "phase4_fsa_f1_config_summary.csv", config_summary)
    write_report(
        args.output_dir / "phase4_future_state_anchor_gate_report.md",
        main_summary,
        main_dataset_summary,
        segment_region_summary,
        segment_deltas,
        training_summary,
        future_summary,
        checkpoint_diagnostics,
    )


if __name__ == "__main__":
    main()
