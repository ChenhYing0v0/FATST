from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


DATASETS = ["ETTh2", "ETTm1", "Weather"]
SHORT_HORIZONS = [96, 192, 336]
MAX_HORIZON = 720
HORIZON_LABEL = "mixed_h96_h192_h336_h720"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mse(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    return float(np.mean(diff * diff))


def mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(x - y)))


def cosine_mean(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.transpose(left, (0, 2, 1)).reshape(-1, left.shape[1])
    right_flat = np.transpose(right, (0, 2, 1)).reshape(-1, right.shape[1])
    numerator = np.sum(left_flat * right_flat, axis=1)
    denominator = np.linalg.norm(left_flat, axis=1) * np.linalg.norm(right_flat, axis=1)
    valid = denominator > 1e-12
    if not np.any(valid):
        return float("nan")
    return float(np.mean(numerator[valid] / denominator[valid]))


def load_prediction(run_root: Path, dataset: str, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    path = run_root / dataset / HORIZON_LABEL / "seed2021" / f"h{horizon}" / "predictions_test.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    return np.asarray(data["pred"], dtype=np.float64), np.asarray(data["true"], dtype=np.float64)


def keyed(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], dict[str, str]]:
    out: dict[tuple[Any, ...], dict[str, str]] = {}
    for row in rows:
        key_parts: list[Any] = []
        for key in keys:
            value: Any = row[key]
            if key in {"horizon", "prefix_horizon"}:
                value = int(value)
            key_parts.append(value)
        out[tuple(key_parts)] = row
    return out


def gap_from_relative(row: dict[str, str] | None) -> bool:
    if row is None:
        return False
    return float(row["relative_mse_pct"]) > 0.0


def short_rows(
    run_root: Path,
    vs_fixed: dict[tuple[Any, ...], dict[str, str]],
    h720_prefix: dict[tuple[Any, ...], dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        max_pred, max_true = load_prediction(run_root, dataset, MAX_HORIZON)
        for horizon in SHORT_HORIZONS:
            pred, true = load_prediction(run_root, dataset, horizon)
            n_aligned = min(pred.shape[0], max_pred.shape[0])
            pred_aligned = pred[:n_aligned, :horizon, :]
            true_aligned = true[:n_aligned, :horizon, :]
            max_pred_prefix = max_pred[:n_aligned, :horizon, :]
            max_true_prefix = max_true[:n_aligned, :horizon, :]
            residual_aligned = pred_aligned - true_aligned
            max_residual_prefix = max_pred_prefix - max_true_prefix

            extra_count = max(pred.shape[0] - n_aligned, 0)
            if extra_count > 0:
                pred_extra = pred[n_aligned:, :horizon, :]
                true_extra = true[n_aligned:, :horizon, :]
                extra_mse = mse(pred_extra, true_extra)
                extra_mae = mae(pred_extra, true_extra)
            else:
                extra_mse = float("nan")
                extra_mae = float("nan")

            full_mse = mse(pred, true)
            aligned_mse = mse(pred_aligned, true_aligned)
            max_prefix_mse = mse(max_pred_prefix, max_true_prefix)
            full_gap = gap_from_relative(vs_fixed.get((dataset, horizon)))
            h720_prefix_gap = gap_from_relative(h720_prefix.get((dataset, horizon)))
            if full_gap and not h720_prefix_gap and extra_count > 0:
                gap_type = "short_extra_window_gap"
            elif full_gap and h720_prefix_gap:
                gap_type = "shared_prefix_gap"
            elif not full_gap and h720_prefix_gap:
                gap_type = "h720_prefix_only_gap"
            else:
                gap_type = "no_mse_gap"

            rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "n_full": pred.shape[0],
                    "n_aligned_with_h720": n_aligned,
                    "n_short_only_extra": extra_count,
                    "full_mse": full_mse,
                    "aligned_mse": aligned_mse,
                    "short_only_extra_mse": extra_mse,
                    "h720_prefix_mse": max_prefix_mse,
                    "full_vs_aligned_mse_pct": (full_mse / max(aligned_mse, 1e-12) - 1.0) * 100.0,
                    "extra_vs_aligned_mse_pct": (extra_mse / max(aligned_mse, 1e-12) - 1.0) * 100.0
                    if finite(extra_mse)
                    else float("nan"),
                    "pred_prefix_mismatch_mse": mse(pred_aligned, max_pred_prefix),
                    "true_prefix_alignment_mse": mse(true_aligned, max_true_prefix),
                    "residual_prefix_mismatch_mse": mse(residual_aligned, max_residual_prefix),
                    "residual_prefix_cosine": cosine_mean(residual_aligned, max_residual_prefix),
                    "full_is_specialist_gap": full_gap,
                    "h720_prefix_is_specialist_gap": h720_prefix_gap,
                    "gap_type": gap_type,
                    "full_vs_fixed_relative_mse_pct": float(
                        vs_fixed.get((dataset, horizon), {}).get("relative_mse_pct", "nan")
                    ),
                    "h720_prefix_vs_fixed_relative_mse_pct": float(
                        h720_prefix.get((dataset, horizon), {}).get("relative_mse_pct", "nan")
                    ),
                }
            )
    return rows


def h720_segment_rows(segments: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in segments:
        if int(row["horizon"]) != MAX_HORIZON:
            continue
        relative = float(row["relative_mse_pct"])
        rows.append(
            {
                "dataset": row["dataset"],
                "segment": row["segment"],
                "target_mse": float(row["target_mse"]),
                "fixed_mse": float(row["fixed_mse"]),
                "relative_mse_pct": relative,
                "target_wins_mse": row["target_wins_mse"],
                "is_segment_gap": relative > 0.0,
            }
        )
    return rows


def summarize(short: list[dict[str, Any]], segments: list[dict[str, Any]]) -> dict[str, Any]:
    full_gap_rows = [row for row in short if row["full_is_specialist_gap"]]
    extra_gap_rows = [row for row in short if row["gap_type"] == "short_extra_window_gap"]
    shared_gap_rows = [row for row in short if row["gap_type"] == "shared_prefix_gap"]
    segment_gap_rows = [row for row in segments if row["is_segment_gap"]]
    max_pred_mismatch = max(float(row["pred_prefix_mismatch_mse"]) for row in short)
    max_true_mismatch = max(float(row["true_prefix_alignment_mse"]) for row in short)
    max_residual_mismatch = max(float(row["residual_prefix_mismatch_mse"]) for row in short)
    extra_ratios = [
        float(row["extra_vs_aligned_mse_pct"])
        for row in short
        if row["full_is_specialist_gap"] and finite(row["extra_vs_aligned_mse_pct"])
    ]
    summary = {
        "short_rows": len(short),
        "short_full_specialist_gap_count": len(full_gap_rows),
        "short_extra_window_gap_count": len(extra_gap_rows),
        "short_shared_prefix_gap_count": len(shared_gap_rows),
        "h720_segment_gap_count": len(segment_gap_rows),
        "short_full_specialist_gaps": [
            {"dataset": row["dataset"], "horizon": row["horizon"], "gap_type": row["gap_type"]}
            for row in full_gap_rows
        ],
        "h720_segment_gaps": [
            {
                "dataset": row["dataset"],
                "segment": row["segment"],
                "relative_mse_pct": row["relative_mse_pct"],
            }
            for row in segment_gap_rows
        ],
        "max_pred_prefix_mismatch_mse": max_pred_mismatch,
        "max_true_prefix_alignment_mse": max_true_mismatch,
        "max_residual_prefix_mismatch_mse": max_residual_mismatch,
        "mean_extra_vs_aligned_mse_pct_for_short_gaps": mean(extra_ratios)
        if extra_ratios
        else float("nan"),
    }
    summary["gate"] = {
        "prefix_identity_pass": max_pred_mismatch <= 1e-10
        and max_true_mismatch <= 1e-10
        and max_residual_mismatch <= 1e-10,
        "short_gaps_are_extra_window_effects": len(full_gap_rows) > 0
        and len(full_gap_rows) == len(extra_gap_rows),
        "long_gaps_are_segment_localized": len(segment_gap_rows) > 0,
    }
    summary["gate"]["supports_tradeoff_diagnostic"] = (
        summary["gate"]["prefix_identity_pass"]
        and (summary["gate"]["short_gaps_are_extra_window_effects"] or summary["gate"]["long_gaps_are_segment_localized"])
    )
    return summary


def format_float(value: Any) -> str:
    return f"{float(value):.6f}" if finite(value) else "nan"


def write_report(path: Path, summary: dict[str, Any], short: list[dict[str, Any]], segments: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase3-A Prefix Specialist Tradeoff Diagnostic",
        "",
        "## Decision",
        "",
    ]
    if summary["gate"]["supports_tradeoff_diagnostic"]:
        lines.append("[Decision] diagnostic supports continuing from objective-matrix failure to a prefix/specialist tradeoff analysis.")
    else:
        lines.append("[Decision] diagnostic does not yet support a prefix/specialist tradeoff mechanism.")
    lines.extend(["", "## Gate", ""])
    for key, value in summary["gate"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Short-Horizon Alignment",
            "",
            "| Dataset | Horizon | Gap type | Full gap | H720-prefix gap | Full MSE | Aligned MSE | Extra MSE | Extra vs aligned | Pred mismatch |",
            "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in short:
        lines.append(
            "| {dataset} | {horizon} | {gap_type} | {full_gap} | {prefix_gap} | {full_mse} | {aligned_mse} | {extra_mse} | {extra_pct} | {pred_mismatch} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                gap_type=row["gap_type"],
                full_gap=row["full_is_specialist_gap"],
                prefix_gap=row["h720_prefix_is_specialist_gap"],
                full_mse=format_float(row["full_mse"]),
                aligned_mse=format_float(row["aligned_mse"]),
                extra_mse=format_float(row["short_only_extra_mse"]),
                extra_pct=format_float(row["extra_vs_aligned_mse_pct"]),
                pred_mismatch=format_float(row["pred_prefix_mismatch_mse"]),
            )
        )
    lines.extend(
        [
            "",
            "## H720 Segment Gaps",
            "",
            "| Dataset | Segment | Relative MSE vs fixed | Is segment gap |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for row in segments:
        if row["is_segment_gap"]:
            lines.append(
                "| {dataset} | {segment} | {relative} | {gap} |".format(
                    dataset=row["dataset"],
                    segment=row["segment"],
                    relative=format_float(row["relative_mse_pct"]),
                    gap=row["is_segment_gap"],
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "[Fact] Prefix prediction and truth alignment are checked on the overlapping `h720`-compatible windows. If mismatch is near zero, standalone short-horizon differences are not caused by inconsistent prediction prefixes on the same inputs.",
            "",
            "[Decision Rule] If short-horizon gaps disappear on the `h720`-aligned subset but appear on the short-only extra windows, treat them as coverage/regime effects. If h720 gaps are concentrated in late segments, treat them as long-tail calibration effects.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze prefix consistency versus horizon-specialist gaps.")
    parser.add_argument(
        "--prediction-root",
        default="analysis/phase2_qdf_alignment_diagnostic_20260623/raw/PatchEncoderPrefixRiskWeighted",
    )
    parser.add_argument("--analysis-root", default="analysis/phase3_prefix_specialist_tradeoff_20260624")
    parser.add_argument(
        "--vs-fixed-csv",
        default="analysis/phase1_prefix_risk_weighted_gate_20260622/phase1_prefix_risk_weighted_vs_fixed.csv",
    )
    parser.add_argument(
        "--h720-prefix-csv",
        default="analysis/phase1_prefix_risk_weighted_gate_20260622/phase1_prefix_risk_weighted_h720_prefix_reference.csv",
    )
    parser.add_argument(
        "--segments-csv",
        default="analysis/phase1_prefix_risk_weighted_gate_20260622/phase1_prefix_risk_weighted_vs_fixed_segments.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root = Path(args.analysis_root)
    analysis_root.mkdir(parents=True, exist_ok=True)
    vs_fixed = keyed(read_csv(Path(args.vs_fixed_csv)), ("dataset", "horizon"))
    h720_prefix = keyed(read_csv(Path(args.h720_prefix_csv)), ("dataset", "prefix_horizon"))
    segment_metrics = read_csv(Path(args.segments_csv))
    short = short_rows(Path(args.prediction_root), vs_fixed, h720_prefix)
    segments = h720_segment_rows(segment_metrics)
    summary = summarize(short, segments)
    write_csv(analysis_root / "phase3_prefix_specialist_short_alignment.csv", short)
    write_csv(analysis_root / "phase3_prefix_specialist_h720_segments.csv", segments)
    (analysis_root / "phase3_prefix_specialist_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(analysis_root / "phase3_prefix_specialist_report.md", summary, short, segments)
    print(f"phase3_prefix_specialist_report={analysis_root / 'phase3_prefix_specialist_report.md'}")


if __name__ == "__main__":
    main()
