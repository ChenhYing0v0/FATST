from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

BASELINE_DIR = Path(__file__).resolve().parents[1] / "baselines" / "patch_encoder_target_set_decoder"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from dataset import ForecastDataset


DATASETS = ["ETTh2", "ETTm1", "Weather"]
SHORT_HORIZONS = [96, 192, 336]
MAX_HORIZON = 720
HORIZON_LABEL = "mixed_h96_h192_h336_h720"
SEGMENTS = [
    ("1-96", 0, 96),
    ("97-192", 96, 192),
    ("193-336", 192, 336),
    ("337-720", 336, 720),
]
SHORT_GAP_KEYS = {("ETTm1", 96), ("Weather", 96)}
H720_GAP_KEYS = {("ETTh2", "193-336"), ("ETTh2", "337-720"), ("ETTm1", "337-720")}


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


def rank_auc(values: np.ndarray, labels: np.ndarray) -> float:
    positives = values[labels == 1]
    negatives = values[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    pos_rank_sum = float(np.sum(ranks[labels == 1]))
    auc = (pos_rank_sum - len(positives) * (len(positives) + 1) / 2.0) / (
        len(positives) * len(negatives)
    )
    return float(max(auc, 1.0 - auc))


def standardized_mean_diff(values: np.ndarray, labels: np.ndarray) -> float:
    positives = values[labels == 1]
    negatives = values[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    pooled = math.sqrt((float(np.var(positives)) + float(np.var(negatives))) * 0.5)
    if pooled <= 1e-12:
        return 0.0
    return float((float(np.mean(positives)) - float(np.mean(negatives))) / pooled)


def history_features(x: np.ndarray) -> dict[str, np.ndarray]:
    first_half = x[:, : x.shape[1] // 2, :]
    second_half = x[:, x.shape[1] // 2 :, :]
    recent = x[:, -48:, :]
    previous = x[:, -96:-48, :]
    t = np.linspace(-1.0, 1.0, x.shape[1], dtype=np.float64)
    centered_t = t - t.mean()
    denom = float(np.sum(centered_t * centered_t))
    x_centered = x - np.mean(x, axis=1, keepdims=True)
    slope = np.sum(x_centered * centered_t.reshape(1, -1, 1), axis=1) / max(denom, 1e-12)
    return {
        "history_mean": np.mean(x, axis=(1, 2)),
        "history_std": np.std(x, axis=(1, 2)),
        "history_abs_mean": np.mean(np.abs(x), axis=(1, 2)),
        "history_last_abs_mean": np.mean(np.abs(x[:, -1, :]), axis=1),
        "history_recent_mean": np.mean(recent, axis=(1, 2)),
        "history_recent_std": np.std(recent, axis=(1, 2)),
        "history_recent_minus_previous_mean": np.mean(recent, axis=(1, 2))
        - np.mean(previous, axis=(1, 2)),
        "history_second_minus_first_mean": np.mean(second_half, axis=(1, 2))
        - np.mean(first_half, axis=(1, 2)),
        "history_slope_abs_mean": np.mean(np.abs(slope), axis=1),
        "window_index_norm": np.linspace(0.0, 1.0, x.shape[0], dtype=np.float64),
    }


def feature_rows(
    dataset: str,
    comparison: str,
    labels: np.ndarray,
    features: dict[str, np.ndarray],
    extra_fields: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, values in features.items():
        values = np.asarray(values, dtype=np.float64)
        pos = values[labels == 1]
        neg = values[labels == 0]
        rows.append(
            {
                "dataset": dataset,
                "comparison": comparison,
                "feature": name,
                "positive_count": int(len(pos)),
                "negative_count": int(len(neg)),
                "positive_mean": float(np.mean(pos)) if len(pos) else float("nan"),
                "negative_mean": float(np.mean(neg)) if len(neg) else float("nan"),
                "standardized_mean_diff": standardized_mean_diff(values, labels),
                "rank_auc_abs": rank_auc(values, labels),
                **extra_fields,
            }
        )
    return rows


def load_prediction(prediction_root: Path, dataset: str, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    path = prediction_root / dataset / HORIZON_LABEL / "seed2021" / f"h{horizon}" / "predictions_test.npz"
    data = np.load(path)
    return np.asarray(data["pred"], dtype=np.float64), np.asarray(data["true"], dtype=np.float64)


def short_regime_rows(dataset_root: Path, prediction_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        n_max = len(ForecastDataset(dataset_root, dataset, "test", 336, MAX_HORIZON))
        for horizon in SHORT_HORIZONS:
            ds = ForecastDataset(dataset_root, dataset, "test", 336, horizon)
            x = np.stack([ds[index][0].numpy() for index in range(len(ds))], axis=0).astype(np.float64)
            labels = np.zeros(len(ds), dtype=np.int64)
            labels[n_max:] = 1
            pred, true = load_prediction(prediction_root, dataset, horizon)
            aligned_mse = float(np.mean((pred[:n_max] - true[:n_max]) ** 2))
            extra_mse = float(np.mean((pred[n_max:] - true[n_max:]) ** 2)) if len(ds) > n_max else float("nan")
            comparison = f"h{horizon}_short_extra_vs_h720_aligned"
            rows.extend(
                feature_rows(
                    dataset,
                    comparison,
                    labels,
                    history_features(x),
                    {
                        "horizon": horizon,
                        "is_known_short_gap": (dataset, horizon) in SHORT_GAP_KEYS,
                        "aligned_mse": aligned_mse,
                        "positive_group_mse": extra_mse,
                        "positive_group": "short_only_extra_windows",
                        "negative_group": "h720_aligned_windows",
                    },
                )
            )
    return rows


def h720_late_rows(dataset_root: Path, prediction_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        ds = ForecastDataset(dataset_root, dataset, "test", 336, MAX_HORIZON)
        x = np.stack([ds[index][0].numpy() for index in range(len(ds))], axis=0).astype(np.float64)
        features = history_features(x)
        pred, true = load_prediction(prediction_root, dataset, MAX_HORIZON)
        squared_error = np.mean((pred - true) ** 2, axis=2)
        for segment, start, end in SEGMENTS:
            segment_error = np.mean(squared_error[:, start:end], axis=1)
            threshold = float(np.quantile(segment_error, 0.75))
            labels = (segment_error >= threshold).astype(np.int64)
            comparison = f"h720_{segment}_top_quartile_error"
            rows.extend(
                feature_rows(
                    dataset,
                    comparison,
                    labels,
                    features,
                    {
                        "horizon": MAX_HORIZON,
                        "segment": segment,
                        "is_known_h720_segment_gap": (dataset, segment) in H720_GAP_KEYS,
                        "positive_group_mse": float(np.mean(segment_error[labels == 1])),
                        "aligned_mse": float(np.mean(segment_error[labels == 0])),
                        "positive_group": "top_quartile_segment_error",
                        "negative_group": "lower_three_quartiles_segment_error",
                    },
                )
            )
    return rows


def best_feature(rows: list[dict[str, Any]], predicate_key: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get(predicate_key) is True or row.get(predicate_key) == "True"]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: max(abs(float(row["standardized_mean_diff"])), float(row["rank_auc_abs"]) - 0.5),
    )


def summarize(short_rows: list[dict[str, Any]], late_rows: list[dict[str, Any]]) -> dict[str, Any]:
    short_gap_rows = [row for row in short_rows if row["is_known_short_gap"]]
    late_gap_rows = [row for row in late_rows if row["is_known_h720_segment_gap"]]
    short_max_auc = max(float(row["rank_auc_abs"]) for row in short_gap_rows)
    short_max_smd = max(abs(float(row["standardized_mean_diff"])) for row in short_gap_rows)
    late_max_auc = max(float(row["rank_auc_abs"]) for row in late_gap_rows)
    late_max_smd = max(abs(float(row["standardized_mean_diff"])) for row in late_gap_rows)
    summary = {
        "short_regime_rows": len(short_rows),
        "late_segment_rows": len(late_rows),
        "short_known_gap_feature_max_auc": short_max_auc,
        "short_known_gap_feature_max_abs_smd": short_max_smd,
        "late_known_gap_feature_max_auc": late_max_auc,
        "late_known_gap_feature_max_abs_smd": late_max_smd,
        "best_short_gap_feature": best_feature(short_rows, "is_known_short_gap"),
        "best_late_gap_feature": best_feature(late_rows, "is_known_h720_segment_gap"),
    }
    summary["gate"] = {
        "short_regime_pre_input_signal": short_max_auc >= 0.65 or short_max_smd >= 0.5,
        "late_segment_pre_input_signal": late_max_auc >= 0.65 or late_max_smd >= 0.5,
        "no_output_residual_mechanism_used": True,
    }
    summary["gate"]["supports_conditioned_target_operator_design"] = (
        summary["gate"]["short_regime_pre_input_signal"]
        or summary["gate"]["late_segment_pre_input_signal"]
    ) and summary["gate"]["no_output_residual_mechanism_used"]
    return summary


def format_float(value: Any) -> str:
    return f"{float(value):.6f}" if finite(value) else "nan"


def top_rows(rows: list[dict[str, Any]], predicate_key: str, n: int = 8) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row.get(predicate_key) is True or row.get(predicate_key) == "True"]
    return sorted(
        candidates,
        key=lambda row: max(abs(float(row["standardized_mean_diff"])), float(row["rank_auc_abs"]) - 0.5),
        reverse=True,
    )[:n]


def write_report(path: Path, summary: dict[str, Any], short_rows: list[dict[str, Any]], late_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase3-B Regime/Segment Mechanism Diagnostic",
        "",
        "## Decision",
        "",
    ]
    if summary["gate"]["supports_conditioned_target_operator_design"]:
        lines.append("[Decision] pre-input regime/segment signals exist; continue to conditioned target-operator design, not output residual correction.")
    else:
        lines.append("[Decision] pre-input signals are weak; do not implement conditioned target operators yet.")
    lines.extend(["", "## Gate", ""])
    for key, value in summary["gate"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Known Short-Gap Feature Signals",
            "",
            "| Dataset | Horizon | Feature | AUC | SMD | Extra MSE | Aligned MSE |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in top_rows(short_rows, "is_known_short_gap"):
        lines.append(
            "| {dataset} | {horizon} | {feature} | {auc} | {smd} | {pos_mse} | {neg_mse} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                feature=row["feature"],
                auc=format_float(row["rank_auc_abs"]),
                smd=format_float(row["standardized_mean_diff"]),
                pos_mse=format_float(row["positive_group_mse"]),
                neg_mse=format_float(row["aligned_mse"]),
            )
        )
    lines.extend(
        [
            "",
            "## Known H720 Late-Gap Feature Signals",
            "",
            "| Dataset | Segment | Feature | AUC | SMD | High-error MSE | Other MSE |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in top_rows(late_rows, "is_known_h720_segment_gap"):
        lines.append(
            "| {dataset} | {segment} | {feature} | {auc} | {smd} | {pos_mse} | {neg_mse} |".format(
                dataset=row["dataset"],
                segment=row["segment"],
                feature=row["feature"],
                auc=format_float(row["rank_auc_abs"]),
                smd=format_float(row["standardized_mean_diff"]),
                pos_mse=format_float(row["positive_group_mse"]),
                neg_mse=format_float(row["aligned_mse"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "[Fact] Features in this diagnostic are computed only from historical input windows and window position. Prediction errors are used only as labels for analysis.",
            "",
            "[Decision Rule] A future model candidate may use these signals to condition target states or segment operators before output generation. It should not add a free output residual correction head.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose pre-input regime signals for Phase3-B.")
    parser.add_argument("--dataset-root", default="/Users/river/PaperResearch/Project/datasets")
    parser.add_argument(
        "--prediction-root",
        default="analysis/phase2_qdf_alignment_diagnostic_20260623/raw/PatchEncoderPrefixRiskWeighted",
    )
    parser.add_argument("--analysis-root", default="analysis/phase3_regime_segment_mechanism_20260624")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root = Path(args.analysis_root)
    analysis_root.mkdir(parents=True, exist_ok=True)
    short_rows = short_regime_rows(Path(args.dataset_root), Path(args.prediction_root))
    late_rows = h720_late_rows(Path(args.dataset_root), Path(args.prediction_root))
    summary = summarize(short_rows, late_rows)
    write_csv(analysis_root / "phase3_short_regime_preinput_features.csv", short_rows)
    write_csv(analysis_root / "phase3_h720_late_segment_preinput_features.csv", late_rows)
    (analysis_root / "phase3_regime_segment_mechanism_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(analysis_root / "phase3_regime_segment_mechanism_report.md", summary, short_rows, late_rows)
    print(f"phase3_regime_segment_mechanism_report={analysis_root / 'phase3_regime_segment_mechanism_report.md'}")


if __name__ == "__main__":
    main()
