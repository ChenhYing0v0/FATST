from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from statistics import mean

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="fatst-mpl-"))

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "baselines" / "patch_encoder_target_set_decoder"
sys.path.insert(0, str(BASELINE_ROOT))

from dataset import ForecastDataset  # noqa: E402


DATASETS = ["ETTh2", "ETTm1", "Weather"]
HORIZONS = [96, 192, 336, 720]
SEGMENTS = [(1, 96), (97, 192), (193, 336), (337, 720)]
SEGMENT_LABELS = [f"{start}-{end}" for start, end in SEGMENTS]


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


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


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


def rank(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        avg_rank = (index + end - 1) / 2.0
        for original_index, _ in ordered[index:end]:
            ranks[original_index] = avg_rank
        index = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    return pearson(rank(left), rank(right))


def expected_uniform_step_pressure(max_pred_len: int, target_horizons: list[int]) -> list[float]:
    pressure = []
    for step in range(1, max_pred_len + 1):
        coeff = 0.0
        for horizon in target_horizons:
            if step <= horizon:
                coeff += 1.0 / float(horizon)
        pressure.append(coeff / float(len(target_horizons)))
    return pressure


def prefix_risk_step_weights(max_pred_len: int, alpha: float) -> list[float]:
    raw = [(step / float(max_pred_len)) ** (-alpha) for step in range(1, max_pred_len + 1)]
    scale = mean(raw)
    return [value / scale for value in raw]


def pressure_share_maps(max_pred_len: int, alpha: float) -> dict[str, dict[str, float]]:
    uniform_pressure = expected_uniform_step_pressure(max_pred_len, HORIZONS)
    prefix_weights = prefix_risk_step_weights(max_pred_len, alpha)
    prefix_pressure = [
        value * prefix_weights[index]
        for index, value in enumerate(uniform_pressure)
    ]
    uniform_total = sum(uniform_pressure)
    prefix_total = sum(prefix_pressure)
    maps: dict[str, dict[str, float]] = {}
    for start, end in SEGMENTS:
        label = f"{start}-{end}"
        uniform_share = sum(uniform_pressure[start - 1 : end]) / uniform_total
        prefix_share = sum(prefix_pressure[start - 1 : end]) / prefix_total
        maps[label] = {
            "uniform_pressure_share": uniform_share,
            "prefix_pressure_share": prefix_share,
            "region_balanced_share": 1.0 / float(len(SEGMENTS)),
        }
    return maps


def region_mean_series(data: np.ndarray, seq_len: int, pred_len: int) -> dict[str, np.ndarray]:
    n_windows = len(data) - seq_len - pred_len + 1
    if n_windows <= 0:
        raise ValueError("Dataset split is shorter than seq_len + pred_len.")

    cumsum = np.vstack([np.zeros((1, data.shape[1]), dtype=np.float64), data.cumsum(axis=0, dtype=np.float64)])
    means: dict[str, np.ndarray] = {}
    window_index = np.arange(n_windows)
    target_start = window_index + seq_len
    for start, end in SEGMENTS:
        label = f"{start}-{end}"
        start_index = target_start + start - 1
        end_index = target_start + end
        region_sum = cumsum[end_index] - cumsum[start_index]
        means[label] = region_sum / float(end - start + 1)
    return means


def region_value_stats(data: np.ndarray, seq_len: int, pred_len: int) -> dict[str, dict[str, float]]:
    n_windows = len(data) - seq_len - pred_len + 1
    cumsum = np.vstack([np.zeros((1, data.shape[1]), dtype=np.float64), data.cumsum(axis=0, dtype=np.float64)])
    cumsum_sq = np.vstack(
        [
            np.zeros((1, data.shape[1]), dtype=np.float64),
            np.square(data, dtype=np.float64).cumsum(axis=0, dtype=np.float64),
        ]
    )
    stats: dict[str, dict[str, float]] = {}
    window_index = np.arange(n_windows)
    target_start = window_index + seq_len
    for start, end in SEGMENTS:
        label = f"{start}-{end}"
        start_index = target_start + start - 1
        end_index = target_start + end
        region_sum = cumsum[end_index] - cumsum[start_index]
        region_sum_sq = cumsum_sq[end_index] - cumsum_sq[start_index]
        count = float(n_windows * (end - start + 1) * data.shape[1])
        value_mean = float(region_sum.sum() / count)
        value_mean_sq = float(region_sum_sq.sum() / count)
        value_var = max(value_mean_sq - value_mean * value_mean, 0.0)
        stats[label] = {
            "target_value_mean": value_mean,
            "target_value_std": math.sqrt(value_var),
            "target_value_rms": math.sqrt(max(value_mean_sq, 0.0)),
        }
    return stats


def correlation_squared(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    left_centered = left_flat - left_flat.mean()
    right_centered = right_flat - right_flat.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0.0:
        return 0.0
    corr = float(np.dot(left_centered, right_centered) / denominator)
    return corr * corr


def dataset_novelty_rows(
    dataset_root: Path,
    dataset: str,
    seq_len: int,
    pred_len: int,
    pressure_maps: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    train_set = ForecastDataset(dataset_root, dataset, "train", seq_len, pred_len)
    data = train_set.data.astype(np.float64)
    means = region_mean_series(data, seq_len, pred_len)
    value_stats = region_value_stats(data, seq_len, pred_len)
    raw_rows: list[dict[str, object]] = []
    for region_index, label in enumerate(SEGMENT_LABELS):
        current = means[label]
        current_centered = current - current.mean(axis=0, keepdims=True)
        pooled_region_mean_std = float(np.sqrt(np.mean(current_centered * current_centered)))
        prev_r2 = []
        for prev_label in SEGMENT_LABELS[:region_index]:
            prev_r2.append(correlation_squared(current, means[prev_label]))
        max_prev_r2 = max(prev_r2) if prev_r2 else 0.0
        transition_rms = 0.0
        if region_index > 0:
            prev = means[SEGMENT_LABELS[region_index - 1]]
            transition_rms = float(np.sqrt(np.mean((current - prev) ** 2)))
        novelty = pooled_region_mean_std * (1.0 - max_prev_r2)
        raw_rows.append(
            {
                "dataset": dataset,
                "segment": label,
                "region_index": region_index,
                "train_windows": len(train_set),
                "channels": data.shape[1],
                "pooled_region_mean_std": pooled_region_mean_std,
                "target_value_std": value_stats[label]["target_value_std"],
                "target_value_rms": value_stats[label]["target_value_rms"],
                "max_prev_region_r2": max_prev_r2,
                "novelty_score": novelty,
                "transition_mean_rms": transition_rms,
                **pressure_maps[label],
            }
        )

    total_novelty = sum(float(row["novelty_score"]) for row in raw_rows)
    for row in raw_rows:
        novelty_share = float(row["novelty_score"]) / max(total_novelty, 1e-12)
        row["novelty_share"] = novelty_share
        row["novelty_minus_region_balanced_share"] = novelty_share - float(row["region_balanced_share"])
        row["prefix_minus_novelty_share"] = float(row["prefix_pressure_share"]) - novelty_share
    return raw_rows


def keyed(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def align_effect_rows(
    novelty_rows: list[dict[str, object]],
    objective_root: Path,
    region_balanced_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    novelty = keyed(
        [{key: str(value) for key, value in row.items()} for row in novelty_rows],
        ("dataset", "segment"),
    )
    r3_segment_rows: list[dict[str, object]] = []
    for row in read_csv(objective_root / "r3_vs_uniform_segments.csv"):
        key = (row["dataset"], row["segment"])
        if key not in novelty:
            continue
        novelty_row = novelty[key]
        r3_segment_rows.append(
            {
                "dataset": row["dataset"],
                "horizon": int(row["horizon"]),
                "segment": row["segment"],
                "r3_vs_uniform_mse_pct": float(row["r3_vs_uniform_mse_pct"]),
                "r3_vs_fixed_mse_pct": float(row["r3_vs_fixed_mse_pct"]),
                "novelty_score": float(novelty_row["novelty_score"]),
                "novelty_share": float(novelty_row["novelty_share"]),
                "prefix_pressure_share": float(novelty_row["prefix_pressure_share"]),
                "uniform_pressure_share": float(novelty_row["uniform_pressure_share"]),
            }
        )

    region_rows: list[dict[str, object]] = []
    for row in read_csv(region_balanced_root / "phase2_region_balanced_h720_regions_vs_r3.csv"):
        key = (row["dataset"], row["segment"])
        novelty_row = novelty[key]
        region_rows.append(
            {
                "dataset": row["dataset"],
                "segment": row["segment"],
                "region_balanced_vs_r3_mse_pct": float(row["relative_mse_vs_r3_pct"]),
                "region_balanced_wins_mse_vs_r3": row["candidate_wins_mse_vs_r3"] == "True",
                "novelty_score": float(novelty_row["novelty_score"]),
                "novelty_share": float(novelty_row["novelty_share"]),
                "novelty_minus_region_balanced_share": float(
                    novelty_row["novelty_minus_region_balanced_share"]
                ),
                "prefix_minus_novelty_share": float(novelty_row["prefix_minus_novelty_share"]),
                "prefix_pressure_share": float(novelty_row["prefix_pressure_share"]),
                "region_balanced_share": float(novelty_row["region_balanced_share"]),
            }
        )

    aggregate_rows: list[dict[str, object]] = []
    by_region: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in r3_segment_rows:
        by_region.setdefault((str(row["dataset"]), str(row["segment"])), []).append(row)
    region_lookup = {(str(row["dataset"]), str(row["segment"])): row for row in region_rows}
    for key, rows in sorted(by_region.items()):
        rb_row = region_lookup.get(key)
        if rb_row is None:
            continue
        first = rows[0]
        aggregate_rows.append(
            {
                "dataset": key[0],
                "segment": key[1],
                "r3_vs_uniform_mean_mse_pct": mean(float(row["r3_vs_uniform_mse_pct"]) for row in rows),
                "r3_vs_uniform_min_mse_pct": min(float(row["r3_vs_uniform_mse_pct"]) for row in rows),
                "r3_vs_uniform_max_mse_pct": max(float(row["r3_vs_uniform_mse_pct"]) for row in rows),
                "region_balanced_vs_r3_h720_mse_pct": rb_row["region_balanced_vs_r3_mse_pct"],
                "novelty_score": first["novelty_score"],
                "novelty_share": first["novelty_share"],
                "prefix_pressure_share": first["prefix_pressure_share"],
                "uniform_pressure_share": first["uniform_pressure_share"],
                "novelty_minus_region_balanced_share": rb_row["novelty_minus_region_balanced_share"],
                "prefix_minus_novelty_share": rb_row["prefix_minus_novelty_share"],
            }
        )
    return r3_segment_rows, region_rows, aggregate_rows


def summarize(
    novelty_rows: list[dict[str, object]],
    r3_segment_rows: list[dict[str, object]],
    region_rows: list[dict[str, object]],
    aggregate_rows: list[dict[str, object]],
) -> dict[str, object]:
    r3_delta = [float(row["r3_vs_uniform_mse_pct"]) for row in r3_segment_rows]
    r3_novelty = [float(row["novelty_share"]) for row in r3_segment_rows]
    r3_prefix = [float(row["prefix_pressure_share"]) for row in r3_segment_rows]
    rb_delta = [float(row["region_balanced_vs_r3_mse_pct"]) for row in region_rows]
    rb_novelty_deficit = [float(row["novelty_minus_region_balanced_share"]) for row in region_rows]
    rb_prefix_gap = [float(row["prefix_minus_novelty_share"]) for row in region_rows]
    agg_r3_delta = [float(row["r3_vs_uniform_mean_mse_pct"]) for row in aggregate_rows]
    agg_rb_delta = [float(row["region_balanced_vs_r3_h720_mse_pct"]) for row in aggregate_rows]
    agg_novelty = [float(row["novelty_share"]) for row in aggregate_rows]
    agg_deficit = [float(row["novelty_minus_region_balanced_share"]) for row in aggregate_rows]

    dataset_rows = {}
    for dataset in DATASETS:
        subset = [row for row in novelty_rows if row["dataset"] == dataset]
        max_row = max(subset, key=lambda row: float(row["novelty_share"]))
        early_row = next(row for row in subset if row["segment"] == "1-96")
        late_row = next(row for row in subset if row["segment"] == "337-720")
        dataset_rows[dataset] = {
            "max_novelty_segment": max_row["segment"],
            "max_novelty_share": float(max_row["novelty_share"]),
            "early_novelty_share": float(early_row["novelty_share"]),
            "late_novelty_share": float(late_row["novelty_share"]),
            "early_prefix_minus_novelty_share": float(early_row["prefix_minus_novelty_share"]),
            "late_prefix_minus_novelty_share": float(late_row["prefix_minus_novelty_share"]),
        }

    correlations = {
        "r3_segment_delta_vs_novelty_share_pearson": pearson(r3_novelty, r3_delta),
        "r3_segment_delta_vs_novelty_share_spearman": spearman(r3_novelty, r3_delta),
        "r3_segment_delta_vs_prefix_pressure_share_pearson": pearson(r3_prefix, r3_delta),
        "region_balanced_delta_vs_novelty_deficit_pearson": pearson(rb_novelty_deficit, rb_delta),
        "region_balanced_delta_vs_novelty_deficit_spearman": spearman(rb_novelty_deficit, rb_delta),
        "region_balanced_delta_vs_prefix_minus_novelty_pearson": pearson(rb_prefix_gap, rb_delta),
        "aggregate_r3_delta_vs_novelty_share_pearson": pearson(agg_novelty, agg_r3_delta),
        "aggregate_region_balanced_delta_vs_novelty_deficit_pearson": pearson(
            agg_deficit,
            agg_rb_delta,
        ),
    }
    gate = {
        "r3_gain_aligns_with_novelty": correlations["r3_segment_delta_vs_novelty_share_pearson"] <= -0.35,
        "region_balanced_loss_aligns_with_novelty_deficit": correlations[
            "region_balanced_delta_vs_novelty_deficit_pearson"
        ]
        >= 0.35,
        "aggregate_patterns_align": correlations["aggregate_r3_delta_vs_novelty_share_pearson"] <= -0.35
        and correlations["aggregate_region_balanced_delta_vs_novelty_deficit_pearson"] >= 0.35,
    }
    gate["supports_step_covariance_balanced"] = all(gate.values())
    return {
        "datasets": dataset_rows,
        "correlations": correlations,
        "gate": gate,
    }


def make_scatter_plot(output_root: Path, aggregate_rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"ETTh2": "#1f77b4", "ETTm1": "#d62728", "Weather": "#2ca02c"}
    for row in aggregate_rows:
        dataset = str(row["dataset"])
        label = f"{dataset} {row['segment']}"
        axes[0].scatter(
            float(row["novelty_share"]),
            float(row["r3_vs_uniform_mean_mse_pct"]),
            color=colors.get(dataset, "#333333"),
        )
        axes[0].annotate(label, (float(row["novelty_share"]), float(row["r3_vs_uniform_mean_mse_pct"])), fontsize=7)
        axes[1].scatter(
            float(row["novelty_minus_region_balanced_share"]),
            float(row["region_balanced_vs_r3_h720_mse_pct"]),
            color=colors.get(dataset, "#333333"),
        )
        axes[1].annotate(
            label,
            (
                float(row["novelty_minus_region_balanced_share"]),
                float(row["region_balanced_vs_r3_h720_mse_pct"]),
            ),
            fontsize=7,
        )
    axes[0].axhline(0.0, color="#888888", linewidth=0.8)
    axes[0].set_xlabel("novelty share")
    axes[0].set_ylabel("R.3 vs uniform MSE delta (%)")
    axes[0].set_title("R.3 effect")
    axes[1].axhline(0.0, color="#888888", linewidth=0.8)
    axes[1].axvline(0.0, color="#888888", linewidth=0.8)
    axes[1].set_xlabel("novelty share - 0.25")
    axes[1].set_ylabel("region-balanced vs R.3 MSE delta (%)")
    axes[1].set_title("Region-balanced failure")
    fig.tight_layout()
    fig.savefig(output_root / "novelty_effect_alignment.png", dpi=180)
    plt.close(fig)


def write_report(output_root: Path, summary: dict[str, object], aggregate_rows: list[dict[str, object]]) -> None:
    correlations = summary["correlations"]
    gate = summary["gate"]
    datasets = summary["datasets"]
    assert isinstance(correlations, dict)
    assert isinstance(gate, dict)
    assert isinstance(datasets, dict)
    decision = (
        "[Decision] Offline covariance/novelty diagnostic supports entering "
        "`step_covariance_balanced` step 4-6."
        if gate["supports_step_covariance_balanced"]
        else "[Decision] Offline covariance/novelty diagnostic does not yet support "
        "implementing `step_covariance_balanced` as the next training candidate."
    )
    lines = [
        "# Phase2-C.1 Covariance / Novelty Diagnostic",
        "",
        "## Decision",
        "",
        decision,
        "",
        "[Fact] This diagnostic uses only train-split target windows and completed",
        "Phase2 analysis artifacts. It does not train a model and does not use future",
        "labels at inference.",
        "",
        "## What Was Computed",
        "",
        "- Dataset split and scaling follow `ForecastDataset`: the scaler is fit on",
        "  the train split, then train targets are analyzed in normalized space.",
        "- Regions are `1-96`, `97-192`, `193-336`, and `337-720` for H720 target",
        "  windows with `seq_len=336`.",
        "- `pooled_region_mean_std`: RMS variation of each window's region mean",
        "  after channel-wise centering.",
        "- `max_prev_region_r2`: largest squared correlation between the current",
        "  region mean and any earlier region mean.",
        "- `novelty_score = pooled_region_mean_std * (1 - max_prev_region_r2)`.",
        "- `novelty_share`: dataset-local normalized novelty across the four regions.",
        "",
        "## Dataset Novelty Summary",
        "",
        "| Dataset | Max novelty region | Max share | Early share | Late share | Early prefix minus novelty | Late prefix minus novelty |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        row = datasets[dataset]
        assert isinstance(row, dict)
        lines.append(
            "| "
            f"{dataset} | {row['max_novelty_segment']} | "
            f"{float(row['max_novelty_share']):.4f} | "
            f"{float(row['early_novelty_share']):.4f} | "
            f"{float(row['late_novelty_share']):.4f} | "
            f"{float(row['early_prefix_minus_novelty_share']):+.4f} | "
            f"{float(row['late_prefix_minus_novelty_share']):+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Effect Alignment",
            "",
            "| Correlation | Value | Interpretation |",
            "| --- | ---: | --- |",
            "| R.3 segment delta vs novelty share | "
            f"{format_float(float(correlations['r3_segment_delta_vs_novelty_share_pearson']))} | "
            "negative means higher novelty aligns with larger R.3 MSE reduction |",
            "| R.3 segment delta vs prefix pressure share | "
            f"{format_float(float(correlations['r3_segment_delta_vs_prefix_pressure_share_pearson']))} | "
            "negative means R.3's prefix pressure aligns with MSE reduction |",
            "| region-balanced delta vs novelty deficit | "
            f"{format_float(float(correlations['region_balanced_delta_vs_novelty_deficit_pearson']))} | "
            "positive means underweighting high-novelty regions aligns with loss |",
            "| aggregate R.3 delta vs novelty share | "
            f"{format_float(float(correlations['aggregate_r3_delta_vs_novelty_share_pearson']))} | "
            "region-averaged version of the first test |",
            "| aggregate region-balanced delta vs novelty deficit | "
            f"{format_float(float(correlations['aggregate_region_balanced_delta_vs_novelty_deficit_pearson']))} | "
            "region-averaged version of the failure test |",
            "",
            "## Region-Level Alignment Table",
            "",
            "| Dataset | Segment | Novelty share | R.3 vs uniform mean MSE | Region-balanced vs R.3 H720 MSE |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| "
            f"{row['dataset']} | {row['segment']} | "
            f"{float(row['novelty_share']):.4f} | "
            f"{format_pct(float(row['r3_vs_uniform_mean_mse_pct']))} | "
            f"{format_pct(float(row['region_balanced_vs_r3_h720_mse_pct']))} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- r3_gain_aligns_with_novelty: `{gate['r3_gain_aligns_with_novelty']}`",
            "- region_balanced_loss_aligns_with_novelty_deficit: "
            f"`{gate['region_balanced_loss_aligns_with_novelty_deficit']}`",
            f"- aggregate_patterns_align: `{gate['aggregate_patterns_align']}`",
            f"- supports_step_covariance_balanced: `{gate['supports_step_covariance_balanced']}`",
            "",
            "## Decision Impact",
            "",
        ]
    )
    if gate["supports_step_covariance_balanced"]:
        lines.extend(
            [
                "[Inference] The next step can move to loop step 4-6 and define a",
                "`step_covariance_balanced` objective. The design must keep the",
                "diagnostic separation between coverage balance and novelty balance,",
                "then test against R.3 rather than only against uniform target-set.",
            ]
        )
    else:
        lines.extend(
            [
                "[Inference] The current static novelty proxy is not strong enough to",
                "justify another remote objective candidate. The objective-only route",
                "should pause unless a sharper source-grounded novelty definition is",
                "introduced; otherwise return to base architecture or external baseline",
                "selection.",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `region_novelty_stats.csv`",
            "- `r3_novelty_effect_alignment.csv`",
            "- `region_balanced_novelty_effect_alignment.csv`",
            "- `aggregate_novelty_effect_alignment.csv`",
            "- `covariance_novelty_summary.json`",
            "- `novelty_effect_alignment.png`",
        ]
    )
    (output_root / "phase2_covariance_novelty_diagnostic_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/Users/river/PaperResearch/Project/datasets")
    parser.add_argument("--objective-root", default="analysis/phase2_objective_pressure_diagnostic_20260623")
    parser.add_argument("--region-balanced-root", default="analysis/phase2_region_balanced_gate_20260623")
    parser.add_argument("--output-root", default="analysis/phase2_covariance_novelty_diagnostic_20260623")
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--max-pred-len", type=int, default=720)
    parser.add_argument("--step-loss-alpha", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pressure_maps = pressure_share_maps(args.max_pred_len, args.step_loss_alpha)

    novelty_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        novelty_rows.extend(
            dataset_novelty_rows(
                Path(args.dataset_root),
                dataset,
                args.seq_len,
                args.max_pred_len,
                pressure_maps,
            )
        )

    r3_segment_rows, region_rows, aggregate_rows = align_effect_rows(
        novelty_rows,
        Path(args.objective_root),
        Path(args.region_balanced_root),
    )
    summary = summarize(novelty_rows, r3_segment_rows, region_rows, aggregate_rows)
    write_csv(output_root / "region_novelty_stats.csv", novelty_rows)
    write_csv(output_root / "r3_novelty_effect_alignment.csv", r3_segment_rows)
    write_csv(output_root / "region_balanced_novelty_effect_alignment.csv", region_rows)
    write_csv(output_root / "aggregate_novelty_effect_alignment.csv", aggregate_rows)
    (output_root / "covariance_novelty_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    make_scatter_plot(output_root, aggregate_rows)
    write_report(output_root, summary, aggregate_rows)
    print(f"report={output_root / 'phase2_covariance_novelty_diagnostic_report.md'}")


if __name__ == "__main__":
    main()
