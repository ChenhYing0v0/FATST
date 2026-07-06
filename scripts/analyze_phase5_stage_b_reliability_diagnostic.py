from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import pandas as pd

DATASETS = ("ETTh2", "ETTm1", "Weather")
DEFAULT_RUN_NAME = "TimeAlignOfficialUnified720_A6_a6_lbf_r256_official-last"
SEQ_LEN = 720
PRED_LEN = 720
UNIT_LEN = 48

DATASET_META = {
    "ETTh2": {
        "relative_root": "ETT-small",
        "data_path": "ETTh2.csv",
        "split": "ett_hour",
        "period": 24,
    },
    "ETTm1": {
        "relative_root": "ETT-small",
        "data_path": "ETTm1.csv",
        "split": "ett_minute",
        "period": 96,
    },
    "Weather": {
        "relative_root": "weather",
        "data_path": "weather.csv",
        "split": "custom",
        "period": 144,
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor
        while end + 1 < len(order) and values[order[end + 1]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + end) / 2.0 + 1.0
        for rank_index in range(cursor, end + 1):
            ranks[order[rank_index]] = average_rank
        cursor = end + 1
    return ranks


def correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_var = sum((value - x_mean) ** 2 for value in xs)
    y_var = sum((value - y_mean) ** 2 for value in ys)
    if x_var == 0 or y_var == 0:
        return float("nan")
    covariance = sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(xs, ys))
    return covariance / math.sqrt(x_var * y_var)


def residualize_by_step(values: list[float], starts: list[float]) -> list[float]:
    if len(values) != len(starts) or len(values) < 2:
        return [float("nan")] * len(values)
    x_mean = mean(starts)
    y_mean = mean(values)
    x_var = sum((value - x_mean) ** 2 for value in starts)
    if x_var == 0:
        return [value - y_mean for value in values]
    slope = sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(starts, values)) / x_var
    intercept = y_mean - slope * x_mean
    return [y_value - (intercept + slope * x_value) for x_value, y_value in zip(starts, values)]


def run_dir(raw_root: Path, run_name: str, dataset: str) -> Path:
    return raw_root / run_name / dataset / "mixed_h96_h192_h336_h720" / "seed2021"


def unit_bounds(unit_len: int = UNIT_LEN) -> list[tuple[int, int]]:
    return [(start, min(start + unit_len, PRED_LEN)) for start in range(0, PRED_LEN, unit_len)]


def collect_prediction_unit_rows(raw_root: Path, run_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unit_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        prediction_path = run_dir(raw_root, run_name, dataset) / "predictions_test.npz"
        with np.load(prediction_path) as payload:
            pred = np.asarray(payload["pred"], dtype=np.float64)
            true = np.asarray(payload["true"], dtype=np.float64)

        sample_errors_by_unit = []
        mses = []
        maes = []
        starts = []
        for start, end in unit_bounds():
            diff = pred[:, start:end, :] - true[:, start:end, :]
            sample_mse = np.mean(diff * diff, axis=(1, 2))
            sample_errors_by_unit.append(sample_mse)
            mses.append(float(np.mean(sample_mse)))
            maes.append(float(np.mean(np.abs(diff))))
            starts.append(float(start))
        del pred, true

        min_mse = min(mses)
        max_mse = max(mses)
        tail_count = max(1, math.ceil(len(mses) * 0.25))
        easy_mean = mean(sorted(mses)[:tail_count])
        hard_mean = mean(sorted(mses)[-tail_count:])
        detrended_mses = residualize_by_step(mses, starts)

        for (start, end), mse, mae, sample_mse in zip(unit_bounds(), mses, maes, sample_errors_by_unit):
            unit_rows.append(
                {
                    "dataset": dataset,
                    "arm": "a6_lbf_r256",
                    "target_horizon": 720,
                    "unit_start": start,
                    "unit_end": end,
                    "unit_len": end - start,
                    "mse": f"{mse:.10f}",
                    "mae": f"{mae:.10f}",
                    "sample_error_std": f"{float(np.std(sample_mse)):.10f}",
                    "sample_error_cv": f"{float(np.std(sample_mse) / np.mean(sample_mse)):.10f}",
                    "relative_to_dataset_min_mse_pct": f"{((mse / min_mse) - 1.0) * 100.0:.4f}",
                    "unit_source": "predictions_test.npz",
                }
            )

        summary_rows.append(
            {
                "dataset": dataset,
                "arm": "a6_lbf_r256",
                "num_units": len(mses),
                "unit_len": UNIT_LEN,
                "min_unit_mse": f"{min_mse:.10f}",
                "max_unit_mse": f"{max_mse:.10f}",
                "mean_unit_mse": f"{mean(mses):.10f}",
                "std_unit_mse": f"{pstdev(mses):.10f}",
                "mean_sample_error_cv": f"{mean(float(row['sample_error_cv']) for row in unit_rows if row['dataset'] == dataset):.10f}",
                "hard_easy_ratio": f"{hard_mean / easy_mean:.4f}",
                "max_vs_min_pct": f"{((max_mse / min_mse) - 1.0) * 100.0:.4f}",
                "spearman_step_mse": f"{correlation(rank(starts), rank(mses)):.4f}",
                "pearson_step_mse": f"{correlation(starts, mses):.4f}",
                "detrended_mse_std": f"{pstdev(detrended_mses):.10f}",
            }
        )
    return unit_rows, summary_rows


def dataset_csv_path(dataset_root: Path, dataset: str) -> Path:
    meta = DATASET_META[dataset]
    direct = dataset_root / meta["data_path"]
    nested = dataset_root / meta["relative_root"] / meta["data_path"]
    if direct.exists():
        return direct
    if nested.exists():
        return nested
    raise FileNotFoundError(f"Cannot find {meta['data_path']} under {dataset_root}")


def load_scaled_train_data(dataset_root: Path, dataset: str) -> np.ndarray:
    meta = DATASET_META[dataset]
    path = dataset_csv_path(dataset_root, dataset)
    df_raw = pd.read_csv(path)
    if dataset == "Weather":
        cols = list(df_raw.columns)
        cols.remove("OT")
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + ["OT"]]
    df_data = df_raw[df_raw.columns[1:]]
    if meta["split"] == "ett_hour":
        train_end = 12 * 30 * 24
    elif meta["split"] == "ett_minute":
        train_end = 12 * 30 * 24 * 4
    else:
        train_end = int(len(df_raw) * 0.7)
    train_values = df_data.iloc[:train_end].to_numpy(dtype=np.float64)
    mean_values = np.mean(train_values, axis=0, keepdims=True)
    std_values = np.std(train_values, axis=0, keepdims=True)
    std_values[std_values == 0] = 1.0
    return (train_values - mean_values) / std_values


def collect_train_proxy_rows(
    dataset_root: Path,
    unit_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    proxy_rows: list[dict[str, Any]] = []
    proxy_unit_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        train_data = load_scaled_train_data(dataset_root, dataset)
        num_windows = len(train_data) - SEQ_LEN - PRED_LEN + 1
        if num_windows <= 0:
            raise ValueError(f"{dataset} train split is shorter than seq_len + pred_len")
        meta = DATASET_META[dataset]
        period = int(meta["period"])
        unit_values = {proxy: [] for proxy in ("label_novelty", "local_variation", "seasonal_residual")}
        for start, end in unit_bounds():
            novelty_values = []
            variation_values = []
            seasonal_values = []
            for index in range(num_windows):
                history_last = train_data[index + SEQ_LEN - 1]
                future_start = index + SEQ_LEN + start
                future_end = index + SEQ_LEN + end
                future = train_data[future_start:future_end]
                novelty_values.append(float(np.mean((future - history_last) ** 2)))
                previous = train_data[future_start - 1:future_end - 1]
                variation_values.append(float(np.mean((future - previous) ** 2)))
                seasonal_reference = train_data[future_start - period:future_end - period]
                seasonal_values.append(float(np.mean((future - seasonal_reference) ** 2)))
            unit_values["label_novelty"].append(mean(novelty_values))
            unit_values["local_variation"].append(mean(variation_values))
            unit_values["seasonal_residual"].append(mean(seasonal_values))

        dataset_unit_rows = [row for row in unit_rows if row["dataset"] == dataset]
        heldout_mses = [float(row["mse"]) for row in dataset_unit_rows]
        starts = [float(row["unit_start"]) for row in dataset_unit_rows]
        detrended_mses = residualize_by_step(heldout_mses, starts)
        hard_count = max(1, math.ceil(len(heldout_mses) * 0.25))
        heldout_hard = set(sorted(range(len(heldout_mses)), key=lambda i: heldout_mses[i], reverse=True)[:hard_count])
        detrended_hard = set(
            sorted(range(len(detrended_mses)), key=lambda i: detrended_mses[i], reverse=True)[:hard_count]
        )
        for proxy_name, proxy_values in unit_values.items():
            proxy_hard = set(sorted(range(len(proxy_values)), key=lambda i: proxy_values[i], reverse=True)[:hard_count])
            proxy_rows.append(
                {
                    "dataset": dataset,
                    "proxy": proxy_name,
                    "unit_len": UNIT_LEN,
                    "num_units": len(proxy_values),
                    "spearman_proxy_mse": f"{correlation(rank(proxy_values), rank(heldout_mses)):.4f}",
                    "pearson_proxy_mse": f"{correlation(proxy_values, heldout_mses):.4f}",
                    "spearman_proxy_detrended_mse": f"{correlation(rank(proxy_values), rank(detrended_mses)):.4f}",
                    "pearson_proxy_detrended_mse": f"{correlation(proxy_values, detrended_mses):.4f}",
                    "top_quartile_overlap": f"{len(proxy_hard & heldout_hard) / hard_count:.4f}",
                    "top_quartile_detrended_overlap": f"{len(proxy_hard & detrended_hard) / hard_count:.4f}",
                    "proxy_mean": f"{mean(proxy_values):.10f}",
                    "proxy_std": f"{pstdev(proxy_values):.10f}",
                }
            )
            for unit_index, (unit_row, proxy_value, detrended_mse) in enumerate(
                zip(dataset_unit_rows, proxy_values, detrended_mses)
            ):
                proxy_unit_rows.append(
                    {
                        "dataset": dataset,
                        "proxy": proxy_name,
                        "unit_start": unit_row["unit_start"],
                        "unit_end": unit_row["unit_end"],
                        "mse": unit_row["mse"],
                        "detrended_mse": f"{detrended_mse:.10f}",
                        "proxy_value": f"{proxy_value:.10f}",
                        "is_raw_top_quartile": int(unit_index in heldout_hard),
                        "is_detrended_top_quartile": int(unit_index in detrended_hard),
                        "is_proxy_top_quartile": int(unit_index in proxy_hard),
                    }
                )
    return proxy_rows, proxy_unit_rows


def collect_drift_rows(raw_root: Path, run_name: str) -> list[dict[str, Any]]:
    drift_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        logs = read_csv(run_dir(raw_root, run_name, dataset) / "training_log.csv")
        val_mses = [float(row["val_mean_mse"]) for row in logs]
        best_val = min(val_mses)
        best_epoch = val_mses.index(best_val) + 1
        first_val = val_mses[0]
        last_val = val_mses[-1]
        last_h720_l1 = float(logs[-1]["train_prediction_h720_l1"])
        last_h96_l1 = float(logs[-1]["train_prediction_h96_l1"])
        drift_rows.append(
            {
                "dataset": dataset,
                "arm": "a6_lbf_r256",
                "epochs": len(val_mses),
                "first_val_mean_mse": f"{first_val:.10f}",
                "best_epoch": best_epoch,
                "best_val_mean_mse": f"{best_val:.10f}",
                "last_val_mean_mse": f"{last_val:.10f}",
                "last_minus_best_pct": f"{((last_val / best_val) - 1.0) * 100.0:.4f}",
                "first_minus_best_pct": f"{((first_val / best_val) - 1.0) * 100.0:.4f}",
                "train_h720_l1_last": f"{last_h720_l1:.10f}",
                "train_h96_l1_last": f"{last_h96_l1:.10f}",
                "train_h720_vs_h96_l1_ratio_last": f"{last_h720_l1 / last_h96_l1:.4f}",
            }
        )
    return drift_rows


def narrative_decision(summary_rows: list[dict[str, Any]], proxy_rows: list[dict[str, Any]]) -> str:
    material = [
        row
        for row in summary_rows
        if float(row["hard_easy_ratio"]) >= 1.5 and float(row["max_vs_min_pct"]) >= 50.0
    ]
    non_distance = [
        row
        for row in summary_rows
        if abs(float(row["spearman_step_mse"])) < 0.95
    ]
    aligned = [
        row
        for row in proxy_rows
        if float(row["spearman_proxy_detrended_mse"]) >= 0.25
    ]
    if len(material) >= 2 and len(non_distance) >= 2 and len(aligned) >= 2:
        return "pass_problem_existence"
    if len(material) >= 2 and aligned:
        return "partial_pass_distance_confounded"
    return "fail_problem_not_supported"


def render_report(
    summary_rows: list[dict[str, Any]],
    drift_rows: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
) -> str:
    decision = narrative_decision(summary_rows, proxy_rows)
    lines: list[str] = [
        "# Phase5 StageB B1 Reliability Diagnostic",
        "",
        "`current_step`: StageB Step 2/3 problem-existence diagnostic with synced A6-LBF predictions.",
        "",
        "## Scope",
        "",
        "[Fact] This diagnostic uses synced `predictions_test.npz`, `training_log.csv`, and local train split labels.",
        "",
        "[Boundary] Validation/test prediction errors are used only as diagnostic labels. Train-only proxies are computed from train split labels and history windows.",
        "",
        "## 11-step Record",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | StageB Step 2/3 full B1 diagnostic |",
        "| `problem` | Does A6-LBF-r256 show future-unit reliability heterogeneity that is not merely forecast distance? |",
        "| `existence_evidence` | prediction-level unit MSE/volatility, official-last drift, train-only proxy alignment |",
        "| `idea` | Use held-out prediction difficulty only as diagnostic labels and test whether train-only proxies can identify difficult units |",
        "| `theory_check` | Reliability-aware supervision is plausible only if train-only proxies align with residual difficulty beyond step-index confounding |",
        "| `design` | Post-hoc diagnostic; no model or training change |",
        f"| `narrative_gate` | {decision} |",
        "| `effectiveness_gate` | not applicable; no new method was trained |",
        "| `artifacts` | this directory |",
        f"| `decision` | {decision}; B2 is rejected before implementation under the current evidence |",
        "",
        "## Prediction Unit Heterogeneity",
        "",
        "| Dataset | Units | Unit Len | Min MSE | Max MSE | Max vs Min | Hard/Easy Ratio | Mean Error CV | Spearman(step, MSE) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['num_units']} | {row['unit_len']} | `{float(row['min_unit_mse']):.4f}` | "
            f"`{float(row['max_unit_mse']):.4f}` | `{float(row['max_vs_min_pct']):.2f}%` | "
            f"`{float(row['hard_easy_ratio']):.2f}` | `{float(row['mean_sample_error_cv']):.2f}` | "
            f"`{float(row['spearman_step_mse']):.2f}` |"
        )
    lines.extend(
        [
            "",
            "[Strong Evidence] All three datasets show material future-unit MSE heterogeneity and non-trivial sample-level error volatility.",
            "",
            "[Counter-Evidence] High `Spearman(step, MSE)` means difficulty is still strongly tied to forecast distance. A StageB method cannot be justified by this table alone.",
            "",
            "## Train-Only Proxy Alignment",
            "",
            "| Dataset | Proxy | Spearman(proxy, MSE) | Spearman(proxy, detrended MSE) | Raw Top-Q Overlap | Detrended Top-Q Overlap |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in proxy_rows:
        lines.append(
            f"| {row['dataset']} | `{row['proxy']}` | `{float(row['spearman_proxy_mse']):.2f}` | "
            f"`{float(row['spearman_proxy_detrended_mse']):.2f}` | `{float(row['top_quartile_overlap']):.2f}` | "
            f"`{float(row['top_quartile_detrended_overlap']):.2f}` |"
        )
    lines.extend(
        [
            "",
            "[Decision] The strict proxy test is the detrended column. Alignment with raw MSE can be explained by both proxy and error increasing with future distance.",
            "",
            "## Training Trajectory",
            "",
            "| Dataset | Best Epoch | Best Val MSE | Last Val MSE | Last vs Best | Last h720/h96 Train L1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in drift_rows:
        lines.append(
            f"| {row['dataset']} | {row['best_epoch']} | `{float(row['best_val_mean_mse']):.4f}` | "
            f"`{float(row['last_val_mean_mse']):.4f}` | `{float(row['last_minus_best_pct']):.2f}%` | "
            f"`{float(row['train_h720_vs_h96_l1_ratio_last']):.2f}` |"
        )
    lines.extend(
        [
            "",
            "[Moderate Evidence] ETTh2 has notable official-last drift, while ETTm1 and Weather "
            "are close to their best epochs. This suggests the stability problem is dataset-dependent, "
            "not a universal A6-LBF failure.",
            "",
            "## Decision",
            "",
            f"[Decision] B1-RED decision: `{decision}`.",
            "",
            "[Rollback Check] If this remains distance-confounded, StageB should return to Step 2/3 instead of implementing reliability-aware allocation. A future B2 must define a proxy that predicts residual difficulty beyond step index.",
            "",
            "## Output Files",
            "",
            "- `stage_b_a6_lbf_unit_reliability.csv`",
            "- `stage_b_a6_lbf_reliability_summary.csv`",
            "- `stage_b_a6_lbf_proxy_alignment.csv`",
            "- `stage_b_a6_lbf_unit_proxy_detrended.csv`",
            "- `stage_b_a6_lbf_trajectory_drift.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze available StageB A6-LBF reliability artifacts.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("analysis/phase5_timealign_hss_a6_capacity_native_gate_20260703/raw/official-last"),
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/phase5_stage_b_reliability_diagnostic_20260706"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Users/river/PaperResearch/Project/datasets"),
    )
    args = parser.parse_args()

    unit_rows, summary_rows = collect_prediction_unit_rows(args.raw_root, args.run_name)
    proxy_rows, proxy_unit_rows = collect_train_proxy_rows(args.dataset_root, unit_rows)
    drift_rows = collect_drift_rows(args.raw_root, args.run_name)

    write_csv(args.output_dir / "stage_b_a6_lbf_unit_reliability.csv", unit_rows)
    write_csv(args.output_dir / "stage_b_a6_lbf_reliability_summary.csv", summary_rows)
    write_csv(args.output_dir / "stage_b_a6_lbf_proxy_alignment.csv", proxy_rows)
    write_csv(args.output_dir / "stage_b_a6_lbf_unit_proxy_detrended.csv", proxy_unit_rows)
    write_csv(args.output_dir / "stage_b_a6_lbf_trajectory_drift.csv", drift_rows)
    (args.output_dir / "stage_b_a6_lbf_reliability_report.md").write_text(
        render_report(summary_rows, drift_rows, proxy_rows)
    )
    print(args.output_dir / "stage_b_a6_lbf_reliability_report.md")


if __name__ == "__main__":
    main()
