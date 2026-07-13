from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "baselines" / "patch_encoder_target_set_decoder"
sys.path.insert(0, str(BASELINE_ROOT))

from dataset import ForecastDataset  # noqa: E402


DATASETS = ["ETTh2", "ETTm1", "Weather"]
REGIONS = [(1, 96), (97, 192), (193, 336), (337, 720)]
TOP_KS = [1, 2, 4, 8, 16, 32, 64, 128]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def target_rows_chunk(data: np.ndarray, indices: np.ndarray, seq_len: int, pred_len: int) -> np.ndarray:
    targets = np.stack([data[index + seq_len : index + seq_len + pred_len] for index in indices], axis=0)
    return targets.transpose(0, 2, 1).reshape(-1, pred_len).astype(np.float64, copy=False)


def estimate_step_covariance(
    data: np.ndarray,
    seq_len: int,
    pred_len: int,
    chunk_windows: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_windows = len(data) - seq_len - pred_len + 1
    if n_windows <= 0:
        raise ValueError("Dataset split is shorter than seq_len + pred_len.")

    total_rows = 0
    step_sum = np.zeros(pred_len, dtype=np.float64)
    step_cross = np.zeros((pred_len, pred_len), dtype=np.float64)
    for start in range(0, n_windows, chunk_windows):
        stop = min(start + chunk_windows, n_windows)
        rows = target_rows_chunk(data, np.arange(start, stop), seq_len, pred_len)
        step_sum += rows.sum(axis=0)
        step_cross += rows.T @ rows
        total_rows += rows.shape[0]

    mean = step_sum / float(total_rows)
    covariance = step_cross / float(total_rows) - np.outer(mean, mean)
    covariance = (covariance + covariance.T) * 0.5
    return covariance, mean, total_rows


def eigensystem(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    return eigenvalues, eigenvectors


def effective_rank(eigenvalues: np.ndarray) -> float:
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return 0.0
    probs = eigenvalues / total
    probs = probs[probs > 0]
    return float(np.exp(-np.sum(probs * np.log(probs))))


def covariance_summary_rows(dataset: str, covariance: np.ndarray, eigenvalues: np.ndarray) -> dict[str, Any]:
    diag = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    corr = covariance / np.outer(diag, diag)
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
    total = float(eigenvalues.sum())
    row: dict[str, Any] = {
        "dataset": dataset,
        "pred_len": covariance.shape[0],
        "total_variance": total,
        "effective_rank": effective_rank(eigenvalues),
        "mean_abs_offdiag_corr": float(np.mean(np.abs(off_diag))),
        "median_abs_offdiag_corr": float(np.median(np.abs(off_diag))),
        "share_abs_corr_gt_0_25": float(np.mean(np.abs(off_diag) > 0.25)),
        "share_abs_corr_gt_0_50": float(np.mean(np.abs(off_diag) > 0.50)),
    }
    for top_k in TOP_KS:
        active = min(top_k, len(eigenvalues))
        row[f"top{top_k}_variance_ratio"] = float(eigenvalues[:active].sum() / total) if total > 0 else 0.0
    return row


def component_rows(dataset: str, eigenvalues: np.ndarray, eigenvectors: np.ndarray, n_components: int) -> list[dict[str, Any]]:
    total = float(eigenvalues.sum())
    rows: list[dict[str, Any]] = []
    for component_index in range(min(n_components, len(eigenvalues))):
        vector = eigenvectors[:, component_index]
        energy = vector * vector
        row: dict[str, Any] = {
            "dataset": dataset,
            "component": component_index + 1,
            "eigenvalue": float(eigenvalues[component_index]),
            "variance_ratio": float(eigenvalues[component_index] / total) if total > 0 else 0.0,
            "peak_step": int(np.argmax(np.abs(vector)) + 1),
            "signed_sum": float(vector.sum()),
            "total_abs_loading": float(np.sum(np.abs(vector))),
        }
        for start, end in REGIONS:
            row[f"energy_{start}_{end}"] = float(energy[start - 1 : end].sum())
        rows.append(row)
    return rows


def horizon_contribution_rows(dataset: str, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> list[dict[str, Any]]:
    total_energy = eigenvalues[None, :] * (eigenvectors * eigenvectors)
    rows: list[dict[str, Any]] = []
    for start, end in REGIONS:
        region_energy = total_energy[start - 1 : end, :].sum(axis=0)
        denominator = float(region_energy.sum())
        row: dict[str, Any] = {
            "dataset": dataset,
            "region": f"{start}-{end}",
            "region_total_variance": denominator,
        }
        for top_k in TOP_KS:
            active = min(top_k, len(eigenvalues))
            row[f"top{top_k}_region_variance_ratio"] = (
                float(region_energy[:active].sum() / denominator) if denominator > 0 else 0.0
            )
        rows.append(row)
    return rows


def analyze_dataset(
    dataset_root: Path,
    dataset: str,
    seq_len: int,
    pred_len: int,
    chunk_windows: int,
    n_components: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    train_set = ForecastDataset(dataset_root, dataset, "train", seq_len, pred_len)
    covariance, _, total_rows = estimate_step_covariance(
        train_set.data.astype(np.float64),
        seq_len,
        pred_len,
        chunk_windows,
    )
    eigenvalues, eigenvectors = eigensystem(covariance)
    summary = covariance_summary_rows(dataset, covariance, eigenvalues)
    summary["train_rows_used"] = total_rows
    summary["train_windows"] = len(train_set)
    summary["channels"] = train_set.data.shape[1]
    components = component_rows(dataset, eigenvalues, eigenvectors, n_components)
    horizons = horizon_contribution_rows(dataset, eigenvalues, eigenvectors)
    return summary, components, horizons


def write_report(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase4 Label Basis Audit",
        "",
        "## Purpose",
        "",
        "[Fact] This diagnostic uses train split future labels only. It does not train a model and does not inspect validation/test labels.",
        "",
        "[Question] Is there enough nontrivial future-label covariance / low-rank component structure to justify horizon-agnostic component-space supervision before implementing a training loss?",
        "",
        "## Metrics",
        "",
        "- `effective_rank`: entropy rank of the step covariance eigenvalue distribution; smaller than `pred_len` means label variation concentrates in fewer components.",
        "- `topK_variance_ratio`: cumulative variance captured by the top K eigen-components.",
        "- `mean_abs_offdiag_corr`: average absolute step-to-step label correlation outside the diagonal.",
        "- `share_abs_corr_gt_0_25`: fraction of off-diagonal step correlations whose absolute value exceeds `0.25`.",
        "",
        "## Summary",
        "",
        "| Dataset | Effective rank | Top16 var | Top32 var | Top64 var | Mean abs offdiag corr | Share abs corr > 0.25 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {rank:.2f} | {top16:.3f} | {top32:.3f} | {top64:.3f} | {corr:.3f} | {share:.3f} |".format(
                dataset=row["dataset"],
                rank=row["effective_rank"],
                top16=row["top16_variance_ratio"],
                top32=row["top32_variance_ratio"],
                top64=row["top64_variance_ratio"],
                corr=row["mean_abs_offdiag_corr"],
                share=row["share_abs_corr_gt_0_25"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "[Decision] Candidate C remains viable if top components capture meaningful variance and off-diagonal correlations are nontrivial. If the covariance is close to diagonal/full-rank, component supervision should be deprioritized in favor of random interval supervision.",
            "",
            "## Current Interpretation",
            "",
        ]
    )
    if all(float(row["top64_variance_ratio"]) >= 0.5 for row in summary_rows) and all(
        float(row["mean_abs_offdiag_corr"]) >= 0.1 for row in summary_rows
    ):
        lines.append(
            "[Strong Evidence] The train-label covariance is structured enough to justify the next residual-projection diagnostic for component-space supervision."
        )
    else:
        lines.append(
            "[Inference] The label basis evidence is mixed; inspect component-region rows before deciding whether to implement component-space training."
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit future-label component basis for horizon-agnostic supervision.")
    parser.add_argument("--dataset-root", default="/Users/river/PaperResearch/Project/datasets")
    parser.add_argument("--datasets", default="ETTh2,ETTm1,Weather")
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, default=720)
    parser.add_argument("--chunk-windows", type=int, default=512)
    parser.add_argument("--n-components", type=int, default=16)
    parser.add_argument("--analysis-root", default="analysis/phase4_label_basis_audit_20260624")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root = Path(args.analysis_root)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    summary_rows: list[dict[str, Any]] = []
    component_rows_all: list[dict[str, Any]] = []
    horizon_rows_all: list[dict[str, Any]] = []
    for dataset in datasets:
        summary, components, horizons = analyze_dataset(
            Path(args.dataset_root),
            dataset,
            args.seq_len,
            args.pred_len,
            args.chunk_windows,
            args.n_components,
        )
        summary_rows.append(summary)
        component_rows_all.extend(components)
        horizon_rows_all.extend(horizons)

    analysis_root.mkdir(parents=True, exist_ok=True)
    write_csv(analysis_root / "phase4_label_basis_summary.csv", summary_rows)
    write_csv(analysis_root / "phase4_label_basis_components.csv", component_rows_all)
    write_csv(analysis_root / "phase4_label_basis_region_contribution.csv", horizon_rows_all)
    (analysis_root / "phase4_label_basis_summary.json").write_text(json.dumps(summary_rows, indent=2))
    write_report(analysis_root / "phase4_label_basis_report.md", summary_rows)


if __name__ == "__main__":
    main()
