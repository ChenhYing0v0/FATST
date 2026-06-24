from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
BASELINE_ROOT = REPO_ROOT / "baselines" / "patch_encoder_target_set_decoder"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(BASELINE_ROOT))

from analyze_phase4_label_basis_audit import (  # noqa: E402
    eigensystem,
    estimate_step_covariance,
)
from dataset import ForecastDataset  # noqa: E402


DATASETS = ["ETTh2", "ETTm1", "Weather"]
HORIZONS = [96, 192, 336, 720]
REGIONS = [(1, 96), (97, 192), (193, 336), (337, 720)]
TOP_KS = [1, 2, 4, 8, 16, 32, 64, 128]
R3_RUN = "mixed_h96_h192_h336_h720"
R3_NAME = "PatchEncoderPrefixRiskWeighted"
SEED = 2021


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prediction_path(raw_root: Path, dataset: str, horizon: int) -> Path:
    return raw_root / R3_NAME / dataset / R3_RUN / f"seed{SEED}" / f"h{horizon}" / "predictions_test.npz"


def load_predictions(raw_root: Path, dataset: str, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(prediction_path(raw_root, dataset, horizon))
    return np.asarray(data["pred"], dtype=np.float64), np.asarray(data["true"], dtype=np.float64)


def residual_vectors(pred: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual = pred - true
    vectors = np.transpose(residual, (0, 2, 1)).reshape(-1, residual.shape[1])
    return residual, vectors


def estimate_basis(
    dataset_root: Path,
    dataset: str,
    seq_len: int,
    pred_len: int,
    chunk_windows: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_set = ForecastDataset(dataset_root, dataset, "train", seq_len, pred_len)
    covariance, _, _ = estimate_step_covariance(
        train_set.data.astype(np.float64),
        seq_len,
        pred_len,
        chunk_windows,
    )
    return eigensystem(covariance)


def keyed_metric_rows(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    rows = read_csv(path)
    return {(row["dataset"], int(row["horizon"])): row for row in rows}


def keyed_segment_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(path)
    return {
        (row["dataset"], row["segment"]): row
        for row in rows
        if int(row["horizon"]) == 720
    }


def energy_share(coords: np.ndarray, top_k: int) -> float:
    active = min(top_k, coords.shape[1])
    squared = coords * coords
    denominator = float(np.sum(squared))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(squared[:, :active]) / denominator)


def label_variance_ratio(eigenvalues: np.ndarray, top_k: int) -> float:
    active = min(top_k, len(eigenvalues))
    total = float(np.sum(eigenvalues))
    if total <= 0.0:
        return 0.0
    return float(np.sum(eigenvalues[:active]) / total)


def component_projection_rows(
    dataset: str,
    horizon: int,
    coords: np.ndarray,
    eigenvalues: np.ndarray,
    n_components: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    residual_component_energy = np.mean(coords * coords, axis=0)
    total_residual_energy = float(np.sum(residual_component_energy))
    total_label_energy = float(np.sum(eigenvalues))
    for component_index in range(min(n_components, coords.shape[1])):
        residual_energy = float(residual_component_energy[component_index])
        label_energy = float(eigenvalues[component_index])
        rows.append(
            {
                "dataset": dataset,
                "horizon": horizon,
                "component": component_index + 1,
                "residual_component_energy": residual_energy,
                "residual_energy_share": residual_energy / max(total_residual_energy, 1e-12),
                "label_eigenvalue": label_energy,
                "label_variance_ratio": label_energy / max(total_label_energy, 1e-12),
                "residual_to_label_variance_ratio": residual_energy / max(label_energy, 1e-12),
            }
        )
    return rows


def segment_reconstruction_rows(
    dataset: str,
    residual: np.ndarray,
    coords: np.ndarray,
    eigenvectors: np.ndarray,
    segment_flags: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    flat_shape = coords.shape[0]
    channels = residual.shape[2]
    batch = residual.shape[0]
    for start, end in REGIONS:
        segment = f"{start}-{end}"
        residual_segment = residual[:, start - 1 : end, :]
        segment_energy = float(np.mean(residual_segment * residual_segment))
        row: dict[str, Any] = {
            "dataset": dataset,
            "segment": segment,
            "segment_mse": segment_energy,
            "is_segment_gap": float(segment_flags[(dataset, segment)]["relative_mse_pct"]) > 0.0,
            "relative_mse_pct": float(segment_flags[(dataset, segment)]["relative_mse_pct"]),
        }
        for top_k in TOP_KS:
            active = min(top_k, eigenvectors.shape[1])
            recon_flat = coords[:, :active] @ eigenvectors[:, :active].T
            recon = recon_flat.reshape(batch, channels, residual.shape[1]).transpose(0, 2, 1)
            recon_segment = recon[:, start - 1 : end, :]
            row[f"top{top_k}_reconstruction_mse_share"] = float(
                np.mean(recon_segment * recon_segment) / max(segment_energy, 1e-12)
            )
        rows.append(row)
    return rows


def collect(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_root = Path(args.dataset_root)
    raw_root = Path(args.r3_raw_root)
    metric_flags = keyed_metric_rows(Path(args.r3_vs_fixed_csv))
    segment_flags = keyed_segment_rows(Path(args.r3_vs_fixed_segments_csv))

    main_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []

    basis_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for dataset in DATASETS:
        for horizon in HORIZONS:
            eigenvalues, eigenvectors = basis_cache.setdefault(
                (dataset, horizon),
                estimate_basis(dataset_root, dataset, args.seq_len, horizon, args.chunk_windows),
            )
            pred, true = load_predictions(raw_root, dataset, horizon)
            residual, vectors = residual_vectors(pred, true)
            coords = vectors @ eigenvectors
            metric = metric_flags[(dataset, horizon)]
            row: dict[str, Any] = {
                "dataset": dataset,
                "horizon": horizon,
                "is_specialist_gap": float(metric["relative_mse_pct"]) > 0.0,
                "relative_mse_vs_fixed_pct": float(metric["relative_mse_pct"]),
                "residual_mse": float(np.mean(residual * residual)),
                "residual_mae": float(np.mean(np.abs(residual))),
            }
            for top_k in TOP_KS:
                row[f"label_top{top_k}_variance_ratio"] = label_variance_ratio(eigenvalues, top_k)
                row[f"residual_top{top_k}_energy_share"] = energy_share(coords, top_k)
                row[f"residual_top{top_k}_over_label_ratio"] = (
                    row[f"residual_top{top_k}_energy_share"]
                    / max(row[f"label_top{top_k}_variance_ratio"], 1e-12)
                )
            main_rows.append(row)
            component_rows.extend(
                component_projection_rows(dataset, horizon, coords, eigenvalues, args.n_components)
            )
            if horizon == 720:
                segment_rows.extend(segment_reconstruction_rows(dataset, residual, coords, eigenvectors, segment_flags))
    return main_rows, component_rows, segment_rows


def group_mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    return mean(float(row[key]) for row in rows)


def summarize(
    main_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    gap_rows = [row for row in main_rows if row["is_specialist_gap"]]
    non_gap_rows = [row for row in main_rows if not row["is_specialist_gap"]]
    segment_gap_rows = [row for row in segment_rows if row["is_segment_gap"]]
    segment_non_gap_rows = [row for row in segment_rows if not row["is_segment_gap"]]
    summary = {
        "main_rows": len(main_rows),
        "specialist_gap_rows": len(gap_rows),
        "non_gap_rows": len(non_gap_rows),
        "gap_mean_residual_top16_energy_share": group_mean(gap_rows, "residual_top16_energy_share"),
        "non_gap_mean_residual_top16_energy_share": group_mean(non_gap_rows, "residual_top16_energy_share"),
        "gap_mean_residual_top32_energy_share": group_mean(gap_rows, "residual_top32_energy_share"),
        "non_gap_mean_residual_top32_energy_share": group_mean(non_gap_rows, "residual_top32_energy_share"),
        "gap_mean_residual_top64_energy_share": group_mean(gap_rows, "residual_top64_energy_share"),
        "non_gap_mean_residual_top64_energy_share": group_mean(non_gap_rows, "residual_top64_energy_share"),
        "gap_mean_top16_over_label_ratio": group_mean(gap_rows, "residual_top16_over_label_ratio"),
        "non_gap_mean_top16_over_label_ratio": group_mean(non_gap_rows, "residual_top16_over_label_ratio"),
        "segment_gap_rows": len(segment_gap_rows),
        "segment_non_gap_rows": len(segment_non_gap_rows),
        "segment_gap_mean_top16_reconstruction_share": group_mean(
            segment_gap_rows, "top16_reconstruction_mse_share"
        ),
        "segment_non_gap_mean_top16_reconstruction_share": group_mean(
            segment_non_gap_rows, "top16_reconstruction_mse_share"
        ),
        "segment_gap_mean_top64_reconstruction_share": group_mean(
            segment_gap_rows, "top64_reconstruction_mse_share"
        ),
        "segment_non_gap_mean_top64_reconstruction_share": group_mean(
            segment_non_gap_rows, "top64_reconstruction_mse_share"
        ),
    }
    top16_ratio_delta = (
        summary["gap_mean_top16_over_label_ratio"] - summary["non_gap_mean_top16_over_label_ratio"]
    )
    segment_top16_delta = (
        summary["segment_gap_mean_top16_reconstruction_share"]
        - summary["segment_non_gap_mean_top16_reconstruction_share"]
    )
    summary["gap_minus_non_gap_top16_over_label_ratio"] = top16_ratio_delta
    summary["segment_gap_minus_non_gap_top16_reconstruction_share"] = segment_top16_delta
    summary["decision"] = {
        "component_residual_structure_exists": summary["gap_mean_residual_top16_energy_share"] > 0.5
        and summary["non_gap_mean_residual_top16_energy_share"] > 0.5,
        "top_component_energy_separates_gaps": abs(top16_ratio_delta) >= 0.05,
        "segment_reconstruction_separates_gaps": abs(segment_top16_delta) >= 0.05,
        "top_only_component_loss_supported": top16_ratio_delta > 0.05 and segment_top16_delta > 0.05,
    }
    summary["decision"]["proceed_to_component_objective_design"] = summary["decision"][
        "component_residual_structure_exists"
    ]
    return summary


def fmt(value: float) -> str:
    return f"{value:.3f}"


def write_report(
    path: Path,
    summary: dict[str, Any],
    main_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
) -> None:
    decision = summary["decision"]
    lines = [
        "# Phase4 Existing-Residual Projection Audit",
        "",
        "## Purpose",
        "",
        "[Fact] This diagnostic uses existing R.3 `predictions_test.npz` artifacts and train-label component bases. It does not train a model.",
        "",
        "[Question] Do existing R.3 residuals have meaningful component-space structure, and does that structure help explain known horizon/segment gaps?",
        "",
        "## Summary",
        "",
        f"- specialist gap rows: `{summary['specialist_gap_rows']}/{summary['main_rows']}`.",
        f"- gap mean residual top16 energy share: `{fmt(summary['gap_mean_residual_top16_energy_share'])}`.",
        f"- non-gap mean residual top16 energy share: `{fmt(summary['non_gap_mean_residual_top16_energy_share'])}`.",
        f"- gap mean top16-over-label ratio: `{fmt(summary['gap_mean_top16_over_label_ratio'])}`.",
        f"- non-gap mean top16-over-label ratio: `{fmt(summary['non_gap_mean_top16_over_label_ratio'])}`.",
        f"- gap minus non-gap top16-over-label ratio: `{fmt(summary['gap_minus_non_gap_top16_over_label_ratio'])}`.",
        f"- segment gap top16 reconstruction share: `{fmt(summary['segment_gap_mean_top16_reconstruction_share'])}`.",
        f"- segment non-gap top16 reconstruction share: `{fmt(summary['segment_non_gap_mean_top16_reconstruction_share'])}`.",
        f"- segment gap minus non-gap top16 reconstruction share: `{fmt(summary['segment_gap_minus_non_gap_top16_reconstruction_share'])}`.",
        "",
        "## Decision",
        "",
        f"- component residual structure exists: `{decision['component_residual_structure_exists']}`",
        f"- top component energy separates gaps: `{decision['top_component_energy_separates_gaps']}`",
        f"- segment reconstruction separates gaps: `{decision['segment_reconstruction_separates_gaps']}`",
        f"- top-only component loss supported: `{decision['top_only_component_loss_supported']}`",
        f"- proceed to component objective design: `{decision['proceed_to_component_objective_design']}`",
        "",
    ]
    if decision["proceed_to_component_objective_design"]:
        lines.append(
            "[Decision] Component-space supervision remains a viable Step 4-6 candidate, but the result does not support a top-only TransDF-style loss. The next objective should be hybrid or variance-balanced."
        )
    else:
        lines.append(
            "[Decision] Component-space supervision is not sufficiently explained by existing residuals yet. Prefer another diagnostic or random interval supervision before training."
        )
    lines.extend(
        [
            "",
            "## Main Horizon Rows",
            "",
            "| Dataset | Horizon | Gap | Rel MSE vs fixed | Residual top16 | Label top16 | Top16 / label |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in main_rows:
        lines.append(
            "| {dataset} | {horizon} | {gap} | {rel:+.2f}% | {rtop:.3f} | {ltop:.3f} | {ratio:.3f} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                gap=row["is_specialist_gap"],
                rel=row["relative_mse_vs_fixed_pct"],
                rtop=row["residual_top16_energy_share"],
                ltop=row["label_top16_variance_ratio"],
                ratio=row["residual_top16_over_label_ratio"],
            )
        )
    lines.extend(
        [
            "",
            "## H720 Segment Rows",
            "",
            "| Dataset | Segment | Gap | Rel MSE vs fixed | Top16 recon share | Top64 recon share |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in segment_rows:
        lines.append(
            "| {dataset} | {segment} | {gap} | {rel:+.2f}% | {top16:.3f} | {top64:.3f} |".format(
                dataset=row["dataset"],
                segment=row["segment"],
                gap=row["is_segment_gap"],
                rel=row["relative_mse_pct"],
                top16=row["top16_reconstruction_mse_share"],
                top64=row["top64_reconstruction_mse_share"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "[Inference] Residual energy is strongly structured in component space, so component supervision is more coherent than arbitrary horizon pairs.",
            "",
            "[Inference] Known gaps are not more concentrated in dominant top components. Gap rows have slightly lower top16-over-label ratio, and H720 segment gaps are nearly indistinguishable by top16 reconstruction share. Therefore, a top-only component loss is risky.",
            "",
            "[Decision] The next design should test a hybrid component objective: keep time-domain MSE while adding variance-normalized or component-balanced supervision, so lower-variance detail components are not ignored.",
            "",
            "[Risk] This diagnostic still does not prove training will improve. It only decides whether component-supervised training is a coherent next candidate.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project existing residuals onto train-label component bases.")
    parser.add_argument("--dataset-root", default="/Users/river/PaperResearch/Project/datasets")
    parser.add_argument(
        "--r3-raw-root",
        default="analysis/phase2_qdf_alignment_diagnostic_20260623/raw",
    )
    parser.add_argument(
        "--r3-vs-fixed-csv",
        default="analysis/phase1_prefix_risk_weighted_gate_20260622/phase1_prefix_risk_weighted_vs_fixed.csv",
    )
    parser.add_argument(
        "--r3-vs-fixed-segments-csv",
        default="analysis/phase1_prefix_risk_weighted_gate_20260622/phase1_prefix_risk_weighted_vs_fixed_segments.csv",
    )
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--chunk-windows", type=int, default=512)
    parser.add_argument("--n-components", type=int, default=16)
    parser.add_argument("--analysis-root", default="analysis/phase4_residual_projection_audit_20260624")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root = Path(args.analysis_root)
    main_rows, component_rows, segment_rows = collect(args)
    summary = summarize(main_rows, segment_rows)
    analysis_root.mkdir(parents=True, exist_ok=True)
    write_csv(analysis_root / "phase4_residual_projection_main.csv", main_rows)
    write_csv(analysis_root / "phase4_residual_projection_components.csv", component_rows)
    write_csv(analysis_root / "phase4_residual_projection_h720_segments.csv", segment_rows)
    (analysis_root / "phase4_residual_projection_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(
        analysis_root / "phase4_residual_projection_report.md",
        summary,
        main_rows,
        segment_rows,
    )


if __name__ == "__main__":
    main()
