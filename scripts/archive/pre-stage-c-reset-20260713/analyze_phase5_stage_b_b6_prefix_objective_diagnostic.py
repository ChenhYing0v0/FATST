from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASETS = ("ETTh2", "ETTm1", "Weather")
HORIZONS = (96, 192, 336, 720)
ENERGY_KS = (8, 16, 32, 64, 128, 256)
STABILITY_KS = (8, 16, 32)
SEQ_LEN = 720
PRED_LEN = 720
DEFAULT_DATASET_ROOT = Path("/Users/river/PaperResearch/Project/datasets")
DEFAULT_ANALYSIS_ROOT = Path("analysis/phase5_stage_b_prefix_native_objective_diagnostic_20260706")
DEFAULT_A6_ROOT = Path(
    "analysis/phase5_timealign_hss_a6_capacity_native_gate_20260703/raw/official-last/"
    "TimeAlignOfficialUnified720_A6_a6_lbf_r256_official-last"
)


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    data_path: str
    relative_root: str
    kind: str


SPECS = {
    "ETTh2": DatasetSpec("ETTh2", "ETTh2.csv", "ETT-small", "ett_hour"),
    "ETTm1": DatasetSpec("ETTm1", "ETTm1.csv", "ETT-small", "ett_minute"),
    "Weather": DatasetSpec("Weather", "weather.csv", "weather", "custom"),
}


def resolve_data_path(dataset_root: Path, spec: DatasetSpec) -> Path:
    direct = dataset_root / spec.data_path
    nested = dataset_root / spec.relative_root / spec.data_path
    if direct.exists():
        return direct
    if nested.exists():
        return nested
    raise FileNotFoundError(f"Cannot find {spec.data_path} under {dataset_root}")


def train_boundaries(num_rows: int, spec: DatasetSpec) -> tuple[int, int]:
    if spec.kind == "ett_hour":
        return 0, 12 * 30 * 24
    if spec.kind == "ett_minute":
        return 0, 12 * 30 * 24 * 4
    if spec.kind == "custom":
        return 0, int(num_rows * 0.7)
    raise ValueError(f"Unsupported dataset kind: {spec.kind}")


def load_train_values(dataset_root: Path, dataset: str) -> np.ndarray:
    spec = SPECS[dataset]
    path = resolve_data_path(dataset_root, spec)
    df = pd.read_csv(path)
    if spec.kind == "custom":
        cols = list(df.columns)
        cols.remove("OT")
        cols.remove("date")
        df = df[["date"] + cols + ["OT"]]
    values = df[df.columns[1:]].to_numpy(dtype=np.float32)
    start, end = train_boundaries(len(df), spec)
    train = values[start:end]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((train - mean) / std).astype(np.float32)


def sample_start_indices(num_windows: int, max_windows: int, seed: int) -> np.ndarray:
    if num_windows <= max_windows:
        return np.arange(num_windows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_windows, size=max_windows, replace=False)).astype(np.int64)


def build_future_matrix(train_values: np.ndarray, starts: np.ndarray, horizon: int) -> np.ndarray:
    channels = train_values.shape[1]
    matrix = np.empty((len(starts) * channels, horizon), dtype=np.float32)
    row = 0
    for start in starts:
        future = train_values[start + SEQ_LEN : start + SEQ_LEN + horizon]
        matrix[row : row + channels] = future.T
        row += channels
    return matrix


def center_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0, keepdims=True)
    return (matrix - mean).astype(np.float32), mean.astype(np.float32)


def covariance_basis(centered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    return eigvals, eigvecs


def energy_shares_from_eigvals(eigvals: np.ndarray, ks: tuple[int, ...]) -> dict[int, float]:
    total = float(eigvals.sum())
    return {k: float(eigvals[: min(k, len(eigvals))].sum() / total) if total > 0 else 0.0 for k in ks}


def projection_energy(centered: np.ndarray, basis: np.ndarray, ks: tuple[int, ...]) -> dict[int, float]:
    total = float(np.sum(centered.astype(np.float64) ** 2))
    if total <= 0:
        return {k: 0.0 for k in ks}
    shares: dict[int, float] = {}
    for k in ks:
        k_eff = min(k, basis.shape[1])
        coeff = centered @ basis[:, :k_eff]
        shares[k] = float(np.sum(coeff.astype(np.float64) ** 2) / total)
    return shares


def dct_basis(length: int) -> np.ndarray:
    steps = np.arange(length, dtype=np.float64) + 0.5
    freqs = np.arange(length, dtype=np.float64)
    basis = np.cos(np.pi * np.outer(steps, freqs) / length)
    basis[:, 0] *= np.sqrt(1.0 / length)
    basis[:, 1:] *= np.sqrt(2.0 / length)
    return basis


def mean_lag_corr(centered: np.ndarray, lag: int) -> float:
    if lag >= centered.shape[1]:
        return float("nan")
    left = centered[:, :-lag].reshape(-1)
    right = centered[:, lag:].reshape(-1)
    left_std = float(left.std())
    right_std = float(right.std())
    if left_std < 1e-12 or right_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    rx = rankdata(np.asarray(x, dtype=np.float64))
    ry = rankdata(np.asarray(y, dtype=np.float64))
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def subspace_overlap(prefix_basis: np.ndarray, full_basis: np.ndarray, horizon: int, k: int) -> float:
    prefix = prefix_basis[:, :k]
    restricted = full_basis[:horizon, :k]
    q_restricted, _ = np.linalg.qr(restricted)
    singular_values = np.linalg.svd(prefix.T @ q_restricted[:, :k], compute_uv=False)
    return float(np.mean(singular_values**2))


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


def read_prediction_residual(a6_root: Path, dataset: str, max_rows: int, seed: int) -> np.ndarray:
    path = a6_root / dataset / "mixed_h96_h192_h336_h720" / "seed2021" / "predictions_test.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = np.load(path)
    residual = (payload["pred"][:, :PRED_LEN, :] - payload["true"][:, :PRED_LEN, :]).astype(np.float32)
    rows = residual.reshape(-1, PRED_LEN)
    if rows.shape[0] > max_rows:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(rows.shape[0], size=max_rows, replace=False))
        rows = rows[selected]
    centered, _mean = center_rows(rows)
    return centered


def load_learned_basis(checkpoint_root: Path | None, dataset: str) -> np.ndarray | None:
    if checkpoint_root is None:
        return None
    path = checkpoint_root / dataset / "checkpoint.pt"
    if not path.exists():
        return None
    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    basis = state["learned_temporal_basis"].detach().cpu().numpy().astype(np.float64)
    left, _singular_values, _right = np.linalg.svd(basis, full_matrices=False)
    return left


def analyze_dataset(args: argparse.Namespace, dataset: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    train_values = load_train_values(args.dataset_root, dataset)
    num_windows = train_values.shape[0] - SEQ_LEN - PRED_LEN + 1
    starts = sample_start_indices(num_windows, args.max_train_windows, args.seed)

    full_matrix = build_future_matrix(train_values, starts, PRED_LEN)
    full_centered, _full_mean = center_rows(full_matrix)
    eigvals_720, basis_720 = covariance_basis(full_centered)
    label_pca_shares = energy_shares_from_eigvals(eigvals_720, ENERGY_KS)
    label_dct_shares = projection_energy(full_centered, dct_basis(PRED_LEN), ENERGY_KS)
    learned_basis = load_learned_basis(args.checkpoint_root, dataset)
    label_learned_shares = (
        projection_energy(full_centered, learned_basis, ENERGY_KS)
        if learned_basis is not None
        else {k: float("nan") for k in ENERGY_KS}
    )

    label_row: dict[str, Any] = {
        "dataset": dataset,
        "train_windows_total": num_windows,
        "train_windows_sampled": len(starts),
        "matrix_rows": full_centered.shape[0],
        "matrix_cols": full_centered.shape[1],
        "lag1_corr": f"{mean_lag_corr(full_centered, 1):.6f}",
        "lag24_corr": f"{mean_lag_corr(full_centered, 24):.6f}",
        "lag96_corr": f"{mean_lag_corr(full_centered, 96):.6f}",
    }
    for k, value in label_pca_shares.items():
        label_row[f"label_pca_top{k}_energy"] = f"{value:.6f}"
    for k, value in label_dct_shares.items():
        label_row[f"label_dct_top{k}_energy"] = f"{value:.6f}"
        label_row[f"label_pca_minus_dct_top{k}"] = f"{label_pca_shares[k] - value:.6f}"
    for k, value in label_learned_shares.items():
        label_row[f"label_a6_basis_top{k}_energy"] = f"{value:.6f}"
        label_row[f"label_a6_minus_dct_top{k}"] = f"{value - label_dct_shares[k]:.6f}"

    stability_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS[:-1]:
        prefix_matrix = full_matrix[:, :horizon]
        prefix_centered, _prefix_mean = center_rows(prefix_matrix)
        _prefix_eigvals, prefix_basis = covariance_basis(prefix_centered)
        for k in STABILITY_KS:
            stability_rows.append(
                {
                    "dataset": dataset,
                    "prefix_horizon": horizon,
                    "k": k,
                    "full_to_prefix_subspace_overlap": f"{subspace_overlap(prefix_basis, basis_720, horizon, k):.6f}",
                }
            )

    residual_centered = read_prediction_residual(args.a6_root, dataset, args.max_residual_rows, args.seed)
    residual_pca = projection_energy(residual_centered, basis_720, ENERGY_KS)
    residual_dct = projection_energy(residual_centered, dct_basis(PRED_LEN), ENERGY_KS)
    residual_learned = (
        projection_energy(residual_centered, learned_basis, ENERGY_KS)
        if learned_basis is not None
        else {k: float("nan") for k in ENERGY_KS}
    )
    residual_step_mse = np.mean(residual_centered.astype(np.float64) ** 2, axis=0)
    residual_row: dict[str, Any] = {
        "dataset": dataset,
        "residual_rows": residual_centered.shape[0],
        "residual_step_spearman": f"{spearman(np.arange(1, PRED_LEN + 1), residual_step_mse):.6f}",
    }
    for k in ENERGY_KS:
        residual_row[f"residual_label_pca_top{k}_energy"] = f"{residual_pca[k]:.6f}"
        residual_row[f"residual_dct_top{k}_energy"] = f"{residual_dct[k]:.6f}"
        residual_row[f"residual_pca_minus_dct_top{k}"] = f"{residual_pca[k] - residual_dct[k]:.6f}"
        residual_row[f"residual_a6_basis_top{k}_energy"] = f"{residual_learned[k]:.6f}"
        residual_row[f"residual_a6_minus_dct_top{k}"] = f"{residual_learned[k] - residual_dct[k]:.6f}"
    return label_row, stability_rows, residual_row


def render_report(
    label_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    residual_rows: list[dict[str, Any]],
) -> str:
    def f(row: dict[str, Any], key: str) -> float:
        return float(row[key])

    pass_label = [
        row["dataset"]
        for row in label_rows
        if f(row, "label_pca_top32_energy") >= 0.80 and f(row, "label_pca_minus_dct_top32") >= 0.03
    ]
    pass_stability = []
    for dataset in DATASETS:
        rows = [row for row in stability_rows if row["dataset"] == dataset and int(row["k"]) == 16]
        if rows and min(f(row, "full_to_prefix_subspace_overlap") for row in rows) >= 0.70:
            pass_stability.append(dataset)
    pass_residual = [
        row["dataset"]
        for row in residual_rows
        if f(row, "residual_label_pca_top32_energy") >= 0.20 and f(row, "residual_pca_minus_dct_top32") >= -0.02
    ]
    pass_a6_specific = [
        row["dataset"]
        for row in residual_rows
        if f(row, "residual_a6_basis_top32_energy") >= 0.20 and f(row, "residual_a6_minus_dct_top32") >= 0.03
    ]
    pass_non_distance = [
        row["dataset"]
        for row in residual_rows
        if abs(f(row, "residual_step_spearman")) < 0.95
    ]

    if (
        len(pass_label) >= 2
        and len(pass_stability) >= 2
        and len(pass_residual) >= 2
        and len(pass_a6_specific) >= 2
        and len(pass_non_distance) >= 2
    ):
        decision = "problem_exists_enter_step46_design"
    elif len(pass_label) >= 2 and len(pass_residual) >= 2:
        decision = "partial_pass_problem_exists_but_distance_or_generic_basis_risk"
    else:
        decision = "diagnostic_not_enough_pause_b6"

    lines = [
        "# Phase5 StageB B6 Prefix-Native Objective Diagnostic Report",
        "",
        "`current_step`: StageB Step 2/3 problem-existence diagnostic.",
        "",
        "## Scope",
        "",
        "[Fact] This diagnostic uses train split labels and existing A6-LBF-r256 prediction artifacts.",
        "",
        "[Boundary] It does not implement a new objective and does not use validation/test labels to build the label basis.",
        "",
        "## 11-step Record",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | StageB Step 2/3 prefix-native objective diagnostic |",
        "| `problem` | Does clean A6-LBF need an objective matched to prefix-native label/basis structure? |",
        "| `existence_evidence` | Train-label PCA/DCT comparison, prefix subspace stability, residual projection into train-label basis |",
        "| `idea` | A basis-native forecast operator may need a basis-native objective instead of generic time-domain point loss |",
        "| `theory_check` | Evidence is positive only if label/residual structure is stable and not just generic low-frequency smoothness |",
        "| `design` | Offline diagnostic; no model training |",
        f"| `narrative_gate` | {decision} |",
        "| `effectiveness_gate` | not applicable before method implementation |",
        "| `artifacts` | this directory |",
        f"| `decision` | {decision} |",
        "",
        "## Label Basis Summary",
        "",
            "| Dataset | PCA top32 | DCT top32 | A6 basis top32 | PCA-DCT | A6-DCT | Lag1 | Lag24 | Lag96 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in label_rows:
        lines.append(
            f"| {row['dataset']} | `{f(row, 'label_pca_top32_energy'):.3f}` | "
            f"`{f(row, 'label_dct_top32_energy'):.3f}` | `{f(row, 'label_a6_basis_top32_energy'):.3f}` | "
            f"`{f(row, 'label_pca_minus_dct_top32'):.3f}` | `{f(row, 'label_a6_minus_dct_top32'):.3f}` | "
            f"`{f(row, 'lag1_corr'):.3f}` | `{f(row, 'lag24_corr'):.3f}` | `{f(row, 'lag96_corr'):.3f}` |"
        )

    lines.extend(
        [
            "",
            "## Prefix Stability",
            "",
            "| Dataset | H96 top16 overlap | H192 top16 overlap | H336 top16 overlap |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for dataset in DATASETS:
        values = {
            int(row["prefix_horizon"]): f(row, "full_to_prefix_subspace_overlap")
            for row in stability_rows
            if row["dataset"] == dataset and int(row["k"]) == 16
        }
        lines.append(
            f"| {dataset} | `{values.get(96, float('nan')):.3f}` | "
            f"`{values.get(192, float('nan')):.3f}` | `{values.get(336, float('nan')):.3f}` |"
        )

    lines.extend(
        [
            "",
            "## Residual Basis Summary",
            "",
            "| Dataset | Residual PCA top32 | Residual DCT top32 | Residual A6 top32 | PCA-DCT | A6-DCT | Step Spearman |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in residual_rows:
        lines.append(
            f"| {row['dataset']} | `{f(row, 'residual_label_pca_top32_energy'):.3f}` | "
            f"`{f(row, 'residual_dct_top32_energy'):.3f}` | `{f(row, 'residual_a6_basis_top32_energy'):.3f}` | "
            f"`{f(row, 'residual_pca_minus_dct_top32'):.3f}` | `{f(row, 'residual_a6_minus_dct_top32'):.3f}` | "
            f"`{f(row, 'residual_step_spearman'):.3f}` |"
        )

    lines.extend(
        [
            "",
            "## Gate Evaluation",
            "",
            f"- Label compressibility / non-generic basis evidence passes on: `{', '.join(pass_label) or 'none'}`.",
            f"- Prefix subspace stability passes on: `{', '.join(pass_stability) or 'none'}`.",
            f"- Residual basis structure passes on: `{', '.join(pass_residual) or 'none'}`.",
            f"- A6-specific learned-basis advantage passes on: `{', '.join(pass_a6_specific) or 'none'}`.",
            f"- Non-distance residual check passes on: `{', '.join(pass_non_distance) or 'none'}`.",
            "",
            "## Interpretation",
            "",
            "[Fact] `label_pca_top32_energy` measures train-label temporal energy captured by the top 32 train-only PCA components.",
            "",
            "[Fact] `label_dct_top32_energy` is a generic low-frequency control. A small PCA-DCT gap weakens the novelty of a learned label-basis objective.",
            "",
            "[Fact] `label_a6_basis_top32_energy` and `residual_a6_basis_top32_energy` use the learned temporal basis from the pure no-align/no-recon A6 checkpoint when available.",
            "",
            "[Fact] `residual_label_pca_top32_energy` measures how much A6-LBF residual energy lies in train-label PCA directions.",
            "",
            "[Fact] `residual_step_spearman` measures whether residual energy is still mostly a forecast-distance effect.",
            "",
            f"[Decision] `{decision}`.",
            "",
        ]
    )
    if decision == "problem_exists_enter_step46_design":
        lines.extend(
            [
                "[Next] B6 may move to Step 4-6 method design. The method should optimize basis/coefficient residuals tied to A6-LBF, not add a generic frequency auxiliary loss.",
                "",
            ]
        )
    elif decision == "partial_pass_problem_exists_but_distance_or_generic_basis_risk":
        lines.extend(
            [
                "[Next] B6 has enough evidence to continue, but not enough to implement a method. The next action is to sharpen the proxy boundary against generic DCT/frequency losses and step-distance confounding.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "[Next] Do not implement a B6 objective. Pause StageB or redefine the problem.",
                "",
            ]
        )

    lines.extend(
        [
            "## Output Files",
            "",
            "- `stage_b_b6_label_basis_summary.csv`",
            "- `stage_b_b6_prefix_stability.csv`",
            "- `stage_b_b6_residual_basis_summary.csv`",
            "- `stage_b_b6_report.md`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase5 StageB B6 prefix-native objective diagnostics.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--a6-root", type=Path, default=DEFAULT_A6_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--max-train-windows", type=int, default=4096)
    parser.add_argument("--max-residual-rows", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        label_row, dataset_stability_rows, residual_row = analyze_dataset(args, dataset)
        label_rows.append(label_row)
        stability_rows.extend(dataset_stability_rows)
        residual_rows.append(residual_row)

    args.analysis_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.analysis_root / "stage_b_b6_label_basis_summary.csv", label_rows)
    write_csv(args.analysis_root / "stage_b_b6_prefix_stability.csv", stability_rows)
    write_csv(args.analysis_root / "stage_b_b6_residual_basis_summary.csv", residual_rows)
    report = render_report(label_rows, stability_rows, residual_rows)
    report_path = args.analysis_root / "stage_b_b6_report.md"
    report_path.write_text(report)
    (args.analysis_root / "stage_b_b6_config.json").write_text(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root),
                "a6_root": str(args.a6_root),
                "checkpoint_root": str(args.checkpoint_root) if args.checkpoint_root else None,
                "max_train_windows": args.max_train_windows,
                "max_residual_rows": args.max_residual_rows,
                "seed": args.seed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(report_path)


if __name__ == "__main__":
    main()
