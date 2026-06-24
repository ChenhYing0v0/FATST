from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_phase2_qdf_matrix_audit import find_single, load_qdf_matrix


DATASETS = ["ETTh2", "ETTm1", "Weather"]
HORIZONS = [96, 192, 336, 720]
QDF_META_TYPES = ["diag", "off_diag", "all"]
HORIZON_LABEL = "mixed_h96_h192_h336_h720"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def prefix_risk_weights(horizon: int, max_pred_len: int = 720, alpha: float = 0.5) -> np.ndarray:
    step = np.arange(1, max_pred_len + 1, dtype=np.float64)
    weights = np.power(step / float(max_pred_len), -alpha)
    weights = weights / np.mean(weights)
    return weights[:horizon]


def residual_vectors(prediction_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(prediction_path)
    pred = np.asarray(data["pred"], dtype=np.float64)
    true = np.asarray(data["true"], dtype=np.float64)
    residual = pred - true
    vectors = np.transpose(residual, (0, 2, 1)).reshape(-1, residual.shape[1])
    return residual, vectors


def normalized_precision_loss(vectors: np.ndarray, matrix: torch.Tensor) -> float:
    mat = matrix.detach().cpu().double().numpy()
    trace = float(np.trace(mat))
    if abs(trace) > 1e-12:
        mat = mat * (mat.shape[0] / trace)
    values = np.einsum("bi,ij,bj->b", vectors, mat, vectors, optimize=True) / float(mat.shape[0])
    return float(np.mean(values))


def static_block_penalty(
    vectors: np.ndarray,
    matrix: np.ndarray,
    horizon: int,
    block_size: int = 48,
) -> float:
    blocks = []
    for start in range(0, horizon, block_size):
        end = min(start + block_size, horizon)
        blocks.append(np.mean(vectors[:, start:end], axis=1))
    block_values = np.stack(blocks, axis=1)
    active = matrix[: block_values.shape[1], : block_values.shape[1]]
    projected = block_values @ active.T
    penalty = float(np.mean(projected * projected))
    return penalty


def load_static_matrix(static_root: Path, dataset: str) -> np.ndarray | None:
    path = (
        static_root
        / "raw"
        / "PatchEncoderOffdiagBlockQuadratic"
        / dataset
        / HORIZON_LABEL
        / "seed2021"
        / "offdiag_block_matrix.csv"
    )
    if not path.exists():
        return None
    rows = read_csv(path)
    size = 1 + max(max(int(row["row_block"]), int(row["col_block"])) for row in rows)
    matrix = np.zeros((size, size), dtype=np.float64)
    for row in rows:
        matrix[int(row["row_block"]), int(row["col_block"])] = float(row["value"])
    return matrix


def load_specialist_flags(path: Path) -> dict[tuple[str, int], bool]:
    if not path.exists():
        return {}
    flags: dict[tuple[str, int], bool] = {}
    for row in read_csv(path):
        if "is_specialist_gap" in row:
            is_gap = str(row["is_specialist_gap"]).lower() == "true"
        elif "relative_mse_pct" in row:
            is_gap = float(row["relative_mse_pct"]) > 0.0
        elif "target_wins_mse" in row:
            is_gap = str(row["target_wins_mse"]).lower() != "true"
        else:
            is_gap = False
        flags[(row["dataset"], int(row["horizon"]))] = is_gap
    return flags


def qdf_matrix_path(qdf_raw_root: Path, meta_type: str, dataset: str, horizon: int) -> Path | None:
    run_dir = qdf_raw_root / meta_type / dataset / f"h{horizon}" / "seed2023"
    return find_single(run_dir / "checkpoints", "*/A.pth")


def collect(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    analysis_root = Path(args.analysis_root)
    r3_root = analysis_root / "raw" / args.r3_run_name
    qdf_raw_root = Path(args.qdf_analysis_root) / "raw"
    static_root = Path(args.static_offdiag_analysis_root)
    specialist_flags = load_specialist_flags(Path(args.r3_vs_fixed_csv))

    rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for dataset in DATASETS:
        static_matrix = load_static_matrix(static_root, dataset)
        if static_matrix is None:
            missing.append(f"static_offdiag_matrix:{dataset}")
        for horizon in HORIZONS:
            qdf_paths: dict[str, Path] = {}
            for meta_type in QDF_META_TYPES:
                matrix_path = qdf_matrix_path(qdf_raw_root, meta_type, dataset, horizon)
                if matrix_path is None:
                    missing.append(f"qdf_matrix:{meta_type}:{dataset}:h{horizon}")
                else:
                    qdf_paths[meta_type] = matrix_path

            pred_path = (
                r3_root
                / dataset
                / HORIZON_LABEL
                / "seed2021"
                / f"h{horizon}"
                / "predictions_test.npz"
            )
            if not pred_path.exists():
                missing.append(f"r3_prediction:{dataset}:h{horizon}")
                continue

            residual, vectors = residual_vectors(pred_path)
            residual_mse = float(np.mean(residual * residual))
            residual_mae = float(np.mean(np.abs(residual)))
            weights = prefix_risk_weights(horizon)
            prefix_loss = float(np.mean(vectors * vectors * weights.reshape(1, -1)))
            key = (dataset, horizon)
            base = {
                "dataset": dataset,
                "horizon": horizon,
                "is_specialist_gap": specialist_flags.get(key, False),
                "residual_mse": residual_mse,
                "residual_mae": residual_mae,
                "matrix_family": "identity",
                "matrix_source": "identity",
                "normalized_quadratic_loss": residual_mse,
                "ratio_to_residual_mse": 1.0,
            }
            rows.append(base)
            rows.append(
                {
                    **base,
                    "matrix_family": "prefix_risk",
                    "matrix_source": "analytic_prefix_risk_alpha_0.5",
                    "normalized_quadratic_loss": prefix_loss,
                    "ratio_to_residual_mse": prefix_loss / max(residual_mse, 1e-12),
                }
            )
            if static_matrix is not None:
                penalty = static_block_penalty(vectors, static_matrix, horizon)
                rows.append(
                    {
                        **base,
                        "matrix_family": "static_train_target_offdiag",
                        "matrix_source": str(
                            static_root
                            / "raw"
                            / "PatchEncoderOffdiagBlockQuadratic"
                            / dataset
                            / HORIZON_LABEL
                            / "seed2021"
                            / "offdiag_block_matrix.csv"
                        ),
                        "normalized_quadratic_loss": penalty,
                        "ratio_to_residual_mse": penalty / max(residual_mse, 1e-12),
                    }
                )

            for meta_type in QDF_META_TYPES:
                matrix_path = qdf_paths.get(meta_type)
                if matrix_path is None:
                    continue
                loaded_meta, _, covariance, precision = load_qdf_matrix(matrix_path, horizon)
                if loaded_meta != meta_type:
                    raise ValueError(f"QDF artifact meta_type mismatch: {matrix_path}")
                loss = normalized_precision_loss(vectors, precision)
                rows.append(
                    {
                        **base,
                        "matrix_family": f"qdf_{meta_type}_precision",
                        "matrix_source": str(matrix_path),
                        "normalized_quadratic_loss": loss,
                        "ratio_to_residual_mse": loss / max(residual_mse, 1e-12),
                    }
                )
                matrix_rows.append(
                    {
                        "meta_type": meta_type,
                        "dataset": dataset,
                        "horizon": horizon,
                        "matrix_dim": int(precision.shape[0]),
                        "precision_trace": float(torch.trace(precision)),
                        "covariance_trace": float(torch.trace(covariance)),
                        "source": str(matrix_path),
                    }
                )
    return rows, matrix_rows, missing


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > 0 else float("nan")


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    out = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2.0
        for pos in range(index, end):
            out[order[pos]] = rank
        index = end
    return out


def summarize(rows: list[dict[str, Any]], matrix_rows: list[dict[str, Any]], missing: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "row_count": len(rows),
        "matrix_row_count": len(matrix_rows),
        "missing_count": len(missing),
        "missing": missing,
        "matrix_families": sorted({str(row["matrix_family"]) for row in rows}),
    }
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(str(row["matrix_family"]), []).append(row)
    summary["by_matrix_family"] = {}
    for family, family_rows in sorted(by_family.items()):
        ratios = [float(row["ratio_to_residual_mse"]) for row in family_rows if finite(row["ratio_to_residual_mse"])]
        losses = [float(row["normalized_quadratic_loss"]) for row in family_rows if finite(row["normalized_quadratic_loss"])]
        residuals = [float(row["residual_mse"]) for row in family_rows if finite(row["residual_mse"])]
        specialist = [
            float(row["ratio_to_residual_mse"])
            for row in family_rows
            if row["is_specialist_gap"] and finite(row["ratio_to_residual_mse"])
        ]
        non_specialist = [
            float(row["ratio_to_residual_mse"])
            for row in family_rows
            if not row["is_specialist_gap"] and finite(row["ratio_to_residual_mse"])
        ]
        summary["by_matrix_family"][family] = {
            "count": len(family_rows),
            "mean_ratio_to_residual_mse": mean(ratios) if ratios else float("nan"),
            "std_ratio_to_residual_mse": float(np.std(ratios)) if ratios else float("nan"),
            "pearson_loss_vs_residual_mse": pearson(losses, residuals),
            "spearman_loss_vs_residual_mse": pearson(ranks(losses), ranks(residuals)) if losses else float("nan"),
            "specialist_gap_mean_ratio": mean(specialist) if specialist else float("nan"),
            "non_specialist_mean_ratio": mean(non_specialist) if non_specialist else float("nan"),
        }
    complete_predictions = len([item for item in missing if item.startswith("r3_prediction:")]) == 0
    complete_qdf = len([item for item in missing if item.startswith("qdf_matrix:")]) == 0
    summary["gate"] = {
        "prediction_artifacts_complete": complete_predictions,
        "qdf_matrices_complete": complete_qdf,
        "ready_for_alignment_decision": complete_predictions and complete_qdf,
    }
    return summary


def format_float(value: Any) -> str:
    return f"{float(value):.6f}" if finite(value) else "nan"


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase2-E2 QDF-to-FATST Residual Alignment Diagnostic",
        "",
        "## Decision Status",
        "",
    ]
    if summary["gate"]["ready_for_alignment_decision"]:
        lines.append("[Decision] residual-level artifacts are complete; use the tables below to judge whether QDF learned matrices align with FATST R.3 residuals.")
    else:
        lines.append("[Decision] diagnostic tooling is ready, but residual-level artifacts are incomplete. Do not make an alignment decision yet.")
    lines.extend(["", "## Gate", ""])
    for key, value in summary["gate"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", f"- missing_count: `{summary['missing_count']}`"])
    if summary["missing"]:
        lines.extend(["", "## Missing Artifacts", ""])
        for item in summary["missing"][:80]:
            lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "## Matrix Family Summary",
            "",
            "| Matrix family | Count | Mean ratio | Std ratio | Pearson loss~MSE | Spearman loss~MSE | Specialist ratio | Non-specialist ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for family, family_summary in summary["by_matrix_family"].items():
        lines.append(
            "| {family} | {count} | {mean_ratio} | {std_ratio} | {pearson} | {spearman} | {spec} | {non_spec} |".format(
                family=family,
                count=family_summary["count"],
                mean_ratio=format_float(family_summary["mean_ratio_to_residual_mse"]),
                std_ratio=format_float(family_summary["std_ratio_to_residual_mse"]),
                pearson=format_float(family_summary["pearson_loss_vs_residual_mse"]),
                spearman=format_float(family_summary["spearman_loss_vs_residual_mse"]),
                spec=format_float(family_summary["specialist_gap_mean_ratio"]),
                non_spec=format_float(family_summary["non_specialist_mean_ratio"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "[Fact] `ratio_to_residual_mse` normalizes each matrix loss by the plain R.3 residual MSE in the same dataset-horizon setting.",
            "",
            "[Decision Rule] If QDF `off_diag/all` ratios separate specialist gaps or hard horizons better than `static_train_target_offdiag`, then the next local mechanism should be learned or validation-informed. If they do not, stop the objective route.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze QDF learned objective alignment on FATST R.3 residuals.")
    parser.add_argument("--analysis-root", default="analysis/phase2_qdf_alignment_diagnostic_20260623")
    parser.add_argument("--r3-run-name", default="PatchEncoderPrefixRiskWeighted")
    parser.add_argument("--qdf-analysis-root", default="analysis/phase2_qdf_upstream_gate_20260623")
    parser.add_argument(
        "--static-offdiag-analysis-root",
        default="analysis/phase2_offdiag_block_quadratic_gate_20260623",
    )
    parser.add_argument(
        "--r3-vs-fixed-csv",
        default="analysis/phase1_prefix_risk_weighted_gate_20260622/phase1_prefix_risk_weighted_vs_fixed.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.analysis_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows, matrix_rows, missing = collect(args)
    summary = summarize(rows, matrix_rows, missing)
    write_csv(output_root / "phase2_qdf_residual_alignment_losses.csv", rows)
    write_csv(output_root / "phase2_qdf_residual_alignment_matrix_sources.csv", matrix_rows)
    (output_root / "phase2_qdf_residual_alignment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(output_root / "phase2_qdf_residual_alignment_report.md", summary)
    print(f"alignment_report={output_root / 'phase2_qdf_residual_alignment_report.md'}")


if __name__ == "__main__":
    main()
