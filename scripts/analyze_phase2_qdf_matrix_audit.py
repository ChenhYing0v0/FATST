from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import types
from pathlib import Path
from statistics import mean
from typing import Any

import torch


DATASETS = ["ETTh2", "ETTm1", "Weather"]
HORIZONS = [96, 192, 336, 720]
META_TYPES = ["all", "diag", "off_diag"]
SEGMENTS = [
    ("1-96", 0, 96),
    ("97-192", 96, 192),
    ("193-336", 192, 336),
    ("337-720", 336, 720),
]


def register_qdf_stub() -> None:
    """Register the class path needed to unpickle QDF A.pth artifacts."""
    if "exp.exp_long_term_forecasting_meta_ml3" in sys.modules:
        return
    package = types.ModuleType("exp")
    module = types.ModuleType("exp.exp_long_term_forecasting_meta_ml3")

    class CovarianceMatrix(torch.nn.Module):
        pass

    CovarianceMatrix.__module__ = "exp.exp_long_term_forecasting_meta_ml3"
    module.CovarianceMatrix = CovarianceMatrix
    sys.modules["exp"] = package
    sys.modules["exp.exp_long_term_forecasting_meta_ml3"] = module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def find_single(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    return matches[0] if matches else None


def load_qdf_matrix(path: Path, horizon: int) -> tuple[str, str, torch.Tensor, torch.Tensor]:
    register_qdf_stub()
    obj = torch.load(path, map_location="cpu", weights_only=False)
    meta_type = str(getattr(obj, "meta_type", "all"))
    pred_len = int(getattr(obj, "pred_len", horizon))
    eps = float(getattr(obj, "eps", 1e-6))
    state = obj.state_dict()
    if meta_type == "all":
        param_key = "L_param"
        l_param = state[param_key].float()
        lower = torch.tril(l_param)
        diag = torch.diag(lower) + eps
        lower = lower - torch.diag_embed(torch.diag(lower)) + torch.diag_embed(diag)
    elif meta_type == "diag":
        param_key = "diag_param"
        diag_param = state[param_key].float()
        diag_values = torch.sqrt(torch.abs(diag_param) + eps)
        lower = torch.diag(diag_values)
    elif meta_type == "off_diag":
        param_key = "L_param"
        l_param = state[param_key].float()
        lower = torch.tril(l_param, diagonal=-1)
        lower = lower + torch.eye(pred_len, dtype=lower.dtype)
        row_norms = torch.linalg.norm(lower, dim=1, keepdim=True).clamp(min=eps)
        lower = lower / row_norms
    else:
        raise ValueError(f"Unsupported QDF meta_type: {meta_type}")
    covariance = lower @ lower.T
    precision = torch.linalg.inv(covariance)
    return meta_type, param_key, covariance, precision


def matrix_stats(prefix: str, matrix: torch.Tensor) -> dict[str, float]:
    diag = torch.diag(matrix)
    offdiag = matrix - torch.diag_embed(diag)
    total_fro_sq = float(torch.sum(matrix * matrix))
    offdiag_fro_sq = float(torch.sum(offdiag * offdiag))
    abs_offdiag = torch.abs(offdiag)
    n = matrix.shape[0]
    distance = torch.abs(torch.arange(n).view(n, 1) - torch.arange(n).view(1, n)).float()
    offdiag_mass = float(torch.sum(abs_offdiag))
    weighted_bandwidth = (
        float(torch.sum(abs_offdiag * distance)) / offdiag_mass if offdiag_mass > 0 else 0.0
    )
    bandwidth_ratio = weighted_bandwidth / float(max(n - 1, 1))
    return {
        f"{prefix}_diag_mean": float(torch.mean(diag)),
        f"{prefix}_diag_std": float(torch.std(diag, unbiased=False)),
        f"{prefix}_diag_min": float(torch.min(diag)),
        f"{prefix}_diag_max": float(torch.max(diag)),
        f"{prefix}_offdiag_abs_mean": float(torch.sum(abs_offdiag) / max(n * (n - 1), 1)),
        f"{prefix}_offdiag_abs_max": float(torch.max(abs_offdiag)) if n > 1 else 0.0,
        f"{prefix}_offdiag_fro_share": offdiag_fro_sq / total_fro_sq if total_fro_sq > 0 else 0.0,
        f"{prefix}_weighted_bandwidth": weighted_bandwidth,
        f"{prefix}_weighted_bandwidth_ratio": bandwidth_ratio,
        f"{prefix}_condition_number": float(torch.linalg.cond(matrix)),
    }


def active_segments(horizon: int) -> list[tuple[str, int, int]]:
    return [(name, start, min(end, horizon)) for name, start, end in SEGMENTS if start < horizon]


def region_rows(
    meta_type: str,
    dataset: str,
    horizon: int,
    matrix_name: str,
    matrix: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for src_name, src_start, src_end in active_segments(horizon):
        for dst_name, dst_start, dst_end in active_segments(horizon):
            block = matrix[src_start:src_end, dst_start:dst_end]
            rows.append(
                {
                    "meta_type": meta_type,
                    "dataset": dataset,
                    "horizon": horizon,
                    "matrix": matrix_name,
                    "src_region": src_name,
                    "dst_region": dst_name,
                    "is_diag_block": src_name == dst_name,
                    "block_mean": float(torch.mean(block)),
                    "block_abs_mean": float(torch.mean(torch.abs(block))),
                    "block_abs_max": float(torch.max(torch.abs(block))),
                    "block_fro": float(torch.linalg.norm(block)),
                }
            )
    return rows


def collect(raw_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    region_metric_rows: list[dict[str, Any]] = []
    for meta_type in META_TYPES:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                run_dir = raw_root / meta_type / dataset / f"h{horizon}" / "seed2023"
                path = find_single(run_dir / "checkpoints", "*/A.pth")
                if path is None:
                    continue
                loaded_meta, param_key, covariance, precision = load_qdf_matrix(path, horizon)
                if loaded_meta != meta_type:
                    raise ValueError(f"Path meta_type={meta_type} but artifact meta_type={loaded_meta}: {path}")
                row: dict[str, Any] = {
                    "meta_type": meta_type,
                    "dataset": dataset,
                    "horizon": horizon,
                    "matrix_dim": covariance.shape[0],
                    "param_key": param_key,
                    "source": str(path),
                    "covariance_trace": float(torch.trace(covariance)),
                    "precision_trace": float(torch.trace(precision)),
                }
                row.update(matrix_stats("covariance", covariance))
                row.update(matrix_stats("precision", precision))
                metric_rows.append(row)
                region_metric_rows.extend(region_rows(meta_type, dataset, horizon, "covariance", covariance))
                region_metric_rows.extend(region_rows(meta_type, dataset, horizon, "precision", precision))
    return metric_rows, region_metric_rows


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "matrix_rows": len(rows),
        "meta_types_present": sorted({str(row["meta_type"]) for row in rows}),
    }
    by_meta: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_meta.setdefault(str(row["meta_type"]), []).append(row)
    summary["by_meta_type"] = {}
    for meta_type, meta_rows in sorted(by_meta.items()):
        summary["by_meta_type"][meta_type] = {
            "count": len(meta_rows),
            "mean_precision_offdiag_fro_share": mean(
                float(row["precision_offdiag_fro_share"]) for row in meta_rows if finite(row["precision_offdiag_fro_share"])
            ),
            "mean_covariance_offdiag_fro_share": mean(
                float(row["covariance_offdiag_fro_share"]) for row in meta_rows if finite(row["covariance_offdiag_fro_share"])
            ),
            "mean_precision_weighted_bandwidth": mean(
                float(row["precision_weighted_bandwidth"]) for row in meta_rows if finite(row["precision_weighted_bandwidth"])
            ),
            "mean_precision_weighted_bandwidth_ratio": mean(
                float(row["precision_weighted_bandwidth_ratio"])
                for row in meta_rows
                if finite(row["precision_weighted_bandwidth_ratio"])
            ),
            "mean_covariance_condition_number": mean(
                float(row["covariance_condition_number"]) for row in meta_rows if finite(row["covariance_condition_number"])
            ),
        }
    diag_share = summary["by_meta_type"].get("diag", {}).get("mean_precision_offdiag_fro_share", 0.0)
    offdiag_share = summary["by_meta_type"].get("off_diag", {}).get("mean_precision_offdiag_fro_share", 0.0)
    all_share = summary["by_meta_type"].get("all", {}).get("mean_precision_offdiag_fro_share", 0.0)
    summary["gate"] = {
        "all_36_matrices_available": len(rows) == len(META_TYPES) * len(DATASETS) * len(HORIZONS),
        "diag_precision_offdiag_near_zero": abs(float(diag_share)) <= 1e-8,
        "off_diag_precision_has_interaction": float(offdiag_share) > 0.001,
        "all_precision_has_interaction": float(all_share) > 0.001,
    }
    summary["gate"]["supports_local_offdiag_audit"] = all(summary["gate"].values())
    return summary


def format_float(value: Any) -> str:
    return f"{float(value):.6f}" if finite(value) else "nan"


def write_report(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Phase2-E0 QDF Learned Matrix Audit",
        "",
        "## Decision",
        "",
        f"[Decision] learned matrix audit {'supports' if summary['gate']['supports_local_offdiag_audit'] else 'does not support'} continuing toward a local off-diagonal objective probe.",
        "",
        "## Gate",
        "",
    ]
    for key, value in summary["gate"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Meta-Type Matrix Summary",
            "",
            "| Meta type | Count | Precision offdiag fro share | Covariance offdiag fro share | Precision bandwidth | Covariance condition |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for meta_type, meta_summary in summary["by_meta_type"].items():
        lines.append(
            "| {meta} | {count} | {p_share} | {c_share} | {bandwidth} | {cond} |".format(
                meta=meta_type,
                count=meta_summary["count"],
                p_share=format_float(meta_summary["mean_precision_offdiag_fro_share"]),
                c_share=format_float(meta_summary["mean_covariance_offdiag_fro_share"]),
                bandwidth=format_float(meta_summary["mean_precision_weighted_bandwidth_ratio"]),
                cond=format_float(meta_summary["mean_covariance_condition_number"]),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Run Precision Off-Diagonal Share",
            "",
            "| Meta type | Dataset | Horizon | Precision offdiag fro share | Precision bandwidth | Covariance condition |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {meta} | {dataset} | {horizon} | {share} | {bandwidth} | {condition} |".format(
                meta=row["meta_type"],
                dataset=row["dataset"],
                horizon=row["horizon"],
                share=format_float(row["precision_offdiag_fro_share"]),
                bandwidth=format_float(row["precision_weighted_bandwidth_ratio"]),
                condition=format_float(row["covariance_condition_number"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "[Fact] QDF stores a `CovarianceMatrix` module. Its loss solves `L x = E`, so the effective residual weighting matrix is the inverse of `Sigma = L L^T`.",
            "",
            "[Decision] Local FATST experiments should audit and use the precision/off-diagonal structure, not only the saved covariance visualization.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit learned QDF covariance/precision matrices.")
    parser.add_argument("--analysis-root", default="analysis/phase2_qdf_upstream_gate_20260623")
    parser.add_argument("--output-root", default="analysis/phase2_qdf_matrix_audit_20260623")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root = Path(args.analysis_root)
    output_root = Path(args.output_root)
    raw_root = analysis_root / "raw"
    rows, region_metric_rows = collect(raw_root)
    summary = summarize(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "phase2_qdf_matrix_audit_metrics.csv", rows)
    write_csv(output_root / "phase2_qdf_matrix_audit_region_blocks.csv", region_metric_rows)
    (output_root / "phase2_qdf_matrix_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(output_root / "phase2_qdf_matrix_audit_report.md", rows, summary)
    print(f"matrix_audit_report={output_root / 'phase2_qdf_matrix_audit_report.md'}")


if __name__ == "__main__":
    main()
