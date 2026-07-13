from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean

import numpy as np


DATASETS = ["ETTh2", "ETTm1", "Weather"]
HORIZONS = [96, 192, 336, 720]
SPECIALIST_GAPS = {
    ("ETTm1", 96),
    ("ETTm1", 720),
    ("ETTh2", 720),
    ("Weather", 96),
}
METRIC_NAMES = ["mae", "mse", "cov_loss", "rmse", "mape", "mspe", "mre"]
QDF_REF_RE = re.compile(r"^\s*(?P<horizon>\d+)\s*\|\s*mse:(?P<mse>[^,]+),\s*mae:(?P<mae>[^,]+),\s*cov:(?P<cov>.+)$")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def find_single(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        return matches[0]
    return matches[0]


def parse_result_log(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    for line in reversed(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        match = QDF_REF_RE.match(line)
        if match:
            return {
                "mse": float(match.group("mse")),
                "mae": float(match.group("mae")),
                "cov_loss": float(match.group("cov")),
            }
    return None


def parse_run(raw_root: Path, meta_type: str, dataset: str, horizon: int, seed: int) -> dict[str, object] | None:
    run_dir = raw_root / meta_type / dataset / f"h{horizon}" / f"seed{seed}"
    if not run_dir.exists():
        return None
    metrics_path = find_single(run_dir / "results", "*/metrics.npy")
    result_log = parse_result_log(run_dir / "result_long_term_forecast.txt")
    stdout_path = raw_root / "_logs" / "phase2_qdf_upstream_gate" / f"QDF_{meta_type}_{dataset}_h{horizon}_seed{seed}.log"
    cov_pdf = find_single(run_dir / "results", "*/cov_matrix.pdf")
    has_a = find_single(run_dir / "checkpoints", "*/A.pth") is not None
    has_marker = (run_dir / "run.done").exists()
    row: dict[str, object] = {
        "meta_type": meta_type,
        "dataset": dataset,
        "horizon": horizon,
        "seed": seed,
        "run_dir": str(run_dir),
        "has_run_done": has_marker,
        "has_cov_matrix_pdf": cov_pdf is not None,
        "has_A_pth": has_a,
        "has_stdout_log": stdout_path.exists(),
    }
    if metrics_path is not None:
        metrics = np.load(metrics_path)
        for index, name in enumerate(METRIC_NAMES):
            row[name] = float(metrics[index]) if index < len(metrics) else float("nan")
        row["metrics_source"] = str(metrics_path)
    elif result_log is not None:
        row.update(result_log)
        for name in METRIC_NAMES:
            row.setdefault(name, float("nan"))
        row["metrics_source"] = str(run_dir / "result_long_term_forecast.txt")
    else:
        for name in METRIC_NAMES:
            row[name] = float("nan")
        row["metrics_source"] = ""
    return row


def collect_rows(raw_root: Path, seed: int) -> list[dict[str, object]]:
    meta_types = sorted(
        path.name
        for path in raw_root.iterdir()
        if path.is_dir() and path.name != "_logs"
    )
    rows: list[dict[str, object]] = []
    for meta_type in meta_types:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                row = parse_run(raw_root, meta_type, dataset, horizon, seed)
                if row is not None:
                    rows.append(row)
    return rows


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def compare_meta_types(rows: list[dict[str, object]], baseline_meta: str, candidate_meta: str) -> list[dict[str, object]]:
    by_key = {
        (str(row["meta_type"]), str(row["dataset"]), int(row["horizon"])): row
        for row in rows
        if finite(row.get("mse"))
    }
    comparisons: list[dict[str, object]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            base = by_key.get((baseline_meta, dataset, horizon))
            cand = by_key.get((candidate_meta, dataset, horizon))
            if base is None or cand is None:
                continue
            base_mse = float(base["mse"])
            cand_mse = float(cand["mse"])
            base_mae = float(base["mae"])
            cand_mae = float(cand["mae"])
            comparisons.append(
                {
                    "candidate_meta_type": candidate_meta,
                    "baseline_meta_type": baseline_meta,
                    "dataset": dataset,
                    "horizon": horizon,
                    "candidate_mse": cand_mse,
                    "baseline_mse": base_mse,
                    "relative_mse_pct": (cand_mse / base_mse - 1.0) * 100.0,
                    "candidate_mae": cand_mae,
                    "baseline_mae": base_mae,
                    "relative_mae_pct": (cand_mae / base_mae - 1.0) * 100.0,
                    "candidate_wins_mse": cand_mse < base_mse,
                    "is_specialist_gap": (dataset, horizon) in SPECIALIST_GAPS,
                }
            )
    return comparisons


def summarize(rows: list[dict[str, object]], comparisons: list[dict[str, object]]) -> dict[str, object]:
    completed_rows = [row for row in rows if finite(row.get("mse"))]
    meta_types_present = sorted({str(row["meta_type"]) for row in rows})
    all_rows = [row for row in rows if row["meta_type"] == "all" and finite(row.get("mse"))]
    all_dataset_means = {
        dataset: mean(float(row["mse"]) for row in all_rows if row["dataset"] == dataset)
        for dataset in DATASETS
        if any(row["dataset"] == dataset for row in all_rows)
    }
    specialist_rows = [
        row for row in comparisons
        if row["candidate_meta_type"] == "all" and row["is_specialist_gap"]
    ]
    all_vs_diag = [row for row in comparisons if row["candidate_meta_type"] == "all" and row["baseline_meta_type"] == "diag"]
    gate = {
        "all_12_runs_complete": len(all_rows) == len(DATASETS) * len(HORIZONS),
        "diag_control_available": "diag" in meta_types_present,
        "off_diag_control_available": "off_diag" in meta_types_present,
        "all_vs_diag_mean_mse_improves": bool(all_vs_diag) and mean(float(row["relative_mse_pct"]) for row in all_vs_diag) < 0.0,
        "all_vs_diag_wins_at_least_7": bool(all_vs_diag) and sum(1 for row in all_vs_diag if row["candidate_wins_mse"]) >= 7,
        "specialist_gap_wins_at_least_2": bool(specialist_rows) and sum(1 for row in specialist_rows if row["candidate_wins_mse"]) >= 2,
        "covariance_artifacts_present": all(bool(row["has_cov_matrix_pdf"]) for row in all_rows) if all_rows else False,
    }
    gate["pass"] = all(
        [
            gate["all_12_runs_complete"],
            gate["diag_control_available"],
            gate["all_vs_diag_mean_mse_improves"],
            gate["all_vs_diag_wins_at_least_7"],
            gate["specialist_gap_wins_at_least_2"],
            gate["covariance_artifacts_present"],
        ]
    )
    return {
        "completed_metric_rows": len(completed_rows),
        "meta_types_present": meta_types_present,
        "all_runs_completed": len(all_rows),
        "all_dataset_mean_mse": all_dataset_means,
        "comparisons_available": len(comparisons),
        "gate": gate,
    }


def pct(value: float) -> str:
    return f"{value:+.2f}%"


def write_report(
    path: Path,
    rows: list[dict[str, object]],
    comparisons: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    gate = summary["gate"]
    meta_types = ", ".join(str(item) for item in summary["meta_types_present"]) or "none"
    decision = "passes" if gate["pass"] else "is incomplete or fails"
    lines = [
        "# Phase2-D QDF Upstream Reproduction Gate Report",
        "",
        "## Decision",
        "",
        f"[Decision] QDF upstream reproduction gate {decision}.",
        "",
        f"- meta_types_present: `{meta_types}`",
        f"- completed_metric_rows: `{summary['completed_metric_rows']}`",
        f"- all_runs_completed: `{summary['all_runs_completed']}/12`",
        "",
        "## Gate",
        "",
    ]
    for key, value in gate.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## All Meta-Type Metrics",
            "",
            "| Dataset | Horizon | MSE | MAE | Cov loss | Cov artifact |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        if row["meta_type"] != "all" or not finite(row.get("mse")):
            continue
        lines.append(
            "| {dataset} | {horizon} | {mse:.6f} | {mae:.6f} | {cov:.6f} | {cov_artifact} |".format(
                dataset=row["dataset"],
                horizon=row["horizon"],
                mse=float(row["mse"]),
                mae=float(row["mae"]),
                cov=float(row["cov_loss"]),
                cov_artifact=row["has_cov_matrix_pdf"],
            )
        )
    if comparisons:
        lines.extend(
            [
                "",
                "## Meta-Type Comparisons",
                "",
                "| Candidate | Baseline | Dataset | Horizon | Specialist gap | Relative MSE | Candidate MSE | Baseline MSE |",
                "| --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for row in comparisons:
            lines.append(
                "| {cand} | {base} | {dataset} | {horizon} | {gap} | {rel} | {cand_mse:.6f} | {base_mse:.6f} |".format(
                    cand=row["candidate_meta_type"],
                    base=row["baseline_meta_type"],
                    dataset=row["dataset"],
                    horizon=row["horizon"],
                    gap=row["is_specialist_gap"],
                    rel=pct(float(row["relative_mse_pct"])),
                    cand_mse=float(row["candidate_mse"]),
                    base_mse=float(row["baseline_mse"]),
                )
            )
    else:
        lines.extend(
            [
                "",
                "## Meta-Type Comparisons",
                "",
                "[Fact] No meta-type control comparison is available yet. Run `META_TYPES=\"diag off_diag\"` to complete controls.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "[Fact] This report parses native QDF upstream outputs. It does not compare directly against FATST R.3 because the upstream model and training protocol are different.",
            "",
            "[Decision Rule] QDF should only be localized into FATST if `meta_type=all` beats its own diagonal control and learned covariance artifacts are present. If only `all` has run, this gate remains incomplete even when metrics exist.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase2-D QDF upstream reproduction gate.")
    parser.add_argument("--analysis-root", default="analysis/phase2_qdf_upstream_gate_20260623")
    parser.add_argument("--seed", type=int, default=2023)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root = Path(args.analysis_root)
    raw_root = analysis_root / "raw"
    rows = collect_rows(raw_root, args.seed) if raw_root.exists() else []
    comparisons = []
    comparisons.extend(compare_meta_types(rows, "diag", "all"))
    comparisons.extend(compare_meta_types(rows, "off_diag", "all"))
    summary = summarize(rows, comparisons)
    write_csv(analysis_root / "phase2_qdf_upstream_metrics.csv", rows)
    write_csv(analysis_root / "phase2_qdf_upstream_meta_type_comparison.csv", comparisons)
    (analysis_root / "phase2_qdf_upstream_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(
        analysis_root / "phase2_qdf_upstream_decision_report.md",
        rows,
        comparisons,
        summary,
    )
    print(f"decision_report={analysis_root / 'phase2_qdf_upstream_decision_report.md'}")


if __name__ == "__main__":
    main()
