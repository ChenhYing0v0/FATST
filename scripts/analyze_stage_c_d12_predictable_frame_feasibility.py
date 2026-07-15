#!/usr/bin/env python3
"""Aggregate the five-dataset D12-A predictable-frame diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_smoke:
        required = ("input_root", "design", "output_dir")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"missing required arguments: {', '.join(missing)}")
    return args


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def analyze(
    input_root: Path,
    design: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    rows = []
    for dataset in design["datasets"]:
        path = input_root / dataset / "dataset_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"missing D12-A summary: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["dataset"] != dataset:
            raise ValueError(f"dataset mismatch in {path}")
        rows.append(payload)
    support_count = sum(bool(row["dataset_support"]) for row in rows)
    invariant_pass = all(bool(row["invariants_pass"]) for row in rows)
    required = int(design["gates"]["dataset_required"])
    if not invariant_pass:
        decision = "diagnostic_invalid"
        d12_b_authorized = False
    elif support_count >= required:
        decision = "cape_problem_supported_d12b_authorized"
        d12_b_authorized = True
    else:
        decisions = Counter(row["decision"] for row in rows)
        if decisions["raw_label_subspace_already_sufficient_cape_closed"] >= required:
            decision = "raw_label_subspace_already_sufficient_cape_closed"
        elif decisions["predictable_signal_not_established"] >= required:
            decision = "predictable_signal_not_established_cape_closed"
        elif decisions["nonstationary_or_estimator_unstable"] >= required:
            decision = "nonstationary_or_estimator_unstable_cape_closed"
        elif decisions["pilot_specific_predictable_subspace"] >= required:
            decision = "pilot_specific_predictable_subspace_cape_closed"
        else:
            decision = "cape_problem_not_cross_dataset_rollback_step2"
        d12_b_authorized = False

    gate = {
        "diagnostic_id": design["diagnostic_id"],
        "dataset_count": len(rows),
        "dataset_support_count": support_count,
        "dataset_required": required,
        "all_invariants_pass": invariant_pass,
        "decision": decision,
        "d12_b_authorized": d12_b_authorized,
        "method_implementation_authorized": False,
        "uses_validation_split": False,
        "uses_test_split": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_fields = (
        "dataset",
        "a6_mean_oof_r2",
        "a6_min_fold_oof_r2",
        "ridge_mean_oof_r2",
        "a6_predictable_trace_fraction",
        "a6_fold_top32_overlap",
        "a6_ridge_rank32_overlap",
        "a6_rank256_optimal_capture",
        "a6_rank256_raw_capture",
        "a6_rank256_raw_relative_gap",
        "weight_effective_sample_fraction",
        "weight_max_share",
        "a6_predictability_pass",
        "a6_trace_pass",
        "a6_fold_stability_pass",
        "rank256_headroom_pass",
        "pilot_robustness_pass",
        "invariants_pass",
        "dataset_support",
        "decision",
    )
    write_csv(
        output_dir / "d12_a_dataset_gate.csv",
        [{field: row[field] for field in selected_fields} for row in rows],
    )
    (output_dir / "d12_a_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        f"# StageC {design['diagnostic_id']} Predictable-Covariance Diagnostic Result",
        "",
        "## Decision Summary",
        "",
        "| Field | Result |",
        "| --- | --- |",
        f"| `current_step` | joint Contribution 1/2 Step 2-3 |",
        f"| `dataset_support` | `{support_count}/{len(rows)}`；required=`{required}` |",
        f"| `all_invariants_pass` | `{str(invariant_pass).lower()}` |",
        f"| `decision` | `{decision}` |",
        f"| `D12-B_authorized` | `{str(d12_b_authorized).lower()}` |",
        f"| `risk_weight_mode` | `{design.get('risk_weight_mode', 'uniform')}` |",
        "| `method/test_authorized` | `false / false` |",
        "",
        "## Dataset Results",
        "",
        "| Dataset | A6 OOF R2 | Min fold R2 | Ridge R2 | Pred trace/label | "
        "A6 fold overlap@32 | A6-ridge overlap@32 | Raw gap@256 | Weight ESS | Support | Decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        report_lines.append(
            "| {dataset} | {a6_r2} | {min_r2} | {ridge_r2} | {trace} | "
            "{fold_overlap} | {pilot_overlap} | {gap} | {weight_ess} | {support} | `{decision}` |".format(
                dataset=row["dataset"],
                a6_r2=fmt(row["a6_mean_oof_r2"]),
                min_r2=fmt(row["a6_min_fold_oof_r2"]),
                ridge_r2=fmt(row["ridge_mean_oof_r2"]),
                trace=fmt(row["a6_predictable_trace_fraction"]),
                fold_overlap=fmt(row["a6_fold_top32_overlap"]),
                pilot_overlap=fmt(row["a6_ridge_rank32_overlap"]),
                gap=fmt(row["a6_rank256_raw_relative_gap"], 6),
                weight_ess=fmt(row["weight_effective_sample_fraction"]),
                support=fmt(row["dataset_support"]),
                decision=row["decision"],
            )
        )
    report_lines.extend(
        [
            "",
            "## Metric Meaning",
            "",
            "- `A6 OOF R2`：purged forward OOF prediction相对fold-centered zero predictor的改善；",
            "- `Pred trace/label`：A6 prediction covariance与label covariance trace之比；",
            "- `fold overlap@32`：两个forward folds的top-32 predictable subspace overlap；",
            "- `A6-ridge overlap@32`：nonlinear A6与DCT-ridge pilot的aggregate top-32 overlap；",
            "- `Raw gap@256`：optimal predictable frame相对raw-label PCA frame多捕获的predictable energy；",
            "- `Weight ESS`：risk weights的effective sample size除以row count；",
            "- support要求上述五项与所有covariance invariants同时通过。",
            "",
            "## Failure Attribution",
            "",
        ]
    )
    if decision == "cape_problem_supported_d12b_authorized":
        report_lines.extend(
            [
                "[Decision] D12-A只建立predictable-frame problem headroom，并授权D12-B offline/probe。",
                "它没有验证PRISM architecture、CAPE effectiveness或paper novelty。",
            ]
        )
    elif decision == "raw_label_subspace_already_sufficient_cape_closed":
        report_lines.extend(
            [
                "[Decision] predictable signal与稳定性可能存在，但在实际rank256下raw-label frame已经",
                "覆盖了predictable covariance。CAPE作为独立training contribution关闭；不得用rank32/64",
                "diagnostic结果替换primary rank256 gate。",
            ]
        )
    elif decision == "diagnostic_invalid":
        report_lines.append(
            "[Decision] covariance/normalization invariant失败；本结果不能用于方向级判断。"
        )
    else:
        report_lines.extend(
            [
                "[Decision] CAPE problem没有跨dataset通过冻结gate，D12-B不授权。",
                "失败只关闭当前predictable-frame estimator problem，不自动否定D6 prefix-locality问题。",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Protocol Boundary",
            "",
            "- train split only；validation/test未读取；",
            "- two-fold forward cross-fitting，history+future raw intervals之间purge 1439 windows；",
            "- pilot固定20 epochs，不用OOF选择checkpoint；",
            f"- risk weight=`{design.get('risk_weight_mode', 'uniform')}`；",
            "- final paper model不复用pilot weights；",
            "- 详细fold/covariance statistics保存在各dataset目录。",
            "",
        ]
    )
    (output_dir / "d12_a_result_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    return gate


def synthetic_smoke() -> None:
    design = {
        "diagnostic_id": "D12-A",
        "datasets": ["A", "B", "C", "D", "E"],
        "gates": {"dataset_required": 3},
    }
    base = {
        "a6_mean_oof_r2": 0.2,
        "a6_min_fold_oof_r2": 0.1,
        "ridge_mean_oof_r2": 0.1,
        "a6_predictable_trace_fraction": 0.3,
        "a6_fold_top32_overlap": 0.6,
        "a6_ridge_rank32_overlap": 0.5,
        "a6_rank256_optimal_capture": 0.99,
        "a6_rank256_raw_capture": 0.98,
        "a6_rank256_raw_relative_gap": 0.01,
        "weight_effective_sample_fraction": 0.8,
        "weight_max_share": 0.001,
        "a6_predictability_pass": True,
        "a6_trace_pass": True,
        "a6_fold_stability_pass": True,
        "rank256_headroom_pass": True,
        "pilot_robustness_pass": True,
        "invariants_pass": True,
        "dataset_support": True,
        "decision": "cape_problem_supported",
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for dataset in design["datasets"]:
            directory = root / dataset
            directory.mkdir()
            (directory / "dataset_summary.json").write_text(
                json.dumps({"dataset": dataset, **base}),
                encoding="utf-8",
            )
        gate = analyze(root, design, root / "analysis")
        if gate["decision"] != "cape_problem_supported_d12b_authorized":
            raise RuntimeError("D12-A analyzer smoke decision failed")
    print("stage_c_d12_analyzer_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    design = json.loads(args.design.read_text(encoding="utf-8"))
    gate = analyze(args.input_root, design, args.output_dir)
    print(
        f"stage_c_d12_analyzer_done decision={gate['decision']} "
        f"support={gate['dataset_support_count']}/{gate['dataset_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
