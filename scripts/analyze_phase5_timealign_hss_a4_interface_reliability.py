from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


DATASETS = ["ETTh2", "ETTm1", "Weather"]
HORIZONS = [96, 192, 336, 720]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def relative_pct(value: float, baseline: float) -> float:
    if baseline == 0:
        return float("nan")
    return (value / baseline - 1.0) * 100.0


def fmt(value: float) -> str:
    return f"{value:.3f}"


def load_reference(path: Path, path_id: str, family: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        rows.append(
            {
                "dataset": row["dataset"],
                "target_horizon": int(row["target_horizon"]),
                "path_id": path_id,
                "family": family,
                "mse": float(row.get("mse", row.get("fixed_mse", "nan"))),
                "mae": float(row.get("mae", row.get("fixed_mae", "nan"))),
                "source_path": row["source_path"],
            }
        )
    return rows


def load_a3e_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    arm_map = {
        "target_conditioned_nested_warm": ("a3e_target_conditioned_warm", "target_conditioned_nested"),
        "target_conditioned_nested_scratch": ("a3e_target_conditioned_scratch", "target_conditioned_nested"),
    }
    for row in read_csv(path):
        path_id, family = arm_map[row["arm"]]
        rows.append(
            {
                "dataset": row["dataset"],
                "target_horizon": int(row["target_horizon"]),
                "path_id": path_id,
                "family": family,
                "mse": float(row["mse"]),
                "mae": float(row["mae"]),
                "source_path": row["source_path"],
            }
        )
    return rows


def index_by_setting(rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["target_horizon"]), []).append(row)
    return grouped


def summarize_paths(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = index_by_setting(rows)
    best_rows: list[dict[str, Any]] = []
    enriched_rows: list[dict[str, Any]] = []

    for dataset in DATASETS:
        for horizon in HORIZONS:
            items = sorted(grouped[(dataset, horizon)], key=lambda item: item["mse"])
            best = items[0]
            second = items[1]
            fixed = next(item for item in items if item["path_id"] == "fixed_specialist")
            h1 = next(item for item in items if item["path_id"] == "h1_target_set")
            best_rows.append(
                {
                    "dataset": dataset,
                    "target_horizon": horizon,
                    "best_path": best["path_id"],
                    "best_family": best["family"],
                    "best_mse": best["mse"],
                    "second_path": second["path_id"],
                    "second_mse": second["mse"],
                    "best_margin_vs_second_pct": relative_pct(best["mse"], second["mse"]),
                    "best_relative_vs_fixed_pct": relative_pct(best["mse"], fixed["mse"]),
                    "best_relative_vs_h1_pct": relative_pct(best["mse"], h1["mse"]),
                }
            )
            for item in items:
                enriched = dict(item)
                enriched["relative_vs_setting_best_pct"] = relative_pct(item["mse"], best["mse"])
                enriched["relative_vs_fixed_pct"] = relative_pct(item["mse"], fixed["mse"])
                enriched["relative_vs_h1_pct"] = relative_pct(item["mse"], h1["mse"])
                enriched["is_best"] = item["path_id"] == best["path_id"]
                enriched["within_0p2pct_of_best"] = enriched["relative_vs_setting_best_pct"] <= 0.2
                enriched_rows.append(enriched)

    summary_rows: list[dict[str, Any]] = []
    path_ids = sorted({row["path_id"] for row in rows})
    for dataset in DATASETS + ["ALL"]:
        subset = enriched_rows if dataset == "ALL" else [row for row in enriched_rows if row["dataset"] == dataset]
        for path_id in path_ids:
            path_subset = [row for row in subset if row["path_id"] == path_id]
            if not path_subset:
                continue
            summary_rows.append(
                {
                    "dataset": dataset,
                    "path_id": path_id,
                    "family": path_subset[0]["family"],
                    "settings": len(path_subset),
                    "mean_mse": sum(row["mse"] for row in path_subset) / len(path_subset),
                    "wins_as_best": sum(1 for row in path_subset if row["is_best"]),
                    "within_0p2pct_best": sum(1 for row in path_subset if row["within_0p2pct_of_best"]),
                    "mean_gap_to_best_pct": sum(row["relative_vs_setting_best_pct"] for row in path_subset)
                    / len(path_subset),
                    "mean_relative_vs_fixed_pct": sum(row["relative_vs_fixed_pct"] for row in path_subset)
                    / len(path_subset),
                    "mean_relative_vs_h1_pct": sum(row["relative_vs_h1_pct"] for row in path_subset)
                    / len(path_subset),
                }
            )
    return enriched_rows, best_rows, summary_rows


def oracle_rows(enriched_rows: list[dict[str, Any]], best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path_summary = summarize_paths_from_enriched(enriched_rows)
    grouped = index_by_setting(enriched_rows)
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS + ["ALL"]:
        candidate_static = [row for row in path_summary if row["dataset"] == dataset]
        best_static = min(candidate_static, key=lambda row: row["mean_mse"])
        static_path = best_static["path_id"]
        settings = best_rows if dataset == "ALL" else [row for row in best_rows if row["dataset"] == dataset]
        oracle_mse = sum(row["best_mse"] for row in settings) / len(settings)
        static_mse_values = []
        h1_mse_values = []
        fixed_mse_values = []
        for row in settings:
            items = grouped[(row["dataset"], row["target_horizon"])]
            static_mse_values.append(next(item["mse"] for item in items if item["path_id"] == static_path))
            h1_mse_values.append(next(item["mse"] for item in items if item["path_id"] == "h1_target_set"))
            fixed_mse_values.append(next(item["mse"] for item in items if item["path_id"] == "fixed_specialist"))
        static_mse = sum(static_mse_values) / len(static_mse_values)
        h1_mse = sum(h1_mse_values) / len(h1_mse_values)
        fixed_mse = sum(fixed_mse_values) / len(fixed_mse_values)
        rows.append(
            {
                "dataset": dataset,
                "oracle_settings": len(settings),
                "oracle_mean_mse": oracle_mse,
                "best_static_path": static_path,
                "best_static_mean_mse": static_mse,
                "oracle_relative_vs_best_static_pct": relative_pct(oracle_mse, static_mse),
                "oracle_relative_vs_h1_pct": relative_pct(oracle_mse, h1_mse),
                "oracle_relative_vs_fixed_pct": relative_pct(oracle_mse, fixed_mse),
            }
        )
    return rows


def summarize_paths_from_enriched(enriched_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    path_ids = sorted({row["path_id"] for row in enriched_rows})
    for dataset in DATASETS + ["ALL"]:
        subset = enriched_rows if dataset == "ALL" else [row for row in enriched_rows if row["dataset"] == dataset]
        for path_id in path_ids:
            path_subset = [row for row in subset if row["path_id"] == path_id]
            if not path_subset:
                continue
            summary_rows.append(
                {
                    "dataset": dataset,
                    "path_id": path_id,
                    "family": path_subset[0]["family"],
                    "settings": len(path_subset),
                    "mean_mse": sum(row["mse"] for row in path_subset) / len(path_subset),
                    "wins_as_best": sum(1 for row in path_subset if row["is_best"]),
                    "within_0p2pct_best": sum(1 for row in path_subset if row["within_0p2pct_of_best"]),
                    "mean_gap_to_best_pct": sum(row["relative_vs_setting_best_pct"] for row in path_subset)
                    / len(path_subset),
                    "mean_relative_vs_fixed_pct": sum(row["relative_vs_fixed_pct"] for row in path_subset)
                    / len(path_subset),
                    "mean_relative_vs_h1_pct": sum(row["relative_vs_h1_pct"] for row in path_subset)
                    / len(path_subset),
                }
            )
    return summary_rows


def family_rows(enriched_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    families = sorted({row["family"] for row in enriched_rows})
    for dataset in DATASETS + ["ALL"]:
        subset = enriched_rows if dataset == "ALL" else [row for row in enriched_rows if row["dataset"] == dataset]
        for family in families:
            family_subset = [row for row in subset if row["family"] == family]
            if not family_subset:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "path_count": len({row["path_id"] for row in family_subset}),
                    "settings": len(family_subset),
                    "wins_as_best": sum(1 for row in family_subset if row["is_best"]),
                    "within_0p2pct_best": sum(1 for row in family_subset if row["within_0p2pct_of_best"]),
                    "mean_gap_to_best_pct": sum(row["relative_vs_setting_best_pct"] for row in family_subset)
                    / len(family_subset),
                    "mean_relative_vs_fixed_pct": sum(row["relative_vs_fixed_pct"] for row in family_subset)
                    / len(family_subset),
                    "mean_relative_vs_h1_pct": sum(row["relative_vs_h1_pct"] for row in family_subset)
                    / len(family_subset),
                }
            )
    return rows


def write_report(
    path: Path,
    best_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    family_summary: list[dict[str, Any]],
    oracle_summary: list[dict[str, Any]],
) -> None:
    all_summary = [row for row in summary_rows if row["dataset"] == "ALL"]
    all_summary = sorted(all_summary, key=lambda row: row["mean_gap_to_best_pct"])
    all_family = [row for row in family_summary if row["dataset"] == "ALL"]
    all_family = sorted(all_family, key=lambda row: row["mean_gap_to_best_pct"])

    lines = [
        "# Phase5 A4 Interface Reliability Diagnostic",
        "",
        "## 诊断目标",
        "",
        "[Step 2/3] A3E 失败后，本诊断不提出新 head，也不把 dataset/horizon 手工选择写成最终方法。",
        "它只回答：现有 capacity-preserving / prefix-aware paths 是否存在稳定可靠性差异，以及这种差异是否大到值得设计 learned reliability routing。",
        "",
        "Dataset universe: `ETTh2 + ETTm1 + Weather`；每个 dataset 使用 `96/192/336/720` 四个 target horizons。",
        "",
        "## Path-Level Reliability",
        "",
        "| rank | path_id | family | wins_as_best | within_0.2%_best | mean_gap_to_best_% | vs_fixed_% | vs_h1_% |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, row in enumerate(all_summary, start=1):
        lines.append(
            "| {rank} | `{path_id}` | `{family}` | {wins} | {near} | {gap} | {fixed} | {h1} |".format(
                rank=idx,
                path_id=row["path_id"],
                family=row["family"],
                wins=row["wins_as_best"],
                near=row["within_0p2pct_best"],
                gap=fmt(row["mean_gap_to_best_pct"]),
                fixed=fmt(row["mean_relative_vs_fixed_pct"]),
                h1=fmt(row["mean_relative_vs_h1_pct"]),
            )
        )

    lines.extend(
        [
            "",
            "## Family-Level Reliability",
            "",
            "| rank | family | path_count | wins_as_best | within_0.2%_best | mean_gap_to_best_% | vs_fixed_% | vs_h1_% |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for idx, row in enumerate(all_family, start=1):
        lines.append(
            "| {rank} | `{family}` | {count} | {wins} | {near} | {gap} | {fixed} | {h1} |".format(
                rank=idx,
                family=row["family"],
                count=row["path_count"],
                wins=row["wins_as_best"],
                near=row["within_0p2pct_best"],
                gap=fmt(row["mean_gap_to_best_pct"]),
                fixed=fmt(row["mean_relative_vs_fixed_pct"]),
                h1=fmt(row["mean_relative_vs_h1_pct"]),
            )
        )

    lines.extend(
        [
            "",
            "## Best Path Map",
            "",
            "| dataset | horizon | best_path | second_path | margin_vs_second_% | best_vs_fixed_% | best_vs_h1_% |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in best_rows:
        lines.append(
            "| {dataset} | {horizon} | `{best}` | `{second}` | {margin} | {fixed} | {h1} |".format(
                dataset=row["dataset"],
                horizon=row["target_horizon"],
                best=row["best_path"],
                second=row["second_path"],
                margin=fmt(row["best_margin_vs_second_pct"]),
                fixed=fmt(row["best_relative_vs_fixed_pct"]),
                h1=fmt(row["best_relative_vs_h1_pct"]),
            )
        )

    lines.extend(
        [
            "",
            "## Oracle Routing Upper Bound",
            "",
            "| dataset | settings | best_static_path | oracle_vs_best_static_% | oracle_vs_h1_% | oracle_vs_fixed_% |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in oracle_summary:
        lines.append(
            "| {dataset} | {settings} | `{path}` | {static} | {h1} | {fixed} |".format(
                dataset=row["dataset"],
                settings=row["oracle_settings"],
                path=row["best_static_path"],
                static=fmt(row["oracle_relative_vs_best_static_pct"]),
                h1=fmt(row["oracle_relative_vs_h1_pct"]),
                fixed=fmt(row["oracle_relative_vs_fixed_pct"]),
            )
        )

    all_oracle = next(row for row in oracle_summary if row["dataset"] == "ALL")
    lines.extend(
        [
            "",
            "## 机制判断",
            "",
            f"- [Fact] 当前 best static path 是 `{all_oracle['best_static_path']}`；per-setting oracle 相对它仍有 `{fmt(all_oracle['oracle_relative_vs_best_static_pct'])}%` MSE 改善上限。",
            "- [Strong Evidence] 没有单一路径在 12 个 setting 上稳定最优；best path map 同时出现 dense target-set、teacher-preserved nested、target-conditioned nested 和 warm-started nested。",
            "- [Strong Evidence] 这说明 Stage A 的核心问题不应再写成寻找一个 universal prefix-aware head；更合理的问题是 capacity-preserving path 的可靠性在不同 future context 下变化。",
            "- [Self-Critique] 但该诊断仍基于 test horizon MSE 的离线 oracle。它只能证明 reliability 差异存在，不能证明可部署的 routing signal 已经存在。",
            "",
            "## 下一步",
            "",
            "进入 A4R：`Reliability-Aware Capacity-Preserving Interface Diagnostic`。",
            "下一轮不应该按 dataset/horizon id 手工选择路径，而应导出可训练前或 validation-time 获得的 reliability signals，例如 teacher-student disagreement、prefix residual、validation gap、segment volatility。",
            "只有当这些 signals 能解释 best-path map 或 gap-to-best，才进入 learned/estimated interface routing method。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("analysis/phase5_timealign_hss_a3e_ettm1_replacement_gate_20260701"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis/phase5_timealign_hss_a4_interface_reliability_diagnostic_20260701"),
    )
    args = parser.parse_args()

    root = args.input_root
    rows: list[dict[str, Any]] = []
    rows.extend(load_reference(root / "phase5_timealign_hss_a3e_ettm1_fixed_reference.csv", "fixed_specialist", "fixed"))
    rows.extend(load_reference(root / "phase5_timealign_hss_a3e_ettm1_h1_reference.csv", "h1_target_set", "dense_target_set"))
    rows.extend(load_reference(root / "phase5_timealign_hss_a3e_ettm1_h1c_reference.csv", "h1c_row_gated", "dense_target_set"))
    rows.extend(load_reference(root / "phase5_timealign_hss_a3e_ettm1_a2_reference.csv", "a2_nested", "primary_nested"))
    rows.extend(load_reference(root / "phase5_timealign_hss_a3e_ettm1_a3c_reference.csv", "a3c_warm_nested", "primary_nested"))
    rows.extend(load_reference(root / "phase5_timealign_hss_a3e_ettm1_a3d_reference.csv", "a3d_teacher_preserved", "teacher_preserved_nested"))
    rows.extend(load_a3e_metrics(root / "phase5_timealign_hss_a3e_ettm1_metrics.csv"))

    enriched_rows, best_rows, summary_rows = summarize_paths(rows)
    family_summary = family_rows(enriched_rows)
    oracle_summary = oracle_rows(enriched_rows, best_rows)

    output_root = args.output_root
    write_csv(output_root / "phase5_timealign_hss_a4_all_paths.csv", enriched_rows)
    write_csv(output_root / "phase5_timealign_hss_a4_best_path_map.csv", best_rows)
    write_csv(output_root / "phase5_timealign_hss_a4_path_reliability_summary.csv", summary_rows)
    write_csv(output_root / "phase5_timealign_hss_a4_family_reliability_summary.csv", family_summary)
    write_csv(output_root / "phase5_timealign_hss_a4_oracle_routing_summary.csv", oracle_summary)
    write_report(
        output_root / "phase5_timealign_hss_a4_interface_reliability_diagnostic.md",
        best_rows,
        summary_rows,
        family_summary,
        oracle_summary,
    )


if __name__ == "__main__":
    main()
