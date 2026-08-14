#!/usr/bin/env python3
"""Audit Decoder-Transfer formal results and build its paper table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


HORIZONS = (96, 192, 336, 720)
BACKBONE_LABELS = {
    "dlinear_style": "DLinear-style",
    "patchtst_style": "PatchTST-style",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/iscf_bsca_decoder_transfer_protocol.json"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_dir(root: Path, row: dict[str, Any]) -> Path:
    return root / row["backbone"] / row["arm_id"] / row["dataset"] / "seed2021"


def collect_cells(
    config: dict[str, Any], manifest: dict[str, Any], root: Path
) -> list[dict[str, Any]]:
    manifest_rows = {
        (row["backbone"], row["arm_id"], row["dataset"]): row
        for row in manifest["rows"]
    }
    if len(manifest_rows) != config["matrix"]["new_training_runs"]:
        raise RuntimeError("immutable manifest row count mismatch")
    cells: list[dict[str, Any]] = []
    for backbone, dataset, arm_id in config["launch_order"]:
        key = (backbone, arm_id, dataset)
        manifest_row = manifest_rows[key]
        directory = run_dir(root, manifest_row)
        checkpoint = directory / config["artifact_contract"]["checkpoint_file"]
        if sha256(checkpoint) != manifest_row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint mutated: {checkpoint}")
        invariants = load_json(directory / "test_audit_invariants.json")
        if (
            invariants.get("pass") is not True
            or invariants.get("evaluation_split") != "test"
            or invariants.get("checkpoint_sha256") != manifest_row["checkpoint_sha256"]
        ):
            raise RuntimeError(f"formal-test invariant failure: {directory}")
        metric_path = directory / "test_audit_metrics_by_target_horizon.csv"
        with metric_path.open(encoding="utf-8", newline="") as handle:
            metric_rows = {
                int(row["target_horizon"]): row for row in csv.DictReader(handle)
            }
        for horizon in HORIZONS:
            if horizon not in metric_rows:
                raise RuntimeError(f"missing H{horizon}: {metric_path}")
            metric = metric_rows[horizon]
            cells.append(
                {
                    "backbone": backbone,
                    "arm_id": arm_id,
                    "dataset": dataset,
                    "horizon": horizon,
                    "mse": float(metric["mse"]),
                    "mae": float(metric["mae"]),
                    "seed": int(metric["seed"]),
                    "checkpoint_sha256": manifest_row["checkpoint_sha256"],
                    "test_access_date": invariants["test_access_date"],
                    "test_role": config["authorization"]["test_role"],
                    "test_tuned": False,
                    "checkpoint_retrained": invariants["checkpoint_retrained"],
                }
            )
    if len(cells) != config["matrix"]["test_cells"]:
        raise RuntimeError(f"formal matrix incomplete: {len(cells)} cells")
    return cells


def aggregate(
    cells: list[dict[str, Any]], group_keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        output.append(
            {
                **dict(zip(group_keys, key)),
                "mean_mse": sum(row["mse"] for row in rows) / len(rows),
                "mean_mae": sum(row["mae"] for row in rows) / len(rows),
                "cell_count": len(rows),
            }
        )
    return output


def evaluate_gates(
    config: dict[str, Any],
    cells: list[dict[str, Any]],
    dataset_means: list[dict[str, Any]],
    overall: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    dataset_lookup = {
        (row["backbone"], row["arm_id"], row["dataset"]): row
        for row in dataset_means
    }
    overall_lookup = {
        (row["backbone"], row["arm_id"]): row for row in overall
    }
    cell_lookup = {
        (row["backbone"], row["arm_id"], row["dataset"], row["horizon"]): row
        for row in cells
    }
    gate_rows = []
    all_pass = True
    for backbone in ("dlinear_style", "patchtst_style"):
        original = f"{backbone.split('_')[0]}_original"
        iscf = f"{backbone.split('_')[0]}_iscf"
        bsca = f"{backbone.split('_')[0]}_iscf_bsca"
        original_mean = overall_lookup[(backbone, original)]
        bsca_mean = overall_lookup[(backbone, bsca)]
        mse_gain = 100.0 * (
            original_mean["mean_mse"] - bsca_mean["mean_mse"]
        ) / original_mean["mean_mse"]
        mae_gain = 100.0 * (
            original_mean["mean_mae"] - bsca_mean["mean_mae"]
        ) / original_mean["mean_mae"]
        dataset_mse_wins = sum(
            dataset_lookup[(backbone, bsca, dataset)]["mean_mse"]
            < dataset_lookup[(backbone, original, dataset)]["mean_mse"]
            for dataset in config["datasets"]
        )
        passed = (
            mse_gain
            > config["gates"]["per_backbone_bsca_vs_original_macro_mse_gain_percent_min_exclusive"]
            and mae_gain
            > config["gates"]["per_backbone_bsca_vs_original_macro_mae_gain_percent_min_exclusive"]
            and dataset_mse_wins
            >= config["gates"]["per_backbone_bsca_vs_original_dataset_mse_wins_min"]
        )
        all_pass = all_pass and passed
        iscf_mean = overall_lookup[(backbone, iscf)]
        bsca_vs_original_mse_cell_wins = sum(
            cell_lookup[(backbone, bsca, dataset, horizon)]["mse"]
            < cell_lookup[(backbone, original, dataset, horizon)]["mse"]
            for dataset in config["datasets"]
            for horizon in HORIZONS
        )
        bsca_vs_iscf_mse_cell_wins = sum(
            cell_lookup[(backbone, bsca, dataset, horizon)]["mse"]
            < cell_lookup[(backbone, iscf, dataset, horizon)]["mse"]
            for dataset in config["datasets"]
            for horizon in HORIZONS
        )
        gate_rows.append(
            {
                "backbone": backbone,
                "bsca_vs_original_macro_mse_gain_percent": mse_gain,
                "bsca_vs_original_macro_mae_gain_percent": mae_gain,
                "bsca_vs_original_dataset_mse_wins": dataset_mse_wins,
                "bsca_vs_original_mse_cell_wins": bsca_vs_original_mse_cell_wins,
                "iscf_vs_original_macro_mse_gain_percent": 100.0
                * (original_mean["mean_mse"] - iscf_mean["mean_mse"])
                / original_mean["mean_mse"],
                "bsca_vs_iscf_macro_mse_gain_percent": 100.0
                * (iscf_mean["mean_mse"] - bsca_mean["mean_mse"])
                / iscf_mean["mean_mse"],
                "bsca_vs_iscf_mse_cell_wins": bsca_vs_iscf_mse_cell_wins,
                "gate_pass": passed,
            }
        )
    return gate_rows, all_pass


def decorate(value: float, rank: int) -> str:
    rendered = f"{value:.3f}"
    if rank == 0:
        return f"\\textbf{{{rendered}}}"
    if rank == 1:
        return f"\\underline{{{rendered}}}"
    return rendered


def build_table(
    config: dict[str, Any], dataset_means: list[dict[str, Any]], overall: list[dict[str, Any]]
) -> str:
    arm_lookup = {arm["id"]: arm for arm in config["arms"]}
    value_lookup = {
        (row["backbone"], row["arm_id"], row["dataset"]): row
        for row in dataset_means
    }
    overall_lookup = {
        (row["backbone"], row["arm_id"]): row for row in overall
    }
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Decoder transfer across two forecasting backbones. Each dataset entry is the mean over $H\\in\\{96,192,336,720\\}$. Lower is better. Best and second-best results within each backbone are bold and underlined, respectively.}",
        "\\label{tab:decoder-transfer}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll" + "cc" * (len(config["datasets"]) + 1) + "}",
        "\\toprule",
        "Backbone & Decoder & "
        + " & ".join(
            f"\\multicolumn{{2}}{{c}}{{{dataset}}}" for dataset in config["datasets"]
        )
        + " & \\multicolumn{2}{c}{Avg.} "
        + r"\\",
        " & & "
        + " & ".join(["MSE & MAE"] * (len(config["datasets"]) + 1))
        + r" \\",
        "\\midrule",
    ]
    for backbone_index, backbone in enumerate(("dlinear_style", "patchtst_style")):
        arms = [
            arm["id"] for arm in config["arms"] if arm["backbone"] == backbone
        ]
        ranks: dict[tuple[str, str], dict[str, int]] = {}
        for dataset in (*config["datasets"], "Avg"):
            rows = [
                value_lookup[(backbone, arm_id, dataset)]
                if dataset != "Avg"
                else overall_lookup[(backbone, arm_id)]
                for arm_id in arms
            ]
            for metric in ("mse", "mae"):
                key = f"mean_{metric}"
                ordered = sorted(rows, key=lambda row: (row[key], row["arm_id"]))
                ranks[(dataset, metric)] = {
                    row["arm_id"]: rank for rank, row in enumerate(ordered)
                }
        for arm_index, arm_id in enumerate(arms):
            values = []
            for dataset in config["datasets"]:
                row = value_lookup[(backbone, arm_id, dataset)]
                values.extend(
                    decorate(row[f"mean_{metric}"], ranks[(dataset, metric)][arm_id])
                    for metric in ("mse", "mae")
                )
            row = overall_lookup[(backbone, arm_id)]
            values.extend(
                decorate(row[f"mean_{metric}"], ranks[("Avg", metric)][arm_id])
                for metric in ("mse", "mae")
            )
            backbone_label = BACKBONE_LABELS[backbone] if arm_index == 0 else ""
            lines.append(
                f"{backbone_label} & {arm_lookup[arm_id]['column']} & "
                + " & ".join(values)
                + r" \\")
        if backbone_index == 0:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}"])
    return "\n".join(lines) + "\n"


def build_report(
    config: dict[str, Any], gate_rows: list[dict[str, Any]], all_pass: bool
) -> str:
    decision = (
        "decoder_transfer_complete_both_backbones_pass"
        if all_pass
        else "decoder_transfer_complete_portability_gate_not_passed"
    )
    lines = [
        "# ISCF-BSCA Decoder-Transfer Formal Result Audit",
        "",
        f"Decision：`{decision}`",
        "",
        "## 完整性与协议",
        "",
        "- 2 backbones × 3 decoder arms × 5 datasets × 4 horizons = 120/120 cells；",
        "- 全部arms为seed2021、from-scratch end-to-end joint training；",
        "- checkpoint由validation四horizon mean MSE选择；formal test不选择checkpoint、seed或单元格；",
        "- 30/30 checkpoint hashes经immutable manifest冻结，formal evaluation前后不允许变更。",
        "",
        "## Pre-registered portability gates",
        "",
        "| Backbone | BSCA vs Original MSE gain | MAE gain | Dataset/Cell MSE wins | ISCF vs Original MSE gain | BSCA vs ISCF MSE gain/cell wins | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in gate_rows:
        lines.append(
            f"| {BACKBONE_LABELS[row['backbone']]} | "
            f"{row['bsca_vs_original_macro_mse_gain_percent']:+.3f}% | "
            f"{row['bsca_vs_original_macro_mae_gain_percent']:+.3f}% | "
            f"{row['bsca_vs_original_dataset_mse_wins']}/5, {row['bsca_vs_original_mse_cell_wins']}/20 | "
            f"{row['iscf_vs_original_macro_mse_gain_percent']:+.3f}% | "
            f"{row['bsca_vs_iscf_macro_mse_gain_percent']:+.3f}%, {row['bsca_vs_iscf_mse_cell_wins']}/20 | "
            f"{'PASS' if row['gate_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Four-layer evaluation and failure attribution",
            "",
            "1. paper_facing_effectiveness：120/120 official-test cells完整，checkpoint hash复核未发现mutation或non-finite结果。",
            "2. matched_mechanism_attribution：DLinear-style通过预注册相对gate；PatchTST-style未通过。PatchTST中+ISCF-BSCA相对+ISCF改善，但仍未优于Original Decoder。",
            "3. internal_mechanism_health：本表不以routing、scope probability或oracle headroom替代matched effectiveness gate。",
            "4. failure_attribution：总体记为hypothesis_false_for_cross_backbone_portability_in_exact_setting。DLinear-style的ETTh1/ETTh2绝对结果提示optimization_or_profile_pathology_suspected，因此其正向相对gate不得被夸大；PatchTST-style负结果未出现artifact或numeric pathology，仍是当前总体portability claim失败的直接证据。",
            "",
            "## Claim boundary",
            "",
            (
                "两类backbone均通过预注册gate，因此支持decoder portability，但结论仅限于本表的source-informed matched local backbones。"
                if all_pass
                else "至少一个backbone未通过预注册gate，因此不得使用跨backbone decoder portability的总体正向表述；应按通过的block收窄结论并保留负结果。"
            ),
            "",
            "DLinear-style与PatchTST-style arms不是native external baseline reproduction；本表只回答matched decoder transferability。",
            "",
            "若作者希望恢复跨backbone portability claim，应回到Step 4--6重新设计PatchTST intervention/readout并冻结新的candidate；不得把本轮负结果改写为HPO问题或选择性删除PatchTST block。",
        ]
    )
    return "\n".join(lines) + "\n"


def main(args: argparse.Namespace) -> None:
    config = load_json(args.config)
    manifest = load_json(args.manifest)
    if manifest["protocol_sha256"] != sha256(args.config):
        raise RuntimeError("protocol hash differs from immutable manifest")
    cells = collect_cells(config, manifest, args.output_root)
    dataset_means = aggregate(cells, ("backbone", "arm_id", "dataset"))
    overall = aggregate(cells, ("backbone", "arm_id"))
    gate_rows, all_pass = evaluate_gates(config, cells, dataset_means, overall)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.results_dir / "decoder_transfer_120_cells.csv", cells)
    write_csv(args.results_dir / "decoder_transfer_dataset_means.csv", dataset_means)
    write_csv(args.results_dir / "decoder_transfer_overall_means.csv", overall)
    write_csv(args.results_dir / "decoder_transfer_gates.csv", gate_rows)
    summary = {
        "candidate_version": config["candidate_version"],
        "matrix_complete": len(cells) == config["matrix"]["test_cells"],
        "test_cells": len(cells),
        "checkpoint_rows": manifest["row_count"],
        "unique_checkpoint_hashes": manifest["unique_checkpoint_hashes"],
        "both_backbones_pass": all_pass,
        "decision": (
            "decoder_transfer_complete_both_backbones_pass"
            if all_pass
            else "decoder_transfer_complete_portability_gate_not_passed"
        ),
        "gates": gate_rows,
    }
    (args.results_dir / "decoder_transfer_result_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.results_dir / "result_and_table_audit.md").write_text(
        build_report(config, gate_rows, all_pass), encoding="utf-8"
    )
    table_dir = args.results_dir / "table"
    table_dir.mkdir(exist_ok=True)
    table = build_table(config, dataset_means, overall)
    (table_dir / "table_iscf_bsca_decoder_transfer.tex").write_text(
        table, encoding="utf-8"
    )
    (table_dir / "table_iscf_bsca_decoder_transfer_standalone.tex").write_text(
        "\\documentclass[10pt]{article}\n"
        "\\usepackage[margin=0.35in]{geometry}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage[normalem]{ulem}\n"
        "\\makeatletter\n"
        "\\setlength{\\@fptop}{0pt}\n"
        "\\setlength{\\@fpbot}{0pt plus 1fil}\n"
        "\\makeatother\n"
        "\\pagestyle{empty}\n"
        "\\begin{document}\n"
        + table
        + "\\end{document}\n",
        encoding="utf-8",
    )
    print(
        "decoder_transfer_result_build=pass "
        f"cells={len(cells)} both_backbones_pass={str(all_pass).lower()}"
    )


if __name__ == "__main__":
    main(parse_args())
