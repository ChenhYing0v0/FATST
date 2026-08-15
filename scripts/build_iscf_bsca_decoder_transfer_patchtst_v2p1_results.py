#!/usr/bin/env python3
"""Build the amended Decoder-Transfer table from v1 controls and v2.1 rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


HORIZONS = (96, 192, 336, 720)
BACKBONES = ("dlinear_style", "patchtst_style")
BACKBONE_LABELS = {
    "dlinear_style": "DLinear-style",
    "patchtst_style": "PatchTST-style",
}
ARM_ORDER = {
    "dlinear_style": (
        "dlinear_original",
        "dlinear_iscf",
        "dlinear_iscf_bsca",
    ),
    "patchtst_style": (
        "patchtst_original",
        "patchtst_iscf",
        "patchtst_iscf_bsca",
    ),
}
ARM_LABELS = {
    "dlinear_original": "Original Decoder",
    "dlinear_iscf": "+ISCF",
    "dlinear_iscf_bsca": "+ISCF-BSCA",
    "patchtst_original": "Original Decoder",
    "patchtst_iscf": "+ISCF",
    "patchtst_iscf_bsca": "+ISCF-BSCA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/iscf_bsca_decoder_transfer_patchtst_v2p1_formal.json"
        ),
    )
    parser.add_argument("--new-output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--v1-results-dir", type=Path, required=True)
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def collect_reused_v1_cells(
    config: dict[str, Any], v1_results_dir: Path
) -> list[dict[str, Any]]:
    old_manifest = load_json(v1_results_dir / "immutable_training_manifest.json")
    manifest_hashes = {
        (row["backbone"], row["arm_id"], row["dataset"]): row[
            "checkpoint_sha256"
        ]
        for row in old_manifest["rows"]
    }
    old_rows = read_csv(v1_results_dir / "decoder_transfer_120_cells.csv")
    cells: list[dict[str, Any]] = []
    for row in old_rows:
        keep = row["backbone"] == "dlinear_style" or (
            row["backbone"] == "patchtst_style"
            and row["arm_id"] == "patchtst_original"
        )
        if not keep:
            continue
        key = (row["backbone"], row["arm_id"], row["dataset"])
        if row["checkpoint_sha256"] != manifest_hashes[key]:
            raise RuntimeError(f"v1 checkpoint provenance mismatch: {key}")
        cells.append(
            {
                "backbone": row["backbone"],
                "arm_id": row["arm_id"],
                "dataset": row["dataset"],
                "horizon": int(row["horizon"]),
                "mse": float(row["mse"]),
                "mae": float(row["mae"]),
                "seed": int(row["seed"]),
                "checkpoint_sha256": row["checkpoint_sha256"],
                "test_access_date": row["test_access_date"],
                "test_role": row["test_role"],
                "test_tuned": row["test_tuned"],
                "checkpoint_retrained": row["checkpoint_retrained"],
                "evidence_source": "reused_decoder_transfer_v1",
                "profile_id": "v1_shared_backbone_dataset_profile",
            }
        )
    expected = (
        config["reused_table_evidence"]["dlinear_three_arms_cells"]
        + config["reused_table_evidence"]["patchtst_original_cells"]
    )
    if len(cells) != expected:
        raise RuntimeError(f"reused v1 matrix incomplete: {len(cells)}/{expected}")
    return cells


def collect_new_cells(
    config: dict[str, Any], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    if manifest["protocol_sha256"] != sha256(Path(config["_config_path"])):
        raise RuntimeError("protocol differs from immutable v2.1 manifest")
    if manifest["row_count"] != 10 or manifest["unique_checkpoint_hashes"] != 10:
        raise RuntimeError("v2.1 manifest is incomplete")
    cells: list[dict[str, Any]] = []
    for manifest_row in manifest["rows"]:
        checkpoint = Path(manifest_row["run_dir"]) / config[
            "artifact_contract"
        ]["checkpoint_file"]
        if sha256(checkpoint) != manifest_row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint mutated: {checkpoint}")
        artifact_dir = Path(manifest_row["formal_artifact_dir"])
        invariants = load_json(artifact_dir / "test_audit_invariants.json")
        if (
            invariants.get("pass") is not True
            or invariants.get("evaluation_split") != "test"
            or invariants.get("checkpoint_sha256")
            != manifest_row["checkpoint_sha256"]
        ):
            raise RuntimeError(f"formal-test invariant failure: {artifact_dir}")
        metric_rows = {
            int(row["target_horizon"]): row
            for row in read_csv(
                artifact_dir / "test_audit_metrics_by_target_horizon.csv"
            )
        }
        for horizon in HORIZONS:
            if horizon not in metric_rows:
                raise RuntimeError(f"missing H{horizon}: {artifact_dir}")
            metric = metric_rows[horizon]
            cells.append(
                {
                    "backbone": "patchtst_style",
                    "arm_id": manifest_row["arm_id"],
                    "dataset": manifest_row["dataset"],
                    "horizon": horizon,
                    "mse": float(metric["mse"]),
                    "mae": float(metric["mae"]),
                    "seed": int(metric["seed"]),
                    "checkpoint_sha256": manifest_row["checkpoint_sha256"],
                    "test_access_date": invariants["test_access_date"],
                    "test_role": config["authorization"]["test_role"],
                    "test_tuned": False,
                    "checkpoint_retrained": invariants[
                        "checkpoint_retrained"
                    ],
                    "evidence_source": "patchtst_v2p1_formal_test",
                    "profile_id": manifest_row["profile_id"],
                }
            )
    if len(cells) != config["matrix"]["formal_test_cells"]:
        raise RuntimeError(f"new matrix incomplete: {len(cells)}/40")
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
    rows = []
    both_pass = True
    for backbone in BACKBONES:
        original, iscf, bsca = ARM_ORDER[backbone]
        original_mean = overall_lookup[(backbone, original)]
        iscf_mean = overall_lookup[(backbone, iscf)]
        bsca_mean = overall_lookup[(backbone, bsca)]
        mse_gain = 100.0 * (
            original_mean["mean_mse"] - bsca_mean["mean_mse"]
        ) / original_mean["mean_mse"]
        mae_gain = 100.0 * (
            original_mean["mean_mae"] - bsca_mean["mean_mae"]
        ) / original_mean["mean_mae"]
        dataset_wins = sum(
            dataset_lookup[(backbone, bsca, dataset)]["mean_mse"]
            < dataset_lookup[(backbone, original, dataset)]["mean_mse"]
            for dataset in config["datasets"]
        )
        gate_pass = (
            mse_gain
            > config["gates"][
                "bsca_vs_original_macro_mse_gain_percent_min_exclusive"
            ]
            and mae_gain
            > config["gates"][
                "bsca_vs_original_macro_mae_gain_percent_min_exclusive"
            ]
            and dataset_wins
            >= config["gates"]["bsca_vs_original_dataset_mse_wins_min"]
        )
        both_pass = both_pass and gate_pass
        rows.append(
            {
                "backbone": backbone,
                "bsca_vs_original_macro_mse_gain_percent": mse_gain,
                "bsca_vs_original_macro_mae_gain_percent": mae_gain,
                "bsca_vs_original_dataset_mse_wins": dataset_wins,
                "bsca_vs_original_mse_cell_wins": sum(
                    cell_lookup[(backbone, bsca, dataset, horizon)]["mse"]
                    < cell_lookup[(backbone, original, dataset, horizon)]["mse"]
                    for dataset in config["datasets"]
                    for horizon in HORIZONS
                ),
                "iscf_vs_original_macro_mse_gain_percent": 100.0
                * (original_mean["mean_mse"] - iscf_mean["mean_mse"])
                / original_mean["mean_mse"],
                "bsca_vs_iscf_macro_mse_gain_percent": 100.0
                * (iscf_mean["mean_mse"] - bsca_mean["mean_mse"])
                / iscf_mean["mean_mse"],
                "bsca_vs_iscf_mse_cell_wins": sum(
                    cell_lookup[(backbone, bsca, dataset, horizon)]["mse"]
                    < cell_lookup[(backbone, iscf, dataset, horizon)]["mse"]
                    for dataset in config["datasets"]
                    for horizon in HORIZONS
                ),
                "gate_pass": gate_pass,
            }
        )
    return rows, both_pass


def decorate(value: float, rank: int) -> str:
    rendered = f"{value:.3f}"
    if rank == 0:
        return f"\\textbf{{{rendered}}}"
    if rank == 1:
        return f"\\underline{{{rendered}}}"
    return rendered


def build_table(
    config: dict[str, Any],
    dataset_means: list[dict[str, Any]],
    overall: list[dict[str, Any]],
) -> str:
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
        "\\caption{Decoder transfer across two forecasting backbones. Each dataset entry is the mean over $H\\in\\{96,192,336,720\\}$. PatchTST-style $+$ISCF and $+$ISCF-BSCA use validation-selected, dataset-level decoder profiles shared across all four horizons; the Original Decoder and the complete DLinear-style block reuse the unchanged v1 evidence. Lower is better. Best and second-best results within each backbone are bold and underlined, respectively.}",
        "\\label{tab:decoder-transfer}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll" + "cc" * (len(config["datasets"]) + 1) + "}",
        "\\toprule",
        "Backbone & Decoder & "
        + " & ".join(
            f"\\multicolumn{{2}}{{c}}{{{dataset}}}"
            for dataset in config["datasets"]
        )
        + " & \\multicolumn{2}{c}{Avg.} \\\\",
        " & & "
        + " & ".join(["MSE & MAE"] * (len(config["datasets"]) + 1))
        + " \\\\",
        "\\midrule",
    ]
    for backbone_index, backbone in enumerate(BACKBONES):
        arms = ARM_ORDER[backbone]
        ranks: dict[tuple[str, str], dict[str, int]] = {}
        for dataset in (*config["datasets"], "Avg"):
            entries = [
                value_lookup[(backbone, arm, dataset)]
                if dataset != "Avg"
                else overall_lookup[(backbone, arm)]
                for arm in arms
            ]
            for metric in ("mse", "mae"):
                ordered = sorted(
                    entries,
                    key=lambda row: (row[f"mean_{metric}"], row["arm_id"]),
                )
                ranks[(dataset, metric)] = {
                    row["arm_id"]: rank
                    for rank, row in enumerate(ordered)
                }
        for arm_index, arm in enumerate(arms):
            values = []
            for dataset in config["datasets"]:
                entry = value_lookup[(backbone, arm, dataset)]
                values.extend(
                    decorate(
                        entry[f"mean_{metric}"],
                        ranks[(dataset, metric)][arm],
                    )
                    for metric in ("mse", "mae")
                )
            entry = overall_lookup[(backbone, arm)]
            values.extend(
                decorate(
                    entry[f"mean_{metric}"], ranks[("Avg", metric)][arm]
                )
                for metric in ("mse", "mae")
            )
            label = BACKBONE_LABELS[backbone] if arm_index == 0 else ""
            lines.append(
                f"{label} & {ARM_LABELS[arm]} & "
                + " & ".join(values)
                + " \\\\"
            )
        if backbone_index == 0:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}"])
    return "\n".join(lines) + "\n"


def build_report(
    config: dict[str, Any],
    gates: list[dict[str, Any]],
    both_pass: bool,
    historical: dict[str, float],
) -> str:
    decision = (
        "decoder_transfer_v2p1_complete_both_backbones_pass"
        if both_pass
        else "decoder_transfer_v2p1_complete_portability_gate_not_passed"
    )
    lines = [
        "# PatchTST Decoder-Transfer v2.1 Formal Result Audit",
        "",
        f"Decision：`{decision}`",
        "",
        "## HPO与修订边界",
        "",
        "- parent v2完成50/50训练，validation gate通过：macro MSE改善0.813%，5/5 datasets改善超过0.1%；",
        "- parent v2仅有40/50 unique hashes，两组极小decoder weight decay配置在各dataset上收敛为bitwise-identical checkpoints；因此parent artifact gate保持FAIL，未被事后放宽；",
        "- v2.1只冻结5个互异的validation-selected BSCA checkpoints，并from scratch补训5个matched ISCF checkpoints；每个dataset只选一个profile且四个horizon共用；",
        "- 新formal access覆盖10 checkpoints/40 cells；DLinear完整block与PatchTST Original Decoder的80 cells复用v1 evidence，不再次访问test。",
        "",
        "## Formal integrity",
        "",
        "- 10/10 rows、10 unique checkpoint hashes、5/5 matched initialization pairs；",
        "- 40/40 new official-test cells与120/120 combined cells完整；",
        "- checkpoint由four-H validation mean MSE选择，test不选择epoch、seed、horizon或table cell；",
        "- candidate为test-informed validation-HPO rescue，不声称untouched holdout。",
        "",
        "## Pre-registered gates",
        "",
        "| Backbone | BSCA vs Original MSE gain | MAE gain | Dataset/Cell MSE wins | ISCF vs Original MSE gain | BSCA vs ISCF MSE gain/cell wins | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in gates:
        lines.append(
            f"| {BACKBONE_LABELS[row['backbone']]} | "
            f"{row['bsca_vs_original_macro_mse_gain_percent']:+.3f}% | "
            f"{row['bsca_vs_original_macro_mae_gain_percent']:+.3f}% | "
            f"{row['bsca_vs_original_dataset_mse_wins']}/5, "
            f"{row['bsca_vs_original_mse_cell_wins']}/20 | "
            f"{row['iscf_vs_original_macro_mse_gain_percent']:+.3f}% | "
            f"{row['bsca_vs_iscf_macro_mse_gain_percent']:+.3f}%, "
            f"{row['bsca_vs_iscf_mse_cell_wins']}/20 | "
            f"{'PASS' if row['gate_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## HPO effect relative to v1",
            "",
            f"- PatchTST +ISCF macro MSE由{historical['v1_iscf_mse']:.6f}降至{historical['v2p1_iscf_mse']:.6f}（改善{historical['iscf_mse_gain_percent']:.3f}%），MAE改善{historical['iscf_mae_gain_percent']:.3f}%；",
            f"- PatchTST +ISCF-BSCA macro MSE由{historical['v1_bsca_mse']:.6f}降至{historical['v2p1_bsca_mse']:.6f}（改善{historical['bsca_mse_gain_percent']:.3f}%），但MAE恶化{-historical['bsca_mae_gain_percent']:.3f}%；",
            f"- BSCA相对Original的MSE deficit从{historical['v1_bsca_vs_original_mse_gain_percent']:.3f}%缩小到{historical['v2p1_bsca_vs_original_mse_gain_percent']:.3f}%，但MAE deficit从{historical['v1_bsca_vs_original_mae_gain_percent']:.3f}%扩大到{historical['v2p1_bsca_vs_original_mae_gain_percent']:.3f}%；",
            "- 新BSCA只在ETTm1和ETTm2的dataset-mean MSE上超过Original；Weather、ETTh1与ETTh2仍落后。",
            "",
            "## Four-layer decision",
            "",
            "1. `paper_facing_effectiveness`：完整120-cell表面决定最终performance viability；validation HPO本身不构成正式有效性证据。",
            "2. `matched_mechanism_attribution`：PatchTST +ISCF和+ISCF-BSCA共享encoder、rank、optimizer scale、seed与initialization class，仅objective不同；Original Decoder仍是同backbone native-readout control。",
            "3. `internal_mechanism_health`：diagnostics只用于解释，不替代相对Original的formal gate。",
            "4. `failure_attribution`：完整结果无numeric/artifact pathology。BSCA相对matched ISCF改善0.912% MSE并赢16/20 MSE cells，说明BSCA objective在该replacement head内部仍有作用；但两者都未超过native Original Decoder。因此exact two-backbone portability claim记为`hypothesis_false_for_cross_backbone_portability_after_decoder_HPO`，设计层更具体地指向`readout_or_head_design_wrong_for_PatchTST_representation_compatibility`，不能据此否定BSCA objective本身。",
            "",
            "## Claim boundary",
            "",
            (
                "两类backbone均通过gate，可恢复受限的decoder portability表述；必须同时披露v2.1 validation HPO、parent test exposure及v1 historical negative result。"
                if both_pass
                else "至少一个backbone未通过gate，因此不得恢复cross-backbone decoder portability总体正向结论；v1负结果的主要结论保持有效。"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    config["_config_path"] = str(args.config)
    manifest = load_json(args.manifest)
    cells = collect_reused_v1_cells(config, args.v1_results_dir)
    cells.extend(collect_new_cells(config, manifest))
    keys = {
        (row["backbone"], row["arm_id"], row["dataset"], row["horizon"])
        for row in cells
    }
    if len(cells) != 120 or len(keys) != 120:
        raise RuntimeError(
            f"combined matrix incomplete or duplicated: rows={len(cells)} "
            f"keys={len(keys)}"
        )
    dataset_means = aggregate(cells, ("backbone", "arm_id", "dataset"))
    overall = aggregate(cells, ("backbone", "arm_id"))
    gates, both_pass = evaluate_gates(config, cells, dataset_means, overall)
    old_overall_rows = read_csv(
        args.v1_results_dir / "decoder_transfer_overall_means.csv"
    )
    old_overall = {
        (row["backbone"], row["arm_id"]): row
        for row in old_overall_rows
    }
    new_overall = {
        (row["backbone"], row["arm_id"]): row for row in overall
    }
    original_mse = float(
        old_overall[("patchtst_style", "patchtst_original")]["mean_mse"]
    )
    original_mae = float(
        old_overall[("patchtst_style", "patchtst_original")]["mean_mae"]
    )
    historical: dict[str, float] = {}
    for short, arm in (
        ("iscf", "patchtst_iscf"),
        ("bsca", "patchtst_iscf_bsca"),
    ):
        old_mse = float(old_overall[("patchtst_style", arm)]["mean_mse"])
        old_mae = float(old_overall[("patchtst_style", arm)]["mean_mae"])
        new_mse = float(new_overall[("patchtst_style", arm)]["mean_mse"])
        new_mae = float(new_overall[("patchtst_style", arm)]["mean_mae"])
        historical[f"v1_{short}_mse"] = old_mse
        historical[f"v1_{short}_mae"] = old_mae
        historical[f"v2p1_{short}_mse"] = new_mse
        historical[f"v2p1_{short}_mae"] = new_mae
        historical[f"{short}_mse_gain_percent"] = 100.0 * (
            old_mse - new_mse
        ) / old_mse
        historical[f"{short}_mae_gain_percent"] = 100.0 * (
            old_mae - new_mae
        ) / old_mae
    historical["v1_bsca_vs_original_mse_gain_percent"] = 100.0 * (
        original_mse - historical["v1_bsca_mse"]
    ) / original_mse
    historical["v1_bsca_vs_original_mae_gain_percent"] = 100.0 * (
        original_mae - historical["v1_bsca_mae"]
    ) / original_mae
    historical["v2p1_bsca_vs_original_mse_gain_percent"] = 100.0 * (
        original_mse - historical["v2p1_bsca_mse"]
    ) / original_mse
    historical["v2p1_bsca_vs_original_mae_gain_percent"] = 100.0 * (
        original_mae - historical["v2p1_bsca_mae"]
    ) / original_mae

    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.results_dir / "decoder_transfer_120_cells.csv", cells)
    write_csv(
        args.results_dir / "decoder_transfer_dataset_means.csv", dataset_means
    )
    write_csv(args.results_dir / "decoder_transfer_overall_means.csv", overall)
    write_csv(args.results_dir / "decoder_transfer_gates.csv", gates)
    summary = {
        "candidate_version": config["candidate_version"],
        "matrix_complete": True,
        "combined_test_cells": len(cells),
        "new_formal_test_cells": config["matrix"]["formal_test_cells"],
        "reused_v1_cells": len(cells) - config["matrix"]["formal_test_cells"],
        "new_checkpoint_rows": manifest["row_count"],
        "new_unique_checkpoint_hashes": manifest["unique_checkpoint_hashes"],
        "both_backbones_pass": both_pass,
        "historical_v1_comparison": historical,
        "failure_attribution": {
            "claim_level": "hypothesis_false_for_cross_backbone_portability_after_decoder_HPO",
            "design_level": "readout_or_head_design_wrong_for_PatchTST_representation_compatibility",
            "bsca_objective_rejected": False,
        },
        "decision": (
            "decoder_transfer_v2p1_complete_both_backbones_pass"
            if both_pass
            else "decoder_transfer_v2p1_complete_portability_gate_not_passed"
        ),
        "gates": gates,
    }
    (args.results_dir / "decoder_transfer_result_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.results_dir / "result_and_table_audit.md").write_text(
        build_report(config, gates, both_pass, historical), encoding="utf-8"
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
        "patchtst_v2p1_result_build=pass "
        f"cells={len(cells)} both_backbones_pass={str(both_pass).lower()}"
    )


if __name__ == "__main__":
    main()
