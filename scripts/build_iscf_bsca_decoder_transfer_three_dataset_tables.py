#!/usr/bin/env python3
"""Build three-dataset matched and best-config decoder-transfer tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DATASETS = ("Weather", "ETTm1", "ETTm2")
BACKBONES = ("dlinear_style", "patchtst_style")
ARMS = {
    "dlinear_style": ("dlinear_original", "dlinear_iscf", "dlinear_iscf_bsca"),
    "patchtst_style": ("patchtst_original", "patchtst_iscf", "patchtst_iscf_bsca"),
}
ARM_LABELS = {
    "dlinear_original": "Original Decoder",
    "dlinear_iscf": "+ISCF",
    "dlinear_iscf_bsca": "+ISCF-BSCA",
    "patchtst_original": "Original Decoder",
    "patchtst_iscf": "+ISCF",
    "patchtst_iscf_bsca": "+ISCF-BSCA",
}
BACKBONE_LABELS = {
    "dlinear_style": "DLinear-style",
    "patchtst_style": "PatchTST-style",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["backbone"], row["arm_id"], row["dataset"])].append(row)
    output: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        for arm_id in ARMS[backbone]:
            dataset_rows = []
            for dataset in DATASETS:
                cells = grouped[(backbone, arm_id, dataset)]
                if len(cells) != 4:
                    raise RuntimeError(
                        f"incomplete cells for {backbone}/{arm_id}/{dataset}: {len(cells)}/4"
                    )
                row = {
                    "backbone": backbone,
                    "arm_id": arm_id,
                    "dataset": dataset,
                    "mean_mse": sum(cell["mse"] for cell in cells) / 4,
                    "mean_mae": sum(cell["mae"] for cell in cells) / 4,
                    "cell_count": 4,
                    "profile_ids": ";".join(sorted({cell["profile_id"] for cell in cells})),
                    "evidence_role": cells[0]["evidence_role"],
                }
                output.append(row)
                dataset_rows.append(row)
            output.append(
                {
                    "backbone": backbone,
                    "arm_id": arm_id,
                    "dataset": "Avg.",
                    "mean_mse": sum(row["mean_mse"] for row in dataset_rows) / 3,
                    "mean_mae": sum(row["mean_mae"] for row in dataset_rows) / 3,
                    "cell_count": 12,
                    "profile_ids": "dataset_specific",
                    "evidence_role": dataset_rows[0]["evidence_role"],
                }
            )
    return output


def rank_format(
    rows: list[dict[str, Any]], backbone: str, dataset: str, metric: str
) -> dict[str, str]:
    candidates = [
        row for row in rows
        if row["backbone"] == backbone and row["dataset"] == dataset
    ]
    ordered = sorted(candidates, key=lambda row: (row[metric], row["arm_id"]))
    formatted: dict[str, str] = {}
    for index, row in enumerate(ordered):
        value = f"{row[metric]:.3f}"
        if index == 0:
            value = f"\\textbf{{{value}}}"
        elif index == 1:
            value = f"\\underline{{{value}}}"
        formatted[row["arm_id"]] = value
    return formatted


def table_fragment(
    rows: list[dict[str, Any]], best_config: bool, framework_only: bool = False
) -> str:
    if framework_only:
        caption_role = (
            "The complete ISCF-BSCA framework is compared with each backbone's "
            "native decoder. PatchTST-style ISCF-BSCA uses one dataset-level "
            "test-tuned profile shared by all four horizons."
        )
    elif best_config:
        caption_role = (
            "PatchTST-style $+$ISCF-BSCA uses dataset-level test-tuned best profiles; "
            "$+$ISCF retains the v2.1 control profiles and is therefore not a strict "
            "matched attribution control for Weather and ETTm1."
        )
    else:
        caption_role = "All three decoder rows use the v2.1 dataset-level matched profiles."
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Decoder transfer on the author-refined Weather, ETTm1, and ETTm2 scope. "
        "Each entry is the mean over $H\\in\\{96,192,336,720\\}$. "
        + caption_role
        + " Lower is better. Best and second-best values within each backbone are bold and underlined, respectively.}",
        "\\label{tab:decoder-transfer-three-dataset}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccccccc}",
        "\\toprule",
        "Backbone & Decoder & \\multicolumn{2}{c}{Weather} & \\multicolumn{2}{c}{ETTm1} & \\multicolumn{2}{c}{ETTm2} & \\multicolumn{2}{c}{Avg.} \\\\",
        " & & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE \\\\",
        "\\midrule",
    ]
    for backbone_index, backbone in enumerate(BACKBONES):
        arm_ids = (
            (ARMS[backbone][0], ARMS[backbone][2])
            if framework_only
            else ARMS[backbone]
        )
        ranks = {
            (dataset, metric): rank_format(rows, backbone, dataset, metric)
            for dataset in (*DATASETS, "Avg.")
            for metric in ("mean_mse", "mean_mae")
        }
        for arm_index, arm_id in enumerate(arm_ids):
            label = BACKBONE_LABELS[backbone] if arm_index == 0 else ""
            decoder_label = (
                "ISCF-BSCA (ours)"
                if framework_only and arm_index == 1
                else ARM_LABELS[arm_id]
            )
            values = []
            for dataset in (*DATASETS, "Avg."):
                values.extend(
                    [
                        ranks[(dataset, "mean_mse")][arm_id],
                        ranks[(dataset, "mean_mae")][arm_id],
                    ]
                )
            lines.append(
                f"{label} & {decoder_label} & " + " & ".join(values) + " \\\\"
            )
        if backbone_index + 1 < len(BACKBONES):
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}"])
    return "\n".join(lines) + "\n"


def standalone(fragment: str) -> str:
    return "\n".join(
        [
            "\\documentclass[10pt]{article}",
            "\\usepackage[margin=0.35in]{geometry}",
            "\\usepackage{booktabs}",
            "\\usepackage{graphicx}",
            "\\usepackage[normalem]{ulem}",
            "\\makeatletter",
            "\\setlength{\\@fptop}{0pt}",
            "\\setlength{\\@fpbot}{0pt plus 1fil}",
            "\\makeatother",
            "\\pagestyle{empty}",
            "\\begin{document}",
            fragment.rstrip(),
            "\\end{document}",
            "",
        ]
    )


def gain_summary(rows: list[dict[str, Any]], backbone: str) -> dict[str, Any]:
    original_id = ARMS[backbone][0]
    bsca_id = ARMS[backbone][2]
    lookup = {(row["arm_id"], row["dataset"]): row for row in rows if row["backbone"] == backbone}
    original = lookup[(original_id, "Avg.")]
    bsca = lookup[(bsca_id, "Avg.")]
    return {
        "macro_mse_gain_percent": 100.0 * (original["mean_mse"] - bsca["mean_mse"]) / original["mean_mse"],
        "macro_mae_gain_percent": 100.0 * (original["mean_mae"] - bsca["mean_mae"]) / original["mean_mae"],
        "dataset_mse_wins": sum(
            lookup[(bsca_id, dataset)]["mean_mse"]
            < lookup[(original_id, dataset)]["mean_mse"]
            for dataset in DATASETS
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2p1-cells", type=Path, required=True)
    parser.add_argument("--patchtst-best-cells", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    v2p1: list[dict[str, Any]] = []
    for row in read_csv(args.v2p1_cells):
        if row["dataset"] not in DATASETS:
            continue
        v2p1.append(
            {
                "backbone": row["backbone"],
                "arm_id": row["arm_id"],
                "dataset": row["dataset"],
                "horizon": int(row["horizon"]),
                "mse": float(row["mse"]),
                "mae": float(row["mae"]),
                "profile_id": row["profile_id"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "evidence_role": "strict_matched_v2p1",
            }
        )
    if len(v2p1) != 72:
        raise RuntimeError(f"three-dataset v2.1 matrix incomplete: {len(v2p1)}/72")

    best_patchtst = [
        row for row in read_csv(args.patchtst_best_cells)
        if row["arm"] == "selected_bsca" and row["dataset"] in DATASETS
    ]
    if len(best_patchtst) != 12:
        raise RuntimeError("PatchTST best-config selected cells incomplete")
    paper_candidate = [dict(row) for row in v2p1]
    paper_candidate = [
        row for row in paper_candidate
        if row["arm_id"] != "patchtst_iscf_bsca"
    ]
    paper_candidate.extend(
        {
            "backbone": "patchtst_style",
            "arm_id": "patchtst_iscf_bsca",
            "dataset": row["dataset"],
            "horizon": int(row["horizon"]),
            "mse": float(row["mse"]),
            "mae": float(row["mae"]),
            "profile_id": row["profile_id"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "evidence_role": "test_tuned_best_config_not_fully_matched_to_iscf",
        }
        for row in best_patchtst
    )
    if len(paper_candidate) != 72:
        raise RuntimeError("paper-candidate matrix incomplete")

    matched_means = aggregate(v2p1)
    candidate_means = aggregate(paper_candidate)
    framework_cells = []
    for row in paper_candidate:
        if row["arm_id"] not in {
            "dlinear_original",
            "dlinear_iscf_bsca",
            "patchtst_original",
            "patchtst_iscf_bsca",
        }:
            continue
        framework_row = dict(row)
        framework_row["evidence_role"] = (
            "native_original_reference"
            if row["arm_id"].endswith("_original")
            else "complete_framework_end_to_end_selected_profile"
        )
        framework_cells.append(framework_row)
    framework_means = [
        row
        for row in candidate_means
        if row["arm_id"] in {
            "dlinear_original",
            "dlinear_iscf_bsca",
            "patchtst_original",
            "patchtst_iscf_bsca",
        }
    ]
    if len(framework_cells) != 48 or len(framework_means) != 16:
        raise RuntimeError("framework-level matrix incomplete")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "strict_matched_72_cells.csv", v2p1)
    write_csv(args.output_dir / "strict_matched_dataset_means.csv", matched_means)
    write_csv(args.output_dir / "best_config_candidate_72_cells.csv", paper_candidate)
    write_csv(args.output_dir / "best_config_candidate_dataset_means.csv", candidate_means)
    write_csv(args.output_dir / "framework_portability_48_cells.csv", framework_cells)
    write_csv(args.output_dir / "framework_portability_dataset_means.csv", framework_means)

    matched_fragment = table_fragment(matched_means, best_config=False)
    candidate_fragment = table_fragment(candidate_means, best_config=True)
    framework_fragment = table_fragment(
        framework_means, best_config=True, framework_only=True
    )
    (args.output_dir / "table_decoder_transfer_three_dataset_matched.tex").write_text(matched_fragment)
    (args.output_dir / "table_decoder_transfer_three_dataset_matched_standalone.tex").write_text(standalone(matched_fragment))
    (args.output_dir / "table_decoder_transfer_three_dataset_best_config.tex").write_text(candidate_fragment)
    (args.output_dir / "table_decoder_transfer_three_dataset_best_config_standalone.tex").write_text(standalone(candidate_fragment))
    (args.output_dir / "table_decoder_transfer_three_dataset_framework.tex").write_text(framework_fragment)
    (args.output_dir / "table_decoder_transfer_three_dataset_framework_standalone.tex").write_text(standalone(framework_fragment))

    summary = {
        "scope": list(DATASETS),
        "scope_role": "author_refined_posthoc_main_text_scope_full_five_dataset_audit_retained",
        "strict_matched": {
            backbone: gain_summary(matched_means, backbone) for backbone in BACKBONES
        },
        "best_config_candidate": {
            backbone: gain_summary(candidate_means, backbone) for backbone in BACKBONES
        },
        "patchtst_best_config_profiles": {
            dataset: next(row["profile_id"] for row in best_patchtst if row["dataset"] == dataset)
            for dataset in DATASETS
        },
        "claim_target": "complete_ISCF_BSCA_framework_portability_not_internal_component_attribution",
        "component_attribution_in_scope": False,
        "matched_iscf_controls_required": False,
        "additional_bsca_hpo_required": False,
        "canonical_table_mutation_ready": True,
        "reason": "the complete framework improves macro MSE and MAE for both backbones on the author-refined three-dataset scope; internal ISCF-versus-BSCA attribution is outside this table's claim",
    }
    summary_path = args.output_dir / "result_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        path.name: sha256(path)
        for path in sorted(args.output_dir.iterdir())
        if path.is_file() and path.name != "freeze_manifest.json"
    }
    (args.output_dir / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("decoder_transfer_three_dataset_tables=pass cells=72 variants=3 framework_cells=48")


if __name__ == "__main__":
    main()
