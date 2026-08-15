#!/usr/bin/env python3
"""Build the author-corrected 2026-08-15 Main I and Main II tables."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import build_iscf_bsca_timealign_table6_style as main_i_builder  # noqa: E402


ANALYSIS_ROOT = (
    REPO_ROOT
    / "analysis/iscf_bsca_paper_experiment_consolidation_20260731"
    / "main_tables_author_corrected_20260815"
)
MAIN_I_SOURCE = (
    REPO_ROOT
    / "analysis/iscf_bsca_paper_experiment_consolidation_20260731"
    / "main_i_h5d_bs16_lr2p4_synced_20260813/table_data_long.csv"
)
MAIN_II_SOURCE = (
    REPO_ROOT
    / "analysis/iscf_bsca_paper_experiment_consolidation_20260731"
    / "main_ii_horizon_loader_reaudit_20260813/formal_results/aggregate_audit"
    / "main_ii_aggregate_cells.csv"
)
MAIN_I_SCREENSHOT_SHA256 = (
    "5c8c2f3e9f0fecb28392095bf28d26240542cd373d5de65cfa36353b4ee2d38b"
)
MAIN_II_SCREENSHOT_SHA256 = (
    "8de2d10fafaad540fbe2a713f528a8985a1ad90b1f657fa3e0216467e4fa8b8d"
)
DATASETS = ("ETTm1", "ETTm2", "ETTh1", "ETTh2", "Weather", "ECL", "Solar")
HORIZONS = (96, 192, 336, 720)
METRICS = ("mse", "mae")
PROVENANCE = "author_corrected_screenshot_20260815_three_decimal"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metric_map(
    values: dict[str, tuple[tuple[float, float], ...]],
) -> dict[tuple[str, int], tuple[float, float]]:
    return {
        (dataset, horizon): pair
        for dataset, dataset_values in values.items()
        for horizon, pair in zip(HORIZONS, dataset_values, strict=True)
    }


ISCF_VALUES = metric_map(
    {
        "ETTm1": ((.265, .326), (.304, .349), (.343, .371), (.404, .405)),
        "ETTm2": ((.159, .242), (.218, .285), (.270, .318), (.343, .371)),
        "ETTh1": ((.348, .386), (.377, .403), (.393, .414), (.443, .458)),
        "ETTh2": ((.241, .314), (.282, .345), (.310, .369), (.392, .429)),
        "Weather": ((.140, .180), (.181, .223), (.231, .263), (.302, .313)),
        "ECL": ((.116, .213), (.136, .231), (.157, .251), (.195, .280)),
        "Solar": ((.166, .195), (.185, .206), (.195, .213), (.204, .218)),
    }
)

MAIN_I_TIMEALIGN_VALUES = metric_map(
    {
        "ETTm1": ((.280, .331), (.323, .357), (.352, .376), (.408, .409)),
        "ETTm2": ((.157, .244), (.210, .280), (.265, .318), (.346, .374)),
        "ETTh1": ((.380, .407), (.406, .419), (.427, .432), (.459, .464)),
        "ETTh2": ((.270, .331), (.334, .374), (.376, .405), (.406, .436)),
        "Weather": ((.143, .182), (.185, .224), (.235, .263), (.308, .319)),
        "ECL": ((.128, .218), (.145, .235), (.162, .251), (.190, .278)),
        "Solar": ((.182, .205), (.198, .217), (.203, .222), (.201, .223)),
    }
)

MAIN_II_TIMEALIGN_VALUES = metric_map(
    {
        "ETTm1": ((.290, .338), (.325, .358), (.355, .378), (.408, .409)),
        "ETTm2": ((.168, .253), (.216, .286), (.268, .320), (.346, .374)),
        "ETTh1": ((.375, .404), (.411, .425), (.433, .437), (.459, .464)),
        "ETTh2": ((.296, .349), (.364, .393), (.405, .418), (.406, .436)),
        "Weather": ((.148, .187), (.187, .226), (.238, .268), (.308, .319)),
        "ECL": ((.129, .220), (.145, .235), (.163, .252), (.190, .278)),
        "Solar": ((.179, .211), (.192, .218), (.198, .223), (.201, .223)),
    }
)

MAIN_I_SIMPLETM_SOLAR = metric_map(
    {"Solar": ((.169, .229), (.189, .252), (.199, .262), (.207, .259))}
)
MAIN_I_TVNET_ETTH2 = metric_map(
    {"ETTh2": ((.263, .329), (.319, .372), (.311, .373), (.401, .434))}
)
MAIN_II_SIMPLETM_SOLAR = metric_map(
    {"Solar": ((.171, .233), (.187, .246), (.193, .251), (.207, .259))}
)
MAIN_II_PATCHTST_ETTH2 = metric_map(
    {"ETTh2": ((.277, .339), (.345, .382), (.379, .414), (.407, .443))}
)


def replace_caption(path: Path, caption: str) -> None:
    text = path.read_text(encoding="utf-8")
    before, tail = text.split("\\caption{", 1)
    _old_caption, after = tail.split("\n\\label", 1)
    path.write_text(
        before + "\\caption{" + caption + "}\n\\label" + after,
        encoding="utf-8",
    )


def build_main_i() -> dict[str, Any]:
    output_dir = ANALYSIS_ROOT / "main_i"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = [row for row in read_csv(MAIN_I_SOURCE) if row["horizon"] != "Avg."]
    rows: list[dict[str, Any]] = []
    override_maps = {
        "ISCF-BSCA": ISCF_VALUES,
        "TimeAlign": MAIN_I_TIMEALIGN_VALUES,
        "SimpleTM": MAIN_I_SIMPLETM_SOLAR,
        "TVNet": MAIN_I_TVNET_ETTH2,
    }
    expected_overrides = 28 + 28 + 4 + 4
    overrides = 0
    for source in raw_rows:
        row: dict[str, Any] = dict(source)
        row["horizon"] = int(source["horizon"])
        row["mse"] = float(source["mse"])
        row["mae"] = float(source["mae"])
        values = override_maps.get(source["model"], {})
        key = (source["dataset"], int(source["horizon"]))
        if key in values:
            row["mse"], row["mae"] = values[key]
            row["value_origin"] = PROVENANCE
            overrides += 1
        rows.append(row)
    if overrides != expected_overrides:
        raise RuntimeError(f"Main I override count mismatch: {overrides}")
    main_i_builder.validate_matrix(rows)
    styled = main_i_builder.add_averages_and_styles(rows)
    data_path = output_dir / "table_data_long.csv"
    table_path = output_dir / "table_iscf_bsca_main_i_qdf.tex"
    main_i_builder.write_long_csv(data_path, styled)
    main_i_builder.build_latex(table_path, styled)
    standalone_path = output_dir / "table_iscf_bsca_main_i_standalone.tex"
    caption = (
        "Long-term forecasting results in the TimeAlign Table-6 layout. All entries "
        "report MSE/MAE, and Avg. is recomputed over the four displayed horizons. "
        "Best and second-best displayed values are bold and underlined. ISCF-BSCA "
        "uses one unified model per dataset, whereas the baselines are separately "
        "optimized horizon-specific systems. The ISCF-BSCA and TimeAlign rows, the "
        "SimpleTM Solar rows, and the TVNet ETTh2 rows use the author-corrected "
        "rerun values supplied at three-decimal precision on 2026-08-15. QDF, AMD, "
        "and the remaining SimpleTM rows retain the preceding official-code local "
        "reproductions; all other baseline rows retain published context. This is "
        "a system-level accuracy comparison rather than matched mechanism attribution."
    )
    replace_caption(table_path, caption)
    replace_caption(standalone_path, caption)
    iscf_rows = [
        row
        for row in styled
        if row["model"] == "ISCF-BSCA" and row["horizon"] != "Avg."
    ]
    summary = {
        "table_id": "ISCF-BSCA-MAIN-I-AUTHOR-CORRECTED-20260815",
        "supersedes": str(MAIN_I_SOURCE.relative_to(REPO_ROOT)),
        "systems": len(main_i_builder.DISPLAY_MODELS),
        "datasets": list(DATASETS),
        "horizons": list(HORIZONS),
        "standard_rows": len(rows),
        "rows_with_dataset_averages": len(styled),
        "author_corrected_standard_rows": overrides,
        "author_correction_scope": {
            "ISCF-BSCA": "7 datasets x 4 horizons",
            "TimeAlign": "7 datasets x 4 horizons",
            "SimpleTM": "Solar x 4 horizons",
            "TVNet": "ETTh2 x 4 horizons",
        },
        "display_precision": 3,
        "iscf_best_metric_cells": sum(
            row[f"{metric}_style"] == "best" for row in iscf_rows for metric in METRICS
        ),
        "iscf_second_metric_cells": sum(
            row[f"{metric}_style"] == "second"
            for row in iscf_rows
            for metric in METRICS
        ),
        "source_hashes": {
            "previous_table_data": sha256(MAIN_I_SOURCE),
            "author_screenshot": MAIN_I_SCREENSHOT_SHA256,
        },
        "claim_boundary": (
            "author_corrected_three_decimal_reruns_and_preserved_prior_baselines; "
            "system-level accuracy context, not matched mechanism attribution"
        ),
    }
    (output_dir / "table_build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_main_ii() -> dict[str, Any]:
    output_dir = ANALYSIS_ROOT / "main_ii"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    override_maps = {
        "ISCF-BSCA-MAIN-v1": ISCF_VALUES,
        "TimeAlign": MAIN_II_TIMEALIGN_VALUES,
        "SimpleTM": MAIN_II_SIMPLETM_SOLAR,
        "PatchTST": MAIN_II_PATCHTST_ETTH2,
    }
    expected_overrides = 28 + 28 + 4 + 4
    overrides = 0
    for source in read_csv(MAIN_II_SOURCE):
        row: dict[str, Any] = dict(source)
        values = override_maps.get(source["system"], {})
        key = (source["dataset"], int(source["horizon"]))
        if key in values:
            row["mse"], row["mae"] = values[key]
            row["checkpoint_sha256"] = ""
            row["system_role"] = (
                "author_corrected_H720_model_on_official_fixed_H_loader"
            )
            row["value_origin"] = PROVENANCE
            overrides += 1
        else:
            row["value_origin"] = "formal_horizon_loader_reaudit_20260813"
        rows.append(row)
    if overrides != expected_overrides:
        raise RuntimeError(f"Main II override count mismatch: {overrides}")
    aggregate_path = output_dir / "main_ii_aggregate_cells_author_corrected.csv"
    write_csv(aggregate_path, rows)
    audit_path = output_dir / "author_correction_gate.json"
    audit = {
        "gate": "pass",
        "aggregate_cells": 224,
        "author_corrected_standard_rows": overrides,
        "source_aggregate_sha256": sha256(MAIN_II_SOURCE),
        "author_screenshot_sha256": MAIN_II_SCREENSHOT_SHA256,
        "provenance": PROVENANCE,
        "claim_boundary": (
            "the corrected cells are author-provided three-decimal rerun values; "
            "their prior checkpoint hashes are intentionally not reused"
        ),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/build_iscf_bsca_main_ii_table.py"),
            "--aggregate-cells",
            str(aggregate_path),
            "--result-audit",
            str(audit_path),
            "--output-dir",
            str(output_dir),
            "--table-id",
            "ISCF-BSCA-MAIN-II-AUTHOR-CORRECTED-20260815",
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    table_path = output_dir / "table_iscf_bsca_main_ii.tex"
    standalone_path = output_dir / "table_iscf_bsca_main_ii_standalone.tex"
    caption = (
        "One-model-for-all-horizons comparison. Each system reuses one fixed "
        "$H=720$ model per dataset; each requested horizon is scored by comparing "
        "the corresponding output prefix with labels from that system's official "
        "fixed-$H$ test loader. Avg. is the arithmetic mean over the four displayed "
        "horizons, and best/second-best styles follow common three-decimal rounding. "
        "The ISCF-BSCA and TimeAlign rows, the SimpleTM Solar rows, and the PatchTST "
        "ETTh2 rows use the author-corrected rerun values supplied at three-decimal "
        "precision on 2026-08-15; all other rows retain the complete 2026-08-13 "
        "horizon-loader reaudit. The corrected cells supersede their prior displayed "
        "values without reusing the superseded checkpoint hashes. External source "
        "contracts remain unmatched, so this is a system-level unified-horizon "
        "benchmark rather than matched mechanism attribution."
    )
    replace_caption(table_path, caption)
    replace_caption(standalone_path, caption)
    summary_path = output_dir / "table_build_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "supersedes": str(MAIN_II_SOURCE.relative_to(REPO_ROOT)),
            "author_corrected_standard_rows": overrides,
            "author_correction_scope": {
                "ISCF-BSCA-MAIN-v1": "7 datasets x 4 horizons",
                "TimeAlign": "7 datasets x 4 horizons",
                "SimpleTM": "Solar x 4 horizons",
                "PatchTST": "ETTh2 x 4 horizons",
            },
            "source_hashes": {
                "previous_aggregate_cells": sha256(MAIN_II_SOURCE),
                "author_screenshot": MAIN_II_SCREENSHOT_SHA256,
                "corrected_aggregate_cells": sha256(aggregate_path),
                "author_correction_gate": sha256(audit_path),
            },
            "claim_boundary": (
                "author-corrected three-decimal reruns plus preserved complete "
                "horizon-loader reaudit; not matched mechanism attribution"
            ),
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    main_i = build_main_i()
    main_ii = build_main_ii()
    report = {
        "freeze_id": "ISCF-BSCA-MAIN-TABLES-AUTHOR-CORRECTED-20260815",
        "status": "table_build_complete_freeze_recorded_separately",
        "main_i": main_i,
        "main_ii": main_ii,
    }
    (ANALYSIS_ROOT / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "author_corrected_main_tables=pass "
        f"main_i_best={main_i['iscf_best_metric_cells']}/56 "
        f"main_i_second={main_i['iscf_second_metric_cells']}/56 "
        f"main_ii_best={main_ii['iscf_best_cells']}/56 "
        f"main_ii_second={main_ii['iscf_second_cells']}/56"
    )


if __name__ == "__main__":
    main()
