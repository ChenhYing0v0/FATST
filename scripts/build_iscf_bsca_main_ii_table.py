#!/usr/bin/env python3
"""Build the paper-facing Main II one-H720-model table artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


SYSTEMS = (
    "ISCF-BSCA-MAIN-v1",
    "TimeAlign",
    "QDF",
    "AMD",
    "SimpleTM",
    "iTransformer",
    "PatchTST",
    "DLinear",
)
DISPLAY_NAMES = {"ISCF-BSCA-MAIN-v1": "ISCF-BSCA"}
YEARS = {
    "ISCF-BSCA-MAIN-v1": "Ours",
    "TimeAlign": "2026",
    "QDF": "2026",
    "AMD": "2025",
    "SimpleTM": "2025",
    "iTransformer": "2024b",
    "PatchTST": "2023",
    "DLinear": "2023",
}
DATASETS = ("ETTm1", "ETTm2", "ETTh1", "ETTh2", "Weather", "ECL", "Solar")
HORIZONS = (96, 192, 336, 720)
METRICS = ("mse", "mae")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-cells", type=Path, required=True)
    parser.add_argument("--result-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--table-id",
        default="ISCF-BSCA-MAIN-II-H720-HORIZON-LOADER-20260813",
    )
    return parser.parse_args()


def style_for(values: list[float], value: float) -> str:
    displayed = sorted({round(item, 3) for item in values})
    rounded = round(value, 3)
    if rounded == displayed[0]:
        return "best"
    if len(displayed) > 1 and rounded == displayed[1]:
        return "second"
    return "normal"


def latex_value(row: dict[str, object], metric: str) -> str:
    value = str(row[f"{metric}_display"])
    style = row[f"{metric}_style"]
    if style == "best":
        return f"\\textcolor{{red}}{{\\textbf{{{value}}}}}"
    if style == "second":
        return f"\\textcolor{{blue}}{{\\underline{{{value}}}}}"
    return value


def build_latex(path: Path, rows: list[dict[str, object]]) -> None:
    lookup = {
        (str(row["system"]), str(row["dataset"]), str(row["horizon"])): row
        for row in rows
    }
    lines = [
        "% Required packages: booktabs, multirow, graphicx, xcolor",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{1.2pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{cc" + "|cc" * len(SYSTEMS) + "}",
        "\\toprule",
    ]
    headers = ["Dataset", "H"]
    for system in SYSTEMS:
        name = DISPLAY_NAMES.get(system, system)
        headers.append(f"\\multicolumn{{2}}{{c}}{{{name} ({YEARS[system]})}}")
    lines.append(" & ".join(headers) + " \\\\")
    metric_headers = ["", ""]
    for _system in SYSTEMS:
        metric_headers.extend(["MSE", "MAE"])
    lines.extend([" & ".join(metric_headers) + " \\\\", "\\midrule"])
    for dataset_index, dataset in enumerate(DATASETS):
        for row_index, horizon in enumerate((*HORIZONS, "Avg.")):
            values = [
                f"\\multirow{{5}}{{*}}{{{dataset}}}" if row_index == 0 else "",
                str(horizon),
            ]
            for system in SYSTEMS:
                row = lookup[(system, dataset, str(horizon))]
                values.extend([latex_value(row, "mse"), latex_value(row, "mae")])
            lines.append(" & ".join(values) + " \\\\")
            if row_index == 3:
                lines.append(f"\\cmidrule(lr){{2-{2 + 2 * len(SYSTEMS)}}}")
        if dataset_index < len(DATASETS) - 1:
            lines.append("\\midrule")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\caption{One-model-for-all-horizons comparison. Each system reuses one fixed $H=720$ model per dataset. For every requested $H\\in\\{96,192,336,720\\}$, we reconstruct that system's horizon-specific official test loader, feed only its historical input to the frozen $H=720$ checkpoint, and compare the first $H$ output steps with the loader's $H$-step labels. Thus, each entry uses the same split, preprocessing, batch semantics, and \\texttt{drop\\_last} rule as the corresponding fixed-$H$ evaluation. All entries are locally evaluated MSE/MAE, and Avg. is the arithmetic mean over the four horizons. Best and second-best values after common three-decimal rounding are bold and underlined. ISCF-BSCA uses its frozen dataset-level test-tuned profile; SimpleTM averages its three native repetitions. External repositories retain their official source-native lookbacks, optimizers, and checkpoint selectors, so this is a system-level unified-horizon benchmark rather than matched mechanism attribution.}",
            "\\label{tab:main_ii_h720_horizon_loader}",
            "\\end{table*}",
            "",
        ]
    )
    table_text = "\n".join(lines)
    path.write_text(table_text, encoding="utf-8")
    standalone_lines = [
        "% Auto-generated review copy; edit the table builder, not this file.",
        "\\documentclass[10pt]{article}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage[a3paper,landscape,margin=10mm]{geometry}",
        "\\usepackage{booktabs}",
        "\\usepackage{multirow}",
        "\\usepackage{graphicx}",
        "\\usepackage{xcolor}",
        "\\pagestyle{empty}",
        "\\renewcommand{\\arraystretch}{0.82}",
        "\\setlength{\\abovecaptionskip}{5pt}",
        "\\begin{document}",
        table_text,
        "\\end{document}",
        "",
    ]
    path.with_name("table_iscf_bsca_main_ii_standalone.tex").write_text(
        "\n".join(standalone_lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    audit = json.loads(args.result_audit.read_text(encoding="utf-8"))
    if audit.get("gate") != "pass" or audit.get("aggregate_cells") != 224:
        raise RuntimeError("Main II complete result gate has not passed")
    raw = read_csv(args.aggregate_cells)
    if len(raw) != 224:
        raise RuntimeError("aggregate input must contain exactly 224 cells")
    base = {
        (row["system"], row["dataset"], int(row["horizon"])): {
            **row,
            "mse": float(row["mse"]),
            "mae": float(row["mae"]),
        }
        for row in raw
    }
    expected = {
        (system, dataset, horizon)
        for system in SYSTEMS
        for dataset in DATASETS
        for horizon in HORIZONS
    }
    if set(base) != expected:
        raise RuntimeError("Main II aggregate cell identity mismatch")

    table_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            candidates = [base[(system, dataset, horizon)] for system in SYSTEMS]
            for row in candidates:
                table_rows.append(
                    {
                        "system": row["system"],
                        "dataset": dataset,
                        "horizon": horizon,
                        "mse": row["mse"],
                        "mae": row["mae"],
                        "mse_display": f"{row['mse']:.3f}",
                        "mae_display": f"{row['mae']:.3f}",
                        "mse_style": style_for(
                            [float(item["mse"]) for item in candidates],
                            float(row["mse"]),
                        ),
                        "mae_style": style_for(
                            [float(item["mae"]) for item in candidates],
                            float(row["mae"]),
                        ),
                        "checkpoint_repetitions": row["checkpoint_repetitions"],
                        "system_role": row["system_role"],
                    }
                )
        averages = {}
        for system in SYSTEMS:
            rows = [base[(system, dataset, horizon)] for horizon in HORIZONS]
            averages[system] = {
                "mse": sum(float(row["mse"]) for row in rows) / 4,
                "mae": sum(float(row["mae"]) for row in rows) / 4,
            }
        for system in SYSTEMS:
            row = averages[system]
            table_rows.append(
                {
                    "system": system,
                    "dataset": dataset,
                    "horizon": "Avg.",
                    "mse": row["mse"],
                    "mae": row["mae"],
                    "mse_display": f"{row['mse']:.3f}",
                    "mae_display": f"{row['mae']:.3f}",
                    "mse_style": style_for(
                        [float(item["mse"]) for item in averages.values()], row["mse"]
                    ),
                    "mae_style": style_for(
                        [float(item["mae"]) for item in averages.values()], row["mae"]
                    ),
                    "checkpoint_repetitions": 3 if system == "SimpleTM" else 1,
                    "system_role": "four_horizon_arithmetic_mean",
                }
            )
    if len(table_rows) != 280:
        raise RuntimeError("table with dataset averages must contain 280 rows")

    iscf_standard = [
        row
        for row in table_rows
        if row["system"] == "ISCF-BSCA-MAIN-v1" and row["horizon"] != "Avg."
    ]
    macro = {}
    for system in SYSTEMS:
        rows = [base[(system, dataset, horizon)] for dataset in DATASETS for horizon in HORIZONS]
        macro[system] = {
            "mse": sum(float(row["mse"]) for row in rows) / len(rows),
            "mae": sum(float(row["mae"]) for row in rows) / len(rows),
        }
    summary = {
        "table_id": args.table_id,
        "systems": list(SYSTEMS),
        "datasets": list(DATASETS),
        "horizons": list(HORIZONS),
        "aggregate_cells": 224,
        "rows_with_dataset_averages": 280,
        "metric_scalars": 448,
        "display_precision": 3,
        "ranking_rule": "best_and_second_distinct_displayed_values_after_three_decimal_rounding",
        "presentation_template": "Main_I_TimeAlign_Table_6_style",
        "standalone_latex": "table_iscf_bsca_main_ii_standalone.tex",
        "standalone_page": "A3_landscape_review_copy",
        "best_style": "red_bold",
        "second_style": "blue_underline",
        "iscf_best_cells": sum(
            row[f"{metric}_style"] == "best"
            for row in iscf_standard
            for metric in METRICS
        ),
        "iscf_second_cells": sum(
            row[f"{metric}_style"] == "second"
            for row in iscf_standard
            for metric in METRICS
        ),
        "macro_over_28_dataset_horizon_cells": macro,
        "source_hashes": {
            "aggregate_cells": sha256(args.aggregate_cells),
            "result_audit": sha256(args.result_audit),
        },
        "claim_boundary": "source-native H720 checkpoints evaluated on official fixed-H loaders; not matched mechanism attribution",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "table_data_long.csv", table_rows)
    build_latex(args.output_dir / "table_iscf_bsca_main_ii.tex", table_rows)
    (args.output_dir / "table_build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "main_ii_table_build=pass aggregate_cells=224 rows_with_averages=280 "
        f"iscf_best={summary['iscf_best_cells']}/56 "
        f"iscf_second={summary['iscf_second_cells']}/56"
    )


if __name__ == "__main__":
    main()
