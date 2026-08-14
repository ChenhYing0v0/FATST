#!/usr/bin/env python3
"""Build the manuscript and standalone LaTeX Core-Ablation table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/iscf_bsca_core_ablation_protocol.json"),
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def decorate(value: float, rank: int, precision: int) -> str:
    rendered = f"{value:.{precision}f}"
    if rank == 0:
        return f"\\textbf{{{rendered}}}"
    if rank == 1:
        return f"\\underline{{{rendered}}}"
    return rendered


def metric_ranks(
    rows: list[dict[str, Any]], key: str
) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: (float(row[key]), row["arm_id"]))
    return {row["arm_id"]: index for index, row in enumerate(ordered)}


def build_table(config: dict[str, Any], results_dir: Path) -> str:
    dataset_rows_raw = read_csv(results_dir / "core_ablation_dataset_means.csv")
    overall_rows_raw = read_csv(results_dir / "core_ablation_overall_means.csv")
    dataset_rows = [
        {**row, "mean_mse": float(row["mean_mse"]), "mean_mae": float(row["mean_mae"])}
        for row in dataset_rows_raw
    ]
    overall_rows = [
        {**row, "macro_mse": float(row["macro_mse"]), "macro_mae": float(row["macro_mae"])}
        for row in overall_rows_raw
    ]
    row_order = config["table"]["row_order"]
    dataset_order = config["table"]["dataset_order"]
    precision = int(config["table"]["precision"])
    labels = {arm["id"]: arm["table_label"] for arm in config["arms"]}
    dataset_lookup = {
        (row["arm_id"], row["dataset"]): row for row in dataset_rows
    }
    overall_lookup = {row["arm_id"]: row for row in overall_rows}
    ranks: dict[tuple[str, str], dict[str, int]] = {}
    for dataset in dataset_order:
        subset = [row for row in dataset_rows if row["dataset"] == dataset]
        ranks[(dataset, "mse")] = metric_ranks(subset, "mean_mse")
        ranks[(dataset, "mae")] = metric_ranks(subset, "mean_mae")
    ranks[("Avg", "mse")] = metric_ranks(overall_rows, "macro_mse")
    ranks[("Avg", "mae")] = metric_ranks(overall_rows, "macro_mae")

    column_spec = "l" + "cc" * (len(dataset_order) + 1)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Core ablation of ISCF-BSCA. Each dataset entry is the mean over prediction horizons $H\\in\\{96,192,336,720\\}$. Lower is better. Best and second-best results are shown in bold and underlined, respectively; rankings use unrounded values.}",
        "\\label{tab:core-ablation}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        "Variant & "
        + " & ".join(f"\\multicolumn{{2}}{{c}}{{{dataset}}}" for dataset in dataset_order)
        + " & \\multicolumn{2}{c}{Avg} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9} \\cmidrule(lr){10-11} \\cmidrule(lr){12-13}",
        " & " + " & ".join(["MSE & MAE"] * (len(dataset_order) + 1)) + " \\\\",
        "\\midrule",
    ]
    for index, arm_id in enumerate(row_order):
        values = []
        for dataset in dataset_order:
            row = dataset_lookup[(arm_id, dataset)]
            values.extend(
                [
                    decorate(row["mean_mse"], ranks[(dataset, "mse")][arm_id], precision),
                    decorate(row["mean_mae"], ranks[(dataset, "mae")][arm_id], precision),
                ]
            )
        overall = overall_lookup[arm_id]
        values.extend(
            [
                decorate(overall["macro_mse"], ranks[("Avg", "mse")][arm_id], precision),
                decorate(overall["macro_mae"], ranks[("Avg", "mae")][arm_id], precision),
            ]
        )
        lines.append(f"{labels[arm_id]} & " + " & ".join(values) + " \\\\")
        if index == 0:
            lines.append("\\midrule")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def main(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    table = build_table(config, args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fragment = args.output_dir / "table_iscf_bsca_core_ablation.tex"
    standalone = args.output_dir / "table_iscf_bsca_core_ablation_standalone.tex"
    fragment.write_text(table, encoding="utf-8")
    standalone.write_text(
        "\\documentclass[10pt]{article}\n"
        "\\usepackage[margin=0.45in]{geometry}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{ulem}\n"
        "\\pagestyle{empty}\n"
        "\\begin{document}\n"
        + table
        + "\\end{document}\n",
        encoding="utf-8",
    )
    print(f"core_ablation_table_build=pass fragment={fragment} standalone={standalone}")


if __name__ == "__main__":
    main(parse_args())
