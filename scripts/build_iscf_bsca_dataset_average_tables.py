#!/usr/bin/env python3
"""Build compact dataset-average Main-I and Main-II manuscript tables."""

from __future__ import annotations

import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    REPO_ROOT
    / "analysis/iscf_bsca_paper_experiment_consolidation_20260731"
    / "main_tables_author_corrected_20260815"
)
DATASETS = ("ETTm1", "ETTm2", "ETTh1", "ETTh2", "Weather", "ECL", "Solar")
SPECS = {
    "main_i": {
        "key": "model",
        "systems": (
            "ISCF-BSCA", "TimeAlign", "QDF", "AMD", "SimpleTM", "TVNet",
            "iTransformer", "TimeMixer", "Leddam", "ModernTCN", "PatchTST",
            "Crossformer", "TimesNet", "DLinear",
        ),
        "labels": {"ISCF-BSCA": "ISCF-BSCA (Ours)"},
        "stem": "table_iscf_bsca_main_i_dataset_average",
        "latex_label": "tab:main_iscf_bsca",
        "caption": (
            "Comparison with horizon-specific forecasters. Each dataset entry reports "
            "MSE/MAE averaged over $H\\in\\{96,192,336,720\\}$, and the final row is "
            "the unweighted mean over seven datasets. ISCF-BSCA uses one unified model "
            "per dataset, whereas each baseline follows its horizon-specific protocol. "
            "Full horizon-wise results, source roles and protocol details are provided "
            "in Appendix A. Best and second-best displayed values are marked in red bold "
            "and blue underline, respectively."
        ),
    },
    "main_ii": {
        "key": "system",
        "systems": (
            "ISCF-BSCA-MAIN-v1", "TimeAlign", "QDF", "AMD", "SimpleTM",
            "iTransformer", "PatchTST", "DLinear",
        ),
        "labels": {"ISCF-BSCA-MAIN-v1": "ISCF-BSCA (Ours)"},
        "stem": "table_iscf_bsca_main_ii_dataset_average",
        "latex_label": "tab:main_iscf_bsca_one_model",
        "caption": (
            "One-model-all-horizons comparison. Each method uses one unified model per "
            "dataset, and each entry reports MSE/MAE averaged over "
            "$H\\in\\{96,192,336,720\\}$. The final row is the unweighted mean over "
            "seven datasets. Full horizon-wise results and protocol details are provided "
            "in Appendix A. Best and second-best displayed values are marked in red bold "
            "and blue underline, respectively."
        ),
    },
}


def displayed(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)


def format_value(value: Decimal, style: str) -> str:
    text = f"{displayed(value):.3f}"
    if style == "best":
        return rf"\textcolor{{red}}{{\textbf{{{text}}}}}"
    if style == "second":
        return rf"\textcolor{{blue}}{{\underline{{{text}}}}}"
    return text


def rank(values: dict[str, Decimal]) -> dict[str, str]:
    rounded = {system: displayed(value) for system, value in values.items()}
    levels = sorted(set(rounded.values()))
    return {
        system: "best" if value == levels[0] else "second" if value == levels[1] else "normal"
        for system, value in rounded.items()
    }


def build_table(folder: str, spec: dict[str, object]) -> str:
    source = SOURCE_ROOT / folder / "table_data_long.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    key = str(spec["key"])
    systems = tuple(spec["systems"])
    averages = {
        (row[key], row["dataset"]): row for row in rows if row["horizon"] == "Avg."
    }
    expected = {(system, dataset) for system in systems for dataset in DATASETS}
    if missing := expected - set(averages):
        raise ValueError(f"Missing dataset-average cells: {sorted(missing)}")

    table_rows = []
    for dataset in DATASETS:
        cells = []
        for system in systems:
            row = averages[(system, dataset)]
            mse = format_value(Decimal(row["mse"]), row["mse_style"])
            mae = format_value(Decimal(row["mae"]), row["mae_style"])
            cells.append(f"{mse}/{mae}")
        table_rows.append(dataset + " & " + " & ".join(cells) + r" \\")

    macros = {
        metric: {
            system: sum(Decimal(averages[(system, d)][metric]) for d in DATASETS)
            / Decimal(len(DATASETS))
            for system in systems
        }
        for metric in ("mse", "mae")
    }
    styles = {metric: rank(values) for metric, values in macros.items()}
    macro_cells = [
        f"{format_value(macros['mse'][s], styles['mse'][s])}/"
        f"{format_value(macros['mae'][s], styles['mae'][s])}"
        for s in systems
    ]
    labels = dict(spec["labels"])
    header = "Dataset & " + " & ".join(labels.get(s, s) for s in systems) + r" \\"
    table_rows.extend((r"\midrule", "Average & " + " & ".join(macro_cells) + r" \\"))
    return "\n".join(
        (
            "% Required packages: booktabs, graphicx, xcolor",
            r"\begin{table*}[t]", r"\centering", r"\scriptsize",
            r"\setlength{\tabcolsep}{2.0pt}", r"\resizebox{\textwidth}{!}{%",
            rf"\begin{{tabular}}{{l|{'c' * len(systems)}}}", r"\toprule", header,
            r"\midrule", *table_rows, r"\bottomrule", r"\end{tabular}%", r"}",
            rf"\caption{{{spec['caption']}}}", rf"\label{{{spec['latex_label']}}}",
            r"\end{table*}", "",
        )
    )


def main() -> None:
    for folder, spec in SPECS.items():
        target_dir = SOURCE_ROOT / folder
        stem = str(spec["stem"])
        (target_dir / f"{stem}.tex").write_text(
            build_table(folder, spec), encoding="utf-8", newline="\n"
        )
        standalone = "\n".join(
            (
                r"\documentclass{article}",
                r"\usepackage[a3paper,landscape,margin=10mm]{geometry}",
                r"\usepackage{booktabs}", r"\usepackage{graphicx}", r"\usepackage{xcolor}",
                r"\pagestyle{empty}", r"\begin{document}", rf"\input{{{stem}.tex}}",
                r"\end{document}", "",
            )
        )
        (target_dir / f"{stem}_standalone.tex").write_text(
            standalone, encoding="utf-8", newline="\n"
        )


if __name__ == "__main__":
    main()
