#!/usr/bin/env python3
"""Build the Elsevier LaTeX manuscript from frozen manuscript-facing drafts."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_DIR = Path(__file__).resolve().parent
SECTIONS_DIR = MANUSCRIPT_DIR / "sections"
FIGURES_DIR = MANUSCRIPT_DIR / "figures"
TABLES_DIR = MANUSCRIPT_DIR / "tables"


SECTION_SOURCES = (
    (
        "01_introduction.tex",
        "docs/paper-drafts/iscf-bsca-introduction-initial-draft.md",
        "## 1. Introduction",
        None,
    ),
    (
        "02_related_work.tex",
        "docs/paper-drafts/iscf-bsca-related-work-initial-draft.md",
        "## 2. Related Work",
        "## Editorial citation and claim audit",
    ),
    (
        "03_problem_formulation.tex",
        "docs/paper-drafts/iscf-bsca-problem-formulation-initial-draft.md",
        "## 3. Problem Formulation and Empirical Motivation",
        "## Editorial evidence and claim audit",
    ),
    (
        "04_method.tex",
        "docs/paper-drafts/iscf-bsca-method-initial-draft.md",
        "## 4. HoriScope",
        "## Editorial implementation and claim audit",
    ),
    (
        "05_experiments.tex",
        "docs/paper-drafts/iscf-bsca-experiments-initial-draft.md",
        "## 5. Experiments",
        "## Editorial evidence and claim audit",
    ),
    (
        "06_discussion.tex",
        "docs/paper-drafts/iscf-bsca-discussion-initial-draft.md",
        "## 6. Discussion and Limitations",
        None,
    ),
    (
        "07_conclusion.tex",
        "docs/paper-drafts/iscf-bsca-conclusion-initial-draft.md",
        "## 7. Conclusion",
        None,
    ),
    (
        "appendices.tex",
        "docs/paper-drafts/iscf-bsca-appendix-initial-draft.md",
        "## A. EXPERIMENT DETAILS",
        None,
    ),
)


FIGURES = {
    "1": {
        "source": "paper-figures/figure_intro_conceptual_problem.pdf",
        "target": "figure_01_conceptual_problem.pdf",
        "label": "fig:conceptual-problems",
        "width": "0.92\\textwidth",
    },
    "2": {
        "source": "paper-figures/figure_intro_prefix_disagreement.pdf",
        "target": "figure_02_prefix_disagreement.pdf",
        "label": "fig:prefix-disagreement",
        "width": "0.96\\textwidth",
    },
    "3": {
        "source": "paper-figures/figure_intro_sharing_heterogeneity.pdf",
        "target": "figure_03_sharing_heterogeneity.pdf",
        "label": "fig:sharing-heterogeneity",
        "width": "0.96\\textwidth",
    },
    "4": {
        # pdflatex-compatible export of the editable HoriScope overview source.
        "source": "paper-figures/ISCF_overview.pdf",
        "target": "figure_04_method_overview.pdf",
        "label": "fig:horiscope-method",
        "width": "1.00\\textwidth",
    },
    # Automatic LaTeX numbering follows manuscript order. The efficiency figure
    # appears before the scope diagnostic in Section 5.
    "6": {
        "source": "paper-figures/figure_6_accuracy_system_cost.pdf",
        "target": "figure_05_accuracy_system_cost.pdf",
        "label": "fig:accuracy-system-cost",
        "width": "0.90\\textwidth",
    },
    "5": {
        "source": "paper-figures/figure_5_scope_allocation_behavior.pdf",
        "target": "figure_06_scope_allocation_behavior.pdf",
        "label": "fig:scope-allocation-behavior",
        "width": "1.00\\textwidth",
    },
    "7": {
        "source": "paper-figures/figure_7_decoder_transfer.pdf",
        "target": "figure_07_decoder_transfer.pdf",
        "label": "fig:decoder-transfer",
        "width": "0.90\\textwidth",
    },
    "C1": {
        "source": (
            "analysis/iscf_bsca_appendix_c_prediction_export_20260825/outputs/"
            "figure_c1_varied_horizon_forecasts.pdf"
        ),
        "target": "figure_c1_varied_horizon_forecasts.pdf",
        "label": "fig:appendix-visualization",
        "width": "0.90\\textwidth",
        "environment": "figure",
        "placement": "H",
    },
}


TABLE_SOURCES = {
    "table_1.tex": (
        "analysis/iscf_bsca_paper_experiment_consolidation_20260731/"
        "main_tables_author_corrected_20260815/main_i/"
        "table_iscf_bsca_main_i_dataset_average.tex"
    ),
    "table_2.tex": (
        "analysis/iscf_bsca_paper_experiment_consolidation_20260731/"
        "main_tables_author_corrected_20260815/main_ii/"
        "table_iscf_bsca_main_ii_dataset_average.tex"
    ),
    "table_3.tex": (
        "analysis/iscf_bsca_paper_experiment_consolidation_20260731/"
        "core_ablation_20260814/formal_results/table/"
        "table_iscf_bsca_core_ablation.tex"
    ),
    "table_b1.tex": (
        "analysis/iscf_bsca_paper_experiment_consolidation_20260731/"
        "main_tables_author_corrected_20260815/main_i/"
        "table_iscf_bsca_main_i_qdf.tex"
    ),
    "table_b2.tex": (
        "analysis/iscf_bsca_paper_experiment_consolidation_20260731/"
        "main_tables_author_corrected_20260815/main_ii/"
        "table_iscf_bsca_main_ii.tex"
    ),
}


SECTION_LABELS = {
    "1": "sec:introduction",
    "2": "sec:related-work",
    "3": "sec:problem-formulation",
    "4": "sec:method",
    "5": "sec:experiments",
    "6": "sec:discussion",
    "7": "sec:conclusion",
}


SUBSECTION_LABELS = {
    "5.2": "sec:horizon-specific-comparison",
    "5.3": "sec:one-model-comparison",
    "5.4": "sec:system-cost",
    "5.5": "sec:ablation",
    "5.6": "sec:scope-behavior",
    "5.7": "sec:generalization",
}


REFERENCE_REPLACEMENTS = (
    (r"Figure~\\ref\{fig:conceptual-problems\}", r"Fig.~\\ref{fig:conceptual-problems}"),
    (r"Figure 2", r"Fig.~\\ref{fig:prefix-disagreement}"),
    (r"Figure 3", r"Fig.~\\ref{fig:sharing-heterogeneity}"),
    (r"Figure 4", r"Fig.~\\ref{fig:horiscope-method}"),
    (r"Figure 5", r"Fig.~\\ref{fig:scope-allocation-behavior}"),
    (r"Figure 6", r"Fig.~\\ref{fig:accuracy-system-cost}"),
    (r"Figure 7", r"Fig.~\\ref{fig:decoder-transfer}"),
    (r"Figure C1", r"Fig.~\\ref{fig:appendix-visualization}"),
    (r"Tables B1 and B2", r"Tables~\\ref{tab:appendix-horizon-specific} and~\\ref{tab:appendix-one-model}"),
    (r"Table A1", r"Table~\\ref{tab:dataset-statistics}"),
    (r"Table A2", r"Table~\\ref{tab:training-settings}"),
    (r"Table A3", r"Table~\\ref{tab:model-settings}"),
    (r"Table B1", r"Table~\\ref{tab:appendix-horizon-specific}"),
    (r"Table B2", r"Table~\\ref{tab:appendix-one-model}"),
    (r"Table 1", r"Table~\\ref{tab:main_horiscope}"),
    (r"Table 2", r"Table~\\ref{tab:main_horiscope_one_model}"),
    (r"Table 3", r"Table~\\ref{tab:core-ablation}"),
    (r"Appendix A", r"\\ref{app:experiment-details}"),
    (r"Appendix B", r"\\ref{app:full-results}"),
)


def extract_body(source: Path, start_marker: str, end_marker: str | None) -> list[str]:
    """Extract only the manuscript-facing body between frozen markers."""
    lines = source.read_text(encoding="utf-8").splitlines()
    start = lines.index(start_marker)
    end = lines.index(end_marker) if end_marker else len(lines)
    return lines[start:end]


def inline(text: str) -> str:
    """Convert the small Markdown subset used by the frozen drafts."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\\)%", r"\\%", text)
    text = text.replace("×", r"$\\times$")
    text = text.replace("–", "--").replace("—", "---")
    for pattern, replacement in REFERENCE_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    text = re.sub(
        r"Sections 5\.2 and 5\.3",
        r"Sections~\\ref{sec:horizon-specific-comparison} and~\\ref{sec:one-model-comparison}",
        text,
    )
    text = re.sub(r"Section~3", r"Section~\\ref{sec:problem-formulation}", text)
    return text


def figure_block(caption_line: str) -> list[str]:
    """Create one LaTeX figure from a manuscript caption line."""
    match = re.match(r"\*\*Figure ([0-9]+|C1) \| (.+?)\*\*(.*)", caption_line)
    if not match:
        raise ValueError(f"Unrecognized figure caption: {caption_line}")
    number, title, remainder = match.groups()
    spec = FIGURES[number]
    caption = inline(f"{title}{remainder}")
    environment = spec.get("environment", "figure*")
    placement = spec.get("placement", "t")
    return [
        f"\\begin{{{environment}}}[{placement}]",
        r"\centering",
        f"\\includegraphics[width={spec['width']}]{{figures/{spec['target']}}}",
        f"\\caption{{{caption}}}",
        f"\\label{{{spec['label']}}}",
        f"\\end{{{environment}}}",
    ]


def table_block(title_line: str, rows: list[list[str]]) -> list[str]:
    """Convert a compact Appendix A Markdown table to LaTeX."""
    match = re.match(r"### Table (A[123]) \| (.+)", title_line)
    if not match:
        raise ValueError(f"Unrecognized table title: {title_line}")
    number, title = match.groups()
    label = {
        "A1": "tab:dataset-statistics",
        "A2": "tab:training-settings",
        "A3": "tab:model-settings",
    }[number]
    columns = len(rows[0])
    alignment = "l" + "c" * (columns - 1)
    output = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        f"\\caption{{{inline(title)}.}}",
        f"\\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(inline(cell) for cell in rows[0]) + r" \\",
        r"\midrule",
    ]
    output.extend(
        " & ".join(inline(cell) for cell in row) + r" \\" for row in rows[1:]
    )
    output.extend(
        [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    )
    return output


def convert(lines: list[str]) -> str:
    """Convert frozen manuscript Markdown to the LaTeX subset used here."""
    output: list[str] = ["% Generated by manuscript/build_manuscript.py."]
    index = 0
    while index < len(lines):
        line = lines[index]

        if line.startswith("<a id=") or line.startswith("!["):
            index += 1
            continue

        if line.startswith("**Figure "):
            output.extend(figure_block(line))
            output.append("")
            index += 1
            continue

        if line.startswith("### Table A"):
            title_line = line
            index += 1
            while index < len(lines) and not lines[index].startswith("|"):
                index += 1
            table_rows: list[list[str]] = []
            while index < len(lines) and lines[index].startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    table_rows.append(cells)
                index += 1
            output.extend(table_block(title_line, table_rows))
            output.append("")
            continue

        section_match = re.match(r"## ([1-7])\. (.+)", line)
        if section_match:
            number, title = section_match.groups()
            output.extend(
                [f"\\section{{{inline(title)}}}", f"\\label{{{SECTION_LABELS[number]}}}", ""]
            )
            index += 1
            continue

        appendix_match = re.match(r"## ([ABC])\. (.+)", line)
        if appendix_match:
            letter, title = appendix_match.groups()
            label = {
                "A": "app:experiment-details",
                "B": "app:full-results",
                "C": "app:visualization",
            }[letter]
            output.extend(
                [
                    r"\setcounter{table}{0}",
                    r"\setcounter{figure}{0}",
                    f"\\section{{{inline(title.title())}}}",
                    f"\\label{{{label}}}",
                    "",
                ]
            )
            index += 1
            continue

        subsection_match = re.match(r"### ([1-7]\.\d+) (.+)", line)
        if subsection_match:
            number, title = subsection_match.groups()
            output.append(f"\\subsection{{{inline(title)}}}")
            if number in SUBSECTION_LABELS:
                output.append(f"\\label{{{SUBSECTION_LABELS[number]}}}")
            output.append("")
            index += 1
            continue

        appendix_subsection_match = re.match(r"### A\.\d+ (.+)", line)
        if appendix_subsection_match:
            output.extend(
                [f"\\subsection{{{inline(appendix_subsection_match.group(1).title())}}}", ""]
            )
            index += 1
            continue

        if line.startswith("<!-- Insert Table 1"):
            output.extend([r"\input{tables/table_1.tex}", ""])
            index += 1
            continue
        if line.startswith("<!-- Insert Table 2"):
            output.extend([r"\input{tables/table_2.tex}", ""])
            index += 1
            continue
        if line.startswith("<!-- Insert Table 3"):
            output.extend([r"\input{tables/table_3.tex}", ""])
            index += 1
            continue
        if "Typeset insertion:" in line and "main_i/" in line:
            output.extend([r"\clearpage", r"\input{tables/table_b1.tex}", ""])
            index += 1
            continue
        if "Typeset insertion:" in line and "main_ii/" in line:
            output.extend([r"\clearpage", r"\input{tables/table_b2.tex}", ""])
            index += 1
            continue
        if line.startswith("**Table B"):
            index += 1
            continue

        if re.match(r"\d+\. ", line):
            output.append(r"\begin{enumerate}")
            while index < len(lines):
                item_match = re.match(r"\d+\. (.+)", lines[index])
                if not item_match:
                    if not lines[index].strip():
                        lookahead = index + 1
                        if lookahead < len(lines) and re.match(r"\d+\. ", lines[lookahead]):
                            index = lookahead
                            continue
                    break
                output.append(f"\\item {inline(item_match.group(1))}")
                index += 1
                while index < len(lines) and not lines[index].strip():
                    if index + 1 < len(lines) and re.match(r"\d+\. ", lines[index + 1]):
                        index += 1
                        break
                    index += 1
            output.extend([r"\end{enumerate}", ""])
            continue

        if line.strip() == "$$":
            output.append(r"\[")
            index += 1
            while index < len(lines) and lines[index].strip() != "$$":
                output.append(lines[index])
                index += 1
            output.append(r"\]")
            output.append("")
            index += 1
            continue

        if line.strip():
            output.append(inline(line))
        else:
            output.append("")
        index += 1

    return "\n".join(output).rstrip() + "\n"


def copy_and_adjust_tables() -> None:
    """Copy canonical tables and align captions with manuscript routing."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for target_name, source_name in TABLE_SOURCES.items():
        source_text = (REPO_ROOT / source_name).read_text(encoding="utf-8")
        source_text = source_text.replace("ISCF-BSCA", "HoriScope")
        source_text = source_text.replace(
            "tab:main_iscf_bsca_one_model", "tab:main_horiscope_one_model"
        )
        source_text = source_text.replace(
            "tab:main_iscf_bsca", "tab:main_horiscope"
        )
        source_text = source_text.replace(
            "provided in Appendix A", "provided in ~\\ref{app:full-results}"
        )
        if target_name == "table_b1.tex":
            replacement = (
                "\\caption{Full results for the horizon-specific comparison. "
                "Results are reported as MSE and MAE for "
                "$H\\in\\{96,192,336,720\\}$, and Avg. is the arithmetic mean over "
                "the four horizons. HoriScope uses one unified model per dataset, "
                "whereas the baselines follow their horizon-specific evaluation "
                "protocols. The best and second-best displayed values are highlighted "
                "in bold and underlined, respectively.}\n"
                "\\label{tab:appendix-horizon-specific}"
            )
            source_text = re.sub(
                r"\\caption\{.*?\}\n\\label\{.*?\}",
                lambda _: replacement,
                source_text,
                flags=re.DOTALL,
            )
        elif target_name == "table_b2.tex":
            replacement = (
                "\\caption{Full results for the one-model-all-horizons comparison. "
                "Each method uses one maximum-horizon model per dataset, and shorter "
                "horizons are evaluated from the corresponding output prefixes. "
                "Results are reported as MSE and MAE, and Avg. denotes the arithmetic "
                "mean over $H\\in\\{96,192,336,720\\}$. The best and second-best "
                "displayed values are highlighted in bold and underlined, "
                "respectively.}\n"
                "\\label{tab:appendix-one-model}"
            )
            source_text = re.sub(
                r"\\caption\{.*?\}\n\\label\{.*?\}",
                lambda _: replacement,
                source_text,
                flags=re.DOTALL,
            )
        if target_name in {"table_b1.tex", "table_b2.tex"}:
            source_text = source_text.replace(
                r"\begin{table*}[t]", "\\begin{landscape}\n\\begin{table}[p]"
            )
            source_text = source_text.replace(
                r"\resizebox{\textwidth}{!}", r"\resizebox{\linewidth}{!}"
            )
            source_text = source_text.replace(
                r"\end{table*}", "\\end{table}\n\\end{landscape}"
            )
        if target_name == "table_b2.tex":
            source_text = source_text.replace(
                r"\resizebox{\linewidth}{!}", r"\resizebox{0.78\linewidth}{!}"
            )
        (TABLES_DIR / target_name).write_text(source_text, encoding="utf-8", newline="\n")


def copy_figures() -> None:
    """Copy canonical vector figures into the self-contained manuscript bundle."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for spec in FIGURES.values():
        shutil.copy2(REPO_ROOT / spec["source"], FIGURES_DIR / spec["target"])


def main() -> None:
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for target, source, start, end in SECTION_SOURCES:
        body = extract_body(REPO_ROOT / source, start, end)
        (SECTIONS_DIR / target).write_text(convert(body), encoding="utf-8", newline="\n")
    copy_figures()
    copy_and_adjust_tables()


if __name__ == "__main__":
    main()
