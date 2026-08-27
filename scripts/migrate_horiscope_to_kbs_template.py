#!/usr/bin/env python3
"""Inject the frozen HoriScope manuscript into the copied KBS template."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "Elsevier_template/elsarticle-template-num.tex"


PREAMBLE = r"""\documentclass[final,3p,times]{elsarticle}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,bm}
\usepackage{booktabs,multirow,array,tabularx}
\usepackage{graphicx}
\usepackage[table]{xcolor}
\usepackage{adjustbox}
\usepackage{float}
\usepackage{placeins}
\usepackage{microtype}
\usepackage{lineno}
\usepackage{xurl}
\usepackage[hidelinks]{hyperref}

\emergencystretch=1em
\hfuzz=2pt

\journal{Knowledge-Based Systems}
"""


FRONTMATTER = r"""\hypersetup{pageanchor=false}
\begin{frontmatter}

\title{HoriScope: Adaptive Multi-Scope Decoding for Unified Varied-Horizon Time-Series Forecasting}

\author{Chenhao Ying}
\ead{chenhying01@zju.edu.cn}
\author{Jiangang Lu\corref{cor1}}
\ead{lujg@zju.edu.cn}
\address{State Key Laboratory of Industrial Control Technology, College of
Control Science and Engineering, Zhejiang University, Hangzhou 310027,
Zhejiang, China}
\cortext[cor1]{Corresponding author}

\begin{abstract}
Most time-series forecasting methods optimize a model for a predefined prediction horizon, whereas practical applications often request forecasts of different lengths from the same history. Under the prevailing horizon-specific paradigm, separate models are optimized for different prediction lengths. This fragments multi-horizon service and can produce inconsistent forecasts for future steps shared across horizon requests. We instead formulate unified varied-horizon forecasting (UVHF), in which a single horizon-agnostic trajectory serves different request endpoints through its nested prefixes. This formulation imposes two coupled requirements on decoder design. First, predictions for shared future steps must remain invariant to the requested horizon, a property we term cross-horizon prefix consistency (CHPC). Second, jointly modeling short-, medium- and long-range futures requires the decoder to regulate how broadly history-conditioned information is shared across the future domain. We therefore propose HoriScope, an adaptive multi-scope decoder for UVHF. HoriScope constructs region-wise forecasts under multiple sharing scopes and integrates them through target-adaptive allocation into one trajectory, thereby satisfying CHPC by construction. We further introduce Balanced Scope Co-Adaptation (BSCA), which balances multi-scope optimization through scope-wise supervision and allocation regularization. Experiments across seven multivariate benchmarks show that HoriScope achieves state-of-the-art accuracy against recent horizon-specific and unified forecasters while serving all supported horizons with a single model. It also offers a favorable trade-off between checkpoint storage and inference memory. Controlled ablations support the proposed components, and generalization studies demonstrate decoder portability across the evaluated backbone families. These results establish output-side multi-scope generation as an effective foundation for UVHF.
\end{abstract}

\input{highlights.tex}

\begin{keyword}
Time series forecasting \sep unified varied-horizon forecasting \sep
cross-horizon prefix consistency \sep multi-scope forecasting \sep
adaptive forecast generation
\end{keyword}

\end{frontmatter}
\hypersetup{pageanchor=true}
"""


MAIN_TEXT = r"""%% Main text

% Uncomment for a line-numbered review copy.
% \linenumbers

\input{sections/01_introduction.tex}
\input{sections/02_related_work.tex}
\input{sections/03_problem_formulation.tex}
\input{sections/04_method.tex}
\input{sections/05_experiments.tex}
\FloatBarrier
\input{sections/06_discussion.tex}
\input{sections/07_conclusion.tex}

"""


BACKMATTER = r"""\section*{CRediT authorship contribution statement}
\textbf{Chenhao Ying:} Writing--original draft, Writing--review \& editing,
Visualization, Validation, Supervision, Software, Resources, Project
administration, Methodology, Investigation. \textbf{Jiangang Lu:}
Writing--review \& editing, Conceptualization.

\section*{Declaration of Competing Interest}
The authors declare that they have no known competing financial interests or
personal relationships that could have appeared to influence the work reported
in this paper.

\section*{Data availability}
Data will be made available on request.

\section*{Acknowledgments}
This work was supported in part by the National Natural Science Foundation of
China under Grant 62293504 and Grant 62293500, and in part by the Zhejiang
Province Science and Technology Plan Project under Grant 2025C01091.

\appendix
\renewcommand*{\theHtable}{appendix.\Alph{section}.\arabic{table}}
\renewcommand*{\theHfigure}{appendix.\Alph{section}.\arabic{figure}}
\input{sections/appendices.tex}

\bibliographystyle{elsarticle-num}
\bibliography{ref}

\end{document}
"""


def replace_once(text: str, pattern: str, replacement: str, name: str) -> str:
    """Replace one structural region and fail if the template drifts."""
    updated, count = re.subn(
        pattern,
        lambda _: replacement,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError(f"Expected one {name} region, found {count}")
    return updated


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    text = replace_once(
        text,
        r"\A.*?(?=\\begin\{document\})",
        PREAMBLE + "\n",
        "preamble",
    )
    text = replace_once(
        text,
        (
            r"(?:\\hypersetup\{pageanchor=false\}\s*)*"
            r"\\begin\{frontmatter\}.*?\\end\{frontmatter\}"
        ),
        FRONTMATTER.strip(),
        "frontmatter",
    )
    text = replace_once(
        text,
        r"\\end\{frontmatter\}.*?(?=\\section\*\{CRediT authorship contribution statement\})",
        "\\end{frontmatter}\n\\hypersetup{pageanchor=true}\n\n" + MAIN_TEXT,
        "main text",
    )
    text = replace_once(
        text,
        r"\\section\*\{CRediT authorship contribution statement\}.*\Z",
        BACKMATTER.strip(),
        "backmatter",
    )
    TARGET.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
