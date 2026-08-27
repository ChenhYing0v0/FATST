#!/usr/bin/env python3
"""Normalize HoriScope references to the accepted PDT bibliography style."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BIB_PATH = ROOT / "manuscript/ref.bib"

BOOKTITLE_MAP = {
    "Proceedings of the 31st International Conference on Machine Learning":
        "Proc. Int. Conf. Mach. Learn. (ICML)",
    "Proceedings of the 39th International Conference on Machine Learning":
        "Proc. Int. Conf. Mach. Learn. (ICML)",
    "Proceedings of the 41st International Conference on Machine Learning":
        "Proc. Int. Conf. Mach. Learn. (ICML)",
    "Proceedings of the 42nd International Conference on Machine Learning":
        "Proc. Int. Conf. Mach. Learn. (ICML)",
    "Proceedings of the 2nd European Symposium on Time Series Prediction":
        "Proc. Eur. Symp. Time Ser. Predict. (ESTSP)",
    "Proceedings of the AAAI Conference on Artificial Intelligence":
        "Proc. AAAI Conf. Artif. Intell.",
    "Advances in Neural Information Processing Systems":
        "Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)",
    "International Conference on Learning Representations":
        "Proc. Int. Conf. Learn. Represent. (ICLR)",
    "Proceedings of the 27th International Conference on Artificial Intelligence and Statistics":
        "Proc. Int. Conf. Artif. Intell. Stat. (AISTATS)",
    "Proceedings of the 28th International Conference on Artificial Intelligence and Statistics":
        "Proc. Int. Conf. Artif. Intell. Stat. (AISTATS)",
    "Proceedings of the 41st International ACM SIGIR Conference on Research and Development in Information Retrieval":
        "Proc. Int. ACM SIGIR Conf. Res. Dev. Inf. Retr. (SIGIR)",
}

JOURNAL_MAP = {
    "Expert Systems with Applications": "Expert Syst. Appl.",
    "Data Mining and Knowledge Discovery": "Data Min. Knowl. Discov.",
    "Transactions on Machine Learning Research": "Trans. Mach. Learn. Res.",
    "Information Sciences": "Inf. Sci.",
}


def entry_blocks(text: str) -> list[tuple[str, str]]:
    """Split comments and BibTeX entries without changing their order."""
    starts = list(re.finditer(r"(?m)^@\w+\{", text))
    if not starts:
        raise RuntimeError("No BibTeX entries found")
    parts: list[tuple[str, str]] = [("text", text[: starts[0].start()])]
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        parts.append(("entry", text[start.start():end]))
    return parts


def entry_key(block: str) -> str:
    """Return a BibTeX entry key."""
    match = re.match(r"@\w+\{([^,]+),", block)
    if match is None:
        raise RuntimeError("Malformed BibTeX entry")
    return match.group(1)


def normalize_entry(block: str) -> str:
    """Remove web metadata and abbreviate venues in one entry."""
    kind_match = re.match(r"@(\w+)\{", block)
    if kind_match is None:
        raise RuntimeError("Malformed BibTeX entry")
    entry_type = kind_match.group(1).lower()
    key = entry_key(block)

    lines = []
    for line in block.splitlines():
        field_match = re.match(r"\s*(url|doi|series|publisher)\s*=", line, re.I)
        if field_match is not None:
            field = field_match.group(1).lower()
            if field in {"url", "doi", "series"}:
                continue
            if field == "publisher" and entry_type == "inproceedings":
                continue
        if key == "wang2026qdf" and re.match(r"\s*pages\s*=", line, re.I):
            continue
        lines.append(line)
    normalized = "\n".join(lines)

    for full, abbreviated in BOOKTITLE_MAP.items():
        normalized = normalized.replace(
            f"booktitle = {{{full}}}",
            f"booktitle = {{{abbreviated}}}",
        )
    for full, abbreviated in JOURNAL_MAP.items():
        normalized = normalized.replace(
            f"journal = {{{full}}}",
            f"journal = {{{abbreviated}}}",
        )
    normalized = normalized.replace("Temporal 2D-Variation", "Temporal {2D}-Variation")
    normalized = normalized.replace(
        "Dynamic Convolution and 3D-Variation",
        "Dynamic Convolution and {3D}-Variation",
    )
    return normalized.rstrip() + "\n\n"


def main() -> None:
    """Rewrite and validate the canonical bibliography."""
    text = BIB_PATH.read_text(encoding="utf-8")
    output: list[str] = []
    for kind, block in entry_blocks(text):
        if kind == "text":
            output.append(block.rstrip() + "\n\n")
            continue
        if entry_key(block) == "yu2024leddam":
            continue
        output.append(normalize_entry(block))

    normalized = "".join(output).rstrip() + "\n"
    if re.search(r"(?mi)^\s*(url|doi|series)\s*=", normalized):
        raise AssertionError("URL, DOI, or series metadata remains")
    if re.search(
        r"(?mi)^\s*booktitle\s*=\s*\{(?:Proceedings|Advances|International Conference)",
        normalized,
    ):
        raise AssertionError("An unabbreviated conference title remains")
    if len(re.findall(r"(?m)^@", normalized)) != 39:
        raise AssertionError("Expected 39 cited bibliography entries")

    BIB_PATH.write_text(normalized, encoding="utf-8", newline="\n")
    print(f"Normalized bibliography: {BIB_PATH}")


if __name__ == "__main__":
    main()
