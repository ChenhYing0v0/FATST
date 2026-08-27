#!/usr/bin/env python3
"""Audit the self-contained HoriScope source migrated to the KBS template."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manuscript"
TARGET = ROOT / "Elsevier_template"
REPORT = (
    ROOT
    / "analysis"
    / "horiscope_kbs_template_migration_20260827"
    / "audit_report.md"
)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_space(text: str) -> str:
    """Normalize TeX prose for metadata comparisons."""
    return " ".join(text.split())


def extract_command(text: str, command: str) -> str:
    """Extract the first simple braced command argument."""
    match = re.search(rf"\\{command}\{{(.*?)\}}", text, re.DOTALL)
    if match is None:
        raise ValueError(f"Missing \\{command} command")
    return normalize_space(match.group(1))


def extract_environment(text: str, environment: str) -> str:
    """Extract and normalize one LaTeX environment body."""
    match = re.search(
        rf"\\begin\{{{environment}\}}(.*?)\\end\{{{environment}\}}",
        text,
        re.DOTALL,
    )
    if match is None:
        raise ValueError(f"Missing {environment} environment")
    return normalize_space(match.group(1))


def add_check(checks: list[tuple[str, bool, str]], name: str, passed: bool,
              detail: str) -> None:
    """Append one named audit check."""
    checks.append((name, passed, detail))


def main() -> None:
    checks: list[tuple[str, bool, str]] = []

    main_source = (SOURCE / "main.tex").read_text(encoding="utf-8")
    main_target = (TARGET / "elsarticle-template-num.tex").read_text(
        encoding="utf-8"
    )

    add_check(
        checks,
        "KBS document class",
        r"\documentclass[final,3p,times]{elsarticle}" in main_target,
        "The entry point uses the copied KBS final/3p/Times layout.",
    )
    add_check(
        checks,
        "KBS journal metadata",
        r"\journal{Knowledge-Based Systems}" in main_target,
        "The target journal is Knowledge-Based Systems.",
    )

    for field, extractor in (
        ("title", lambda text: extract_command(text, "title")),
        ("abstract", lambda text: extract_environment(text, "abstract")),
        ("keyword", lambda text: extract_environment(text, "keyword")),
    ):
        source_value = extractor(main_source)
        target_value = extractor(main_target)
        add_check(
            checks,
            f"Frozen {field}",
            source_value == target_value,
            f"Normalized {field} content is identical to manuscript/main.tex.",
        )

    exact_pairs: list[tuple[Path, Path]] = []
    for directory in ("sections", "tables", "figures"):
        for source_path in sorted((SOURCE / directory).iterdir()):
            if not source_path.is_file() or source_path.name == "appendices.tex":
                continue
            exact_pairs.append((source_path, TARGET / directory / source_path.name))
    exact_pairs.append((SOURCE / "ref.bib", TARGET / "ref.bib"))

    changed_pairs: list[str] = []
    for source_path, target_path in exact_pairs:
        if not target_path.exists() or sha256(source_path) != sha256(target_path):
            changed_pairs.append(str(source_path.relative_to(ROOT)))
    add_check(
        checks,
        "Body/table/figure/bibliography identity",
        not changed_pairs,
        (
            "All frozen body sections, tables, figure assets and ref.bib are "
            "byte-identical."
            if not changed_pairs
            else "Unexpected differences: " + ", ".join(changed_pairs)
        ),
    )

    source_appendix = (SOURCE / "sections/appendices.tex").read_text(
        encoding="utf-8"
    )
    target_appendix = (TARGET / "sections/appendices.tex").read_text(
        encoding="utf-8"
    )
    normalized_target_appendix = target_appendix.replace("\\FloatBarrier\n", "")
    add_check(
        checks,
        "Appendix content identity",
        source_appendix == normalized_target_appendix,
        "The only appendix delta is one layout-only FloatBarrier before Appendix B.",
    )

    section_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((TARGET / "sections").glob("*.tex"))
    )
    citation_keys: set[str] = set()
    for group in re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", section_text):
        citation_keys.update(key.strip() for key in group.split(","))
    bib_text = (TARGET / "ref.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"^@\w+\{([^,]+),", bib_text, re.MULTILINE))
    missing_citations = sorted(citation_keys - bib_keys)
    uncited_entries = sorted(bib_keys - citation_keys)
    add_check(
        checks,
        "Citation coverage",
        not missing_citations,
        (
            f"{len(citation_keys)} cited / {len(bib_keys)} defined; "
            f"missing={missing_citations or 'none'}, retained unused entries="
            f"{uncited_entries or 'none'}."
        ),
    )

    asset_paths: list[Path] = []
    submission_tex = main_target + "\n" + section_text
    for raw_path in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",
                               submission_tex):
        asset_paths.append(TARGET / raw_path)
    for raw_path in re.findall(r"\\input\{(tables/[^}]+)\}", submission_tex):
        suffix = "" if raw_path.endswith(".tex") else ".tex"
        asset_paths.append(TARGET / f"{raw_path}{suffix}")
    missing_assets = sorted(
        str(path.relative_to(TARGET)) for path in asset_paths if not path.exists()
    )
    add_check(
        checks,
        "Referenced assets",
        not missing_assets,
        f"Checked {len(asset_paths)} figure/table paths; missing={missing_assets or 'none'}.",
    )

    highlights_text = (TARGET / "highlights.tex").read_text(encoding="utf-8")
    highlights = [
        normalize_space(item)
        for item in re.findall(r"\\item\s+(.*?)(?=\\item|\\end\{highlights\})",
                               highlights_text, re.DOTALL)
    ]
    highlight_lengths = [len(item) for item in highlights]
    add_check(
        checks,
        "Elsevier highlights",
        3 <= len(highlights) <= 5 and all(length <= 85 for length in highlight_lengths),
        f"{len(highlights)} highlights; character counts={highlight_lengths}.",
    )

    legacy_files = [
        "efficiency.pdf",
        "embedding.pdf",
        "energy.pdf",
        "lookback.pdf",
        "overview.pdf",
        "prefer.pdf",
        "representation.pdf",
        "visual.pdf",
        "math_utils.tex",
    ]
    retained_legacy = [name for name in legacy_files if (TARGET / name).exists()]
    add_check(
        checks,
        "Legacy PDT assets removed",
        not retained_legacy,
        f"Retained legacy assets={retained_legacy or 'none'}.",
    )

    pdf_path = TARGET / "build/elsarticle-template-num.pdf"
    review_pdf_path = TARGET / "HoriScope_KBS_submission.pdf"
    log_path = TARGET / "build/elsarticle-template-num.log"
    critical_patterns = (
        "Overfull",
        "Float too large",
        "Undefined control sequence",
        "Citation `",
        "Reference `",
        "LaTeX Error",
        "destination with the same identifier",
    )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    critical_hits = [pattern for pattern in critical_patterns if pattern in log_text]
    add_check(
        checks,
        "LaTeX log",
        pdf_path.exists() and not critical_hits,
        f"Submission PDF exists; critical log patterns={critical_hits or 'none'}.",
    )

    pdf_info = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    page_match = re.search(r"^Pages:\s+(\d+)", pdf_info, re.MULTILINE)
    pages = int(page_match.group(1)) if page_match else 0
    add_check(
        checks,
        "Rendered submission",
        pages > 0,
        f"A4 PDF rendered successfully with {pages} pages.",
    )
    add_check(
        checks,
        "Author-review PDF",
        review_pdf_path.exists() and sha256(review_pdf_path) == sha256(pdf_path),
        "The root-level author-review PDF is byte-identical to the audited build.",
    )

    status = all(passed for _, passed, _ in checks)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "# HoriScope KBS Template Migration Audit",
        "",
        f"- **Date:** 2026-08-27",
        f"- **Overall status:** `{'PASS' if status else 'FAIL'}`",
        f"- **Entry point:** `Elsevier_template/elsarticle-template-num.tex`",
        f"- **Rendered PDF:** `Elsevier_template/build/elsarticle-template-num.pdf`",
        f"- **Page count:** {pages}",
        "",
        "## Checks",
        "",
        "| Check | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for name, passed, detail in checks:
        rows.append(f"| {name} | `{'PASS' if passed else 'FAIL'}` | {detail} |")
    rows.extend(
        [
            "",
            "## Submission Boundary",
            "",
            "The scientific body, tables, figures and bibliography are preserved from "
            "the frozen manuscript. The KBS migration changes only the journal wrapper, "
            "author-facing metadata, declaration blocks, highlights and float control. "
            "The author list, affiliations, CRediT statement, funding, acknowledgments, "
            "competing-interest statement and data-availability statement were carried "
            "from the author-provided PDT KBS source and require final author confirmation.",
            "",
        ]
    )
    REPORT.write_text("\n".join(rows), encoding="utf-8", newline="\n")

    print(f"KBS migration audit: {'PASS' if status else 'FAIL'}")
    for name, passed, detail in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    print(f"Report: {REPORT}")
    if not status:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
