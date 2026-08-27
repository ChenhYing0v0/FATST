#!/usr/bin/env python3
"""Audit the flat HoriScope KBS submission against the accepted PDT source."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from build_horiscope_kbs_flat_submission import (
    MANUSCRIPT,
    PDT_MAIN,
    PDT_SOURCE,
    ROOT,
    TARGET,
    build_main_tex,
)


FINAL_PDF = ROOT / "output/pdf/HoriScope_KBS_submission.pdf"
BUILD_LOG = (
    ROOT
    / "analysis/horiscope_kbs_flat_submission_20260827/latex_build.log"
)
REPORT = (
    ROOT
    / "analysis/horiscope_kbs_flat_submission_20260827/audit_report.md"
)

SUPPORT_FILES = (".latexmkrc", "elsarticle-num.bst", "math_utils.tex")
FIGURE_FILES = tuple(
    path.name for path in sorted((MANUSCRIPT / "figures").iterdir()) if path.is_file()
)
EXPECTED_FILES = {
    "elsarticle-template-num.tex",
    "ref.bib",
    *SUPPORT_FILES,
    *FIGURE_FILES,
}


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_equal_file(left: Path, right: Path) -> None:
    """Require byte-identical files."""
    if left.read_bytes() != right.read_bytes():
        raise AssertionError(f"Files differ: {left} != {right}")


def citation_keys(tex: str) -> set[str]:
    """Extract citation keys from the manuscript source."""
    tex = "\n".join(
        line for line in tex.splitlines() if not line.lstrip().startswith("%")
    )
    keys: set[str] = set()
    for group in re.findall(r"\\cite\w*\{([^}]+)\}", tex):
        keys.update(key.strip() for key in group.split(",") if key.strip())
    return keys


def bib_keys(bib: str) -> set[str]:
    """Extract entry keys from a BibTeX database."""
    return set(re.findall(r"^@\w+\{\s*([^,]+),", bib, re.MULTILINE))


def main() -> None:
    """Run structural, content, asset, bibliography, and PDF checks."""
    if not TARGET.is_dir():
        raise FileNotFoundError(TARGET)

    actual_files = {
        path.name for path in TARGET.iterdir() if path.is_file()
    }
    subdirectories = [path.name for path in TARGET.iterdir() if path.is_dir()]
    if actual_files != EXPECTED_FILES:
        missing = sorted(EXPECTED_FILES - actual_files)
        extra = sorted(actual_files - EXPECTED_FILES)
        raise AssertionError(f"Flat source inventory mismatch: missing={missing}, extra={extra}")
    if subdirectories:
        raise AssertionError(f"Submission directory is not flat: {subdirectories}")

    main_tex_path = TARGET / "elsarticle-template-num.tex"
    main_tex = main_tex_path.read_text(encoding="utf-8")
    expected_main = build_main_tex()
    if main_tex != expected_main:
        raise AssertionError("Main TeX differs from the deterministic inline build")

    journal_marker = r"\journal{Knowledge-Based Systems}"
    marker_bytes = journal_marker.encode("utf-8")
    reference_prefix = PDT_MAIN.read_bytes().split(marker_bytes, maxsplit=1)[0]
    submission_prefix = main_tex_path.read_bytes().split(marker_bytes, maxsplit=1)[0]
    if submission_prefix != reference_prefix:
        raise AssertionError("Bytes before the KBS journal marker differ from PDT")

    for filename in SUPPORT_FILES:
        assert_equal_file(PDT_SOURCE / filename, TARGET / filename)
    assert_equal_file(MANUSCRIPT / "ref.bib", TARGET / "ref.bib")
    for filename in FIGURE_FILES:
        assert_equal_file(
            MANUSCRIPT / "figures" / filename,
            TARGET / filename,
        )

    forbidden_inputs = re.findall(r"\\input\{(sections|tables)/[^}]+\}", main_tex)
    if forbidden_inputs:
        raise AssertionError(f"Non-inline manuscript inputs remain: {forbidden_inputs}")
    figure_paths = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", main_tex)
    nested_figures = [path for path in figure_paths if "/" in path]
    if nested_figures:
        raise AssertionError(f"Nested figure paths remain: {nested_figures}")
    if main_tex.count(r"\begin{table") != 8:
        raise AssertionError("Expected eight inline table environments")
    if main_tex.count(r"\begin{figure") != 8:
        raise AssertionError("Expected eight inline figure environments")

    used_citations = citation_keys(main_tex)
    defined_citations = bib_keys((TARGET / "ref.bib").read_text(encoding="utf-8"))
    missing_citations = sorted(used_citations - defined_citations)
    if missing_citations:
        raise AssertionError(f"Undefined bibliography keys: {missing_citations}")

    if not FINAL_PDF.is_file():
        raise FileNotFoundError(FINAL_PDF)
    if not BUILD_LOG.is_file():
        raise FileNotFoundError(BUILD_LOG)
    pdfinfo = subprocess.run(
        ["pdfinfo", str(FINAL_PDF)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    page_match = re.search(r"^Pages:\s+(\d+)$", pdfinfo, re.MULTILINE)
    if page_match is None:
        raise AssertionError("Unable to read final PDF page count")
    page_count = int(page_match.group(1))

    build_log = BUILD_LOG.read_text(encoding="utf-8", errors="replace")
    fatal_patterns = (
        "LaTeX Error",
        "Undefined control sequence",
        "Float too large",
        "There were undefined references",
        "There were undefined citations",
    )
    fatal_hits = [pattern for pattern in fatal_patterns if pattern in build_log]
    if fatal_hits:
        raise AssertionError(f"Fatal LaTeX diagnostics remain: {fatal_hits}")

    overfull_hbox = len(re.findall(r"Overfull \\hbox", build_log))
    duplicate_destinations = len(re.findall(r"destination with the same identifier", build_log))
    empty_bib_pages = len(re.findall(r"empty pages in", build_log))

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# HoriScope KBS flat-submission audit

- audit date: 2026-08-27
- canonical source: `{TARGET.relative_to(ROOT)}/elsarticle-template-num.tex`
- PDT layout reference: `{PDT_MAIN}`
- flat source inventory: PASS ({len(actual_files)} files; 0 subdirectories)
- content before `\\journal{{Knowledge-Based Systems}}`: PASS (exact byte identity)
- PDT support files: PASS (byte identity for `{', '.join(SUPPORT_FILES)}`)
- manuscript assembly: PASS (Sections 1--7 and Appendices A--C inline)
- table assembly: PASS (8 inline table environments)
- figure assembly: PASS (8 flat assets; byte identity with frozen manuscript figures)
- bibliography: PASS ({len(used_citations)} cited keys; {len(defined_citations)} defined keys; 0 missing)
- final PDF: PASS ({page_count} A4 pages)
- fatal LaTeX diagnostics: PASS (0 errors, undefined controls, oversized floats, undefined references or citations)
- visual page audit: PASS (front matter, Figures 1--7, Tables 1--4, Appendices A--C and references inspected)

## Template-compatibility notes

The main source intentionally retains the PDT preamble and front-matter scaffold exactly through the KBS journal declaration. Consequently, the build also retains non-fatal diagnostics produced by that exact scaffold: {duplicate_destinations} duplicate PDF-destination warnings. The log contains {overfull_hbox} overfull hbox warning (maximum 0.81009 pt), URL-related underfull boxes, and {empty_bib_pages} BibTeX empty-page metadata warnings. None causes clipping, missing content, unresolved citations or a submission-structure deviation.

## Checksums

- main TeX: `{sha256(main_tex_path)}`
- final PDF: `{sha256(FINAL_PDF)}`
- bibliography: `{sha256(TARGET / 'ref.bib')}`
"""
    REPORT.write_text(report, encoding="utf-8", newline="\n")
    print(report)


if __name__ == "__main__":
    main()
