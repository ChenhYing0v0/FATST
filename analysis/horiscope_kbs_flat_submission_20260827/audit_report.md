# HoriScope KBS flat-submission audit

- audit date: 2026-08-27
- canonical source: `HoriScope_KBS_submission/elsarticle-template-num.tex`
- PDT layout reference: `/Users/river/PaperResearch/Project/R_2026_PDT/manuscript/revision/PDT_revision_v01/source_latex/elsarticle-template-num.tex`
- flat source inventory: PASS (13 files; 0 subdirectories)
- content before `\journal{Knowledge-Based Systems}`: PASS (exact byte identity)
- PDT support files: PASS (byte identity for `.latexmkrc, elsarticle-num.bst, math_utils.tex`)
- manuscript assembly: PASS (Sections 1--7 and Appendices A--C inline)
- table assembly: PASS (8 inline table environments)
- figure assembly: PASS (8 flat assets; byte identity with frozen manuscript figures)
- bibliography: PASS (39 cited keys; 40 defined keys; 0 missing)
- final PDF: PASS (25 A4 pages)
- fatal LaTeX diagnostics: PASS (0 errors, undefined controls, oversized floats, undefined references or citations)
- visual page audit: PASS (front matter, Figures 1--7, Tables 1--4, Appendices A--C and references inspected)

## Template-compatibility notes

The main source intentionally retains the PDT preamble and front-matter scaffold exactly through the KBS journal declaration. Consequently, the build also retains non-fatal diagnostics produced by that exact scaffold: 17 duplicate PDF-destination warnings. The log contains 1 overfull hbox warning (maximum 0.81009 pt), URL-related underfull boxes, and 0 BibTeX empty-page metadata warnings. None causes clipping, missing content, unresolved citations or a submission-structure deviation.

## Checksums

- main TeX: `d753aba616482b88a3a33d4ab2bd6028056bb9599e8735791691b00ce2e25e0a`
- final PDF: `1aed013edb5454fc9124eed2df266a664b63cf32cdc7418bdbe7e1187108676d`
- bibliography: `04ff496097b80314ea45772196cb5fceeb80a7b0c4228941a1003dec84f2f4fb`
