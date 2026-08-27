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
- bibliography: PASS (39 cited keys; 39 defined keys; 0 missing)
- final PDF: PASS (24 A4 pages)
- fatal LaTeX diagnostics: PASS (0 errors, undefined controls, oversized floats, undefined references or citations)
- visual page audit: PASS (front matter, Figures 1--7, Tables 1--4, Appendices A--C and references inspected)

## Template-compatibility notes

The main source intentionally retains the PDT preamble and front-matter scaffold exactly through the KBS journal declaration. Consequently, the build also retains non-fatal diagnostics produced by that exact scaffold: 17 duplicate PDF-destination warnings. The log contains 1 overfull hbox warning (maximum 0.81009 pt), URL-related underfull boxes, and 15 BibTeX empty-page metadata warnings. None causes clipping, missing content, unresolved citations or a submission-structure deviation.

## Checksums

- main TeX: `d9e938b71d9d314b9684ae63e4ce27d1ad7127114fd635f1f1b6bc791eeb5baf`
- final PDF: `8cb789b0910e04d67dea62936bc1565a063d2a93102fbcbc6485c02b6369cb91`
- bibliography: `2b04e03e5c5b3cd3b1bca9b2eca62a1e35471334a89fd88a2fb79a3e7bd76e7d`
