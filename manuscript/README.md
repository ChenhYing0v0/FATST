# ISCF-BSCA Elsevier LaTeX manuscript

This directory contains a self-contained `elsarticle` manuscript assembled from the temporarily frozen manuscript-facing Markdown drafts.

## Current placeholders

- The title is the provisional working title recorded in the paper architecture.
- Author names, affiliations, corresponding-author details and journal name are placeholders.
- The Abstract and Keywords are populated from the current author-review draft; References remain intentionally empty.
- Citation commands are retained from the approved drafts and therefore remain unresolved until `references.bib` is populated.

## Build

From this directory, run:

```bash
python build_manuscript.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

`build_manuscript.py` extracts only the manuscript-facing bodies, copies the canonical PDF figures, and copies the canonical LaTeX tables into this submission bundle. Draft-status tables, terminology ledgers and editorial audits are excluded.

## Numbering note

LaTeX numbers figures by their first appearance in the assembled manuscript. Because the accuracy--cost figure occurs in Section 5.4 before the scope-allocation diagnostic in Section 5.6, they become Figures 5 and 6, respectively. This corrects the non-sequential working labels in the Markdown drafts without changing either figure or its scientific role.
