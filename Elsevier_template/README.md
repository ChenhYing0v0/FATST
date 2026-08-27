# Superseded HoriScope KBS migration

> This split-file migration is retained only as a historical artifact. The
> canonical, PDT-exact, flat submission source is now
> `../HoriScope_KBS_submission/`.

This directory contains the self-contained Knowledge-Based Systems manuscript
source migrated from the frozen HoriScope paper draft.

## Build

Run the following command from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=build elsarticle-template-num.tex
```

The compiled submission preview is written to
`build/elsarticle-template-num.pdf`.
The audited copy intended for author review is
`HoriScope_KBS_submission.pdf`.

## Structure

- `elsarticle-template-num.tex`: KBS `final,3p,times` manuscript entry point;
- `HoriScope_KBS_submission.pdf`: audited author-review preview;
- `highlights.tex`: highlights included in the manuscript front matter;
- `highlights.txt`: upload-ready plain-text highlights source;
- `sections/`: frozen HoriScope main text and appendices;
- `tables/`: main and appendix tables;
- `figures/`: manuscript figure assets;
- `ref.bib`: HoriScope bibliography database;
- `elsarticle.cls` and `elsarticle-num.bst`: local Elsevier template files.

Author contributions, funding, acknowledgments, competing interests and data
availability were carried over from the author-provided PDT KBS source and must
be confirmed by the authors before submission.
