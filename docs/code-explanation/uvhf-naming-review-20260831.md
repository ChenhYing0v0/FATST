# UVHF naming review package

## Purpose

This update creates an independent, highlighted KBS manuscript review package
without changing the frozen `HoriScope_KBS_submission/` source.

## Terminology mapping

- Task: `unified varied-horizon forecasting`
- Framework/model: `UVHF`
- Architecture: `Multi-Scope Decoder (MSD)`
- Training: `Balanced Co-Adaptation (BCA)`
- Structural property: `CHPC`

The manuscript uses `UVHF` only for the proposed framework. Generic references
to the research setting remain written in full.

## Figure path

Review-specific copies or regenerated assets are stored in
`UVHF_KBS_submission_highlighted_review/`. The quantitative plotting sources
under `analysis/uvhf_naming_review_20260831/` preserve the frozen data and alter
only visible framework/component labels.

## Verification

The package is compiled with `latexmk`; the generated PDF is inspected across
all pages. Static terminology checks ensure that legacy method names and
task/model ambiguity patterns do not remain in the revised manuscript.

## Approved submission package

After author approval, the reviewed manuscript was promoted to
`UVHF_KBS_submission/`. The canonical package contains the same accepted text,
figures, tables and references as the highlighted review package. All 94
review-only wrappers and the blue BCA loss label are removed, and the source
preamble before `\journal{Knowledge-Based Systems}` is restored to exact byte
identity with the previously audited PDT-based package. The original HoriScope
package and highlighted UVHF package are retained unchanged for comparison.
