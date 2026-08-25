# Figure C1 QA record

**Figure source:** `scripts/plot_iscf_bsca_appendix_c_figure.py`  
**Backend:** Python/matplotlib only  
**Target dimensions:** 183 mm wide, approximately 188 mm high

## Automated checks

- `python -m py_compile scripts/plot_iscf_bsca_appendix_c_figure.py`: passed.
- `nature-figure/scripts/validate_figure.py --strict`: passed with 14 PASS,
  0 WARN and 0 FAIL.
- SVG export uses `svg.fonttype=none`; editable `<text>` nodes are present.
- PDF export is one page; SVG, PDF, PNG (600 dpi) and TIFF (600 dpi) are
  present.
- Source CSV contains 10,080 rows: 7 datasets × 2 samples × 720 steps.
- All plotted arrays are finite and have shape `(2, 720)` per dataset.

## Visual checks

- Seven dataset rows and two sample columns are aligned on a common future-step
  axis; y-limits are independently scaled per dataset and shared between the
  two samples in each row.
- Ground truth and ISCF-BSCA use stable dark-slate and teal encodings. Paired
  nested-prefix rulers above the sample columns identify the four horizon
  endpoints; faint neutral vertical guides preserve alignment without
  overwhelming the traces.
- The image title is `Representative validation trajectories`; the redundant
  `Appendix C` prefix is absent because the manuscript supplies the section
  hierarchy.
- One prefix ruler is aligned above each sample column, and the selected
  validation-audited channel is fixed within each dataset. Channel IDs and
  sample scores are recorded in the source metadata.
- Dataset names, sample headers, legend and axis labels do not overlap at the
  rendered 600-dpi size. The two right-hand columns intentionally omit
  duplicated y tick labels to preserve the compact grid.
- The figure is explicitly validation-only and qualitative; no error bars,
  population statistics or test-set claims are encoded.

## Reviewer boundary

Samples were selected by a deterministic low-error rule using validation
labels. Figure C1 should therefore be captioned as representative validation
examples, not as a prevalence estimate, an independent generalization result,
or a claim that every forecast follows the same visual pattern.
