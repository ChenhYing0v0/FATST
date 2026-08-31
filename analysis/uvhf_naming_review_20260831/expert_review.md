# Expert review of the UVHF/MSD/BCA naming revision

## Verdict

**Recommendation: adopt after author approval.** The revised hierarchy is
clearer and more paper-defining than the former HoriScope/BSCA naming. It gives
the title direct task salience, separates architectural and optimization
contributions, and aligns the terminology used in the abstract, method,
experiments, figures and appendices.

## Reviewer-facing strengths

1. **Task salience.** The title immediately identifies unified varied-horizon
   time-series forecasting, so the paper topic is legible without first
   decoding a coined model name.
2. **Contribution hierarchy.** `UVHF = MSD + BCA` provides a compact mental
   model: UVHF is the complete framework, MSD is the inference architecture,
   and BCA is its training strategy.
3. **Claim alignment.** The method section, ablations and generalization study
   now refer to the level actually evaluated. Decoder replacement is described
   as MSD transfer with BCA training, avoiding the inaccurate impression that
   BCA is part of the inference decoder.
4. **Task/model disambiguation.** The generic task is always written as
   `unified varied-horizon forecasting`; `UVHF` is reserved for the proposed
   framework. This resolves the main ambiguity created when an acronym names
   both a setting and a model.
5. **Visual consistency.** Every figure containing an old model label was
   regenerated from the same source data, and captions for the component-only
   figures were updated to the new hierarchy.

## Remaining risks and boundaries

1. **Generic acronym risk.** `UVHF` is intentionally close to the task name.
   Its clarity depends on preserving the terminology contract. Future revisions
   should not introduce phrases such as `the UVHF task`, `UVHF setting` or
   `UVHF workflow`.
2. **Novelty wording.** Prior flexible-horizon methods, especially ElasTST,
   prevent an unqualified claim that UVHF creates the first multi-horizon
   paradigm. The manuscript appropriately claims systematic formulation,
   decoder-side problem analysis and a targeted framework.
3. **BCA specificity.** `Balanced Co-Adaptation` is concise but generic in
   isolation. Its first definition therefore explicitly ties co-adaptation to
   the scope-indexed forecast field and allocation process.
4. **MSD versus multi-scale encoding.** Reviewers may initially associate MSD
   with conventional multi-scale history modeling. Related Work and Discussion
   preserve the essential distinction: scope indexes output-side latent-state
   reuse, whereas scale usually indexes input resolution or frequency.

## Verification summary

- Original submission directory: byte-for-byte unchanged by checksum.
- Legacy manuscript terms (`HoriScope`, `ISCF`, `BSCA`): absent from the
  revised TeX and compiled PDF.
- Task/model ambiguity patterns: absent.
- Revision markers: 94 blue-highlight blocks.
- Modified plotting sources: both pass the Nature Figure static preflight.
- Manuscript build: 24 pages, all citations and cross-references resolved, no
  overfull or underfull box warnings.
- Visual QA: all 24 pages inspected; revised labels, tables, captions and
  appendix figures render without clipping or overlap.
