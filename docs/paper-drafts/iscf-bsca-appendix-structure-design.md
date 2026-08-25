# ISCF-BSCA Appendix Structure Design

**Version:** v0.2-three-appendix-author-plan
**Date:** 2026-08-25
**Scope:** minimal appendix routing for the frozen Sections 1--7 manuscript

## 1. Design principle

The appendices should carry only material needed to reproduce the reported protocol, inspect the complete benchmark coverage, or understand the qualitative behaviour of the unified forecast. They should not repeat the method, duplicate main-text figures, or introduce new experiments. Appendix A is therefore reduced to three compact protocol tables; Appendix B retains the complete horizon-wise result cells behind Tables 1 and 2; Appendix C adds one validation-only qualitative figure for the seven datasets.

The attached `PDT_final.pdf` is treated as a reference source for dataset descriptions and figure conventions, not as an instruction document. Its dataset table and four-horizon visualization will be cross-checked against the frozen FATST protocol before any values are transferred.

## 2. Recommended appendix map

### Appendix A. Datasets, training protocol and ISCF configuration

Appendix A should contain tables only, with a short paragraph defining the notation and the split convention.

**Table A1 | Dataset metadata and splits.** One row per paper-core dataset: ETTm1, ETTm2, ETTh1, ETTh2, Weather, ECL and Solar. Recommended columns are `Dataset`, `Variables`, `Sampling frequency`, `Raw observations`, `Split rule`, and `Train/validation/test boundaries or window counts`. Raw series length and window counts must be labelled separately; they are not interchangeable.

**Table A2 | Shared training and evaluation protocol.** One row per dataset, or a compact grouped table when values are identical. Recommended columns are `Look-back $L$`, `Target horizon $T$`, `Evaluated horizons`, `Optimizer`, `Initial learning rate`, `Learning-rate schedule`, `Batch size`, `Gradient accumulation`, `Maximum epochs`, `Early-stopping patience`, `Checkpoint rule` and `Seed`. The table should make clear that checkpoints are selected on the four-horizon validation mean MSE, while the frozen paper profiles are test-informed at the dataset level; this disclosure belongs in the table note rather than the main narrative.

**Table A3 | Dataset-specific ISCF-BSCA configuration.** One row per dataset. Recommended columns are `Scope set`, `Mode rank`, `Scope-wise loss weight`, `Uniform-prefix loss weight/ramp`, `Allocation-balance final weight/ramp`, and the representation settings needed to reproduce the decoder interface (`patch number`, `Encoder width`, `feed-forward width`, `dropout`, `weight decay`). The scopes and loss coefficients are global architectural settings and may be shown as repeated values or marked “shared across datasets”; dataset-specific values must be listed explicitly. The table should not include HPO trial histories, source-role prose or checkpoint hashes.

No separate Appendix A subsection is needed for baseline taxonomy, software provenance or historical HPO. The main text already identifies the baseline families and implementation environment; machine-readable manifests and audit reports remain repository artifacts.

### Appendix B. Complete horizon-wise benchmark results

Appendix B preserves the full-precision cells omitted from the compact dataset-average tables in Sections 5.2 and 5.3:

- **Table B1:** complete Main-I results for the unified ISCF-BSCA model and horizon-specific baselines, for every paper-core dataset, horizon and metric;
- **Table B2:** complete Main-II results when every method serves all horizons with one unified model under the H720-prefix protocol.

Negative cells remain visible. The appendix tables are the complete audit surface behind Tables 1 and 2; they do not change the main-text aggregation or source-role interpretation.

### Appendix C. Qualitative varied-horizon forecasts

**Figure C1** should contain a compact seven-row by two-column grid: two samples for each paper-core dataset. Each panel plots only the ground-truth future and the unified ISCF-BSCA forecast over $T=720$ steps. Vertical markers at $H=96$, $192$, $336$ and $720$ show the nested prefixes used by the four-horizon evaluation. The colour, line weight and axis treatment should follow the Nature-figure contract and the visual language already established for Figures 5--7.

The samples should be selected by a frozen, deterministic validation-only rule rather than manual visual choice: rank eligible forecast origins by the four-horizon normalized MSE of the fused forecast, enforce a minimum separation between selected origins, and retain the two lowest-error origins per dataset. The plotted channel should be fixed before selection (default: channel 0, with the channel identifier recorded in the source-data manifest). This makes “good samples” reproducible while keeping the figure explicitly illustrative rather than a prevalence or test-effectiveness estimate.

The source-data manifest for Figure C1 should record `dataset`, `split`, `origin`, `channel`, `horizon`, `target`, `prediction`, `selection score`, `selection rule`, and `checkpoint/profile identifier`. No baseline curves, scope-wise curves or allocation probabilities should be added to C1; those behaviours are already covered by Figures 5 and 7 and would make the appendix figure redundant.

## 3. Section-by-section routing audit

| Section | Appendix requirement | Routing decision |
| --- | --- | --- |
| Sections 1--4 | Definitions, method equations and Figures 1--4 are self-contained. | No appendix item. |
| 5.1 Experimental setup | Dataset metadata, split boundaries, training settings and dataset-specific ISCF-BSCA configuration. | Appendix A, Tables A1--A3. |
| 5.2 Horizon-specific comparison | Complete dataset--horizon cells behind compact Table 1. | Appendix B, Table B1. |
| 5.3 One-model-all-horizons evaluation | Complete dataset--horizon cells behind compact Table 2. | Appendix B, Table B2. |
| 5.4 Accuracy and system cost | Figure 6 is the sole main-text carrier; its numerical source remains an audit artifact. | No additional appendix table. |
| 5.5 Component and training-objective ablations | Table 3 is the required aggregate attribution surface. | No duplicate appendix table. |
| 5.6 Scope diversity and allocation behavior | Figure 5 is the selected, sample-specific internal diagnostic. | No duplicate figure or prevalence table. |
| 5.7 Generalization studies | Figure 7 is the sole main-text carrier. | No duplicate transfer table. |
| Sections 6--7 | Discussion and Conclusion are self-contained. | No appendix item. |

## 4. Material deliberately excluded

The minimal plan excludes repeated copies of Figures 2, 3, 5, 6 and 7; historical HPO trials; source-role and checkpoint-hash inventories; unpromoted sensitivity analyses; realized-allocation analyses; failure-case panels; and additional qualitative samples beyond the two deterministic validation examples per dataset. These remain in `analysis/` or machine-readable artifact manifests unless a later author decision changes the paper contract.

## 5. Staged execution plan

1. **Protocol audit:** completed for the seven paper-core datasets against the local loaders and frozen configs. Raw lengths, split boundaries, window counts and finite-loader checks are recorded in `docs/paper-drafts/iscf-bsca-appendix-a-c-data-audit.md`.
2. **Configuration extraction:** completed for the selected dataset-level profiles from the frozen manifest, phase configs and trial ledgers. The manuscript-ready tables are drafted in `docs/paper-drafts/iscf-bsca-appendix-a-initial-draft.md`; values were not inferred from trial names.
3. **Qualitative data export:** completed on 2026-08-25 using the frozen Main-I/II selected profiles. The validation-only export, checkpoint hashes, selected origins and raw prediction arrays are recorded in `analysis/iscf_bsca_appendix_c_prediction_export_20260825/`; no ablation checkpoint or test label was used.
4. **Figure C1 generation and QA:** next step is to render the seven-row by two-column figure from these fixed arrays, export editable SVG/PDF plus raster fallback, and run source-data, typography, alignment and `git diff --check` checks.
5. **Manuscript synchronization:** after Figure C1 is approved, update Section 5.1 references, captions and the appendix routing in the architecture, mainline, roadmap and stage ledger documents.

No new architecture, training, remote launch or formal test is implied by this plan. The final-profile prediction arrays are now available; Figure C1 remains pending only figure rendering and author review.
