# ISCF-BSCA Appendix Structure Design

**Version:** v0.6-table-hierarchy-refinement
**Date:** 2026-08-25
**Scope:** minimal appendix routing for the frozen Sections 1--7 manuscript

## 1. Design principle

The appendices should carry only material needed to interpret the evaluation metrics, reproduce the reported protocol, inspect the complete benchmark coverage, or understand the qualitative behaviour of the unified forecast. They should not repeat the method, duplicate main-text figures, or introduce new experiments. Appendix A follows the compact sequence `Metric Details -> Datasets -> Implementation Details`, supported by three protocol tables; Appendix B retains the complete horizon-wise result cells behind Tables 1 and 2; Appendix C is restored as `Visualization` and contains one validation-only qualitative figure for the seven datasets.

The attached TimeAlign paper is treated as a reference for appendix organization and presentation, not as an instruction document or a source of FATST experimental values. Appendix A follows the structural role of its Appendix D, Appendix B follows the concise full-results entry used in Appendix E.1, and Appendix C follows the compact visualization presentation used in Appendix F. All reported values remain governed by the frozen FATST protocol and local audit artifacts.

## 2. Recommended appendix map

### Appendix A. Experiment details

Appendix A is organized into three concise subsections. `A.1 Metric Details` defines MSE and MAE; `A.2 Datasets` introduces the four dataset families and reports their statistics and split construction; `A.3 Implementation Details` records the software/hardware environment, optimization settings and dataset-specific ISCF-BSCA configuration.

**Table A1 | Dataset statistics.** One row is retained for each of the seven paper-core datasets. The compact columns are `Dataset`, `Variables`, `Sampling frequency`, `Dataset size`, and `Domain`. `Dataset size` is reported as a `(Train, Validation, Test)` tuple following the TimeAlign Appendix D convention; exact boundary indices and loader-window counts remain in the audit artifact rather than the manuscript table.

**Table A2 | Training and evaluation settings.** One row is retained per dataset. The compact columns are `Max Length`, `Initial learning rate`, `Batch size`, `Gradient Accumulation Steps`, `Maximum epochs`, and `Early-stopping patience`. The accompanying prose defines gradient accumulation as the number of mini-batches combined before one optimizer update and states that effective batch size equals batch size multiplied by accumulation steps. Optimizer, learning-rate schedule, seed and validation checkpoint selection are stated once in the surrounding prose.

**Table A3 | Dataset-specific ISCF-BSCA configuration.** One row per dataset. Recommended columns are `Scope set`, `Mode rank`, `Scope-wise loss weight`, `Uniform-prefix loss weight/ramp`, `Allocation-balance final weight/ramp`, and the representation settings needed to reproduce the decoder interface (`patch number`, `Encoder width`, `feed-forward width`, `dropout`, `weight decay`). The scopes and loss coefficients are global architectural settings and may be shown as repeated values or marked “shared across datasets”; dataset-specific values must be listed explicitly. The table should not include HPO trial histories, source-role prose or checkpoint hashes.

No separate Appendix A subsection is needed for baseline taxonomy or historical HPO. The main text already identifies the baseline families, while machine-readable manifests and audit reports remain repository artifacts.

### Appendix B. Full results

Because Appendix B contains only the two main-result tables, it uses one `B. FULL RESULTS` section without a nested B.1 subsection. A concise opening paragraph introduces the full-precision cells omitted from the compact dataset-average tables in Sections 5.2 and 5.3:

- **Table B1:** complete Main-I results for the unified ISCF-BSCA model and horizon-specific baselines, for every paper-core dataset, horizon and metric;
- **Table B2:** complete Main-II results when every method serves all horizons with one unified model under the H720-prefix protocol.

Negative cells remain visible. The appendix tables are the complete audit surface behind Tables 1 and 2; they do not change the main-text aggregation or source-role interpretation.

### Appendix C. Visualization

The assembled manuscript restores a visible `C. VISUALIZATION` heading. Figure C1 itself contains only the title `Representative validation trajectories`; the appendix identifier belongs to the manuscript hierarchy and is not repeated inside the image.

**Figure C1** should contain a compact seven-row by two-column grid: two samples for each paper-core dataset. Each panel plots only the ground-truth future and the unified ISCF-BSCA forecast over $T=720$ steps. A shared nested-prefix ruler above the grid, complemented by faint vertical endpoint guides, shows the four-horizon evaluation prefixes at $H=96$, $192$, $336$ and $720$. The colour, line weight and axis treatment should follow the Nature-figure contract and the visual language already established for Figures 5--7.

The samples should be selected by a frozen, deterministic validation-only rule rather than manual visual choice. First select one fixed channel per dataset using the global visual-fidelity score, excluding the lowest-variance 20% of channels. Then rank eligible forecast origins by the four-horizon visual-fidelity score, enforce a minimum separation between selected origins, and retain the two lowest-scoring origins per dataset. The score combines train-scale level error, trajectory correlation, first-difference correlation and amplitude agreement. This makes “good samples” reproducible while keeping the figure explicitly illustrative rather than a prevalence or test-effectiveness estimate.

The source-data manifest for Figure C1 should record `dataset`, `split`, `origin`, `channel`, `horizon`, `target`, `prediction`, `selection score`, `selection rule`, and `checkpoint/profile identifier`. No baseline curves, scope-wise curves or allocation probabilities should be added to C1; those behaviours are already covered by Figures 5 and 7 and would make the appendix figure redundant.

## 3. Section-by-section routing audit

| Section | Appendix requirement | Routing decision |
| --- | --- | --- |
| Sections 1--4 | Definitions, method equations and Figures 1--4 are self-contained. | No appendix item. |
| 5.1 Experimental setup | Metric definitions, dataset descriptions and compact statistics, split protocol, training settings and dataset-specific ISCF-BSCA configuration. | Appendix A.1--A.3 and Tables A1--A3. |
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
4. **Figure C1 generation and QA:** completed on 2026-08-25 and revised after a Weather/ECL candidate audit. The seven-row by two-column figure, source-data CSV, editable SVG/PDF and 600-dpi PNG/TIFF exports passed the strict static preflight and visual QA; the figure now uses a shared nested-prefix ruler and faint endpoint guides.
5. **Manuscript synchronization:** the complete Appendix draft is assembled in `docs/paper-drafts/iscf-bsca-appendix-initial-draft.md`. Appendix A uses the author-refined compact dataset and training tables, Appendix B points to the frozen canonical LaTeX tables, and Appendix C restores the `Visualization` heading while Figure C1 omits the redundant appendix identifier; no new model training or formal test is implied.

No new architecture, training, remote launch or formal test is implied by this plan. The final-profile prediction arrays, candidate audit and accepted Figure C1 exports are available for manuscript assembly.
