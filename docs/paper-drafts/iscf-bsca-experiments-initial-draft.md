# ISCF-BSCA Section 5: Experiments

## Draft status

| Field | Content |
| --- | --- |
| `document_role` | Clean manuscript-facing initial draft of Section 5 |
| `version` | `v0.3-latest-evidence-redraft` |
| `date` | `2026-08-19` |
| `review_status` | `initial_draft_pending_author_review` |
| `upstream_dependency` | Introduction v0.9, Related Work v0.2, Section 3 v0.7 and Section 4 v0.7 remain temporarily frozen and unchanged |
| `evidence_scope` | Main-I, Main-II, Efficiency, Core-Ablation, Figure 5 mechanism diagnostics and author-refined Decoder-Transfer |
| `experiment_change` | None; the prose is regenerated from the latest frozen Tables 1--5 and Figures 5--7, with no new training or formal test |
| `claim_boundary` | Main tables establish system-level effectiveness; matched ablations establish aggregate component utility; Figure 5 provides mixed internal-behavior evidence; transfer claims are restricted to the evaluated three-dataset scope |
| `narrative_spine` | evaluation contract → horizon-specific comparison → one-model capability → system-cost trade-off → matched attribution → internal behavior → bounded transferability |

The status table and artifact map below are editorial metadata and are not part of the manuscript body submitted for review.

## Canonical artifact map

| Manuscript item | Canonical artifact |
| --- | --- |
| Table 1 | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_tables_author_corrected_20260815/main_i/table_iscf_bsca_main_i_qdf.tex` |
| Table 2 | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_tables_author_corrected_20260815/main_ii/table_iscf_bsca_main_ii.tex` |
| Table 3 | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/efficiency_accuracy_memory_storage_20260817/table/table_iscf_bsca_efficiency.tex` |
| Table 4 | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/core_ablation_20260814/formal_results/table/table_iscf_bsca_core_ablation.tex` |
| Figure 5 | `paper-figures/figure_iscf_bsca_mechanism.*` |
| Figure 6 | `paper-figures/figure_6_accuracy_system_cost.*` |
| Table 5 | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/decoder_transfer_three_dataset_scope_20260816/table_decoder_transfer_three_dataset_framework.tex` |
| Figure 7 | `paper-figures/figure_7_decoder_transfer.*` |

## 5. Experiments

The preceding sections introduced ISCF-BSCA as a unified forecaster that constructs one prefix-consistent trajectory through adaptive output-side sharing. We now evaluate whether this design translates into forecasting accuracy, practical service consolidation and the internal behavior for which it was designed. System-level comparisons are kept separate from controlled ablations and validation diagnostics: the former establish effectiveness under the audited benchmark protocols, whereas the latter examine component utility, CHPC and scope-allocation behavior within the frozen ISCF-BSCA design family.

### 5.1 Experimental setup

**Datasets and metrics.** We evaluate ISCF-BSCA on seven public multivariate forecasting benchmarks: ETTm1, ETTm2, ETTh1, ETTh2, Weather, Electricity (ECL) and Solar. Together, they cover electricity transformers, meteorology, electricity consumption and solar generation at different sampling frequencies and variable dimensions. Following the standard long-term forecasting protocol, we use prediction horizons $\mathcal H=\{96,192,336,720\}$ and report mean squared error (MSE) and mean absolute error (MAE), where lower values indicate better forecasts. Dataset splits, normalization and source-specific preprocessing follow the audited protocol of each compared system; complete settings are provided in Appendix A.

**Evaluation protocols.** We use two complementary main comparisons. Main-I compares one ISCF-BSCA model per dataset with horizon-specific systems that train a separate model for each requested horizon. It therefore evaluates whether one unified predictor can replace a four-model service while remaining competitive with horizon-wise optimization. Main-II removes this deployment difference by requiring every system to serve all four horizons from one $H=720$ model; an $H$-step request is scored from the first $H$ outputs of that checkpoint using the system's official fixed-$H$ test loader. The external baselines span linear, patch-based, inverted-token and multi-scale forecasters. Because their results preserve audited source-native or published optimization protocols, Tables 1 and 2 support system-level comparisons rather than matched architectural attribution.

**Training and model selection.** ISCF-BSCA retains one checkpoint per dataset and uses seed 2021 for the primary result. Within each trial, checkpoint selection minimizes validation MSE averaged over the four horizons. Dataset-level hyperparameter profiles are subsequently selected by the preregistered mean official-test MSE over the same horizon set, without horizon-, seed-, metric- or cell-specific selection. The main benchmark is therefore explicitly test-informed and test-tuned rather than an untouched-holdout evaluation. All attempted profiles and negative cells are retained in the experiment record. External systems preserve their native optimization, checkpoint and seed protocols, with the corresponding source roles detailed in Appendix A.

**Controlled and diagnostic evaluations.** The ablation study uses the exact ISCF-BSCA-v1 anchor and end-to-end controls under a shared training protocol on ETTm1, ETTm2, ETTh1, ETTh2 and Weather. The transfer study compares the native Original Decoder with the complete ISCF-BSCA framework under the same backbone family and end-to-end training. Figure 5 uses validation artifacts only and is not treated as test-set effectiveness evidence. The system-cost analysis pairs Main-I accuracy with fresh-process RTX 3090 peak allocated memory and complete four-horizon checkpoint storage. Trained checkpoint files are used when the full inventory is available; the remaining architecture-equivalent resource rows are identified explicitly in Table 3 and Figure 6.

### 5.2 Comparison with horizon-specific forecasters

The first comparison asks whether one unified forecaster can match systems optimized separately for each requested horizon. ISCF-BSCA uses one model per dataset to produce all four forecasts, whereas each baseline entry in Table 1 follows its native horizon-specific protocol. The proposed model therefore competes directly with the flexibility of fitting four independent predictors.

<!-- Insert Table 1 near here. Canonical source: main_i/table_iscf_bsca_main_i_qdf.tex -->

Across the complete seven-dataset scorecard, ISCF-BSCA achieves the best result in 44 of the 56 dataset--horizon metric cells and the second-best result in another nine. Its macro MSE and MAE are 0.2607 and 0.3061. Among baselines with a complete audited result block, TimeAlign provides the strongest aggregate reference; ISCF-BSCA reduces its macro MSE and MAE by 4.94% and 2.54%, respectively. The corresponding reductions relative to QDF are 9.32% and 7.64%. The advantage is most uniform on ETTh2 and Weather, where the unified model ranks first for both metrics at every horizon.

The three remaining cells define the main exceptions: ISCF-BSCA falls outside the top two for MSE on ECL and Solar at $H=720$ and for MAE on ETTh1 at $H=336$. Table 1 also combines audited local reproductions with published-context values and does not align every optimization budget or seed. It therefore establishes a system-level result: one ISCF-BSCA model provides a more accurate alternative to four separately optimized forecasters on the large majority of evaluated cells. The table does not attribute this advantage to an individual decoder component.

### 5.3 One-model-all-horizons evaluation

Main-I compares different deployment structures. Main-II instead applies the one-model constraint to every method, thereby testing unified serving without the four-model advantage available to horizon-specific systems. For each dataset, one $H=720$ checkpoint is retained, and shorter requests are evaluated from its corresponding output prefixes. No model is retrained or reselected for $H\in\{96,192,336\}$.

<!-- Insert Table 2 near here. Canonical source: main_ii/table_iscf_bsca_main_ii.tex -->

Under the shared service constraint, ISCF-BSCA ranks first in 50 of 56 metric cells and second in the remaining six. It achieves the best MSE and MAE at every horizon on ETTm1, ETTh1, ETTh2 and Weather. The second-place results are limited to MSE on ETTm2 at $H\in\{192,336\}$, ECL at $H=720$ and Solar at $H\in\{336,720\}$, together with MAE on ECL at $H=720$. Averaged over all seven datasets and four horizons, ISCF-BSCA reduces MSE and MAE by 6.45% and 3.72% relative to TimeAlign, the next strongest complete system in Table 2.

Together, these results show that one ISCF-BSCA checkpoint can serve all evaluated endpoints while improving aggregate accuracy over the compared one-model systems. The comparison remains source-native rather than fully matched, and several baselines also obtain shorter outputs by slicing an $H=720$ forecast. Main-II therefore establishes one-model-all-horizons effectiveness; the separate contribution of ISCF-BSCA is its explicit future-step-indexed construction and adaptive output-side sharing, which are evaluated through the controlled analyses below.

### 5.4 Accuracy and system cost

The value of one-model forecasting also depends on the cost of serving all requested horizons. Table 3 combines Main-I macro accuracy with the peak GPU memory and checkpoint storage of a complete four-horizon service. ISCF-BSCA contributes one unified checkpoint per dataset, whereas each baseline service retains four fixed-horizon model instances.

<!-- Insert Table 3 near here. Canonical source: efficiency_accuracy_memory_storage_20260817/table/table_iscf_bsca_efficiency.tex -->

<a id="fig:accuracy-system-cost"></a>

![Accuracy, checkpoint storage and peak-memory trade-off.](../../paper-figures/figure_6_accuracy_system_cost.png)

**Figure 6 | Accuracy--storage trade-off for four-horizon forecasting services.** Vertical position shows macro MSE over seven datasets and four horizons; horizontal position shows the complete four-horizon checkpoint storage on a logarithmic scale; bubble area is proportional to peak allocated inference memory. ISCF-BSCA stores one unified checkpoint, whereas each baseline represents four horizon-specific models. Resource footprints for DLinear, iTransformer, PatchTST and TimeMixer use official-configuration architecture-equivalent state dicts because complete trained checkpoint inventories are unavailable. SimpleTM is omitted from this visualization under the author-specified display scope but remains reported in the complete Table 3. Lower-left positions indicate a more favorable accuracy--storage trade-off; the figure does not establish a uniform resource advantage.

ISCF-BSCA attains the lowest macro MSE and MAE among the nine systems, at 0.261 and 0.306, while requiring 17.677 MiB of checkpoint storage and 38.817 MiB of peak allocated memory. Figure 6 places this result in the lower part of the accuracy--storage plane. Compared with TimeAlign, AMD, iTransformer, PatchTST and TimeMixer, the unified service reduces checkpoint storage by 30.12--92.01% and peak memory by 18.01--83.69%. The result is particularly pronounced relative to TimeAlign: ISCF-BSCA lowers MSE by 4.94%, checkpoint storage by 81.48% and peak memory by 64.42%.

The comparison also retains the lightweight counterexamples. DLinear has the lowest peak memory at 13.111 MiB, SimpleTM has the lowest storage at 2.835 MiB, and QDF uses less peak memory than ISCF-BSCA. Peak memory is measured in a fresh process on an exclusive RTX 3090 in FP32 with batch size 1, standardized synthetic inputs and all service models resident. DLinear, iTransformer, PatchTST and TimeMixer use official-configuration architecture-equivalent state dicts because complete trained checkpoint inventories are unavailable; their resource values therefore characterize model-equivalent service footprints rather than completed training artifacts. Table 3 and Figure 6 support one-checkpoint consolidation and a favorable accuracy--resource trade-off against several larger services, but not uniform resource superiority over every lightweight forecaster.

### 5.5 Component and training-objective ablations

The main tables establish system-level effectiveness but do not identify which parts of ISCF-BSCA contribute within the proposed design. Table 4 therefore compares the complete model with four end-to-end controls under a shared training protocol. **w/o BSCA** retains the Uniform-Prefix Forecasting Loss but removes the Scope-Wise Forecasting Loss and Allocation-Balance Regularizer. **w/o Target-Adaptive Allocation** replaces learned Scope Probabilities with equal non-adaptive fusion. **Shared Scope Projection** replaces the scope-specific history projections with one shared projection, whereas **Fixed Scope ($s=144$)** retains only the preregistered middle scope. The fixed value is a budget-aware control rather than a validation-selected optimum.

<!-- Insert Table 4 near here. Canonical source: core_ablation/table/table_iscf_bsca_core_ablation.tex -->

Full ISCF-BSCA achieves a five-dataset macro MSE/MAE of 0.305/0.344 and ranks first in all 12 dataset and average metric columns. Removing BSCA produces the largest aggregate degradation, increasing MSE and MAE by 3.48% and 2.83%. Equal fusion yields 0.310/0.349, corresponding to gains of 1.61% in MSE and 1.43% in MAE from Target-Adaptive Allocation. Shared Scope Projection and Fixed Scope yield MSE values of 0.314 and 0.315, respectively, compared with 0.305 for the complete model. Full ISCF-BSCA improves both metrics over every control on all five datasets.

These results support the aggregate utility of BSCA's Scope-Wise Forecasting Loss and Allocation-Balance Regularizer beyond the retained Uniform-Prefix Forecasting Loss, together with target-adaptive rather than equal allocation, scope-specific history projections and multi-scope generation. The evidence does not imply that the learned probabilities identify an oracle scope, and Table 4 reports four-horizon dataset means rather than a newly synchronized per-horizon checkpoint record. We therefore restrict the attribution to aggregate accuracy within the evaluated design family; the internal behavior of the learned allocation is examined separately in Figure 5.

### 5.6 Forecast consistency and scope-allocation behavior

The preceding results establish forecast accuracy, but they do not show whether the trained system exhibits the consistency and sharing behavior motivated in Section 3. Figure 5 addresses this distinction using validation artifacts. It verifies CHPC numerically, measures how the errors of Scope-conditioned Forecasts vary across future regions, and compares this structure with the learned Scope Probabilities. These diagnostics characterize the implemented mechanism and are not used as test-set effectiveness evidence.

<a id="fig:mechanism-analysis"></a>

![Forecast consistency and scope-allocation behavior.](../../paper-figures/figure_iscf_bsca_mechanism.png)

**Figure 5 | Forecast consistency, scope-allocation behavior and an illustrative trajectory.** **a**, Maximum absolute CHPD for the four supported horizons on ETTm1, ETTm2, ETTh1, ETTh2 and Weather; all 20 dataset-horizon comparisons equal zero and remain below the numerical tolerance. **b**, Scope Probabilities across 720 future steps for the performance-selected Weather validation row. **c**, Dataset-macro Scope Probability in eight future regions. **d**, Excess scope-wise regional MSE above the best scope in each region, averaged equally over the five datasets. **e**, Ground truth and forecasts from Full ISCF-BSCA and Fixed Scope ($s=144$) on Weather validation row 77. The example was selected from all 1,280 aligned validation rows by the largest $H=720$ MSE reduction of Full over Fixed Scope and is illustrative rather than representative; dashed markers indicate the shorter prefix endpoints.

Figure 5a verifies the system contract directly. Across five datasets and four horizons, maximum absolute CHPD equals zero in all 20 comparisons and remains below the predefined tolerance of $2\times10^{-5}$. Changing the requested endpoint therefore leaves every shared prediction unchanged in the implemented inference graph. This result establishes numerical CHPC, independently of forecast accuracy.

The scope diagnostics reveal a more nuanced pattern. All five scopes remain active, but their dataset--region mean probabilities occupy the narrow range 0.1826--0.2148 (Figure 5b,c), indicating soft and broadly distributed allocation rather than sparse specialization. The Scope-conditioned Forecasts nevertheless exhibit distinct regional errors. The lowest-error scope changes across the eight future regions, and the largest macro difference between the best and worst scopes reaches 6.123% excess MSE (Figure 5d). The trained scope field therefore preserves region-dependent predictive differences, consistent with the sharing-demand heterogeneity observed in Section 3.

The allocation weights do not, however, behave as an oracle selector. The highest-probability scope coincides with the lowest-MSE scope in only 8 of 40 dataset--region cells. This internal limitation is compatible with the positive aggregate ablation in Table 4: equal fusion tests whether target-dependent weighting improves the final forecast, whereas the 8/40 statistic asks whether the largest weight recovers the region-wise error winner. Taken together, the results show an aggregate benefit from learned allocation while indicating that its weights remain distributed and should not be interpreted as reliable region-best routing or semantic specialization.

Figure 5e illustrates the resulting forecast on the Weather validation set. For the selected row, Full ISCF-BSCA reduces $H=720$ MSE and MAE by 78.48% and 54.39% relative to Fixed Scope ($s=144$), while the shorter forecasts remain nested prefixes of the same trajectory. The row was selected from all 1,280 aligned validation examples by the largest $H=720$ MSE improvement over the fixed-scope control. It therefore demonstrates a possible qualitative benefit rather than a representative gain, prevalence estimate or independent causal effect of allocation.

### 5.7 Backbone transferability

The final experiment examines whether the complete framework can operate with different history encoders. We replace the Original Decoder of a DLinear-style decomposition backbone and a PatchTST-style contextual patch backbone with ISCF-BSCA, and train each resulting system end to end. The comparison covers Weather, ETTm1 and ETTm2; every dataset value averages MSE or MAE over the four horizons.

<!-- Insert Table 5 near here. Canonical source: decoder_transfer_three_dataset_scope_20260816/table_decoder_transfer_three_dataset_framework.tex -->

<a id="fig:decoder-transfer"></a>

![Decoder transfer across two forecasting backbones.](../../paper-figures/figure_7_decoder_transfer.png)

**Figure 7 | Transfer of the complete ISCF-BSCA framework across two forecasting backbones.** **a**, Four-horizon mean MSE for the DLinear-style backbone. **b**, Corresponding results for the PatchTST-style backbone. Each pair compares ISCF-BSCA with the Original Decoder; arrows and signed percentages report the relative MSE change, with negative values indicating lower error. Avg. is the arithmetic mean over Weather, ETTm1 and ETTm2. The shared MSE axis begins at 0.20, as indicated by the axis break, to resolve paired differences. The corrected aggregate artifacts provide point estimates rather than repeated-run uncertainty, so no error bars are shown.

For the DLinear-style backbone, ISCF-BSCA lowers MSE and MAE on all three datasets and reduces the macro averages from 0.303/0.333 to 0.286/0.321, corresponding to improvements of 5.61% and 3.60%. The largest MSE reduction is 10.3% on ETTm2. The PatchTST-style backbone shows smaller but directionally consistent gains: macro MSE/MAE decreases from 0.282/0.314 to 0.276/0.310, with improvements of 2.13% and 1.27%. Across both backbones, ISCF-BSCA is better in all 16 displayed dataset and average metric comparisons.

This transfer result has two important boundaries. First, the three-dataset reporting scope was refined after a broader audit; ETTh1, ETTh2 and iTransformer-style evaluations did not yield consistent improvements and are retained in Appendix C. Second, the author-corrected aggregate table is not accompanied by a newly synchronized per-horizon checkpoint manifest. Table 5 and Figure 7 therefore support portability of the complete framework across the two evaluated backbone families and three reported datasets, but not universal architecture-agnostic transfer or separate attribution to ISCF and BSCA.

## Editorial evidence and claim audit

| Subsection | Evidence status | Permitted claim | Prohibited promotion |
| --- | --- | --- | --- |
| 5.2 Main-I | Complete seven-dataset, four-horizon table with mixed source roles | One unified ISCF-BSCA model is more accurate on most cells than separately optimized horizon-specific systems | Matched decoder attribution or untouched-holdout generalization |
| 5.3 Main-II | Complete source-native one-model table | One ISCF-BSCA checkpoint serves all evaluated horizons competitively | Fully matched architecture comparison across all systems |
| 5.4 Efficiency | 252/252 Main-I accuracy cells, 63/63 memory/storage service units and 231 table-role objects | Best nine-system macro accuracy and one-checkpoint consolidation; lower memory/storage than TimeAlign, AMD, iTransformer, PatchTST and TimeMixer | Uniform memory/storage advantage, treating architecture-equivalent rows as trained artifacts, or a training-time claim |
| 5.5 Core ablation | Author-corrected five-dataset aggregate table | All four interventions improve aggregate MSE/MAE under the matched design | Per-horizon dominance, oracle routing or independently verified component semantics |
| 5.6 Mechanism analysis | Complete validation diagnostic bundle | Exact CHPC, active scopes and regional scope-error heterogeneity; selected example is illustrative | Reliable region-best routing, sparse specialization, prevalence or causal interpretation of the example |
| 5.7 Decoder transfer | Author-refined two-backbone, three-dataset aggregate table | Complete-framework portability within the displayed scope | Universal or architecture-agnostic transfer; separate ISCF/BSCA attribution |
