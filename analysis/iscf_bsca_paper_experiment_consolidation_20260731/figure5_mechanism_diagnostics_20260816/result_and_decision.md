# Figure 5 mechanism diagnostics：result and decision

## 1. 完整性

- 5 datasets × 2 frozen systems=`10/10` checkpoint-linked validation artifact objects；
- Full 与 Fixed Scope 的 targets 在全部 `5 × 256=1,280` qualitative pool rows 上 exact aligned；
- aggregate statistics 使用各数据集全部 validation series rows，不从 datasets、regions、scopes 或 metrics 中选择有利子集；
- new training=`0`，formal test=`0`，checkpoint mutation=`0`；
- 所有 arrays finite，Scope Probability sum-to-one最大绝对误差=`2.3842e-7`。

## 2. CHPC

五数据集、四个 paper-facing horizons 共 `20/20` comparisons 的 maximum absolute CHPD 均为 `0.0`，通过预冻结 `2e-5` tolerance。该结果验证当前 inference implementation 的 numerical prefix consistency，但它不是 accuracy evidence。

## 3. Scope usage 与regional error

Learned Scope Probability 在 dataset-region aggregate 上接近均匀：40个 dataset-region × scope means 的范围为`0.18258--0.21479`。这说明各 scopes 均保持 active，但没有形成强烈、稀疏的 allocation specialization。

Scope-wise regional MSE 则呈现可复现的区域差异。在五数据集等权 macro 后：

- regions `1--48`与`49--96`由`s=360`最低；
- regions `97--144`、`145--192`、`193--288`与`289--336`由`s=720`最低；
- region `337--512`由`s=360`最低；
- region `513--720`由`s=1`以很小margin最低；
- region-best与最弱scope之间的macro excess MSE最高为`6.123%`。

然而，highest-utilization scope只在`8/40` dataset-region cells与lowest-MSE scope一致。该结果支持“scope arms存在region-dependent error heterogeneity”，但不支持“learned allocation已经可靠识别region-best scope”。它与Core-Ablation中Full未优于equal fusion的结果一致。

## 4. Qualitative trajectory

按照预冻结规则，在完整1,280-row validation pool中选择Full相对`Fixed Scope (s=144)` H720 MSE reduction最大的row：

- dataset=`Weather`；
- sequential probe row=`77`；
- Full MSE/MAE=`0.00002417 / 0.004029`；
- Fixed Scope MSE/MAE=`0.00011228 / 0.008833`；
- relative reduction=`78.477% MSE / 54.385% MAE`。

该row是刻意的performance-selected illustration。它只说明完整framework在一个披露选择规则的validation example上能够明显优于fixed-scope control，不代表典型提升、出现频率或learned allocation的独立因果贡献。

## 5. Four-layer interpretation

1. `paper_facing_effectiveness`：不在本轮重复；沿用Main I/II与Core-Ablation结果。
2. `matched_mechanism_attribution`：Core-Ablation已支持BSCA objective、scope-specific projection与multi-scope design，但不支持learned allocation独立accuracy utility。
3. `internal_mechanism_health`：CHPC pass；五个scopes均active；regional arm errors存在heterogeneity；allocation-to-best-scope agreement较低。
4. `failure_attribution`：allocation alignment不足属于exact learned allocation path的`readout_or_head_design_wrong`/optimization limitation，不否定multi-scope framework，也不能由selected trajectory补救。

## 6. Decision

Decision=`figure5_complete_mixed_mechanism_evidence`。

Section 5.6可写为：

- exact numerical CHPC成立；
- scope forecasts具有region-dependent error differences；
- learned probabilities保持active但总体接近均匀，且与region-best scope的一致率有限；
- performance-selected trajectory仅作illustrative evidence。

禁止写作：learned allocation improves accuracy over equal fusion、causal specialization、universal specialization或representative example。当前不自动重启allocation HPO或architecture search；下一步进入Sections 5--7写作与全文claim synchronization。

## 7. Artifacts

- Figure：`results/figure_5_iscf_bsca_mechanism.{svg,pdf,png,tiff}`；
- canonical paper figure：`paper-figures/figure_iscf_bsca_mechanism.*`；
- source data：`results/chpc_verification.csv`、`scope_utilization.csv`、`scope_regional_error.csv`、`scope_preference_summary.csv`、`qualitative_source_data.csv`；
- complete selection audit：`results/qualitative_selection_pool.csv`；
- provenance：`results/artifact_manifest.csv`、`figure5_summary.json`。
