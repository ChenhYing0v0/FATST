# ISCF-BSCA-MAIN-v1 Post-HPO Main-I Published Audit and Next Gate

## Decision summary

| Field | Value |
| --- | --- |
| `date` | 2026-08-02 |
| `current_step` | HPO Step 9--10 complete；Main I published block complete；next remote blocks gated |
| `main_candidate` | `ISCF-BSCA-MAIN-v1`；8 reusable selected checkpoints；32 official-test cells |
| `published_block` | 5 models × 7 datasets × 4 horizons = 140 MSE/MAE rows |
| `competitiveness` | aggregate MSE competitive；per-cell SOTA and MAE superiority not established |
| `remote_launch` | none；baseline and Main II authorizations remain separate |
| `decision` | freeze HPO；retain complete published rows；request staged baseline authorization |

## 1. Frozen main candidate

[Fact] `ISCF-BSCA-MAIN-v1` H1/H2/H3A/H3B共有53个完整test-tuned trials。最终8个dataset-level profiles及32个standard-horizon MSE/MAE cells已经冻结，所有profiles各自共享四个horizons，不存在per-horizon、per-seed、per-metric或per-cell选择。

- selected profile config：`configs/iscf_bsca_main_v1_selected_profiles.json`；
- canonical scorecard：`analysis/iscf_bsca_main_v1_hpo_20260731/selected_main_scorecard.csv`，SHA256 `a419f840b45d316c9c7a72979211a52e4d50583a915f9871b0c3b50f5c10f427`；
- selected checkpoint provenance：`analysis/iscf_bsca_main_v1_hpo_20260731/selected_profile_manifest.csv`，SHA256 `badddc5f563f49da94e256c0eb11895c8071e7c5642409f54eecaa38da4f650c`。

ECL selected mean MSE/MAE为`0.150669/0.245170`；Solar为`0.191855/0.220518`。二者均达到原预冻结reported-Avg MSE targets 0.154和0.192，因此same-neighborhood HPO停止。

## 2. TimeAlign Table 6 published block

### 2.1 Source and extraction

Primary source为TimeAlign ICLR 2026论文的arXiv PDF；PDF首页明确标注ICLR 2026 conference paper。PDF SHA256为`0a0d7deba16c5b3025902f897dddc70fd1f6efaeb3737c4a751d3b2440670b08`。Table 6位于PDF page 22。

`scripts/extract_timealign_table6_main_i.py`按页面坐标抽取TimeAlign、TimeMixer、DLinear、iTransformer、PatchTST在ETTh1、ETTh2、ETTm1、ETTm2、Weather、Electricity/ECL、Solar上的逐horizon MSE/MAE。输出`timealign_table6_main_i_published.csv`为140/140 rows，SHA256 `c9286468a4ca5977bac40f18635d387eafda0fd14794cabaac9b6f659ae82d8c`。

Verification采用两条路径：逐cell coordinate extraction后与page 22高分辨率渲染目视核对；再将逐horizon均值与Table 1/6的reported Avg行交叉检查。Traffic不属于当前8-dataset contract，完整读取但不写入该140-row artifact。Exchange在源表缺失。

### 2.2 Source-internal anomalies

[Fact] 论文内部存在不能由正常三位小数rounding完全解释的逐horizon均值与reported Avg差异：

| Model / dataset | Mean recomputed from four displayed cells | Reported Avg |
| --- | --- | --- |
| PatchTST / ETTm1 | 0.352750 / 0.376250 | 0.353 / 0.382 |
| PatchTST / ETTh2 | 0.351250 / 0.394500 | 0.351 / 0.404 |
| TimeAlign / Weather | 0.215250 / 0.244500 | 0.214 / 0.244 |
| TimeMixer / ECL | 0.172500 / 0.272250 | 0.185 / 0.284 |
| TimeMixer / Solar | 0.193250 / 0.252000 | 0.193 / 0.264 |

Main I逐horizon表保留PDF中显示的原值，不用reported Avg覆盖，也不猜测作者未公开的“正确值”。这些cells可以作为`published_context`，但必须披露source anomaly，不能用于matched attribution。

Lookback描述也有三种版本：Table 1 caption为`{336,512,720}`，main-text Implementation Details为`{96,192,336,512,720}`，Appendix E.1为`{192,336,512,720}`。因此published rows只能标记为source-native hyperparameter-searched results，不能写成与本地ISCF profile完全matched。

## 3. Current competitiveness judgment

`scripts/analyze_iscf_bsca_main_i_published.py`把ISCF frozen 32-cell scorecard与140-row published block对齐。以下比较只覆盖共同的7 datasets；ISCF是single-seed test-tuned结果，published rows按论文说明为three-run mean，且两者search protocol不同。

Comparison artifacts为`iscf_vs_published_pairwise.csv`（SHA256 `d6874b70abedb6729efcde0b9ce961bc9f51b7e550a4b5756f4cdae349a42234`）、`iscf_vs_best_published_per_cell.csv`（`c34999c85b6b126a8053b52d2cf5bc35d96db7acf09a3c68e55c6c201900eba7`）与`iscf_vs_published_summary.csv`（`f34510103392a721c87c3586f0d65a90157202af4023669bad7927d82d579318`）。

| Reference | MSE gain of ISCF | MSE cells | Dataset means | Horizon means | MAE gain of ISCF |
| --- | ---: | ---: | ---: | ---: | ---: |
| TimeAlign | +2.199% | 15/28 | 4/7 | 3/4 | -0.066% |
| best published model per cell | +2.017% | 14/28 | 4/7 | 3/4 | -0.135% |
| TimeMixer | +6.915% | 26/28 | 7/7 | 4/4 | +6.446% |
| PatchTST | +5.785% | 26/28 | 7/7 | 4/4 | +4.849% |
| iTransformer | +9.751% | 28/28 | 7/7 | 4/4 | +7.569% |
| DLinear | +12.264% | 27/28 | 7/7 | 4/4 | +8.271% |

[Strong Evidence] 当前ISCF-BSCA在aggregate MSE上具有竞争力：对TimeAlign以及逐cell最强published MSE comparator均约有2% aggregate优势，并明显优于另外四个selected published baselines。

[Boundary] 这还不是完整SOTA结论。相对TimeAlign，ETTm2、Weather与Solar的dataset-mean MSE仍弱；Solar按Table 6逐horizon显示值重算为0.191750，ISCF 0.191855略高0.055%，尽管它低于论文reported Avg 0.192。MAE对TimeAlign和逐cell best的aggregate也略弱。必须保留这些negative cells，并等待AMD、SimpleTM、TimePerceiver、SRSNet及Exchange official reproductions。

## 4. Next minimal sufficient experiment blocks

### 4.1 Main I

Published block现为`reusable_with_disclosure`。缺口为：

1. TimeAlign、TimeMixer、DLinear、iTransformer、PatchTST的Exchange，20个fixed-H jobs；
2. AMD、SimpleTM、TimePerceiver、SRSNet在8 datasets上的128个fixed-H jobs。

最小remote block应先执行现有`TimeAlign × Exchange × four H × seed2021`，共4 jobs。它必须标为`official-source local Exchange adapter / ETTh1-derived bootstrap`，不能声称是upstream tuned Exchange preset。该block仍需paper-artifact wrapper/checker、remote training与formal test的单独授权。

### 4.2 Main II

最小Main II为ISCF-BSCA-MAIN-v1、DLinear-Unified、PatchTST-Unified与A6_FULL在8 datasets上的完整矩阵，共128 MSE cells与128 MAE cells。当前ISCF的8 checkpoints可直接复用；A6_FULL可复用原5 datasets、需补3个；DLinear-Unified与PatchTST-Unified各缺8个。

当前可直接安全launch的Main II jobs为0。DLinear缺8-dataset/Solar loader、fixed-H720-to-four-prefix contract、four-H validation selector和test-only evaluator；active tree不存在独立的official-source PatchTST-Unified implementation。不得把TimeAlign carrier改名为PatchTST baseline。Tier A修复后建议先做3个validation-only resource smoke：PatchTST-Unified/ECL、PatchTST-Unified/Solar、DLinear-Unified/ECL。

[Estimate] Main II 19个new checkpoints约需9--19 GPU-hours，三张RTX 3090约3--7小时wall-clock；PatchTST smoke前为低置信度估计。Official test必须在19/19 train/validation artifacts完整后另行授权。

## 5. Four-layer decision

- `paper_facing_effectiveness=performance_partial_pass_competitive_aggregate_mse_pending_complete_baselines`；
- `matched_mechanism_attribution=pending_Main_II_and_exact_ablations`；
- `internal_mechanism_health=pending_selected_checkpoint_replay`；
- `failure_attribution=no_numeric_pathology_remaining_weak_cells_must_be_reported`。

Decision=`HPO_frozen_Main_I_published_140_rows_complete_competitive_not_full_SOTA_request_staged_baseline_authorization`。
