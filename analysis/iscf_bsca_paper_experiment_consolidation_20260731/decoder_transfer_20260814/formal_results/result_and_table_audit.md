# ISCF-BSCA Decoder-Transfer Formal Result Audit

Decision：`decoder_transfer_complete_portability_gate_not_passed`

## 完整性与协议

- 2 backbones × 3 decoder arms × 5 datasets × 4 horizons = 120/120 cells；
- 全部arms为seed2021、from-scratch end-to-end joint training；
- checkpoint由validation四horizon mean MSE选择；formal test不选择checkpoint、seed或单元格；
- 30/30 checkpoint hashes经immutable manifest冻结，formal evaluation前后不允许变更。

## Pre-registered portability gates

| Backbone | BSCA vs Original MSE gain | MAE gain | Dataset/Cell MSE wins | ISCF vs Original MSE gain | BSCA vs ISCF MSE gain/cell wins | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DLinear-style | +15.702% | +8.184% | 4/5, 15/20 | +10.606% | +5.701%, 8/20 | PASS |
| PatchTST-style | -0.733% | -0.062% | 2/5, 8/20 | -2.148% | +1.385%, 16/20 | FAIL |

## Four-layer evaluation and failure attribution

1. paper_facing_effectiveness：120/120 official-test cells完整，checkpoint hash复核未发现mutation或non-finite结果。
2. matched_mechanism_attribution：DLinear-style通过预注册相对gate；PatchTST-style未通过。PatchTST中+ISCF-BSCA相对+ISCF改善，但仍未优于Original Decoder。
3. internal_mechanism_health：本表不以routing、scope probability或oracle headroom替代matched effectiveness gate。
4. failure_attribution：总体记为hypothesis_false_for_cross_backbone_portability_in_exact_setting。DLinear-style的ETTh1/ETTh2绝对结果提示optimization_or_profile_pathology_suspected，因此其正向相对gate不得被夸大；PatchTST-style负结果未出现artifact或numeric pathology，仍是当前总体portability claim失败的直接证据。

## Claim boundary

至少一个backbone未通过预注册gate，因此不得使用跨backbone decoder portability的总体正向表述；应按通过的block收窄结论并保留负结果。

DLinear-style与PatchTST-style arms不是native external baseline reproduction；本表只回答matched decoder transferability。

若作者希望恢复跨backbone portability claim，应回到Step 4--6重新设计PatchTST intervention/readout并冻结新的candidate；不得把本轮负结果改写为HPO问题或选择性删除PatchTST block。
