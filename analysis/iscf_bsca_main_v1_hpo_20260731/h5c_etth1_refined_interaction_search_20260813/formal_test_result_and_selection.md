# H5C ETTh1 Formal-Test Result and Frozen Selection

## 1. Decision

- `decision=H5C_no_eligible_best_cell_improvement_retain_H5B_profile`
- `gate_pass=false`
- H5C完整formal test已结束，但54个H5C profiles中没有任何一个把Main II ETTh1
  best cells从H5B的`4/8`提高到最低目标`5/8`。
- 按冻结selector，继续保留H5B `ETTh1__h5b_seq640_p20`作为ETTh1 paper-row
  fallback。H5C不得触发Main I/Main II table mutation，也不得自动进入H5D、extra seed或
  architecture redesign。

## 2. Formal-test execution audit

| Item | Result |
| --- | --- |
| user authorization | 2026-08-13，继续H5C完整formal test |
| candidate version | `ISCF-BSCA-MAIN-v1-etth1-h5c-test-informed-20260813` |
| exact commit | `d2f9eacd8502d92cec2301a3ef283bc96dc7206a` |
| config SHA256 | `a519c6dc197d090eecddb32df90cc072a46c41d35292e1f8a7211a1a03018dea` |
| checkpoint manifest | 54 rows，54 unique hashes |
| manifest SHA256 | `e94f95a67c748f95d72e1aab6ced4aaae498982105da5234d411d4c5c0c8379f` |
| formal-test interval | 2026-08-13 14:08:17--14:11:20 +08:00 |
| complete profiles | 54/54 |
| standard-horizon rows | 216/216 |
| horizons / metrics | `{96,192,336,720}` / MSE, MAE |
| test artifact errors | 0 |
| temporary artifacts after queue | 0 |
| checkpoint mutation / retraining | 0 / false |
| test role | test-tuned dataset-level profile selection and paper benchmark |

每个checkpoint仍由four-H validation mean MSE选定；official test只对54个冻结
hyperparameter profiles进行dataset-level排名。一个profile必须同时服务四个horizons，未执行
per-horizon、per-metric、per-cell、per-seed选择。

## 3. Frozen selector result

冻结排序为：Main II best cells降序、Main I best cells降序、Main II top-2 cells降序、
four-H mean MSE、mean MAE、validation mean MSE、参数量与profile ID。只有four-H mean
MSE和MAE均不超过H5B `1.002x`的profile eligible。

- 54个H5C profiles中22个通过双guard。
- 全部54个profile的Main II best-cell上限仍为`4/8`；因此即使忽略guard，也没有候选达到
  minimum=`5/8`。
- H5C可达到的Main II top-2上限为`8/8`，但对应候选只有`3/8` best，不能替代primary
  objective。
- 唯一达到Main I `6/8` best的`h5c_ctx630_p21`只有Main II `3/8` best，且MAE guard
  为`1.002038x`、略超冻结上限，故不具备selection资格。

eligible H5C winner为`ETTh1__h5c_do0`，即在H5B `L640/p20`设置上将dropout设为0：

| Horizon | H5B MSE | H5C-do0 MSE | Relative | H5B MAE | H5C-do0 MAE | Relative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 0.346489 | 0.346435 | -0.016% | 0.385206 | 0.385612 | +0.106% |
| 192 | 0.377191 | 0.376652 | -0.143% | 0.403444 | 0.403300 | -0.035% |
| 336 | 0.396507 | 0.395779 | -0.184% | 0.419125 | 0.418715 | -0.098% |
| 720 | 0.445325 | 0.444033 | -0.290% | 0.461247 | 0.460395 | -0.185% |
| Mean | 0.391378 | 0.390725 | -0.167% | 0.417255 | 0.417006 | -0.060% |

`h5c_do0`通过双guard，且四H平均MSE/MAE均小幅优于H5B；但经共同三位小数
rounding后，其Main I/II best/top-2计数与H5B完全相同：Main I=`5/8,7/8`，Main
II=`4/8,7/8`。Main II best仍为H96 MSE/MAE与H720 MSE/MAE。由于冻结primary
objective要求best cells严格增加，不能用均值的事后改善改写selector；最终保留H5B。

## 4. Four-layer evidence and rollback

1. `paper_facing_effectiveness`：完整test surface通过artifact gate，但H5C targeted success gate
   失败；本轮没有新的paper-row selection。
2. `matched_mechanism_attribution`：H5C只改变hyperparameters，未设计architecture/objective
   control，因此不提供BSCA或decoder mechanism attribution。
3. `internal_mechanism_health`：54/54 checkpoints、validation metrics、effective configs与test
   artifacts完整，54 hashes唯一，无NaN/Inf/failure或checkpoint mutation；没有证据表明失败来自
   numeric pathology。
4. `failure_attribution`：`search_space_performance_shortfall_at_best_cell_gate`。它只否定H5C这组
   refined interactions达到`5/8` best的能力，不否定ISCF-BSCA方向。Rollback=`Step 6`：保留
   H5B fallback；若作者未来重启ETTh1 HPO，应先重新设计有实质差异的search strategy并创建
   H5D candidate，而不是继续在H5C邻域加密。

## 5. Canonical artifacts

- Full audit: `analysis/iscf_bsca_main_v1_hpo_20260731/h5c_formal_test_result_20260813/`
- Frozen ranking: `analysis/iscf_bsca_main_v1_hpo_20260731/h5c_formal_test_result_20260813/frozen_selector/all_profile_ranking.csv`
- H5C winner scorecard: `analysis/iscf_bsca_main_v1_hpo_20260731/h5c_formal_test_result_20260813/frozen_selector/best_h5c_profile_scorecard.csv`
- Retained H5B scorecard: `analysis/iscf_bsca_main_v1_hpo_20260731/h5c_formal_test_result_20260813/frozen_selector/selected_profile_scorecard.csv`
- Machine decision: `analysis/iscf_bsca_main_v1_hpo_20260731/h5c_formal_test_result_20260813/frozen_selector/h5c_selection_result.json`

该证据是single-seed、test-tuned且test-informed，不得描述为untouched holdout或严格
confirmatory generalization estimate。
