# H5D ETTh1 Formal-Test Result and Frozen Selection

## 1. Decision

- `decision=H5D_no_eligible_best_cell_improvement_retain_H5B_profile`
- `gate_pass=false`
- H5D完整formal test已结束。48个profiles中只有3个通过相对H5B的mean MSE/MAE
  `1.002x`双guard；eligible pool的Main II best-cell上限仍为`4/8`。
- 唯一达到minimum=`5/8`的`h5d_bs16_lr3p6`因mean MSE退化`0.675%`而不eligible。
- 冻结selector继续保留H5B `ETTh1__h5b_seq640_p20`；Main I/Main II不修改，H5E、
  extra seeds与architecture redesign不自动启动。

## 2. Formal-test execution audit

| Item | Result |
| --- | --- |
| User authorization | 2026-08-13，H5D训练完成后继续formal test并分析结果 |
| Candidate version | `ISCF-BSCA-MAIN-v1-etth1-h5d-test-informed-20260813` |
| Exact commit | `12ddfed7b393be3880a51ef7dbc250e43eff89b5` |
| Formal-test config SHA256 | `4657e966af40d6d970929e75fc3550379b7646c70ec33dc079aaf2fa089b0ede` |
| Checkpoint manifest | 48 rows，48 unique hashes |
| Manifest SHA256 | `480180333de60c3f53d98c894b8854e4169401edcf7ca378d20f1b213e233a9e` |
| Formal-test interval | `2026-08-13 16:28:05--16:30:50 +08:00` |
| Complete profiles / standard rows | `48/48` / `192/192` |
| Artifact errors / temporary files / ABORT | `0 / 0 / absent` |
| Checkpoint mutation / retraining | `0 / false` |
| Test role | test-tuned ETTh1 dataset-level profile selection and paper benchmark |

每个checkpoint仍由four-H validation mean MSE选定；official test只对完整冻结profiles作
dataset-level排名。一个profile同时服务H96/H192/H336/H720，未进行per-H、per-metric、
per-cell或per-seed选择。

## 3. Frozen selector result

排序为Main II best降序、Main I best降序、Main II top-2降序、mean MSE、mean MAE、
validation score、参数量与profile ID。相对H5B的MSE/MAE各自`1.002x`是先验eligibility
约束，而不是tie-break。

| Statistic | H5D result |
| --- | ---: |
| Profiles tested | 48 |
| Eligible profiles | 3 |
| Max Main I best cells | 6/8 |
| Max Main II best cells before guard | 5/8 |
| Max Main II best cells after guard | 4/8 |
| Max Main II top-2 cells | 8/8 |

Eligible H5D winner是`ETTh1__h5d_bs16_lr2p4`（batch16，LR=`2.4e-4`）：

| Horizon | H5B MSE | H5D MSE | Relative | H5B MAE | H5D MAE | Relative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 0.346489 | 0.348176 | +0.487% | 0.385206 | 0.386213 | +0.261% |
| 192 | 0.377191 | 0.376924 | -0.071% | 0.403444 | 0.402917 | -0.131% |
| 336 | 0.396507 | 0.394763 | -0.440% | 0.419125 | 0.417207 | -0.458% |
| 720 | 0.445325 | 0.446464 | +0.256% | 0.461247 | 0.461009 | -0.052% |
| Mean | 0.391378 | 0.391582 | +0.052% | 0.417255 | 0.416836 | -0.100% |

它把Main II top-2从`7/8`提高到`8/8`，但best仍为H96和H720的四项，即`4/8`；按
冻结primary objective不能替换H5B。

唯一的`5/8` profile `h5d_bs16_lr3p6`新增H192 MAE best并达到top-2=`8/8`，mean MAE
改善`0.295%`，但mean MSE从0.391378升至0.394021（`+0.675%`）。因此它违反MSE guard，
不能用一个rounded favorable cell掩盖dataset-average regression。

另两个值得保留的negative/diagnostic结果是：

- `h5d_bs24_lr2p8`取得H5D最低mean MSE=`0.390289`（相对H5B `-0.278%`），但mean
  MAE=`0.417652`且Main II best仅`3/8`；
- `h5d_bs24_lr3p4`同时改善mean MSE/MAE至`0.391030/0.416786`，但best仍只有`3/8`。

## 4. Hyperparameter evidence

- [Strong Evidence] 3个eligible profiles全部来自`dropout0_batch_lr_interaction`；batch size
  与LR的联合效应是本轮唯一形成aggregate-feasible frontier的因素。
- [Strong Evidence] p19/p21 geometry、standalone rank与geometry×rank共36个profiles均未通过
  双guard；尤其p21×high-rank没有产生Main II best cells，继续沿该方向加密缺乏依据。
- [Strong Evidence] 更高有效step size倾向降低MAE并增加rounded best cells，但会提高MSE；
  低LR/batch24更利于mean MSE。这解释了`5/8`候选与aggregate guard的冲突。
- [Inference] 当前瓶颈已不是未覆盖的局部geometry/rank点，而是MSE与MAE/cell-count目标之间
  的Pareto trade-off。若未来继续HPO，应先明确是否改变预注册objective或采用不同训练
  objective；不能在看到H5D test后放宽guard。

## 5. Four-layer evidence and rollback

1. `paper_facing_effectiveness`：formal artifacts完整，但targeted success gate失败；无新paper-row。
2. `matched_mechanism_attribution`：只改变hyperparameters，不提供BSCA/decoder attribution。
3. `internal_mechanism_health`：48 hashes唯一，dense/provenance/NPZ checks、numeric health与
   checkpoint immutability全部通过；失败不是runtime pathology。
4. `failure_attribution`：`objective_tradeoff_and_search_space_performance_shortfall`。结果只否定
   H5D在固定双guard下达到eligible `5/8`的能力，不否定ISCF-BSCA architecture。
   Rollback=`Step 6 -> retain H5B and close H5D`。

## 6. Canonical artifacts

- Full audit: `analysis/iscf_bsca_main_v1_hpo_20260731/h5d_formal_test_result_20260813/`
- Machine decision: `analysis/iscf_bsca_main_v1_hpo_20260731/h5d_formal_test_result_20260813/frozen_selector/h5d_selection_result.json`
- All profiles: `analysis/iscf_bsca_main_v1_hpo_20260731/h5d_formal_test_result_20260813/frozen_selector/all_profile_ranking.csv`
- H5D winner scorecard: `analysis/iscf_bsca_main_v1_hpo_20260731/h5d_formal_test_result_20260813/frozen_selector/best_h5d_profile_scorecard.csv`
- Retained H5B scorecard: `analysis/iscf_bsca_main_v1_hpo_20260731/h5d_formal_test_result_20260813/frozen_selector/selected_profile_scorecard.csv`
