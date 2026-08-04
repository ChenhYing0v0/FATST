# ISCF-BSCA-MAIN-v1 H4L Formal Test and Joint-HPO Decision

## 1. Decision

H4L完整formal test通过artifact与protocol audit，但未通过performance gate。48/48 frozen checkpoints、192/192 standard-horizon rows与MSE/MAE均完整；48个checkpoint在test前后hash一致，无retrain、mutation、partial artifact或ABORT。

把H4L加入H1--H4K后，165个trials的合法dataset-level joint selector由MSE/MAE/combined=`15/28, 15/28, 30/56`提升为`15/28, 16/28, 31/56`。唯一新增领先cell是ETTm2 H720 MAE；Weather selector不变。`20/28, 20/28, 40/56`三项global gates仍全部失败，因此H4L状态为`performance_partial_pass_gate_fail`，不能宣称达到完整per-cell SOTA。

Failure attribution=`search_space_performance_shortfall`。该结论拒绝的是H4L wide HPO contract达到目标的能力，不是ISCF-BSCA architecture或BSCA mechanism。Rollback=`Step 6 HPO/benchmark strategy decision`；H4M、新训练、selected-profile confirmation与3-seed不自动授权。

## 2. Frozen test audit

| Field | Value |
| --- | --- |
| test access date | `2026-08-04` |
| user authorization | complete H4L formal test |
| candidate version | `ISCF-BSCA-MAIN-v1-wide-h4l-test-informed-20260804` |
| test role | `test-tuned-hyperparameter-selection-and-paper-benchmark` |
| datasets | ETTm2, Weather |
| checkpoints | 48, seed2021 |
| horizons | 96, 192, 336, 720 |
| metrics | MSE, MAE |
| expected / complete cells | 192 / 192 |
| checkpoint manifest SHA256 | `c7ce6b6915dbe0323282140c0ed28ecad590b5ea256e8545a7f0fb3217c25584` |
| checkpoint immutable | 48 / 48 |
| checkpoint retrained | 0 / 48 |
| test-tuned / test-informed | true / true |
| test artifact publication | atomic, 48 / 48 |
| ABORT sentinel | absent |
| remote execution | commit `8d44f6cdeefa68d59eff5641701da6c0fcf60821`, GPU 0/1/2, 20:28:12--20:37:09 +08:00 |

Canonical machine artifacts:

- `h4l_test_result/test_audit_completeness.json`；
- `h4l_test_result/test_audit_ledger.jsonl`；
- `h4l_test_result/all_trial_scorecard.csv`；
- `joint_objective_h4l_result_20260804/current_joint_objective_status.json`；
- `joint_objective_h4l_result_20260804/joint_selected_profiles.csv`；
- `joint_objective_h4l_result_20260804/joint_selected_cells.csv`。

## 3. Dataset-level selection result

### 3.1 ETTm2

The frozen selector changes from H4K `ETTm2__h4k_rank64_dropout2` to H4L `ETTm2__h4l_wd1e3`. The only hyperparameter change relative to the H4K anchor is `weight_decay: 0.01 -> 0.001`; all four horizons continue to share this single profile. Its checkpoint SHA256 is `c7d6f903e0e0764496f0225746bd5e7f5df2b8fd769929bcfd0482a2965fdcbf`.

| H | MSE | target | relative gap | MAE | target | relative gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 0.161596 | 0.155 | +4.2553% | 0.247268 | 0.241 | +2.6008% |
| 192 | 0.217939 | 0.210 | +3.7803% | 0.285395 | 0.280 | +1.9267% |
| 336 | 0.270679 | 0.263 | +2.9197% | 0.320226 | 0.315 | +1.6589% |
| 720 | 0.344272 | 0.343 | +0.3709% | 0.371969 | 0.372 | **-0.0082%** |

相对H4K ETTm2 selector，four-H mean MSE从0.248679降至0.248621，mean MAE从0.306326降至0.306214。H96/H192/H336两指标均小幅改善，H720 MSE微幅变差，但H720 MAE跨过published rounded target，产生唯一新增lead。ETTm2仍为MSE 0/4、MAE 1/4，未通过预设4/8 local gate。

### 3.2 Weather

H4L没有替换H4K selector。保留`Weather__h4k_current_lr5e5_patch24_dropout0`，其four-H mean MSE/MAE为0.216530/0.248302；H96/H192/H336仍落后published targets，H720 MSE/MAE分别领先0.3344%/0.0550%。Weather保持MSE 1/4、MAE 1/4、combined 2/8，未通过6/8 local gate。

H4L内部test joint-mean最优profile为`Weather__h4l_seq512_patch16`，但只有MAE 1/4 lead；按冻结1% guard后最大化balanced lead cells的selector，它不能超过H4K当前profile的2/8。因此“不替换Weather”是完整165-trial selector的结果，不是选择性忽略H4L negative trials。

### 3.3 H720 and global score

H4L后selected seven-dataset H720领先数从combined 4/14增至5/14，仍低于6/14 gate。Seven-dataset 28-cell macro MSE/MAE相对H4K selector只改善0.00312%/0.00516%。完整global score为：

- MSE leads=`15/28`；
- MAE leads=`16/28`；
- combined leads=`31/56`；
- legal selector、unrestricted single-profile upper bound与diagnostic per-cell oracle均=`31/56`。

因此，当前search pool内部已经没有selector规则可以合法或非法地把31/56提高到40/56；差距来自候选profile性能，而非joint selector实现。

## 4. Search-space diagnostics

Validation与official-test排序仅部分一致。ETTm2 validation-best为`patch1`，test joint-best为`lr1e5`；Weather validation-best为`patch4`，test joint-best为`seq512_patch16`。H4L内validation与test joint-score的Spearman rank correlation分别约0.582与0.017。该结果支持继续保留validation-only checkpoint selection，同时说明validation不能替代本项目已冻结的dataset-level test-tuned profile selection。

四个明确保持TimeAlign official encoder coupling的profiles未成为dataset selector：ETTm2 short/long profiles在H4L test joint-score中排名10/24与7/24，均为0/8 leads；Weather short/long profiles排名9/24与17/24，分别为1/8与0/8 leads。TimeAlign-inspired parameter prior提供了有界搜索起点，但没有解决ETTm2/Weather的主要published-target gap。

## 5. Four-layer evaluation

1. `paper_facing_effectiveness`: `performance_partial_pass_gate_fail`。完整test matrix健康，但31/56未达到40/56。
2. `matched_mechanism_attribution`: H4L不引入新mechanism，不能从HPO gain扩张ISCF/BSCA attribution claim；正式ablation/transfer证据仍缺失。
3. `internal_mechanism_health`: 48/48 training artifacts、48/48 test artifacts、192/192 cells、48 unique immutable checkpoints全部通过；无numeric或artifact pathology。
4. `failure_attribution`: `search_space_performance_shortfall`。ETTm2 short/mid horizons与Weather H96--H336仍是主要缺口；不是checkpoint selector、joint selector、checkpoint mutation、quota failure或test loader failure。

## 6. Next gate

当前应先在Step 6选择下一策略，而不是自动堆叠H4M。可审议的分支为：

1. 接受31/56与strong aggregate competitor定位，停止针对rounded published-cell targets的HPO，转入Main I完整baseline表与matched Main II；
2. 若仍坚持40/56目标，先冻结一个新的、来源审计充分且与H4L不重复的dataset-level strategy，并明确为何它能同时缩小ETTm2 H96--H336和Weather H96--H336差距；
3. 任何architecture/objective redesign必须创建新的`test_informed` candidate并回到Step 4--6 narrative/design gate，不能伪装成H4L邻域调参。

在用户新授权前，不启动H4M、新训练、selected-profile confirmation、formal test或3-seed。
