# ISCF-BSCA-MAIN-v1 H4M Formal Test and Joint-HPO Decision

## 1. Decision

H4M完整formal test通过artifact与protocol audit，但未通过预注册performance gate。24/24 frozen checkpoints、96/96 standard-horizon rows与MSE/MAE均完整；24个checkpoint在test前后hash一致，无retrain、mutation、partial artifact或ABORT。

把H4M加入H1--H4L后，189个trials的合法dataset-level joint selector由MSE/MAE/combined=`15/28, 16/28, 31/56`提升为`17/28, 16/28, 33/56`。ETTm2切换到`ETTm2__h4m_p6_lr5e5`，仍只有H720 MAE一个lead；Weather切换到`Weather__h4m_seq640_p20`，由2/8提高到4/8。`20/28, 20/28, 40/56`三项global gates仍全部失败，因此H4M状态为`performance_partial_pass_gate_fail`，不能宣称达到完整per-cell SOTA。

Failure attribution=`search_space_performance_shortfall`。H4M证明扩大高影响参数的搜索可以改善Weather，但当前frozen architecture内的patch/LR/context/rank profiles仍未缩小ETTm2 H96--H336的主要差距。该结论拒绝的是H4M search contract达到40/56目标的能力，不是ISCF-BSCA architecture或BSCA mechanism。Rollback=`Step 6 HPO/benchmark strategy decision`；H4N、新训练、selected-profile confirmation与3-seed不自动授权。

## 2. Frozen test audit

| Field | Value |
| --- | --- |
| test access date | `2026-08-05` |
| user authorization | analyze returned H4M results and proceed with formal test |
| candidate version | `ISCF-BSCA-MAIN-v1-targeted-h4m-test-informed-20260804` |
| test role | `test-tuned-hyperparameter-selection-and-paper-benchmark` |
| datasets | ETTm2, Weather |
| checkpoints | 24, seed2021 |
| horizons | 96, 192, 336, 720 |
| metrics | MSE, MAE |
| expected / complete cells | 96 / 96 |
| checkpoint manifest SHA256 | `bd78782e70a911e8c5b118c67a4441cf512d7012b94fdda8a19cf14507e8adb3` |
| checkpoint immutable | 24 / 24 |
| checkpoint retrained | 0 / 24 |
| test-tuned / test-informed | true / true |
| test artifact publication | atomic, 24 / 24 |
| ABORT sentinel | absent |
| remote execution | commit `bb84a7eeaddf2fd50cb3e31b31fc91957890bc7b`, GPU 0/1/2, 09:15:44--09:19:46 +08:00 |

Canonical machine artifacts:

- `h4m_test_result/test_audit_completeness.json`；
- `h4m_test_result/test_audit_ledger.jsonl`；
- `h4m_test_result/all_trial_scorecard.csv`；
- `joint_objective_h4m_result_20260805/current_joint_objective_status.json`；
- `joint_objective_h4m_result_20260805/joint_selected_profiles.csv`；
- `joint_objective_h4m_result_20260805/joint_selected_cells.csv`。

## 3. Dataset-level selection result

### 3.1 ETTm2

冻结selector从H4L `ETTm2__h4l_wd1e3`切换到H4M `ETTm2__h4m_p6_lr5e5`。相对H4L selector，four-H mean MSE从0.248621变为0.248733（+0.0448%），mean MAE从0.306214降至0.305693（-0.1704%）。新profile位于joint-mean 1% guard内，并因H720 MAE lead按冻结leading-cell priority胜出；不能改用本轮MSE更低但0/8 leads的`ETTm2__h4m_p4_lr2e5`作metric-specific选择。

| H | MSE | target | relative gap | MAE | target | relative gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 0.160881 | 0.155 | +3.7943% | 0.245766 | 0.241 | +1.9775% |
| 192 | 0.218428 | 0.210 | +4.0134% | 0.284770 | 0.280 | +1.7036% |
| 336 | 0.271552 | 0.263 | +3.2519% | 0.320394 | 0.315 | +1.7124% |
| 720 | 0.344069 | 0.343 | +0.3117% | 0.371841 | 0.372 | **-0.0427%** |

ETTm2保持MSE 0/4、MAE 1/4、combined 1/8，未通过4/8 local gate。H4M内部`patch_num × learning_rate`响应明显但非单调：`patch4/lr2e-5`的mean MSE最低，`patch6/lr5e-5`的mean MAE与lead coverage更好；rank48/80与seq960补充项均未改善joint frontier。说明patch与LR确是高影响interaction，但当前范围内不存在能同时解决short/mid-horizon MSE与MAE的profile。

### 3.2 Weather

冻结selector从H4K `Weather__h4k_current_lr5e5_patch24_dropout0`切换到H4M `Weather__h4m_seq640_p20`。Four-H mean MSE由0.216530降至0.214752（-0.8211%），mean MAE由0.248302降至0.247324（-0.3941%）。

| H | MSE | target | relative gap | MAE | target | relative gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 0.141174 | 0.140 | +0.8388% | 0.183515 | 0.179 | +2.5221% |
| 192 | 0.181922 | 0.182 | **-0.0429%** | 0.224484 | 0.220 | +2.0380% |
| 336 | 0.231708 | 0.232 | **-0.1258%** | 0.264744 | 0.262 | +1.0474% |
| 720 | 0.304205 | 0.307 | **-0.9105%** | 0.316553 | 0.317 | **-0.1410%** |

Weather由MSE 1/4、MAE 1/4提高到MSE 3/4、MAE 1/4、combined 4/8，但仍未通过6/8 local gate。本轮joint-score最低的是`seq512_p16_lr2e5`，只有2/8 leads；`seq640_p20`在1% guard内具有4/8 leads，因而按冻结规则合法胜出。Context结果同样非单调：640/20有效，768/24明显退化，960/30与1152/36未继续改善；rank16/64与patch8/32 variants也未稳定改善MAE。主要剩余缺口已从MSE转为H96--H336 MAE。

## 4. TimeAlign official-native reproduction cross-check

并行TimeAlign复现已8/8完成，8个checkpoint hash唯一，全部采用official-last、10 epochs、无early stopping并在训练结束后一次test。其执行标签为`official-source model/config + FATST test-hygiene/artifact adapter`，角色是`native_external`，不是matched mechanism attribution；license仍为`unresolved`。复现相对论文three-run mean非常接近：

| Dataset | reproduced mean MSE / MAE | paper mean MSE / MAE | relative gap MSE / MAE |
| --- | --- | --- | --- |
| ETTm2 | 0.242889 / 0.302523 | 0.242750 / 0.302000 | +0.0573% / +0.1733% |
| Weather | 0.215800 / 0.244725 | 0.215250 / 0.244500 | +0.2557% / +0.0921% |

与本次同seed本地复跑的TimeAlign逐cell结果比较，selected ISCF-BSCA在ETTm2 mean MSE/MAE分别落后2.4059%/1.0477%；在Weather mean MSE领先0.4857%，但mean MAE落后1.0618%。Weather的H192/H336/H720 MSE以及H720 MAE已经接近或优于本地TimeAlign；ETTm2只有H720接近，H96--H336仍存在约3.10%--3.90% MSE差距。

## 5. Global result and four-layer evaluation

H4M后seven-dataset 28-cell macro MSE/MAE从0.262687/0.308710改善为0.262449/0.308496，即-0.0906%/-0.0694%。相对TimeAlign published 28-cell macro，当前selected ISCF-BSCA低2.5387% MSE、低0.6225% MAE，但这仍是single-seed test-tuned unified model与published three-run native fixed-H systems的context comparison，不能表述为untouched-holdout或无条件SOTA。

完整leading-cell score为：

- MSE leads=`17/28`；
- MAE leads=`16/28`；
- combined leads=`33/56`；
- H96/H192/H336/H720 combined leads=`9/14, 11/14, 8/14, 5/14`；
- legal selector、unrestricted single-profile upper bound与diagnostic per-cell oracle均=`33/56`。

四层结论：

1. `paper_facing_effectiveness`: `performance_partial_pass_gate_fail`。完整test matrix健康，33/56高于H4L的31/56，但未达到40/56。
2. `matched_mechanism_attribution`: H4M不引入新mechanism，不能从HPO gain扩张ISCF/BSCA attribution claim；正式ablation/transfer证据仍缺失。
3. `internal_mechanism_health`: 24/24 training artifacts、24/24 test artifacts、96/96 cells、24 unique immutable checkpoints全部通过；无numeric或artifact pathology。
4. `failure_attribution`: `search_space_performance_shortfall`。ETTm2 H96--H336与Weather H96--H336 MAE是主要缺口；不是checkpoint selector、joint selector、checkpoint mutation、quota failure或test loader failure。

## 6. Next gate

当前应回到Step 6选择下一策略，不能自动启动H4N。可审议的最小分支是：

1. 接受33/56和strong aggregate competitor定位，停止针对rounded published-cell targets的同架构HPO，转入Main I完整baseline表与matched Main II；
2. 若仍坚持40/56，另行冻结H4N search contract，目标需分别针对ETTm2 short/mid horizons和Weather short/mid MAE，并说明为何新参数自由度不是H4L/H4M的重复邻域；
3. 任何architecture/objective redesign必须创建新的`test_informed` candidate并回到Step 4--6 narrative/design gate。

Decision=`H4M_complete_Weather_gain_global_33_of_56_gate_fail_return_step6_no_automatic_H4N`。
