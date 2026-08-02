# ISCF-BSCA-MAIN-v1 Joint-Objective Reset and H4J Prelaunch

## Decision summary

| Field | Value |
| --- | --- |
| `date` | 2026-08-02 |
| `current_step` | Step 6 H4J contract frozen；Step 7 local tooling pass；remote preflight pending |
| `problem` | previous selector optimized dataset-level four-H mean MSE only；current selected row leads 14/28 MSE and 9/28 MAE cells |
| `new_objective` | one shared profile per dataset；joint MSE/MAE mean quality first，lead-cell count second |
| `target` | MSE `>=20/28`，MAE `>=20/28`，combined `>=40/56` |
| `architecture_search` | false；exact ISCF-BSCA architecture、objective、scale set与inference graph unchanged |
| `matrix` | 40 new H4J jobs；seed2021；8 datasets；28/40 jobs allocated to ETTm2/Weather/Solar |
| `official_test` | training 40/40 and checkpoint manifest freeze后直接执行complete test；不以validation比较profile |
| `decision` | `H4J_frozen_local_gate_pass_remote_resource_smoke_then_training_authorized` |

## 1. Existing evidence and why reselection is insufficient

H1/H2/H3A/H3B共有53个完整test-tuned trials。以TimeAlign Table 6中五个published models的逐cell最优displayed value为target，当前selected profiles为：

- MSE leading cells=`14/28`；
- MAE leading cells=`9/28`；
- combined=`23/56`。

若不训练任何新profile，而在每个dataset现有trials中重新选择一个共享四个horizons的profile，combined cell count的理论上限也只有`25/56`。逐dataset上限为ECL 5/8、ETTh1 5/8、ETTh2 6/8、ETTm1 6/8、ETTm2 0/8、Solar 1/8、Weather 2/8。因此新目标不能通过重排、per-metric挑选或per-horizon拼接实现，必须新增训练；ETTm2、Solar和Weather是主要瓶颈。

Canonical evidence：

- `joint_objective_reset_20260802/all_existing_trial_joint_scorecard.csv`；
- `joint_objective_reset_20260802/dataset_frontier.csv`；
- `joint_objective_reset_20260802/current_joint_objective_status.json`。

## 2. Frozen dataset-level selector

每个common dataset的candidate profile先计算：

$$
J_d(p)=\frac{1}{4}\sum_{H\in\{96,192,336,720\}}
\frac{1}{2}\left(
\frac{\operatorname{MSE}_{d,H}(p)}{\operatorname{MSE}^{\star}_{d,H}}
+
\frac{\operatorname{MAE}_{d,H}(p)}{\operatorname{MAE}^{\star}_{d,H}}
\right),
$$

其中$\operatorname{MSE}^{\star}_{d,H}$和$\operatorname{MAE}^{\star}_{d,H}$分别是frozen published block中该cell的最小displayed value。只有$J_d(p)$不高于该dataset全部retained trials最佳$J_d$的1%的profile进入lead-cell排序。合格profile按以下顺序选择：

1. MSE+MAE总leading cells更多；
2. $\min(\text{MSE leads},\text{MAE leads})$更大；
3. $J_d$更小；
4. validation mean MSE、parameter count、profile id依次打破平局。

这一selector仍只产生一个dataset-level profile并共同服务四个horizons。禁止per-horizon、per-metric、per-seed或per-cell profile selection。Exchange当前没有同协议published target，不进入56-cell denominator；它仍参加8-dataset HPO，并以within-search MSE/MAE mean regret的等权分数选择唯一profile。

Validation仍只负责early stopping和trial checkpoint selection，checkpoint score保持four-H mean validation MSE。Official test只在40/40 training artifacts完整、checkpoint SHA256 manifest冻结后访问，随后一次性评估全部40 profiles × four horizons × MSE/MAE；所有negative trials保留。

## 3. H4J minimal sufficient matrix

H4J共40个from-scratch end-to-end joint-training jobs：

| Dataset | Jobs | Main search freedom |
| --- | ---: | --- |
| ETTm2 | 9 | budget、LR、dropout、decoder rank、lookback、effective batch |
| Weather | 9 | current/TimeAlign/Table5 frontiers上的budget、LR、patch、capacity、dropout |
| Solar | 10 | joint-MSE/MAE winner与MAE frontier的LR × dropout × patch × rank interactions及budget60 |
| ECL | 4 | exact joint winner与5-cell frontier的budget/LR refinements |
| ETTh1 | 2 | LR与dropout refinement |
| ETTh2 | 2 | LR与lookback refinement |
| ETTm1 | 2 | budget与dropout refinement |
| Exchange | 2 | lookback refinement |

28/40 jobs集中在当前总计仅3/24 leading cells的ETTm2、Weather和Solar。其余12 jobs用于改善或保护dataset-level joint mean frontier，避免只追弱dataset而使原有强dataset cells倒退。完整profile表以`configs/iscf_bsca_main_v1_hpo_joint_h4j.json`为唯一machine-readable contract；该列表不是Cartesian product。

## 4. Controls, resources, schedule, and gates

- architecture/objective control：candidate仍为`ISCF-BSCA-MAIN-v1`，不改变ISCF、BSCA、loss、scales或inference graph；本轮只允许frozen config中显式列出的training/encoder profile freedom；
- seed：当前仅2021；不以seed选择profile；three-seed仍是time-permitting完整block；
- checkpoint：best four-H validation mean MSE；test不选择epoch/checkpoint；
- reporting：7 common datasets的56 cells全报；Exchange完整MSE/MAE另报但不伪造published lead target；
- resource estimate：[Estimate] 基于H2/H3A吞吐，40 jobs在3张RTX 3090上约18--30小时wall-clock，具体ETA在resource smoke与首批epochs后更新；
- scheduling：global workload-aware queue，ECL和Solar先入队，Weather分散到后续slots，短ETT/Exchange任务填充空闲GPU；不得使用per-arm paired wait；
- remote gate：先核对pushed commit、remote `git pull --ff-only`、data path、storage及`nvidia-smi`；随后complete resource smoke，再启动40-job train/validation；
- test gate：40/40 checkpoints、finite four-H validation metrics、artifact schema、checkpoint hash均pass后，冻结40-row manifest并直接启动complete official-test audit；
- success gate：最终选择同时满足MSE `>=20/28`、MAE `>=20/28`、combined `>=40/56`，且7个selected profiles全部通过1% joint-mean guard；这是ambitious HPO target，不在结果返回前宣称已达成；
- failure gate：若complete H4J后未达到目标，保留所有negative results，状态记为`joint_objective_target_not_met`。任何H4K扩展必须重新冻结有限search contract；architecture redesign必须创建新candidate version并回到Step 4--6 narrative/design gate。

## 5. Local verification and decision

Local checks cover JSON parse、Python compile、40-job materialization、unique trial IDs、exact dataset counts、28 weak-dataset jobs、frozen evidence hashes、MSE/MAE gates与generic runner dry-run。No remote connection、training或official test is claimed by this prelaunch report.

Four-layer status：

- `paper_facing_effectiveness=pending_H4J_complete_test`；
- `matched_mechanism_attribution=unchanged_pending_Main_II_and_ablations`；
- `internal_mechanism_health=unchanged`；
- `failure_attribution=existing_weak_cells_are_profile_optimization_gap_not_architecture_rejection`。

Decision=`H4J_frozen_local_gate_pass_remote_resource_smoke_then_training_authorized`。
