# ISCF-BSCA-v1 three-seed confirmation Step 9/10

## 结论

Decision=`passed_core_candidate_ready_for_paper_consolidation`。冻结的three-seed official-test matrix完整通过direction
robustness与paper-core promotion两级gate：MSE/MAE macro gain=`+0.3541%/+0.3073%`，MSE为41/60 cells、3/3
seeds、4/5 datasets、4/4 horizons positive。ETTm2 three-seed mean由seed2021的`-1.7375%`收敛到`-0.6506%`，
满足预注册`>=-1%`边界，但仍是必须报告的主要negative dataset。

该结果把BSCA从single-seed partial pass升级为ISCF上的paper-core contribution candidate。它支持的是
`balanced policy access -> balanced fused-loss gradient access -> stable scope co-adaptation`，不支持conditional routing
specialization、universal gain或generic load-balancing novelty。整体论文仍需完成ISCF+BSCA的贡献边界整合与modern baseline
protocol，不能把本次pass等同于整篇论文已完成。

## Protocol 与 test role

| Field | Value |
|---|---|
| `test_access_date` | 2026-07-22 |
| `user_authorization` | 用户回复“继续按计划推进实验”，授权10 trainings与10/10后的single frozen formal test |
| `candidate_version` | `ISCF-BSCA-v1-confirmation-seeds2022-2023-v1` |
| `training_commit` | `72e3356720acba45d9dcd81aa197b138a4e64b59` |
| `checkpoint_retrained` | true；新增seeds2022/2023共10个candidate checkpoints；seed2021复用已冻结candidate |
| `reference_retrained` | false；复用FCC same-seed ISCF-EQUAL checkpoints |
| `checkpoint_selection` | validation mean MSE over H96/H192/H336/H720 |
| `checkpoint_mutated_during_test` | false；15/15 candidate checkpoint hashes匹配test invariants |
| `test_role` | primary mechanism effectiveness and paper-facing benchmark；`test_informed=true` |
| `matrix_complete` | true；3 seeds × 5 datasets × 4 horizons = 60 cells per metric |
| `formal_test_access_count_for_version` | 1 |

15/15 candidate runs的objective均为`equal_uniform_scope_anchor`，missing artifact count均为0；candidate与same-seed
EQUAL的encoder、PCSD、coordinate、partition与SIFF basis initialization hashes逐项配对。Objective、weight=0.1、前25%
ramp、profiles、ranks、optimizer、checkpoint selector与test matrix均未根据seed2021 test结果调整。

## 统计量与artifact定义

- `cell gain (%) = 100 * (EQUAL - BSCA) / EQUAL`；正值表示BSCA更好。
- `macro gain`：60个seed-dataset-horizon cell gain的等权平均。
- `seed/dataset/horizon mean`：固定相应维度后，对其余test cells等权平均MSE gain。
- `win`：对应cell或group mean严格大于0；`joint-positive`表示同一cell的MSE与MAE gain均大于0。
- `leave-one-seed-out`：删除一个seed后，对剩余40个MSE cell gain等权平均；仅作稳健性描述，不属于冻结gate。
- `cluster bootstrap CI`：先把每个seed×dataset的四horizon MSE gain平均成15个cluster effects，再以固定RNG
  `20260722`有放回抽样15 clusters、重复50,000次，取2.5%/97.5%分位数。该post-hoc uncertainty diagnostic不改变
  preregistered gate。
- `comparison_cells.csv`：120 rows，分别为60 validation与60 official-test cells，包含candidate/reference MSE/MAE及gain。
- `run_audit.csv`：15 candidate runs的artifact、checkpoint SHA、initialization pairing、objective与test-invariant audit。
- `internal_health.csv`：15 candidate + 15 EQUAL runs的policy usage、arm diversity与oracle statistics。

## Layer 1：paper-facing effectiveness

### 总体与方向稳定性

| Statistic | Result | Frozen requirement | Pass |
|---|---:|---:|---:|
| Macro MSE gain | **+0.3541%** | >= +0.3% | yes |
| Macro MAE gain | **+0.3073%** | > 0 | yes |
| MSE-positive cells | 41/60 | diagnostic | — |
| MAE-positive cells | 41/60 | diagnostic | — |
| Joint-positive cells | 36/60 | diagnostic | — |
| Positive seed means | 3/3 | >= 2/3 | yes |
| Positive dataset means | 4/5 | >= 3/5 | yes |
| Positive horizon means | 4/4 | >= 3/4 | yes |
| Minimum dataset mean | -0.6506% | > -2% | yes |
| ETTm2 mean | -0.6506% | >= -1% | yes |

Seed mean MSE gains为seed2021 `+0.3104%`、seed2022 `+0.5735%`、seed2023 `+0.1783%`。删除任一seed后
macro仍为正：drop-2021=`+0.3759%`、drop-2022=`+0.2444%`、drop-2023=`+0.4420%`。因此正方向不是由单一
seed维持，但effect size在seeds间仍有约3.2倍差异。

### Dataset 与 horizon 分解

| Dataset | MSE gain | MAE gain | MSE-positive cells |
|---|---:|---:|---:|
| ETTm1 | +1.1814% | +0.9306% | 10/12 |
| ETTh1 | +0.7885% | +0.2049% | 10/12 |
| Weather | +0.4157% | +0.4298% | 9/12 |
| ETTh2 | +0.0355% | +0.0925% | 8/12 |
| ETTm2 | **-0.6506%** | **-0.1216%** | 4/12 |

| Horizon | MSE gain | MAE gain | MSE-positive cells |
|---|---:|---:|---:|
| H96 | +0.6191% | +0.4688% | 11/15 |
| H192 | +0.1716% | +0.2382% | 10/15 |
| H336 | +0.2416% | +0.2668% | 10/15 |
| H720 | +0.3840% | +0.2552% | 10/15 |

ETTm2的主要损失集中在seed2021 H192/H336/H720；新增seeds明显减轻而未完全消除该异质性。Weather seed2023
MSE mean为`-0.4079%`，说明positive dataset mean也不是每seed都成立。不得只报告ETTm1/ETTh1或H96。

Validation macro MSE gain=`+0.5236%`，test=`+0.3541%`；逐cell gain Pearson correlation=`0.4006`，方向一致
40/60。没有出现整体validation/test reversal，但cell-level generalization仍有限。

### Post-hoc uncertainty boundary

15个seed×dataset cluster中11个mean MSE gain为正，中位数为`+0.0902%`。cluster bootstrap 95% interval约为
`[-0.1190%, +0.8423%]`，跨过0。该结果不撤销预注册promotion pass，但要求论文把effect描述为
`small, directionally robust under the frozen three-seed matrix`，不能宣称statistically conclusive universal gain。

## Layer 2：matched mechanism attribution

BSCA与EQUAL共享exact ISCF architecture、参数量、inference graph、initialization class、data、optimizer、checkpoint rule和
evaluation；15/15 candidate-control initialization contracts完全配对。唯一干预是train-only、target/H-free uniform policy
anchor。因此本矩阵可把observed mean difference归因到balanced-anchor objective导致的joint co-adaptation，而不是新增capacity、
router、test-time ensemble或初始化偏差。

边界仍然明确：这是一项same-architecture objective attribution，不证明uniform KL primitive本身新颖；论文贡献必须绑定到
ISCF dense temporal-scope arms中policy同时分配prediction mixture与fused-loss gradient这一特定问题链。

## Layer 3：internal mechanism health

| Metric, 15-run mean | BSCA | EQUAL | Interpretation |
|---|---:|---:|---|
| Policy normalized entropy | 0.9974 | 0.7553 | anchor稳定产生near-uniform pointwise access |
| Marginal usage max | 0.2065 | 0.2751 | 五scope使用接近0.2 |
| Marginal usage min | 0.1943 | 0.1362 | 弱scope access显著提高 |
| Pairwise arm L1 | 0.1197 | 0.1222 | ratio=0.9798；arms未collapse |
| Oracle headroom | 32.5894% | 33.3107% | complementarity基本保留但未由routing兑现 |

Candidate entropy三seed分别为0.9983/0.9965/0.9974；pairwise arm L1为0.1165/0.1265/0.1161；oracle
headroom为32.56%/33.22%/31.99%。这些internal paths跨seed稳定，且all tensors finite。它们支持anchor确实改变
gradient access同时未抹去arm差异；它们不能单独证明性能，也不支持“各arm自动学成明确语义专家”。

## Layer 4：failure attribution、claim与下一步

本轮不是failure，且没有`optimization_or_numeric_pathology`或`capacity_control_explains`证据。最合适的归因是：

- [Strong Evidence] frozen three-seed matched matrix的平均performance、seed/dataset/horizon方向与internal health全部过gate。
- [Strong Evidence] improvement来自train-time objective/co-adaptation，而非architecture capacity或inference path变化。
- [Uncertainty] effect small、cluster bootstrap跨0、ETTm2 negative；不能外推为all-dataset guarantee。
- [Narrative boundary] 支持BSCA作为ISCF-native training contribution；不claimgeneric KL novelty、conditional specialization或
  universal superiority。

下一步不是继续调lambda、补router/loss或按dataset rescue。应进入paper consolidation：冻结`ISCF-BSCA-v1`作为当前
paper-core组合，明确Contribution 1（ISCF architecture）与Contribution 2（BSCA balanced co-adaptation）的依赖关系、
limitations和ablation matrix；完成叙事后再冻结modern baselines与paper-facing main table。任何后续test-informed redesign都必须
成为新candidate version。

## Artifacts

- `primary/summary.json`：machine-readable gate decision。
- `primary/comparison_cells.csv`：validation/test逐cell指标。
- `primary/internal_health.csv`：candidate/control internal health。
- `primary/run_audit.csv`：15-run protocol、hash、pairing与nonmutation audit。
- `remote_records/confirmation_training_launch_record.txt`：training launch provenance。
- `remote_records/confirmation_formal_test_launch_record.txt`：formal-test launch provenance。
