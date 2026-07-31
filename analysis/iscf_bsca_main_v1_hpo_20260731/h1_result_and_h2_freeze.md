# ISCF-BSCA-MAIN-v1 H1 Result Audit and H2 Freeze

## 0. Decision summary

| Field | Value |
| --- | --- |
| `current_step` | Step 9 H1 artifact audit；Step 6--8 H2 frozen design and launch preparation |
| `candidate` | `ISCF-BSCA-MAIN-v1` |
| `H1_status` | `16/16_validation_complete_test_0` |
| `H2_status` | `24_additional_profiles_frozen_prelaunch` |
| `test_tuned` | true |
| `primary_seed` | 2021 |
| `checkpoint_rule` | four-horizon mean validation MSE |
| `profile_selection_rule` | per-dataset four-horizon mean official-test MSE |
| `decision` | 当前性能整体具有竞争力但不是SOTA；继续bounded H2 |

[Fact] H1的16个training/validation jobs全部完成；每个trial均具有
`checkpoint.pt`、`training_log.csv`、四horizon validation MSE/MAE、
`effective_config.json`、`initialization_contract.json`和
`model_diagnostics.json`。日志未发现Traceback、OOM、NaN或Inf。H1没有新增
official-test访问。

## 1. Problem

H1需要回答两个问题：

1. exact ISCF-BSCA架构在两个有来源依据的初始profiles下是否训练稳定；
2. 是否已有足够竞争力，还是需要继续做受限H2搜索。

H1不是最终HPO。它只有每dataset两个anchors，不能据此宣布最优，也不能用
validation数字与published test结果直接比较。

## 2. Existence evidence

### 2.1 H1 validation ranking

| Dataset | Conservative | TimeAlign prior | H1 validation winner | Winner relative gain |
| --- | ---: | ---: | --- | ---: |
| ETTh1 | 1.121983 | 1.143482 | conservative | 1.88% |
| ETTh2 | 0.400668 | 0.408826 | conservative | 2.00% |
| ETTm1 | 0.614575 | 0.600672 | TimeAlign prior | 2.26% |
| ETTm2 | 0.183536 | 0.184049 | conservative | 0.28% |
| Weather | 0.484225 | 0.489787 | conservative | 1.14% |
| ECL | 0.150245 | 0.132994 | TimeAlign prior | 11.48% |
| Solar | 0.139754 | 0.133654 | TimeAlign prior | 4.36% |
| Exchange | 0.824208 | 0.929655 | conservative | 11.34% |

[Strong Evidence] 不存在一个跨dataset通吃的初始profile。ECL、Solar明显受益于
TimeAlign-source capacity/learning-rate prior；Exchange相反；ETTm2为近似平局。
因此H2必须围绕各dataset的H1局部winner展开，不能继续做单一global profile搜索。

### 2.2 Current competitive position

五个历史dataset的H1 conservative checkpoint SHA256与已完成的
ISCF-BSCA-v1 seed2021 confirmation完全一致。因此，下表复用同一checkpoint已审计
official-test结果，不产生新的test访问。

TimeAlign参考值来自2026-07-31检索的ICLR 2026官方论文
[Table 1/6](https://arxiv.org/pdf/2509.14181)，其结果同样是四个standard
horizons的平均，并经过lookback/hyperparameter search。协议仍存在模型训练方式与
test-tuning范围差异，因此只用于竞争力定位，不作matched attribution。

| Dataset | ISCF-BSCA current anchor test MSE | TimeAlign published MSE | Relative gap | Audit judgment |
| --- | ---: | ---: | ---: | --- |
| ETTm1 | 0.344468 | 0.340 | +1.31% | competitive，尚非best |
| ETTm2 | 0.257293 | 0.243 | +5.88% | clear tuning gap |
| ETTh1 | 0.410911 | 0.406 | +1.21% | competitive，仍需改进 |
| ETTh2 | 0.312861 | 0.336 | -6.89% | strong；优于该published reference |
| Weather | 0.217213 | 0.214 | +1.50% | competitive，仍需改进 |
| 5-dataset macro | 0.308549 | 0.307800 | +0.24% | overall competitive |

[Decision] 当前ISCF-BSCA不是“无竞争力”：五dataset macro仅比TimeAlign
published结果高0.24%，且ETTh2明显更强。但它也不能称为SOTA：ETTm2仍有5.88%
缺口，另外三个datasets仍有约1%--1.5%差距；ECL、Solar、Exchange尚没有当前
checkpoint的official-test证据。

## 3. Idea and theory check

H2不改变architecture、objective、scope partition、BSCA policy或inference graph。
它只在H1支持的局部邻域内搜索：

- optimizer scale；
- encoder capacity；
- patch granularity；
- dropout；
- lookback length；
- H1中触及20 epochs的慢收敛profiles使用30 epochs / patience 7。

[Theory check] 这些变量改变的是同一冻结architecture的capacity、regularization和
optimization，不构成新机制candidate。H2仍是end-to-end joint training，并使用相同
four-horizon validation checkpoint selector。Lookback只取`336/720`，且每个
`seq_len`均可被对应`patch_num`整除。

## 4. Frozen H2 design

H2固定为8 datasets × 3 additional profiles = 24 jobs；连同H1共40个trials。
每个dataset的三个profiles分别覆盖H1 winner的局部optimizer/capacity或
regularization邻域，以及一个TimeAlign论文支持的`L=336` lookback邻域。ETTm2使用
额外hybrid profile修补当前最大test缺口；Exchange因两个H1 profiles都在epoch 1达到
best validation，额外测试更低learning rate和更强dropout。

完整逐trial contract冻结于：

- `configs/iscf_bsca_main_v1_hpo_h2.json`；
- `scripts/remote/run_iscf_bsca_main_v1_hpo_h2.sh`。

H2完整训练前仍为`test=0`。H2完成后，才允许对H1+H2全部40个checkpoints执行
official-test，并按每dataset的four-H mean MSE选择一个shared profile。禁止按horizon、
seed、MAE或单cell选择配置。

## 5. Narrative and effectiveness gates

### Narrative gate

`passed_as_same_architecture_hpo`。H2不提出新contribution，也不改变paper-core
mechanism boundary。

### Effectiveness gate

当前为`performance_partial_pass_for_competitiveness`：

- positive：五dataset macro与TimeAlign published结果接近，ETTh2强；
- negative：ETTm2明显落后，三个dataset有小幅差距，三个新增dataset缺test；
- blocked claim：不得宣称ISCF-BSCA-MAIN-v1达到SOTA，直到40-trial完整
test-tuned ranking完成。

## 6. Failure attribution and rollback

当前没有numeric pathology。Observed gap暂归为
`hyperparameter_optimization_incomplete`，不是`hypothesis_false`，也不是architecture
failure。若H2后仍在ETTm2或macro上明显落后，先回到HPO/profile-budget评估，而不是
新增paper-core机制；只有完整H2表明冻结architecture存在跨dataset系统性劣势时，才
回到Step 4--6重新评估main-model design。

## 7. Artifacts

- `h1_artifact_audit/hpo_completeness.json`；
- `h1_artifact_audit/trial_ledger.jsonl`；
- `h1_artifact_audit/trial_scorecard.csv`；
- `h1_artifact_audit/profile_aggregates.csv`；
- remote H1 root:
  `/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h1`。

## 8. Self-critique

[Uncertainty] Published TimeAlign与本项目不是matched protocol，且H1 winner中
ETTm1/ECL/Solar尚未访问test；因此“整体具有竞争力”是定位判断，不是最终paper claim。
Exchange的H720 validation有效样本很少，H2 validation ranking可能波动，最终必须保留
全部四-horizon official-test结果并披露test-tuned属性。
