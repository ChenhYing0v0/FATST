# ISCF-BSCA-MAIN-v1 HPO Tooling

## 1. Role and Boundary

本工具链服务paper-facing `ISCF-BSCA-MAIN-v1`超参数调优，不修改冻结的
ISCF-BSCA architecture forward path，也不替换exact `ISCF-BSCA-v1` ablation
anchor。H0 dataset audit和H1 two-anchor matrix已经完成；H2已冻结为每dataset
三个additional profiles，因此每dataset共有五个total trials。

核心选择顺序固定为：

```text
train one trial
-> validation four-H mean MSE selects checkpoint
-> freeze checkpoint hash
-> complete all frozen trials
-> official-test four-H mean MSE ranks profiles per dataset
-> freeze one profile shared by H={96,192,336,720}
```

Official test不得选择epoch、checkpoint、seed或per-horizon profile。

## 2. Dataset Support

`baselines/timealign_official/train_repo.py`的`OFFICIAL_PRESETS`新增：

- ECL：`Dataset_Custom`，321 channels，hourly；
- Solar：`Dataset_Solar`，137 channels；
- Exchange：`Dataset_Custom`，8 channels，daily。

这些preset只提供可构造的source-informed defaults；最终main profile仍由HPO
contract决定。`Dataset_Solar.__getitem__`同时修正`seq_y_mark`长度，使其与
`seq_y`而不是`seq_x`一致。

`scripts/audit_iscf_bsca_paper_datasets.py`在任何training前记录：

- exact file path、SHA256、bytes、rows和channels；
- CSV的`date`/`OT` contract、timestamp monotonicity、duplicates和cadence；
- NaN/Inf及constant channels；
- chronological split boundaries与train-only scaler fit rows；
- 可选train/validation loader batch shapes。

该audit不构造test loader，也不计算test metrics。

## 3. H1/H2 Tensor and Training Flow

每个job从历史窗口

$$
X \in \mathbb{R}^{B\times720\times C}
$$

开始。`timealign-token-mlp` encoder由profile中的`patch_num`、`d_model`、
`d_ff`、`dropout`和`layer_norm`控制；`siff-independent-scope-control`
readout沿用五个sharing scopes和dataset/job指定的`mode_rank`。最终一次生成

$$
\hat Y \in \mathbb{R}^{B\times720\times C},
$$

再以full-crop读取H96、H192、H336和H720。

H1包含8 datasets × 2 source-audited anchors：

- `h1_conservative_anchor`：原5 datasets复用exact-v1 natural profile；新3
  datasets使用bounded conservative start；
- `h1_timealign_source_prior`：使用对应TimeAlign official encoder/profile
  setting；Exchange使用已披露的ETTh1-derived bootstrap。

所有job固定seed2021、BSCA objective、five scopes、canonical partition和
joint end-to-end training。差异只来自config中显式记录的hyperparameters。

H2的24个jobs以H1 job作为`base_trial_id`，再通过显式`overrides`冻结局部邻域。
Runner在启动时materialize完整job；profile hash基于materialized job，而不是仅对
override计算。H2允许的变化只有lookback、patch count、encoder capacity、dropout、
learning rate和固定的30-epoch/patience-7预算。Architecture invariants保持不变。

## 4. Provenance and Optimizer

`train_repo.py`新增以下provenance arguments，它们进入
`effective_config.json`：

- `hpo_trial_id`；
- `hpo_profile_id`；
- `hpo_profile_hash`；
- `hpo_config_hash`；
- `hpo_search_space_hash`。

`AdamW`原先隐式使用PyTorch默认`weight_decay=0.01`。现在通过
`--weight-decay`显式连接到optimizer并记录；默认值仍为`0.01`，因此旧命令语义
不变。

## 5. Runner and Artifacts

`scripts/remote/run_iscf_bsca_main_v1_hpo.sh`同时支持H1和H2 config：

- `MODE=dry-run`：输出phase-specific frozen manifest和hash；
- `MODE=data-audit`：执行新三dataset H0 audit；
- `MODE=resource-smoke`：执行two-batch construction/resource canary；
- `MODE=train`：执行phase-specific full-budget train/validation；
- `MODE=status`：统计完整job。

H2通过`scripts/remote/run_iscf_bsca_main_v1_hpo_h2.sh`固定config和repo-external
output root。`CANARY_ONLY=1`在H1为6 jobs，在H2为9 jobs。Runner采用global
shared queue，空闲GPU领取剩余最长job，不按dataset或arm静态配对。每个H2 job的
`max_epochs`和`early_stopping_patience`来自materialized config，不再由runner
硬编码。

每个trial目录至少包含：

- `checkpoint.pt`；
- `training_log.csv`；
- `metrics_by_target_horizon.csv`；
- `effective_config.json`；
- `initialization_contract.json`；
- `model_diagnostics.json`。

## 6. Analysis and Selection

`scripts/analyze_iscf_bsca_main_v1_hpo.py`能解析H1完整jobs或H2
`base_trial_id + overrides` profiles，只接受完整standard horizons，输出：

- `trial_ledger.jsonl`；
- `trial_scorecard.csv`；
- `profile_aggregates.csv`；
- `hpo_completeness.json`。

只有使用`--require-test`且全部frozen trials都有完整test scorecard时，才继续
生成`profile_ranking.csv`和`selected_profiles.json`。排序依次使用：

1. lower four-H mean official-test MSE；
2. lower four-H mean validation MSE；
3. lower trainable parameter count；
4. lexical `profile_id`。

MAE完整报告，但不参与默认selector。

## 7. Code-Theory Consistency

Intended theory是：main-table model应在保持ISCF-BSCA computation contract不变
的前提下，通过dataset-level encoder/optimization profile释放架构性能。代码通过
固定readout mode、objective、scopes、partition和training path，仅搜索显式
hyperparameters来实现这一边界。

仍然只是proxy的部分：

- H1只有两个anchors，不能宣称已经找到最优profile；
- TimeAlign设置只是source prior，不保证适配ISCF-BSCA；
- 新dataset的frequency/source identity必须由remote H0 file audit确认；
- test-tuned结果不构成untouched-holdout generalization estimate。

若H2出现OOM/NaN/Inf、validation checkpoint provenance不完整、base-config hash
不一致，或trial hash不一致，则H2 gate失败并回到local protocol/resource repair。
H2 24/24完成之前不得进入official-test ranking。
