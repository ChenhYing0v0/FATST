# H5D ETTh1 Interaction Search: History Analysis and Prelaunch Gate

## 1. Decision

H5D冻结为48个new ETTh1 seed2021 profiles，保持ISCF-BSCA architecture、objective、
H720-prefix inference graph与validation-only checkpoint selector不变。它只搜索现有H5C/H5B
证据尚未覆盖的高信息interaction：

1. `dropout=0 × actual batch size × learning rate`；
2. H192-adjacent `p19 geometry × dropout=0`；
3. H336-frontier `p21 geometry × dropout=0`；
4. `mode rank × dropout=0`及其与p19/p21 geometry的interaction。

User authorization覆盖local H5D protocol patch与remote train/validation。Formal test、profile
selection、extra seeds、H5E、architecture/objective change及table mutation均未授权；48/48
checkpoint manifest闭合后必须重新停在formal-test gate。

## 2. Completed-history audit

Canonical history audit=
`analysis/iscf_bsca_main_v1_hpo_20260731/h5d_etth1_interaction_search_20260813/history_analysis/`。

| Phase | ETTh1 profiles |
| --- | ---: |
| H1 | 2 |
| H2 | 3 |
| H4J | 2 |
| H4K | 2 |
| H5A | 16 |
| H5B | 36 |
| H5C | 54 |
| Total | 115 |

115 profiles对应460/460 complete standard-horizon cells。Main II external best thresholds按共同
三位小数口径为：H96 `0.350/0.386`、H192 `0.375/0.402`、H336
`0.393/0.416`、H720 `0.452/0.461`，前者/后者分别为MSE/MAE。

### 2.1 Confirmed observations

- H5C `h5c_do0`是115-profile four-H mean-MSE frontier：MSE/MAE=
  `0.390725/0.417006`，但Main II best仍为4/8。相对H5B，dropout从0.1降至0
  改善H192/H336 MSE和H720两项，仅H96 MAE轻微退化。
- `h5c_ctx589_p19`是H192-adjacent geometry，其H192 MSE/MAE=
  `0.377097/0.403010`；H96两项与H720 MSE仍为best，因此它适合尝试增加一个H192
  cell。
- `h5c_ctx630_p21`是H336 joint frontier，其H336 MSE/MAE=
  `0.391205/0.416439`，共同三位小数下两项均best，但它牺牲H96性能。
- `h5c_rank160`是历史上唯一把H192 MAE降入best threshold的profile
  (`0.401656`)，同时明显破坏aggregate和H720，说明rank不能再作single-factor结论，
  只能与dropout0/geometry联合测试。
- H5C weight-decay series形成窄平台；H5A/H5B的`d16_ff16`、`d32_ff64`、
  `d48_ff48`均明显负向。继续加密weight decay或capacity的预期信息增益低。
- H5C best epoch范围为1--3。更长training budget不会增加epoch-boundary sampling density，
  因此当前没有证据支持把budget extension作为主rescue axis。
- 115个ETTh1 profiles的actual batch size全部为32。Batch size会同时改变gradient noise、
  每epoch optimizer steps和epoch-level validation位置，是尚未覆盖的optimization axis。

### 2.2 Inference and self-critique

[Strong Evidence] `dropout=0`应替代H5B dropout0.1，作为H5D交互搜索的base。

[Hypothesis] Batch-size/LR coupling可能移动H192 performance frontier；p19/p21与high-rank的
interaction可能把rank160的H192-MAE优势与dropout0的aggregate/H720优势结合起来。

这些判断来自非正交、逐阶段test-informed search，不能当作causal hyperparameter importance。
H5D的作用是用显式interaction matrix检验上述假设，而不是宣称已定位global optimum。连续访问
official test也会提高benchmark overfitting风险，最终结果必须披露为single-seed、test-tuned和
test-informed。

## 3. Frozen 48-profile matrix

| Group | Runs | Purpose |
| --- | ---: | --- |
| `dropout0_batch_lr_interaction` | 12 | 首次覆盖batch 16/24/48/64及相应LR范围 |
| `h192_p19_dropout0_interaction` | 8 | p19局部geometry、LR及batch interaction |
| `h336_p21_dropout0_interaction` | 8 | p21局部geometry、LR及batch interaction |
| `dropout0_rank_refinement` | 8 | p20/do0下rank 136--192细化 |
| `h192_geometry_rank_dropout0` | 8 | L589/p19/do0 × rank 136--192 |
| `h336_geometry_rank_dropout0` | 4 | L630/p21/do0 × rank 144/160/176/192 |

所有jobs与115个historical effective fingerprints零重复。固定项为`d_model=d_ff=32`、
weight decay 0.01、LayerNorm on、canonical scopes/partition、from-scratch joint training、
seed2021、120 epochs/patience24。该matrix不是Cartesian product。

## 4. Selection, success and rollback

- Training只写validation artifacts，official test jobs=`0/48`。
- 每trial checkpoint由validation `{96,192,336,720}` mean MSE选择。
- Future formal test若另获授权，必须完整覆盖48 checkpoints × four H=`192` rows。
- 一个dataset-level profile必须同时服务四H；禁止per-H/per-metric/per-cell/per-seed选择。
- Primary gate：Main II ETTh1 best从4/8提高到至少5/8；stretch=6/8。
- Eligibility：four-H mean MSE与MAE分别不超过H5B `1.002x`。
- 若没有eligible best-cell improvement，保留H5B `h5b_seq640_p20`且不改表。
- Resource/numeric failure只允许修复受影响runtime block；architecture/objective变化必须建立新
  candidate并重过narrative/design gate。

## 5. Resource and scheduling gate

- 预计4--10 GPU-hours，三张3090 wall time约1.5--4.0小时。
- 新增核心artifacts预算不超过6 GiB；remote smoke前先审计quota，并仅清理可重建的旧
  resource-smoke和nonselected dense diagnostics。
- 三GPU queue先放batch16 slow jobs，再按context length调度，最后填充batch48/64 jobs，
  避免fast GPU空闲。
- Resource smoke必须48/48通过，检查OOM/NaN/Inf/Traceback/RuntimeError、artifact completeness、
  unique hashes及test=0后，才可启动full train/validation。

## 6. Local prelaunch status

- JSON parse：pass；
- Python compile：pass；
- shell syntax：pass；
- 48 jobs / 48 order IDs / six group counts：pass；
- 115 historical profiles / 0 effective duplicates：pass；
- frozen evidence hashes：pass；
- generic runner dry-run：48 jobs、test=0、remote authorization=true；
- formal-test authorization=false；automatic table mutation=false。

Decision=`H5D_48_profile_interaction_matrix_frozen_remote_train_validation_authorized_formal_test_pending`。
