# StageC D1 PMFO/PIR Problem-Existence Diagnostic Protocol

## Status

| Field | Value |
| --- | --- |
| `candidate` | `SC1-PMFO` / `SC2-PIR` |
| `role` | `diagnostic_only` |
| `current_step` | Step 2-3 |
| `method_training_authorized` | `false` |
| `carrier` | frozen `A6-LBF-natural-baseline` |
| `data_boundary` | train用于结构/probe fitting/gradients；validation只评估fixed probes/counterfactual；test=false |
| `rollback` | stable diagnostic fails -> exact problem candidate returns Step 2 |

## Corrected A6 Contract

[Fact] natural profiles都产生`memory: [B,C,P,D]`与`hidden: [B,C,768]`：Weather/ETTh2为
`P12×D64`，ETTm1为`P24×D32`。A6再计算`coeff: [B,C,256]`，并使用
`basis[:H]: [H,256]`直接输出`[B,H,C]`。

[Correction] A6不需要先生成H720再裁剪；它已经domain-only且output matmul随H变化。PMFO必须证明的
不是这项已有性质，而是nested refinement、local support和operator-aligned risk。Encoder和当前basis
都不被预设为最终组件。

## D1-A: Future Structure

在evaluation space中比较`future_deviation = y - history_mean`与
`residual = y - frozen_A6(x)`。该定义保留A6 decoder真正需要解释的future deviation，同时避免按每个
window的history std除法把低方差样本放大；dataset loader已有train-fitted channel scaling。

- nested DCT ranks `{8,24,72,144,256}`；
- nested localized block spaces，block sizes `{90,30,10,5,1}`；
- fixed random orthogonal rank control；
- frozen learned-basis rank-256 subspace。

Metrics：

- `increment_energy_share`: 本层正交increment能量 / source总能量；
- `cumulative_energy_share`: 截至本层可重构能量比例；
- `reconstruction_error_share`: `1-cumulative_energy_share`。

Gate：在shared rank 144处，`max(DCT, block)-random`：label至少`0.10`、residual至少`0.02`，分别要求
至少2/3 datasets通过。该gate同时要求3 seeds汇总，不能由单seed或单dataset触发。

## D1-B: Encoder Information Sufficiency

冻结forecast model，只拟合closed-form ridge probes，不更新Encoder或decoder。使用固定
`ridge_lambda=0.01`、train前8 batches拟合、validation前4 batches评估。四种feature：

1. `full_hidden`: `[P,D] -> [768]`；
2. `patch_mean`: `[P,D] -> [D]`；
3. `patch_shuffled`: 每个sample-channel独立随机patch permutation后flatten；
4. `raw_history`: 同channel normalized 720-step history，作为linear recoverability control。

Probe targets为label/residual的DCT coefficients与localized block coefficients。统计每个increment level的
validation `R2`、`NRMSE`、`SSE/SST`。

Linear probe只作recoverability辅助量。其严格有效条件为对label DCT coarse/mid levels
（cumulative rank<=72），`R2(full_hidden)>=0.05`且
`R2(full_hidden)-R2(patch_shuffled)>=0.01`；负R2之间的差值不得形成pass。

Primary Encoder gate使用frozen decoder counterfactual：

1. 用原始有序`memory`、per-sample patch shuffle、patch-mean collapse分别通过同一frozen LBF head；
2. 原始decoder必须相对zero-future-deviation baseline取得positive R2；
3. shuffle或collapse至少使SSE相对原始path增加1%；
4. direct memory decode与正式`forward`的max absolute gap必须`<=1e-5`。

至少2/3 datasets通过才允许“先保留Encoder、只重构decoder”。该counterfactual只证明当前head利用了有序
patch memory，不等价于Encoder已经产生完备multiresolution sufficient statistics。
失败时不能立即判定PMFO失败：若D1-A
通过而D1-B失败，应进入Step 2评估最小multiscale history interface，并保持Encoder改造为辅助贡献。

## D1-C: Basis Geometry And PIR Gradients

### Basis audit

对每个frozen checkpoint的`learned_temporal_basis: [720,256]`报告：

- singular effective/stable rank与condition number；
- normalized temporal entropy与90% energy support fraction；
- first-difference energy；
- 与DCT ranks及localized block rank-144 subspace的principal overlap；
- label/residual projection capture。

[Boundary] 当前basis按构造没有nested/refinement constraint。高capture只能说明容量可能足够，不能说明
PMFO已存在；低capture则说明subspace容量/几何本身也可能是瓶颈。

### PIR gradient audit

固定train前2 batches，使用evaluation-space squared error比较四种deployment measures：

- `delta_720`；
- `uniform_h` over all integer H；
- `log_uniform_h`；
- `benchmark_h={96,192,336,720}` control。

每种measure计算raw step risk和localized nested-block projected-increment risk，记录encoder、coefficient、
basis与all-active-path gradients。`delta_720`下orthogonal increments必须满足Parseval：loss relative gap
`<=1e-4`且gradient cosine`>=0.9999`，否则诊断无效。

PIR problem gate要求至少2/3 datasets同时满足：

- nonuniform raw measure相对delta-720的mean gradient separation `1-cos >= 0.005`；
- projected risk相对same-measure raw risk的excess separation `1-cos >= 0.005`。

若只出现第一项，说明horizon measure改变risk但PIR没有超越raw weighting，判定
`simple_measure_alignment_only`。

## Decision Matrix

| D1-A | D1-B | D1-C/PIR | Decision |
| --- | --- | --- | --- |
| pass | pass | PMFO geometry headroom | 保留A6 Encoder，进入PMFO Step4-6 decoder design |
| pass | fail | any | 回Step2设计最小multiscale history interface，不同步换全backbone |
| fail | any | any | PMFO当前problem formulation失败，不实现 |
| any | any | PIR pass | PIR可进入独立Step4-6，即使PMFO需回滚 |
| any | any | only raw measure differs | horizon measure只保留为protocol，不作为Contribution 2 |

## Failure Attribution

projection重构/Parseval、ridge solve或gradient extraction出现数值异常时，标记
`diagnostic_invalid_for_direction_rejection`。Stable D1失败只否定当前PMFO/PIR problem formulation；不得
扩大为所有multiresolution decoder、Encoder redesign或training strategy的方向级拒绝。

## Protocol Amendment: v1 -> v2

2026-07-13初次remote run已完整保留在
`analysis/stage_c_d1_pmfo_pir_offline_20260713/`，但标记
`diagnostic_invalid_for_direction_rejection`：

- ETTh2 `full_hidden R2=-39.7831`，旧gate却因shuffled更差而误判pass；
- Weather/ETTh2的history-std normalized residual几乎等于label，说明source space被低方差窗口主导，
  没有可靠隔离A6已解释后的residual；
- 因此旧summary中的`pmfo_pir_problem_gate_passed`作废，不能进入论文claim。

v2只修复measurement space与gate，并加入frozen-decoder counterfactual；candidate、checkpoint、dataset、
seed、batch budget和阈值方向均不因结果调参。v1保留为failure-attribution evidence，v2使用独立output root。
