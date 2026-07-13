# StageC D1-v2 Research Interpretation

## Decision Summary

| Field | Decision |
| --- | --- |
| `current_step` | Step 2-3 completed；进入conditional Step 4-6 |
| `SC1-PMFO` | `problem_gate_passed_narrative_pending` |
| `SC2-PIR` | `problem_gate_passed_but_measure_conditional` |
| `method_training_authorized` | `false` |
| `encoder_decision` | 保留A6 Encoder作为首个PMFO carrier，不宣称其已是最终multiresolution Encoder |
| `basis_decision` | 不原样保留single dense learned basis作为PMFO核心；Step 4-6评估nested/localized replacement |
| `rollback` | novelty/theory gate失败 -> Step 2 problem reformulation；不得直接实现或叠加Encoder/MoE |

## What D1-v2 Establishes

### PMFO problem evidence

[Strong Evidence] evaluation-space future deviation与frozen-A6 residual在rank 144下均显示跨dataset、跨seed的
structured capture：相对fixed random rank control的advantage分别为label `0.6600-0.7657`、residual
`0.6115-0.7462`。这说明A6未解释的误差仍有稳定coarse-to-fine temporal structure，PMFO不是只在raw
label smoothness上成立。

[Strong Evidence] frozen A6相对zero-future-deviation baseline在validation子集取得R2
`0.2164/0.3339/0.2284`（Weather/ETTm1/ETTh2）；patch shuffle或collapse令SSE相对增加
`0.3800/0.6834/0.3603`。因此当前有序`memory: [B,C,P,D]`确实承载被frozen head使用的信息，首轮PMFO
无需同步更换Encoder。

[Risk] ETTh2 coarse/mid linear probe仍为`R2=-0.9500`，而Weather/ETTm1为`0.1389/0.3077`。这不是
Encoder方向失败，因为frozen nonlinear head在ETTh2仍有positive R2；但它表明“现有memory可被简单
multiscale readout线性重组”并未跨dataset成立。若PMFO在ETTh2失败，应回滚检查最小history-scale
interface，不能先归因于future operator。

### Current basis headroom

[Fact] 同为rank 256，current learned-basis subspace的label capture为`0.7819-0.8199`，DCT为
`0.9029-0.9777`；learned-basis residual capture仅`0.7201-0.7478`。同时basis effective rank约
`190-212`、column entropy约`0.889`、90% energy support约占时间域`44.5%`，且没有nested/refinement
constraint。

[Inference] current basis并非容量崩溃，但其geometry没有主动服务PMFO叙事，并在本diagnostic projection
criterion下留有明显headroom。因此应保留A6 coefficient/readout作为matched control，而不是把原basis直接
包装成PMFO。DCT capture高也不能单独证明DCT应成为最终method；它可能主要利用了时序smoothness。

### PIR problem evidence and boundary

[Strong Evidence] evaluation-space Parseval invariant在全部dataset/seed通过；改变deployment horizon
measure相对delta-720产生显著raw gradient direction变化。uniform/log-uniform/benchmark的`1-cos`分别为：

| Dataset | Raw uniform | Raw log | Raw benchmark |
| --- | ---: | ---: | ---: |
| Weather | 0.2064 | 0.7890 | 0.2575 |
| ETTm1 | 0.1891 | 0.6223 | 0.2137 |
| ETTh2 | 0.1354 | 0.8279 | 0.2062 |

[Strong Evidence] projected increments相对same-measure raw weighting的额外gradient separation在
log-uniform下很强（`0.2895-0.4059`），uniform下较小但跨dataset超过0.005（`0.0066-0.0117`）。

[Boundary] benchmark measure下的projected excess只有`0.0011-0.0038`，三套dataset均未达到0.005。
所以D1支持的是“PIR在broad continuous horizon measure下可能提供额外训练信号”，不支持“PIR对任意
measure都超越raw weighting”。Step 4-6必须明确deployment measure，并把raw uniform/log weighting设为
mandatory control；不能用log-uniform结果替代benchmark-horizon证据。

## Failure Attribution

D1-v1的问题属于`measurement_or_gate_fault`：negative R2之间的差值可形成false pass，history-std
normalization又使低方差窗口主导residual。v2保持checkpoint、dataset、seed与batch budget不变，仅修复
source/gradient space、gate和frozen counterfactual。v2 forward reconstruction max gap为0，Parseval
invariant通过，未发现可使方向结论失效的numeric pathology。

## Self-Critique

- DCT/block/random是offline geometry controls，不是PMFO method；高capture可能部分来自一般时序smoothness。
- frozen shuffle/collapse是对当前head的OOD counterfactual，混合了memory information与head positional
  dependence；它只支持“先保留Encoder”，不能证明Encoder具备完备multiresolution sufficient statistics。
- ridge只取固定前若干batches，ETTh2 negative R2可能受distribution shift影响；因此它已降级为辅助风险信号。
- gradient direction separation证明objective不同，不证明训练后MSE/MAE一定改善；effectiveness仍完全未知。

## Next Research Plan

1. `SC1-PMFO Step 4-6`：专项审计multiresolution analysis、lifting scheme、multiwavelet/neural operator与
   arbitrary-horizon functional decoder，形成novelty matrix。
2. 给出nested space、restriction/refinement identity、local-support complexity与tensor contract；候选必须
   保持H只定义output domain，不进入learned coefficient path。
3. 首轮设计复用`memory [B,C,P,D]`，通过最小scale readout构造coefficients；不同时更换backbone。current
   dense basis仅作matched control。
4. `SC2-PIR Step 4-6`可与PMFO理论并行，但method implementation应在operator contract稳定后进行；明确
   L2 exact decomposition、Huber/L1 boundary和deployment measure。
5. 预注册controls：A6 dense learned basis、fixed DCT、no-refinement、matched parameters/FLOPs、raw
   uniform/log horizon weighting、benchmark-only weighting。
6. narrative gate通过后先做单dataset/seed最小implementation；PMFO与PIR分别过effectiveness gate，再做
   `2x2` factorial，最后才进入3 datasets x 3 seeds full matrix。

## 11-Step Record

| Field | Record |
| --- | --- |
| `current_step` | Step 2-3 complete；Step 4-6 active |
| `problem` | dense global future basis缺nested refinement/local support；continuous-horizon risk与operator unit不一致 |
| `existence_evidence` | 3 datasets x 3 seeds structure、frozen counterfactual、basis geometry、gradient audit |
| `idea` | domain-only H + nested future coefficients + projective increment risk |
| `theory_check` | L2 Parseval implementation成立；restriction/refinement proof与L1/Huber边界待完成 |
| `design` | Step 4-6 prior-art/theory/control preregistration；尚无method implementation |
| `narrative_gate` | pending |
| `effectiveness_gate` | pending；D1不构成performance evidence |
| `artifacts` | `d1_diagnostic_report.md`、`d1_dataset_gate.csv`、raw per-seed CSVs |
| `decision` | 两项problem gate通过；PIR具有measure conditional boundary；不授权training |
