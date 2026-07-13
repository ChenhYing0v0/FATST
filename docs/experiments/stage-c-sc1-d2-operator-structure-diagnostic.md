# SC1-D2 Frozen-Memory Operator-Structure Diagnostic Protocol

## Status

| Field | Value |
| --- | --- |
| `candidate` | `SC1-D2` |
| `role` | `diagnostic_only` |
| `current_step` | Step 2/3 active |
| `problem` | frozen ordered memory是否包含超出rank expansion与generic nonlinearity的true-scale conditional structure？ |
| `carrier` | frozen `A6-LBF-natural-baseline` Encoder checkpoints |
| `formal_suite` | ETTh1, ETTh2, ETTm1, ETTm2, Weather × seeds 2021/2022/2023 |
| `current_precheck` | ETTh2, ETTm1, Weather；ETTh1/ETTm2 profiles pending |
| `test_used` | `false` |
| `forecast_model_updated` | `false` |
| `method_training_authorized` | `false` |
| `decision_boundary` | core3只能precheck；formal5才可支持或重定义problem |

## 1. What We Plan To Test

Step 6证明linear FPMO-DS与full-affine DA具有同一function class。D2不再问“换一种future basis是否有效”，
而是依次隔离三个问题：

1. `full affine > rank256 affine`：A6 readout是否受rank/capacity限制；
2. `dense nonlinear > full affine`：是否存在generic nonlinear memory-to-future mapping；
3. `true-scale grouped nonlinear > strongest dense nonlinear + random controls`：nonlinearity是否真的与
   balanced interval scales对齐。

D2只训练probe heads。Encoder、A6 checkpoint、data split与dataset profile全部冻结；MIPR、MoE、horizon
embedding和future router均不进入本轮。

## 2. Tensor And Data Construction

对每个frozen checkpoint，A6 history normalization为

$$
x^{(n)}=\frac{x-\mu_x}{\sigma_x},
$$

Encoder输出

$$
M\in\mathbb R^{B\times C\times P\times D},\qquad
h=\operatorname{vec}(M)\in\mathbb R^{BC\times768}.
$$

probe target为同一history statistics下的full future：

$$
u=\frac{y-\mu_x}{\sigma_x}\in\mathbb R^{BC\times720}.
$$

训练loss在evaluation space计算。对于time-space head，error为$\widehat u-u$；对于orthogonal coefficient
head，$\alpha=uQ^\top$。由于每个sample-channel的$\sigma_x$是scalar且$Q$正交，

$$
\|\sigma_x(\widehat u-u)\|_2^2
=\|\sigma_x(\widehat\alpha-\alpha)\|_2^2.
$$

因此coefficient heads可直接在coefficient space训练，validation时再通过$\widehat u=\widehat\alpha Q$
还原，不产生不同loss的confound。

### Split boundary

- fixed `data_seed=20260713`，所有arms与同dataset checkpoints使用相同batch order；
- train前16 batches用于probe development；按**sample ID**而非channel row做80/20 fit/inner-holdout split；
- inner holdout用于early stopping；official validation前8 batches只作一次final evaluation；
- test split不加载；official validation不得用于选epoch、width、learning rate或random seed。

## 3. Probe Matrix

| Arm | Mapping | Parameters | Attribution role |
| --- | --- | ---: | --- |
| `rank256_linear` | `768 -> 256 -> 720`，无activation | 381,904 | A6 affine-family rank control |
| `full_affine` | `768 -> 720` | 553,680 | full-rank affine control |
| `dense_nonlinear_param_h197` | `768 -> GELU(197) -> 720` | 294,053 | parameter-matched dense nonlinear control |
| `dense_nonlinear_units_h352` | `768 -> GELU(352) -> 720` | 524,848 | same-total-hidden-units / stronger dense control |
| `true_scale_grouped` | 11个独立`768 -> GELU(32) -> n_l` | 294,448 | tested scale-aligned hypothesis |
| `random_group_s*` | true $Q$，随机把720 coefficients分成相同group sizes | 294,448 | row-group semantics control ×3 |
| `random_basis_s*` | random orthogonal $Q$，使用相同contiguous group sizes | 294,448 | basis semantics control ×3 |

true scale group sizes固定为
`[1,1,2,4,8,16,32,64,128,256,208]`；random control seeds固定为
`3101/3102/3103`。params差异不作为超参数选择依据；这里报告params只用于解释capacity和matched-control
边界。strongest dense control定义为同一dataset/checkpoint seed上两个预注册dense arms中validation MSE
较低者。这是对candidate的保守比较，不用于修改candidate。

## 4. Probe Optimization

- input feature只使用fit rows统计做z-score；
- 所有arms固定`AdamW(lr=1e-3, weight_decay=1e-4)`；
- `batch_size=1024`、`max_epochs=120`、`patience=15`；
- 相同probe initialization seed与epoch-wise fit permutation；
- 每个arm保存best inner-holdout state，official validation不参与early stopping；
- report fit、inner-holdout、normalized validation与evaluation-space validation MSE/MAE。

如果某类arm明显未收敛，D2必须标记optimization risk；不得把弱optimizer fit误写为function-class失败。

## 5. Metrics And Reading

对control $c$与candidate $m$定义

$$
\operatorname{gain}(m;c)=\frac{\mathrm{MSE}_c-\mathrm{MSE}_m}{\mathrm{MSE}_c}.
$$

- `full_vs_rank256_improvement`：只支持rank/capacity bottleneck；
- `dense_vs_full_improvement`：只支持generic nonlinear readout；
- `true_vs_dense_improvement`：true scale相对strongest dense control；
- `true_vs_random_median_improvement`：true scale相对6个random controls的median；
- `random_controls_beaten`：true scale validation MSE严格低于多少个random controls；
- `true_vs_dense_mae_improvement`：MSE gate之外的MAE guard。

## 6. Formal Five-Dataset Hard Gate

formal pass必须同时满足：

1. 五dataset × 三checkpoint seeds × 11 arms完整，且basis/Parseval/test/freeze invariants全部通过；
2. true scale相对strongest dense的macro MSE gain至少`0.5%`；
3. true scale相对random median的macro MSE gain至少`0.5%`；
4. 上述两项分别至少3/5 datasets有2/3 seeds为正；
5. 平均至少击败`5/6` random controls；
6. true scale相对strongest dense的macro MAE gain不得低于`-0.25%`。

结果解释：

| Observation | Decision |
| --- | --- |
| full affine only wins | `capacity_or_rank_problem_only` |
| dense nonlinear wins，true scale不胜 | `generic_nonlinearity_only` |
| true scale胜dense但不胜random | `grouped_parameterization_explains` |
| true scale同时稳定胜dense与random | problem supported；只允许返回Step 4提出新idea |
| core3任意结果 | `precheck_only`；不得formal pass或方向级reject |

## 7. Failure Attribution

- non-finite loss、>100%异常退化、basis/Parseval gap `>1e-5`：
  `optimization_or_numeric_pathology`，不得否定方向；
- 缺dataset/seed/arm，或official validation进入early stopping：diagnostic invalid；
- full/dense controls的fit与holdout显示明显未收敛：`optimization_protocol_suspected`；
- true scale只超过rank256：`capacity_control_explains`；
- true scale超过dense但不超过random：`grouped_parameterization_explains`；
- formal5 stable failure且controls有效：只否定“scale-aligned nonlinear final-head problem”，不得扩大为所有
  future-aware architecture、minimal Encoder interface或training strategy方向。

## 8. Current Execution Boundary

当前ETTh2/ETTm1/Weather拥有冻结的三seed natural checkpoints，先执行`core3_precheck`检查pipeline、
optimization与明显signal。ETTh1/ETTm2必须按five-dataset policy完成validation-only profile calibration后
再加入formal gate；不得从旧FSA或archive继承配置。

## 9. 11-Step Record

| Field | Record |
| --- | --- |
| `current_step` | Step 2/3 active |
| `problem` | scale-aligned conditional nonlinearity是否真实存在 |
| `existence_evidence` | D1 structure/memory pass；Step6 linear scale factorization fail |
| `idea` | none；本轮只做problem diagnostic |
| `theory_check` | orthogonal coefficient MSE与time-space MSE等价 |
| `design` | 11-arm frozen-memory probe matrix + formal5 hard gate |
| `narrative_gate` | not applicable to diagnostic |
| `effectiveness_gate` | not started |
| `artifacts` | worker metrics/history/metadata + pairwise/dataset/summary report |
| `decision` | core3 precheck next；method implementation unauthorized |
