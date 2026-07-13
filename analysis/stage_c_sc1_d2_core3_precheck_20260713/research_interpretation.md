# SC1-D2 Core3 Precheck Research Interpretation

## Decision Summary

| Field | Decision |
| --- | --- |
| `role` | `diagnostic_only / core3_precheck` |
| `matrix` | 3 datasets × 3 frozen checkpoint seeds × 11 head-only arms = 99/99 |
| `invariant_gate` | pass：test=false、forecast model frozen、validation不参与early stopping、basis/Parseval pass |
| `rank_signal` | not supported in core3：full affine vs rank256 macro `-0.5661%` |
| `generic_nonlinearity_signal` | not supported：best dense nonlinear vs full affine macro `-6.4492%` |
| `true_scale_vs_dense` | mixed：macro `+4.0358%`由ETTh2 dense overfit主导，只2/3 datasets方向为正 |
| `true_scale_vs_random_group` | fail in core3：macro `-0.2212%`，仅Weather稳定为正 |
| `true_scale_vs_random_basis` | positive precheck：macro `+2.3137%`，3/3 datasets、9/9 seeds为正 |
| `decision` | `partial_core3_basis_geometry_signal_only`；formal5 pending |
| `method_implementation_authorized` | `false` |

## 1. What Was Actually Tested

每个A6 Encoder checkpoint完全冻结，只训练从`h: [B*C,768]`到full future
`u: [B*C,720]`的probe head。所有arms共享数据、fit/inner-holdout、optimizer和official validation边界。
true-scale与六个random controls拥有相同group sizes、hidden width与parameter count。

完整raw metrics、training histories与metadata位于`raw/`；aggregate definitions见
`docs/experiments/stage-c-sc1-d2-operator-structure-diagnostic.md`。

## 2. Result Reading

### 2.1 Rank expansion is not the current common bottleneck

full-affine相对rank256的mean validation MSE gain：

- ETTh2：`+0.1288%`；
- ETTm1：`-0.8357%`；
- Weather：`-0.9915%`。

[Strong Evidence] 仅增加affine rank没有形成cross-dataset improvement。当前不能把A6问题收缩为
`rank256 capacity不足`。这也说明未来method只超过A6但未超过rank/full controls仍不能归因于scale mechanism。

### 2.2 Generic nonlinear head is not sufficient

strongest dense nonlinear相对full affine：ETTh2 `-20.6786%`、ETTm1 `+1.1714%`、Weather `+0.1597%`。

[Fact] ETTh2 dense nonlinear的fit/inner-holdout MSE低于full affine，但official validation MSE显著更差。
例如seed2021 parameter-matched dense的fit/holdout为`0.3890/0.6166`，full affine为
`0.5607/0.6820`，validation却从`0.7062`恶化到`0.8723`。

[Inference] 这不是“没训练好”，而是train-inner-holdout与temporal validation shift之间的generalization
failure。D2否定的是“加一个generic MLP即可解决readout问题”，不否定所有nonlinear operator。

### 2.3 Apparent true-scale-vs-dense gain is not a clean scale result

true scale相对strongest dense在ETTh2为`+12.5011%`，但它仍弱于full affine：三seed true-scale MSE约
`0.744-0.749`，full-affine约`0.700-0.715`。因此macro `+4.0358%`主要来自dense nonlinear在ETTh2的
validation overfit，而不是true-scale超过strong affine control。

ETTm1上true scale相对best dense为`-0.8040%`；Weather为`+0.4104%`。只有Weather提供完整的
`true > dense + random-group + random-basis`正向模式。

### 2.4 Basis geometry is positive; depth grouping is not

拆分mandatory random controls后：

| Dataset | True vs random-group median | True vs random-basis median | Group controls beaten | Basis controls beaten |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | `-1.0148%` | `+5.1077%` | `0.00/3` | `3.00/3` |
| ETTm1 | `-0.1263%` | `+0.6057%` | `1.33/3` | `3.00/3` |
| Weather | `+0.4775%` | `+1.2278%` | `3.00/3` | `3.00/3` |

[Strong Evidence] true interval basis相对random orthogonal basis在9/9 checkpoint runs均为正，说明future
coordinate geometry/smoothness具有真实generalization value。这与D1中DCT/localized spaces优于random
subspace一致。

[Strong Evidence] 但true depth grouping相对同一true basis上的random grouping，macro为`-0.2212%`；ETTh2
甚至0/3 random groups被击败。因此当前数据不支持“balanced tree depth就是正确的scale-specific nonlinear
allocation”。这是D2的关键negative finding。

## 3. Gate Amendment

初版analyzer把random-group与random-basis合并为一个median，得到`+1.1280%`，会被较弱的random-basis
controls抬高，从而掩盖true grouping未超过random grouping。

[Decision] 在formal5启动前修复为两个mandatory gates。raw runs、candidate、seeds、control数量、MSE
margin均不改变；修复后的core3分别为：

- true vs random-group：`-0.2212%`，fail；
- true vs random-basis：`+2.3137%`，positive precheck。

该修复标记为`measurement_gate_fault_repaired_before_formal5`。初版combined statistic保留在JSON中只作
audit，不再决定scale alignment。

## 4. Failure Attribution

- `hypothesis_false`：formal5前不能成立；core3已对depth-group hypothesis形成负压力；
- `intervention_point_wrong`：仍可能。final-head independent groups也许不是正确的scale interaction位置；
- `readout_or_head_design_wrong`：generic dense MLP在ETTh2出现temporal generalization failure；
- `optimization_or_numeric_pathology`：未发现non-finite或>100%退化；basis/Parseval与freeze invariants通过；
- `capacity_control_explains`：rank/full results不支持capacity expansion作为统一解释；
- `grouped_parameterization_explains`：random grouping在ETTh2/ETTm1可解释或超过true grouping。

因此当前结论是`partial_core3_basis_geometry_signal_only`，不能升级为paper method，也不能在formal5前作
direction-level rejection。

## 5. Next Decision

1. 不进入Step 4，不实现新的scale-grouped decoder；
2. 先按既定validation-only规则冻结ETTh1与ETTm2 natural profiles；
3. 使用修复后的双random hard gate运行formal5；
4. 若formal5仍只支持true basis而不支持true grouping，关闭“depth-grouped final head”问题，回Step 2重构
   Contribution 1问题；orthogonal/localized basis只保留为representation evidence/control，不单独成文；
5. SC2-MIPR、Encoder、MoE继续held。

## 6. 11-Step Record

| Field | Record |
| --- | --- |
| `current_step` | Step 2/3 active；core3 partial complete |
| `problem` | true future-scale grouping是否超越generic/random grouping |
| `existence_evidence` | basis geometry 9/9 positive；depth grouping core3 negative/mixed |
| `idea` | none；diagnostic only |
| `theory_check` | coefficient/time MSE equivalence与basis invariants pass |
| `design` | 99/99 head-only probes；split random controls after gate audit |
| `narrative_gate` | not applicable |
| `effectiveness_gate` | not started |
| `artifacts` | raw metrics/history/metadata、pairwise/dataset/summary、interpretation |
| `decision` | `partial_core3_basis_geometry_signal_only`；formal5 pending；training false |
