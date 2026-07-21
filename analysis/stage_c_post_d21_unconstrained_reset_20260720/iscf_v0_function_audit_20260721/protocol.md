# ISCF-v0 Function-Level Audit Protocol

## 1. Long-stage record

| Field | Record |
| --- | --- |
| `current_step` | new carrier freeze；Step2/3 low-cost problem audit pre-result |
| `problem` | independent coupling scopes已经形成strong carrier，但scope之间是否存在稳定、可学习且非人为ordered的function relation仍未知 |
| `existence_evidence` | FCC中independent不差于SIFF-v2；由既有完整test表post-hoc派生的independent vs A6_FULL MSE/MAE为`+1.3584%/+0.9144%` |
| `idea` | 冻结`ISCF-v0`，只复用其15个existing checkpoints的已保存function tensors，审计low-dimensional relation、common/private structure、complementarity与topology stability |
| `theory_check` | function-space关系比直接对parameter tensor做SVD更接近实际预测函数；但现有probe仅256 rows且来自已访问test artifacts，因此只能形成test-informed Step4 problem evidence |
| `design` | 5 datasets × 3 seeds；`probe_arms [256,5,720]`、full row-bin losses和policy usage；64次marginal-preserving circular-shift null |
| `narrative_gate` | not evaluated；本轮不是method proposal |
| `effectiveness_gate` | not applicable；不训练、不重新访问test、不评估新method |
| `artifacts` | frozen config、analysis script、pre-registered CSV/JSON outputs |
| `decision` | pending existing-artifact audit |

## 2. Carrier freeze boundary

`ISCF-v0`是原FCC arm `siff_independent_equal`的新research-carrier identity：

- code保持`readout_mode=siff-independent-scope-control`，避免checkpoint不兼容；
- $Q=5$、`scale_basis=I_5`、scopes=`{1,48,144,360,720}`；
- policy=`direct`，objective=`equal_skill`；
- dataset-wise ranks固定为ETTh1/ETTh2/ETTm1/ETTm2/Weather=`109/116/116/106/116`；
- natural profiles、training、checkpoint selector与full-crop protocol保持FCC原样；
- 新identity不继承SIFF-v2失败的ordered-field claim，也不因改名自动成为paper-core method。

机器可读contract为`configs/stage_c_iscf_v0_carrier.json`。

## 3. Source artifacts and split role

本轮只读取已经由FCC formal audit保存的：

- `probe_arms [256,5,720]`；
- `probe_fused [256,720]`与`probe_targets [256,720]`；
- `arm_row_bin_mse [N,8,5]`与`fused_row_bin_mse [N,8]`；
- `policy_row_bin_usage [N,8,5]`。

它不重新调用dataset loader，不重新读取test labels，不改变checkpoint。由于source tensors来自已访问official test，所有
结论标记为`diagnostic_only_test_informed_reuse`。这些结果可以支持新的Step4问题，但不能建立method effectiveness、选择
paper method或拒绝尚未end-to-end训练的方向。

现有NPZ没有保存`component_history_modes`，所以本轮不声称mode/parameter subspace evidence。若function audit通过，
mode activation只能在后续validation-only diagnostic中补充。

## 4. Statistics

设五个probe arm展平后为$A\in\mathbb R^{5\times F}$，target为$y\in\mathbb R^F$。

### 4.1 Aligned low-dimensional relation

对$A$沿scope轴中心化，计算scope Gram matrix的eigenvalues。报告：

- `centered_scope_ev1/ev2`：前1/2 eigenvalues解释的scope-deviation energy；
- `centered_scope_effective_rank`：participation ratio；
- `shift_null_ev2_p95`：每个scope独立circular shift flattened coordinates，重复64次后的EV2 95%分位数。

`centered_scope_ev2 > shift_null_ev2_p95`表示低维结构依赖跨scope对齐关系，不能只由各arm marginal distribution解释。

### 4.2 Common/private residual structure

令$E_s=A_s-y$。定义：

$$
r_{common}=\frac{5\|\frac{1}{5}\sum_s E_s\|_2^2}{\sum_s\|E_s\|_2^2},
\qquad r_{private}=1-r_{common}.
$$

用相同independent circular shifts构造`shift_null_common_energy_p95`。observed高于null说明scopes共享对齐误差；
`private_energy`非零则说明完全共享也不足以描述五个functions。

### 4.3 Complementarity and fusion

- `fused_gain_over_best_fixed_arm_percent`：learned fused prediction相对全局最优固定arm；
- `oracle_headroom_over_fused_percent`：每个row-bin选择最优arm相对fused的剩余headroom；
- `unique_best_scopes_across_bins`：八个future bins中成为mean-MSE最优的scope数量。

oracle只证明互补性，不证明router或新method可实现该收益。

### 4.4 Topology and order

- pairwise `prediction_nrmse`构成每个run的10维scope-distance topology；
- 同一dataset三对seeds的distance-vector Spearman衡量topology stability；
- `scale_distance_spearman`比较prediction distance与absolute log-scale distance；
- `adjacent_to_nonadjacent_distance_ratio`比较相邻与非相邻canonical scales。

order统计只检查canonical ordering是否在未共享参数时重新出现；它不能恢复SIFF-v2的ordered-field attribution。

## 5. Frozen decision rule

| Gate | Pass rule |
| --- | --- |
| aligned low dimension | 15个run中至少12个`EV2 > shift-null p95` |
| common + private | 至少12个common energy超过null，且median private energy至少0.05 |
| complementarity | median oracle headroom至少1%，且median unique best scopes至少2 |
| topology stability | 至少4/5 datasets的median cross-seed topology Spearman至少0.5 |

四项全过：`function_relation_supported_for_new_step4_problem`。部分通过：
`function_relation_unresolved_requires_narrow_step4_audit`。全无支持：`stable_function_relation_not_supported`。

无论哪种结果，本轮均不授权new model implementation、remote training或new formal test access。
