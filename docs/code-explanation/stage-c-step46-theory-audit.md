# StageC Step 4-6 Theory Audit Code Explanation

## Purpose

`scripts/analyze_stage_c_step46_theory_gate.py`在实现模型前验证两件事：

1. `90 -> 30 -> 10 -> 5 -> 1` mixed-radix refinement是否形成exact orthogonal synthesis、可恢复的
   parent/detail coefficients与prefix-local evaluation；
2. deployment horizon measure诱导的raw quadratic risk，与删除cross-scale blocks后的MIPR之间究竟有多大
   几何差异。

脚本不读取checkpoint、dataset或test metrics，不构成method performance evidence。

## PMFO Algebra Flow

### Coefficient groups

canonical length为`T=720`，block sizes为`(90,30,10,5,1)`，对应radices
`(3,3,2,5)`。ordered coefficient vector的groups为：

- coarse scaling：`8`；
- level details：`16/48/72/576`；
- total：`720`。

`helmert_detail(r)`构造`Q_r: [r,r-1]`，其columns与normalized ones vector正交。
`synthesize(coefficients)`按：

```text
parent: [N,R]
detail: [N,parent_count,r-1]
children = parent * ones/sqrt(r) + detail @ Q_r^T
```

逐层得到unit leaves `[N,720]`。

### Prefix evaluation

`synthesize_prefix(coefficients,H)`每层只读取前`ceil(H/parent_support)`个parents/details，并在child level
截到`ceil(H/child_support)`。它与full synthesis后取`[:H]`比较，输出
`prefix_restriction_max_abs`。

### Invariant quantities

`theory_gate.json`中的统计定义：

- `orthogonality_max_abs`：synthesis basis Gram与identity的最大绝对差；
- `projector_sum_max_abs`：所有scale increment projectors之和与$I$的最大差；
- `projector_idempotence_max_abs`：$Q_l^2-Q_l$最大差；
- `projector_cross_max_abs`：$Q_lQ_k,l\ne k$最大差；
- `basis_projector_match_max_abs`：coefficient group生成的projector与nested block projector最大差；
- `prefix_restriction_max_abs`：pruned evaluation与full crop最大差；
- `refinement_recovery_max_abs`：children重新分析得到parent/detail时的最大差。

gate要求上述误差最大值不超过`1e-10`。

## Tree Count CSV

`pmfo_tree_counts.csv`每列含义：

- `horizon`：requested prefix length；
- `active_coarse_coefficients`：与prefix相交的90-step coarse cells数；
- `active_detail_l1..l4`：各level active parents乘以`r_l-1`；
- `active_total_coefficients`：coarse与四层details之和；
- `boundary_overhead_vs_output`：active coefficient count减H，来自boundary parent必须生成完整contrast；
- `out_of_prefix_coefficients_avoided`：`720-active_total_coefficients`；
- `active_fraction_of_full_tree`：active/720；
- `a6_dense_basis_scalar_products`：`H*256`，只表示A6 temporal basis的scalar product count，不是完整
  model FLOPs。

[Boundary] active coefficient count也不是PMFO完整FLOPs；learned state transition尚未实现。该CSV只验证
domain-local algebra具有不计算out-of-prefix atoms的可能性。

## PIR Measure Flow

`measure_weights()`从$\mu(H)$构造：

$$
w_t=\sum_{H\ge t}\mu(H)/H.
$$

对`delta_720/uniform_h/log_uniform_h/benchmark_h`分别构造$W=\operatorname{diag}(w)$。nested block
projectors记为$Q_l$，脚本计算：

$$
\widetilde W=\sum_lQ_lWQ_l,\qquad W_{off}=W-\widetilde W.
$$

`pir_measure_geometry.csv`每列含义：

- `step_weight_sum`：$\sum_tw_t$；所有measure应为1；
- `step_weight_first/last`：第一与最后位置的risk weight；
- `first_to_last_weight_ratio`：temporal skew；
- `offblock_fro_ratio`：$\|W_{off}\|_F/\|W\|_F$；
- `offblock_energy_fraction`：$\|W_{off}\|_F^2/\|W\|_F^2$；
- `max_cross_scale_block_fro`：所有$l\ne k$的$\|Q_lWQ_k\|_F$最大值；
- `trace_preservation_abs`：$|\operatorname{tr}(\widetilde W)-\operatorname{tr}(W)|$；
- `component_quadratic_identity_max_abs`：显式sum component risk与$e^T\widetilde We$的最大差；
- `random_error_relative_risk_gap_mean/max`：固定seed的256个Gaussian error上，
  $|R_{MIPR}-R_{raw}|/R_{raw}$的mean/max。

## Code-Theory Consistency

- intended theory：mixed-radix tree必须exact conservative/projective；MIPR必须被描述为measure-induced
  block-diagonal surrogate；
- code realization：fixed Helmert contrasts与nested block projectors实现上述代数；
- proxy boundary：tree count不是model FLOPs，Gaussian error gap不是训练收益；
- falsification：任一invariant超过`1e-10`则PMFO algebra gate失败；off-block为零时MIPR不得声称提供raw
  risk之外的信号；
- current evidence：全部PMFO invariant在`1.33e-15`以内；`delta_720` off-block energy为0，
  `log_uniform_h`为`0.205154`，与measure-conditional结论一致。

## Verification

```bash
python -m py_compile scripts/analyze_stage_c_step46_theory_gate.py
conda run -n r2026-fsa python scripts/analyze_stage_c_step46_theory_gate.py
```
