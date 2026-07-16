# SC2-PCC Step6 Design Checker 说明

## Purpose

`scripts/check_stage_c_sc2_pcc_step6_design.py`不实现model training。它读取
`configs/stage_c_sc2_pcc_step6.json`，验证source-informed redesign的数学与protocol是否足以授权Step7A。

## Tensor Flow

checker构造nonnegative arm errors：

```text
errors [B,C,T,S]
  -> prefix_risk = cumsum_t(errors) / [1..T] [B,C,T,S]
  -> standardize across S + stop-gradient
  -> prefix capability q [B,C,T,S]
  -> reverse_cumsum_t(q / [1..T]) / harmonic_tail
  -> target credit c [B,C,T,S]
```

`transport_to_targets`没有构造`[T,T]`矩阵，时间与activation均为$O(BCTS)$。`c[...,t,:]`是simplex；若
$q(H)$对所有$H$为常数，transport后保持不变。

## Statistics

- `nested_prefix_transport_identity`：直接计算
  $T^{-1}\sum_H\sum_s q_s^\epsilon(H)R_s(H)$与transported target form之差；
- `positive_affine_error_invariance`：$e\mapsto ae+b$后standardized capability最大差；
- `transport_not_pointwise_credit`：crossed early/late best-arm case中transport与pointwise target最大差；
- `transported_skill_floor`：最小$c_s^\epsilon(t)$，必须不低于$\epsilon/S$；
- `stopgrad_credit_no_gradient_path`：capability target不得携带autograd path；
- `control_matrix_exact/phase_a_run_count`：九个new training arms与45-run matrix必须与config完全相同。

## Code-Theory Consistency

- Intended theory：nested-prefix capability不能直接输入requested $H$，应通过harmonic incidence measure输运到natural
  target coordinates；
- Code realization：prefix cumulative risk + reverse harmonic cumulative sum精确实现identity，float64 gap为`0`；
- Proxy boundary：checker中的errors是synthetic outputs，未证明真实PCSD arms可学或credit可预测；
- Falsification：identity、simplex、floor、stop-gradient、schedule或protocol任一失败，Step7A不得开始。
