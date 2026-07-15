# SC1-D11 Future-Component Responsibility Code Explanation

## Scope

D11只读取A6 natural checkpoints，在fixed train/validation batches上提取gradient；不读取test、不拟合probe、
不更新forecast model。冻结协议见
`analysis/stage_c_sc1_d11_future_component_responsibility_20260715/d11_step23_source_theory_audit.md`。

## Forward And Gradient Flow

1. `batch_x [B,720,C]`经A6 Encoder得到`memory [B,C,P,D]`；
2. flatten为`hidden [B,C,P*D]`，A6 coefficient head得到`coeff [B,C,256]`；
3. `learned_temporal_basis [720,256] × coeff`产生normalized prediction，再由RevIN denormalize为
   `prediction [B,720,C]`；该manual path与正式forward逐batch比较；
4. short/long prefix sets分别转为step weights `q_short/q_long [720]`，每个measure严格sum to 1；
5. 对MSE/L1分别得到`v_mu = dL_mu/dprediction [B,720,C]`；
6. 对每个orthogonal future group应用`P_g v_mu`，再通过vector-Jacobian product回传到
   `coeff [B,C,256]`，得到component responsibility；
7. 所有group responsibility之和必须重构total coeff gradient；此外直接计算short/long loss对
   Encoder、coefficient head、learned basis和all parameters的total gradients。

## Basis Controls

- `rgnb`：16维global root加六层details；
- `dct`：相同group sizes的frequency bands；
- `random_s*`：三个fixed random orthogonal bases；
- A6 learned basis通过QR只审计span/complement residual energy，不被伪装成七层scale basis。

所有basis均以float64构造并审计orthogonality，再转为model dtype执行VJP。

## Artifact Definitions

- `total_gradient_metrics.csv`：每行是dataset/seed/split/batch/loss/target；`short_norm/long_norm`是两个
  regime gradient norm，`norm_ratio`为较大norm除较小norm，`cosine/dot/negative`描述direction；
- `component_metrics.csv`：每行再按basis区分；`responsibility_js`是short/long component norm-share的
  Jensen-Shannon divergence；三个`*_negative_fraction`分别表示short内部、long内部和同component跨regime的
  negative-dot比例；`alignment_efficiency=||sum_g r_g||/sum_g||r_g||`，`cancellation=1-efficiency`；
  `*_additivity_relative_gap`是component sum与total coeff gradient之差除total norm；zero responsibility不计为
  conflict，cosine只在双方非零的active pairs上取均值，`*_zero_group_count`显式记录不可达groups；
- `component_group_metrics.csv`：逐group的short/long norm、share、same-component cosine与negative flag；
- `reachability_metrics.csv`：residual落入A6 learned temporal span及其orthogonal complement的energy share；
- `metadata.json`：checkpoint、profile、hash、runtime、forward/additivity/orthogonality invariants和data boundary；
- analyzer输出`total_seed_summary.csv`、`component_seed_summary.csv`、`dataset_gate_summary.csv`、
  `gate.json`与`research_interpretation.md`。

## Code-Theory Consistency

代码分解的是output gradient而不是component energy。因为complete orthogonal projectors之和为identity，
`sum_g J^T P_g v = J^T v`对prefix mask是否与basis commute没有要求，也同时适用于MSE和L1。

仍是proxy的部分：checkpoint-local gradient只给出first-order optimization geometry；negative dot证明局部下降
方向冲突，但不证明长期训练必然退化。RGNB超越controls也只允许返回Step4，不证明某个decoder或loss有效。
