# SC-D24-CTB Step 2/3：Conditional Trajectory Bias Problem Audit

## 1. 当前节点

| Field | Content |
| --- | --- |
| `current_step` | Post-D23 Step2/3；SC-D24-CTB validation diagnostic frozen |
| `problem` | strong fixed trajectory synthesis是否仍留下可由ordered raw history识别的coarse future deformation？ |
| `existence_evidence` | DENSE几乎恢复A6但无稳定allocation；D22 target access pass；phase probe specificity不足 |
| `idea` | 在frozen A6/DENSE validation forecasts上测量past-identifiable 48-step trajectory-bias surface |
| `theory_check` | 不输入requested H，不改变Bayes target；只检验finite-function-class的history-conditioned output freedom |
| `design` | first-third fit / middle-third purge / last-third evaluate；ordered raw-history对matched controls |
| `narrative_gate` | not applicable；本轮仅problem diagnostic |
| `effectiveness_gate` | not applicable；不访问official test |
| `artifacts` | frozen config、diagnostic evaluator与后续validation summaries |
| `decision` | implementation + frozen-checkpoint validation inference authorized；method/training/test false |

## 2. 为什么D23后仍可提出这个问题

A6 forecast满足

$$
\hat y(x)=B\,c(x)+b,
$$

其中$B\in\mathbb R^{720\times256}$是dataset-global temporal basis。D23的
`DENSE_DUAL_MATCHED`也通过dataset-global low-rank trajectory residual恢复了A6性能。这说明global trajectory
synthesis是必要capacity，却没有回答同一固定output geometry是否对所有history regimes都充分。

一个与requested horizon无关的剩余自由度是：raw history $x$是否能预测未来trajectory在coarse coordinate
blocks上的系统deformation $d(x)$. 若不存在，继续做input-conditioned basis、phase router或MoE没有problem
必要性；若存在，也只允许回Step4设计native future-stage-aware operator，不能把本轮的frozen correction升级方法。

## 3. 已关闭的phase/time-warp子方向

本轮先对D23已保存的256-row validation probes做了held-out row cross-fit。相对global affine calibration：

- 一阶prediction derivative在A6与DENSE上的macro incremental MSE gain均约`+0.03%`；
- curvature方向约`+0.29%–0.48%`；
- coordinate-shifted derivative control约`+0.11%`；
- blocked与interleaved rows差异明显，说明连续probe rows存在window overlap与抽样偏差。

因此证据不能把剩余误差特异归因于phase shift。最新primary sources又已直接覆盖：

- [prediction delay + derivative regularization](https://arxiv.org/abs/2407.01622)；
- [PULSE, ICML 2026](https://openreview.net/forum?id=JJIqZzujgE)的phase evolution与phase router；
- [PhaseFormer, ICLR 2026](https://openreview.net/forum?id=Lk9SqMQzhX)的phase-wise prediction；
- [BasisFormer, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e150e6d0a1e5214740c39c6e4503ba7a-Abstract-Conference.html)
  的adaptive basis selection/consolidation。

Decision：`phase_warp_problem_not_supported_by_current_probe / no_phase_router`。

## 4. D24统计量与数据角色

### 4.1 数据

- carriers：D23 seed2021的`A6_MEASURE`与`DENSE_DUAL_MATCHED`；
- datasets：Weather、ETTm1、ETTh1、ETTh2、ETTm2；
- split：只读取validation loader与冻结checkpoint；
- official test：禁止读取；
- 每个forecast origin的全部channels必须进入同一chronological partition。

### 4.2 Chronological transfer

按forecast origin顺序：

1. first third：拟合diagnostic linear map；
2. middle third：完全丢弃，作为约720-origin purge gap；
3. last third：唯一evaluation区间。

该设计避免256-row probe的interleaved-window leakage。它仍是同一validation split内的时间迁移，不是
paper-facing generalization claim。

### 4.3 Features

对每个history row $x_i\in\mathbb R^{720}$：

- `channel`：channel identity；
- `marginal`：mean、std、endpoint change、first-difference RMS；
- `ordered_history`：row-standardized history的24个连续30-step block means；
- `sorted_history`：逐row排序上述24 values，保留marginal values但销毁order；
- `recent`：最后4个ordered blocks；
- `target_shuffled`：保持ordered features与map capacity，但打乱fit residual rows。

### 4.4 Exact coarse correction statistic

对future 48-step block $b$，令frozen residual为$r_{i,b,t}=y-\hat y$，diagnostic map预测常数
$a_{i,b}=g(z_i)$。无需保存full prediction即可精确计算corrected squared error：

$$
\sum_{t\in b}(r_{i,b,t}-a_{i,b})^2
=\sum_{t\in b}r_{i,b,t}^2
-2a_{i,b}\sum_{t\in b}r_{i,b,t}
+48a_{i,b}^2.
$$

$g$是multi-output ridge regression；$\lambda\in\{0.1,1,10\}$全部报告，primary为1。H96/H192/H336/H720
分别聚合前2/4/7/15 blocks。

## 5. Gate与failure attribution

Primary $\lambda=1$下，两个carriers都必须满足：

1. ordered vs marginal macro MSE gain $\ge0.3\%$；
2. ordered vs sorted $\ge0.2\%$；
3. ordered vs target-shuffled $\ge0.3\%$；
4. 每项至少11/20 cells、3/5 datasets、3/4 horizons为正；
5. $\lambda=0.1,10$的macro sign仍为正。

通过时decision只能是`past_identifiable_coarse_deformation_supported_on_validation`，随后回Step4进行prior与native
operator gate；remote training/test仍需另行冻结和授权。

失败时只关闭exact `coarse linear deformation diagnostic`。由于checkpoint与forecast representation是frozen、
feature map有限，不能方向级拒绝所有history-conditioned output geometry；failure attribution应为
`diagnostic_invalid_for_direction_rejection`或`intervention_point_wrong`，而不是`hypothesis_false`。

## 6. Source boundary

`adaptive basis`、`phase evolution`、`dynamic forecasting`和generic residual correction已有大量prior。D24不claim
这些primitive，也不把frozen residual correction作为paper mechanism。若problem通过，新candidate必须说明：

1. 为什么ordered history信息必须进入future synthesis而非generic encoder；
2. 为什么完整operator没有被BasisFormer/PULSE/Implicit Forecaster覆盖；
3. 如何以same-objective E2E controls排除parameter capacity与普通 calibration；
4. 如何避免router、第二loss与post-hoc adapter叙事。
