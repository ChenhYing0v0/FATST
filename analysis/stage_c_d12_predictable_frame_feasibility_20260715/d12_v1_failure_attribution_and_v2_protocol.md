# StageC D12-A v1 Failure Attribution And v2 Protocol

## Decision Summary

| Field | Decision |
| --- | --- |
| `current_step` | joint Contribution 1/2 Step 2-3 |
| `D12-A-v1 execution` | complete；5/5 numeric/provenance invariants pass |
| `v1 frozen gate` | support `1/5`；surface decision=`predictable_signal_not_established` |
| `direction-level validity` | `diagnostic_invalid_for_direction_rejection` |
| `failure cause` | `intervention_measure_wrong`：normalized synthesis coordinates被等权统计，但A6优化/评估发生在denormalized raw risk |
| `D12-B` | false |
| `next action` | D12-A-v2 risk-aligned weighted covariance；复用v1 pilot checkpoints，只重跑OOF statistics |
| `rollback` | v2仍少于3/5才关闭CAPE；PRISM locality hypothesis仍需独立重估 |

## 1. What v1 Tested

D12-A-v1在五个dataset上完成two-fold purged forward cross-fitting。每fold使用固定20 epochs的A6 pilot，
并以DCT-ridge作model-bias control；train split only，validation/test均未读取。所有covariance symmetry、PSD、
normalization与forward reconstruction检查通过。

remote provenance：commit=`684c34edb553a9a6604c2cedecf078a23953df14`；server=`529_Lab-3090`；
GPU 0/1/2启动时均为15 MiB used；运行区间`2026-07-15T15:26:43+08:00`至
`2026-07-15T15:31:54+08:00`；repo-external output为
`/home/yingch/exp_outputs/r-2026-fatst/stage_c_d12_predictable_frame_feasibility`。

冻结结果为：

| Dataset | A6 OOF $R^2$ | prediction trace / label trace | fold overlap@32 | raw gap@256 | v1 support |
| --- | ---: | ---: | ---: | ---: | --- |
| ETTh1 | -0.2589 | 0.3950 | 0.7094 | 0.019644 | fail |
| ETTh2 | -0.0094 | 0.0001 | 0.6506 | 0.025459 | fail |
| ETTm1 | 0.1618 | 0.2730 | 0.9604 | 0.005341 | pass |
| ETTm2 | -0.0018 | 0.00004 | 0.9087 | 0.030156 | fail |
| Weather | 0.0002 | 0.0004 | 0.7927 | 0.006973 | fail |

若只看冻结gate，CAPE应关闭。但该结论必须先通过failure attribution；v1没有通过这个审计。

## 2. Why v1 Cannot Reject The Direction

### 2.1 Observed pathology

v1 aggregate/fold statistics显示：

- ETTm1 label covariance trace约$1.3\times10^3$；
- ETTm2达到$2.6\times10^6$至$1.5\times10^7$；
- Weather达到$8.1\times10^5$至$8.9\times10^5$；
- ETTm2/Weather的minimum history std触及$\sqrt{10^{-5}}$。

因此少数history近常数、future发生shift的rows在normalized space被放大数百倍，并支配等权covariance。
这不是NaN/PSD错误，但属于risk geometry pathology。

### 2.2 Code-theory mismatch

A6先用history statistics做RevIN：

$$
z=(y-\mu_x)/s_x,
$$

decoder在$z$ coordinates中生成future，随后输出

$$
\hat y=s_x\hat z+\mu_x.
$$

官方training loss对raw output计算L1，paper primary metric在raw output计算MSE/MAE。对raw MSE有exact identity：

$$
\|\hat y-y\|_2^2=s_x^2\|\hat z-z\|_2^2.
$$

v1却令每个normalized row的weight为1，相当于优化另一个risk：

$$
\mathbb E\|\hat z-z\|_2^2,
$$

它会过度强调raw-space几乎无权重的small-$s_x$ rows。故v1的“prediction trace接近0”不能解释为raw
forecast task缺乏predictable signal。

[Strong Evidence] `intervention_measure_wrong`，同时出现极端trace scale。依据Diagnostic Failure Attribution
Rule，v1只能否定`uniform normalized covariance`这个exact estimator，不能关闭CAPE或forecast-frame方向。

## 3. D12-A-v2 Risk-Aligned Estimator

对每个window/channel row $n$，令$w_n=s_{x,n}^2$。weighted mean与covariance为：

$$
\bar z_w=\frac{\sum_n w_nz_n}{\sum_nw_n},
$$

$$
\Sigma_w=\frac{\sum_nw_n(z_n-\bar z_w)(z_n-\bar z_w)^T}{\sum_nw_n}.
$$

该frame仍位于A6 normalized synthesis coordinates，但其energy与raw-space MSE完全对齐。OOF SSE、zero
predictor SST、label/pilot/residual covariance、subspace与capture全部使用同一$w_n$。

为了避免换measure后由极少数large-scale rows支配，新增两个numeric/provenance diagnostics：

$$
\text{ESS fraction}=\frac{(\sum_nw_n)^2}{\sum_nw_n^2}\frac1N\ge0.05,
$$

$$
\text{max weight share}=\max_nw_n/\sum_nw_n\le0.01.
$$

这两个threshold在v2结果返回前冻结；它们只检查estimator concentration，不按dataset调节。

## 4. What Is Kept Fixed

- 完全相同的five natural profiles、fold ranges、purge 1439与OOF sampled indices；
- 完全相同的v1 A6 pilot checkpoints，不重训、不重新选checkpoint；
- 完全相同的DCT-ridge form与rank32/64/256 diagnostics；
- 完全相同的dataset support thresholds与3/5 cross-dataset gate；
- train split only；validation/test=false；method implementation=false。

只有统计risk measure从`uniform normalized`改为`history_std_squared weighted normalized`。因此v1/v2差异可以
归因于风险口径修复，而不是pilot capacity或optimization变化。

## 5. v2 Frozen Gate And Outcomes

dataset support仍要求：A6 OOF $R^2$、predictable trace、cross-fold overlap、rank256 raw-frame headroom、
A6/ridge robustness及所有invariants同时通过。至少3/5才授权D12-B。

- v2通过：只证明risk-aligned predictable-frame problem存在，进入D12-B；
- v2 predictability/trace仍失败：`hypothesis_false_for_CAPE`，CAPE关闭；
- v2 weight concentration失败：`diagnostic_invalid_for_direction_rejection`，但不允许继续调weight/cap做性能搜索；
- v2 raw gap@256失败：raw-label frame已经足够，CAPE关闭；
- 无论v2结果如何，不能由CAPE结果自动关闭或验证PRISM；PRISM的locality problem必须单独判定。

## 6. Local Verification

v2 worker对non-uniform weighted covariance加入direct-reference synthetic check，并验证`[B,1,C]` std到
`[B*C]` weights的row order。Python compile、JSON parse、worker/analyzer smoke、Bash syntax与local runner
dry-run均通过。

[Decision] `D12-A-v2 remote diagnostic authorized`；D12-B、forecast method、validation/test仍false。
