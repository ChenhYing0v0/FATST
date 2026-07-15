# StageC D12 Predictable-Frame Feasibility: Step 2/3 Protocol

## Decision Summary

| Field | Decision |
| --- | --- |
| `current_step` | joint Contribution 1/2 Step 2-3 |
| `diagnostic` | `D12-A` train-only predictable-covariance existence/stability audit |
| `role` | `diagnostic_only`；不是PRISM/CAPE effectiveness gate |
| `problem` | rank-256 future frame是否有必要从history-predictable variation而不是raw label covariance构造 |
| `source_boundary` | reduced-rank/output-subspace regression已覆盖generic predictable subspace；本项目只保留prefix-risk coupling的potential claim |
| `design` | two-fold purged forward cross-fitting；A6 primary pilot + DCT-ridge robustness pilot |
| `data` | five datasets；train split only；validation/test=false |
| `next_gate` | D12-A至少3/5 datasets支持后，才授权D12-B PRISM Pareto audit |
| `rollback` | D12-A失败则CAPE关闭；不能用更复杂pilot或validation tuning挽救同一hypothesis |

## 1. Why D12-A Comes Before PRISM

PRISM-CAPE的frame objective需要一个history-predictable future covariance：

$$
\Sigma_m=\operatorname{Cov}(\mathbb E[y\mid x]).
$$

如果在当前rank $r=256$下，raw-label top-$r$ subspace已经几乎完整覆盖predictable covariance，或者
predictable covariance在不同time folds间不稳定，那么CAPE没有独立problem headroom。此时继续优化
prefix-localized frame只会把一个未经支持的estimator写进architecture。

D12-A因此只回答三个existence问题：

1. OOF pilot是否真正从history预测到非退化的future variation；
2. predictable subspace是否跨forward folds稳定；
3. raw-label rank-256 basis是否遗漏至少0.5%的predictable energy。

它不训练PRISM、不改变A6 paper carrier，也不读取validation/test。

## 2. Source-Informed Boundary

检索日期为2026-07-15；Zotero只作seed，以下来源由external primary-source search发现或复核。

| Source | Primary evidence | Boundary for D12 |
| --- | --- | --- |
| [Vector-Valued Least-Squares Regression, JMLR 2022](https://jmlr.org/papers/v23/21-1357.html) | reduced-rank output methods在structured/vector-valued regression中已有理论与统计收益 | “预测输出应位于低秩subspace”不是新贡献 |
| [Nonparametric Principal Subspace Regression, JMLR 2022](https://www.jmlr.org/papers/v23/20-963.html) | two-step principal-subspace regression结合低秩近似与predictor-dependent signal | “先估计subspace再回归”不是新贡献 |
| [Subspace Fitting Meets Regression, ICML 2020](https://proceedings.mlr.press/v119/dar20a.html) | supervised subspace fitting与orthonormality已有系统研究 | CAPE不能claim supervised orthogonal subspace本身 |
| [Time-series performance estimation study, arXiv 2019](https://arxiv.org/abs/1905.11744) | nonstationary forecasting更适合保持时间顺序的out-of-sample evaluation | 普通random K-fold不适合作为D12 primary protocol |
| [Leave-future-out CV, arXiv 2019](https://arxiv.org/abs/1902.06281) | 允许future影响past prediction会产生optimistic estimate | D12必须forward-only，不能让后段windows进入前段pilot |

[Decision] CAPE的conditional-mean theorem属于经典reduced-rank regression边界内的数学工具。若D12通过，
paper claim只能位于完整链条：

> unified nested-prefix deployment measure → risk-localized projective forecast frame →
> train-only predictable-energy estimation → multi-horizon architecture-training co-design。

## 3. Leakage-Safe Fold Construction

train dataset中的window $i$使用：

- history raw interval：$[i,i+719]$；
- future raw interval：$[i+720,i+1439]$。

因此若pilot training最后一个window index为$j$，OOF window必须至少从$j+1440$开始。等价地，训练
index range与OOF range之间冻结`purge_windows=1439`。

设train-window总数为$N$，两fold为：

1. OOF-1：$[\lfloor0.6N\rfloor,\lfloor0.8N\rfloor)$；
2. OOF-2：$[\lfloor0.8N\rfloor,N)$；
3. fold $k$的pilot只训练于$[0,\text{oof_start}_k-1439)$。

每fold从OOF block中确定性均匀抽取512个windows；所有channels均进入统计。fold间使用相同model
initialization seed，避免把initialization差异误当作temporal instability。

## 4. Coordinate Contract

PRISM future frame位于A6 RevIN-normalized synthesis coordinates，而不是最终denormalized output。
对每个window/channel：

$$
x^{norm}=(x-\bar x)/s_x,\qquad
y^{norm}=(y-\bar x)/s_x.
$$

D12所有label、A6 prediction和ridge prediction covariance均在$y^{norm}\in\mathbb R^{720}$中计算。
rows定义为`window × channel`；frame在variables间共享，与A6 temporal basis contract一致。

统计只保存充分统计量与$720\times720$ covariance，不落地大规模raw predictions。

## 5. Pilot Contract

### 5.1 A6 primary pilot

- architecture/profile：每个dataset冻结的natural A6 profile；
- objective：full-720 L1；
- optimizer：AdamW，learning rate与natural profile一致，cosine schedule；
- epoch：固定20，不使用OOF、validation或test选择checkpoint；
- seed：两个fold均为2021；
- final paper model：不复用任何pilot weights。

固定epoch避免把OOF block变成early-stopping validation。D12只使用OOF predictions估计covariance，
不是比较pilot leaderboard performance。

### 5.2 DCT-ridge robustness pilot

- per-window normalized history先投影到前128个orthonormal DCT coordinates；
- train rows最多8192，按window均匀采样并保留全部channels；
- ridge penalty为
  $\alpha=10^{-3}\operatorname{tr}(X^TX)/128$；
- 与A6使用完全相同的train/OOF ranges。

ridge只是model-bias control。ridge失败不能单独否定CAPE；A6与ridge subspace完全不相干则降低
predictable-subspace claim confidence。

## 6. Statistics

每个dataset/fold/pilot记录：

- `oof_mse`：$\mathbb E\|\hat y^{norm}-y^{norm}\|^2$；
- `zero_mse`：$\mathbb E\|y^{norm}-\bar y_{fold}\|^2$；
- `oof_r2=1-oof_mse/zero_mse`；
- `predictable_trace_fraction=tr(\Sigma_{\hat y})/tr(\Sigma_y)`；
- covariance symmetry、minimum eigenvalue与effective rank；
- rank 32/64/256 predictable-energy capture。

定义raw-label frame $U_y^{(r)}$与pilot frame $U_p^{(r)}$后：

$$
\operatorname{capture}(U;\Sigma_p)
=\frac{\operatorname{tr}(U^T\Sigma_pU)}{\operatorname{tr}(\Sigma_p)},
$$

$$
\operatorname{gap}_{r}
=\frac{\operatorname{capture}(U_p^{(r)};\Sigma_p)
-\operatorname{capture}(U_y^{(r)};\Sigma_p)}
{\operatorname{capture}(U_p^{(r)};\Sigma_p)}.
$$

`gap_256`直接回答CAPE在当前decoder rank下是否有headroom；rank32/64只作geometry解释，不能替代primary。

fold/pilot subspace overlap定义为：

$$
\operatorname{overlap}(U,V)=\|U^TV\|_F^2/r.
$$

## 7. Frozen Gate

dataset支持CAPE existence要求A6同时满足：

1. mean OOF $R^2>0$，且两个fold最差$R^2\ge-0.1$；
2. predictable trace fraction $\ge0.05$；
3. two-fold top-32 overlap $\ge0.35$；
4. rank-256 raw-basis predictable-capture relative gap $\ge0.005$；
5. A6/ridge top-32 overlap $\ge0.25$，且ridge mean OOF $R^2>0$。

至少3/5 datasets支持才得到`cape_problem_supported_d12b_authorized`。

Alternative outcomes：

- 1-3通过、4失败：`raw_label_subspace_already_sufficient_cape_closed`；
- A6通过、ridge robustness失败：`pilot_specific_predictable_subspace`，不授权CAPE；
- OOF $R^2$或trace失败：`predictable_signal_not_established`；
- fold overlap失败：`nonstationary_or_estimator_unstable`；
- covariance invariant失败：`diagnostic_invalid`。

任何positive result只授权D12-B offline/probe，不授权PRISM model implementation。

## 8. Self-Critique

1. 固定epoch A6可能不是每fold的最优pilot，但它避免OOF selection leakage；
2. forward folds测到的subspace差异混合estimation error与temporal nonstationarity，这正是frame能否固定部署的
   practical gate，而不是需要消除的噪声；
3. pooling variables符合A6 shared temporal basis，但可能掩盖variable-specific predictable structure；
4. rank256很高，raw-label subspace可能已经充分；若如此，应关闭CAPE，而不是改用rank32结果挽救；
5. D12-A不评价prefix locality，只有通过后D12-B才测试PRISM。

## 9. Implementation And Local Gate

2026-07-15已完成worker、five-dataset analyzer、remote runner、result sync与code explanation。最小验证链：

- touched Python files通过`python -m py_compile`；
- design JSON通过parse；
- worker synthetic covariance/fold smoke通过；
- analyzer five-dataset decision smoke通过；
- remote/sync scripts通过`bash -n`；
- remote runner在local conda `r2026-fsa`下通过`DRY_RUN=1`。

[Decision] `local_implementation_gate=pass`。只授权3090执行D12-A train-only diagnostic；D12-B、PRISM/CAPE
method implementation、validation与test仍未授权。
