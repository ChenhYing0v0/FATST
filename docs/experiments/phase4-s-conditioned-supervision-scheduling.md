# Phase4-S：State/Difficulty-Conditioned Supervision Scheduling

`current_step`: Step 4-6 draft。

## 11-step 记录

| Field | Content |
| --- | --- |
| `current_step` | Step 4-6 |
| `problem` | 静态 horizon-free supervision units 有局部收益但整体输 R.3，是否需要 train-side condition 来分配 unit pressure |
| `existence_evidence` | Phase4-R remote gate；Phase4-S post-hoc segment diagnostic；QDF/TransDF/ElasTST/SRP++ seed evidence |
| `idea` | 训练时仍使用 horizon-free units，但 unit sampling / auxiliary pressure 由 train-side difficulty 或 state proxy 条件化 |
| `theory_check` | future-step dependency、task conflict、step heterogeneity 和 curriculum/dynamic weighting 共同支持“pressure 不应全局静态” |
| `design` | 先做 post-hoc diagnostic 和最小 proxy 设计；实现前不启动 remote training |
| `gate` | proxy 必须不泄漏 evaluation horizon；local smoke 后再做 small remote gate；若不能定义稳定 proxy，回退到 Step 1 文献调研 |
| `artifacts` | `analysis/phase4_horizon_decoupled_gate_20260624/phase4_s_conditioning_diagnostic_report.md` |
| `decision` | Phase4-S 作为 hypothesis 继续推进；优先设计 R.3 + sparse auxiliary，而不是替换 R.3 objective |

## 为什么提出 Phase4-S

[Fact] Phase4-R 的 `D2-D6` 静态 horizon-decoupled replacement 全部不通过 paper-core
gate。最好的 `D2_random_future_mask` 仍为 `+3.51%` mean relative MSE，只有 `1/12`
MSE wins。

[Fact] 失败不是 prefix consistency 或 unified evaluation interface 损坏导致。所有 strategy
的 prefix mismatch 都在 numerical-zero 量级。

[Strong Evidence] 但 horizon-free units 不是完全无效。segment-level diagnostic 显示：

- `D2_random_future_mask` 有 `4/30` segment wins，全部集中在 R.3 high-residual bucket；
- `D3_interval_supervision` 有 `5/30` segment wins，也全部集中在 R.3 high-residual bucket；
- `D3_interval_supervision` 在 late region 有 `2/3` wins，mean relative MSE 为 `-0.46%`；
- early region 上所有候选均为 `0/12` wins，且显著退化。

[Inference] 这说明 static mask/interval 的问题不是 unit 类型完全错误，而是 pressure 分配错误。
全局固定策略会在 easy/early regions 施加不必要 pressure，同时没有把 pressure 定向给
high-residual 或 late regions。

[Decision] 因此下一步不应继续调 `mask_ratio`、`interval_length` 或 component rank，而应问：

> 什么 train-side condition 能决定何时、对哪些 future units 施加额外 supervision pressure？

## 前人工作基础

### Forecasting 直接基础

[Strong Evidence] `QDF`：standard MSE 等价于 identity weighting，忽略 future-step
autocorrelation 和 heterogeneous task weights。它支持 objective pressure 需要被设计，但不要求
按 benchmark horizon 组织。

[Strong Evidence] `TransDF`：temporal MSE 忽略 label sequence autocorrelation；长 forecast
horizon 会造成过多 step-wise tasks 和 task conflict。它支持把 supervision 从 raw time steps
转向 transformed components 或更少的关键 units。

[Strong Evidence] `ElasTST`：varied-horizon forecasting 强调 horizon-invariant / prefix-stable
inference。它支持 evaluation horizon 是测试接口，而不是训练 schedule 的必要定义。

[Strong Evidence] `SRP++`：multi-step forecasting 存在 future-step / segment heterogeneity。
它支持不同 future regions 可能需要不同 treatment，但它本身是 representation route，不是
Phase4-S 的 training strategy 直接模板。

### 通用 training dynamics 类比基础

[Moderate Evidence] 以下工作不是 TSF 直接证据，但支持“训练 pressure 可随 difficulty / task
state 动态调整”的大原则。进入论文引用前必须导入 Zotero 并完成 fulltext note。

| Work | 可借鉴点 | 与 Phase4-S 的关系 |
| --- | --- | --- |
| [Curriculum Learning](https://dl.acm.org/doi/10.1145/1553374.1553380) | 训练顺序可影响 optimization path 与 local minima | 支持 schedule path 不是无关实现细节 |
| [Self-Paced Learning for Latent Variable Models](https://papers.nips.cc/paper/3923-self-paced-learning-for-latent-variable-models) | 根据样本 easiness 逐步纳入训练 | 支持 difficulty-conditioned sample/unit selection |
| [GradNorm](https://arxiv.org/abs/1711.02257) | 动态平衡 multi-task gradient magnitudes | 支持多 future-unit pressure 不应固定 |
| [Uncertainty loss weighting](https://arxiv.org/abs/1705.07115) | 用 task uncertainty 学习 loss weights | 支持 heterogeneous objectives 的 learned weighting |

[Boundary] 这些通用工作不能直接证明 Phase4-S 会提升 forecasting；它们只证明“conditioned
training pressure”是有理论谱系的优化思想。Phase4-S 的 forecasting 合法性仍必须由 QDF /
TransDF / ElasTST / SRP++ 和本仓库 artifacts 共同支撑。

## 可证伪假设

[Hypothesis] Horizon-free supervision units 可以改善 unified multi-horizon forecasting，但
只有当额外 pressure 被定向到 high-difficulty future units 时才有效；全局静态 pressure 会损害
easy/early regions。

形式化地，设最长训练标签为 $Y_{1:T}$，evaluation set 为：

$$
\mathcal{H}_{eval}=\{96,192,336,720\}.
$$

evaluation 只使用：

$$
\operatorname{Eval}(f_\theta;\mathcal{H}_{eval}).
$$

训练定义 horizon-free units：

$$
\mathcal{U}=\{u_1,\dots,u_K\}.
$$

Phase4-R 的失败策略等价于全局静态分布：

$$
p_t(u \mid x, y, \theta)=p(u).
$$

Phase4-S 改为：

$$
p_t(u \mid x, y, \theta)=p(u \mid c_t(x,y,\theta)),
$$

其中 $c_t$ 是只由 train split 和训练过程构造的 condition，例如 label novelty、running loss
bucket 或 residual proxy。该 condition 不能使用 test/evaluation result，也不能直接把
`96,192,336,720` 写回 training schedule。

## 候选设计

### S2：R.3 + Sparse Unit Auxiliary

[Decision] 优先级最高。

核心：

$$
\mathcal{L}=\mathcal{L}_{R3}+\lambda\mathcal{L}_{aux}(u),
$$

其中 $u$ 是 horizon-free sparse unit，$\lambda$ 小于主 loss。它的目的不是替换 R.3，而是在不破坏
early/easy regions 的情况下给 high-residual/late regions 额外 signal。

为什么优先：

- Phase4-R 已证明 replacement 风险高；
- D2/D3 在局部有效，说明 auxiliary 可能比 replacement 更合理；
- 对 existing carrier 改动最小，rollback 成本低。

### S1：Difficulty-Conditioned Interval

核心：

$$
p(u=\text{interval}_{a:b}) \propto \phi(c_t(a:b)),
$$

其中 $c_t$ 可以来自 train-label novelty、局部 variation、running loss bucket 或 train residual
proxy。interval 不按 evaluation horizon 定义。

风险：

- 若 proxy 实际上只是 future position index，会退化为手工 late weighting；
- 若 proxy 依赖 validation/test residual，会造成 leakage。

### S3：Error-Process Reweighting

核心：先估计 train split 上的 residual/error process，再对 future units 施加附加 pressure。

当前不作为第一实现，因为：

- residual proxy 容易和 evaluation horizon 绑定；
- 需要更严格的 split 管理；
- Phase2 的 QDF/covariance route 已提示直接 transfer objective matrix 风险较高。

## Phase4-S 进入实现前的 gate

实现前必须满足：

1. condition 明确来自 train split 或 online training trace；
2. condition 不直接使用 evaluation horizon identity；
3. 有一份 local diagnostic 说明该 condition 与 high-residual / late-region weakness 有关系；
4. loss 以 R.3 为 base，第一版只加 auxiliary，不替换主 objective；
5. local smoke 必须输出 `supervision_trace.csv`，记录 condition bucket、unit type、active steps、
   auxiliary weight 和 loss；
6. small remote gate 先跑 `ETTh2` + `Weather`，不直接 full matrix。

## 回退规则

- 若 train-side proxy 无法定义，回退到 Step 1：补 literature 和 data diagnostic。
- 若 proxy 定义成立但 local smoke 显示 trace 退化为固定 late weighting，回退到 Step 4。
- 若 small remote gate 只在 `ETTh2` 有效而 `Weather` 明显退化，回退到 Step 6，重新设计
  condition，而不是调大 sweep。
- 若 S2 auxiliary 不损害 R.3 但无收益，才考虑 S1 condition interval。
- 若 S2/S1 均失败，Phase4-S 暂停，HSS 需要重新回到 Step 2-3 判断 carrier 是否错误。
