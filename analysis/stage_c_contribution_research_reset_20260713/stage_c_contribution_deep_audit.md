# StageC 两项 Contribution 深度审计与研究重置

## Executive Decision

[Decision] StageC 不接受以下两个过早命名：

1. 仅以 requested horizon 显式 conditioning 为核心的 `Horizon-adaptive forecast operator`；
2. 仅将不同 horizon 采样概率换算为 step weights 的 `Horizon-measure-aligned training`。

前者既有 horizon-specific shortcut 风险，也受到 target-query / functional decoder prior art 的直接覆盖；
后者有严谨的 risk-definition 价值，但单纯重加权不足以构成 SCI-level contribution。

[Hypothesis] 当前最值得推进的联合主线是：

- `Contribution 1 candidate`: **Projective Multiresolution Forecast Operator (PMFO)**；
- `Contribution 2 candidate`: **Projective Increment Risk (PIR)**。

两者都不把 horizon ID 作为 learned feature。PMFO 定义一个可限制、可增量细化的未来函数；PIR 在同一
嵌套函数空间中分解训练风险，避免把高度重叠的 prefixes 当成独立 tasks 重复计数。两者目前仍是
`problem/design candidate`，未通过 Step 4-6 narrative gate，不授权直接远程训练。

## 1. 过去实验究竟否定了什么

| 历史证据 | 实际结果 | 可否否定显式 target/horizon 方向 |
| --- | --- | --- |
| Phase1 target-set decoder | 旧 PatchEncoder 上 5/12 wins，mean MSE `+0.62%`；prefix consistency 正确 | 否。只说明 exact decoder 在旧 carrier 上没有形成稳定收益 |
| B10 target-specific frozen readout | late linear/ridge readout 出现巨大退化与数值病态 | 否。报告自身已标记 intervention/readout pathology，不能方向级拒绝 |
| B13 future-unit GRU composition | no-transition control 解释收益；ETTh2 两个尺度约 `+5%` | 只否定 exact GRU transition，不否定所有 future evolution operator |
| B14 unit-specific retrieval | model-independent label-patch gate 仅 1/6 | 对跨 dataset patch retrieval problem 是强负证据；不等价于否定所有 future-aware decoder |

[Strong Evidence] 所以“过去已经严格证明 requested horizon 进入 decoder 必然失败”是不成立的。
但“尚未被严格否定”也不等于“应重新作为主线”。用户提出的连续性担忧是独立且合理的设计约束：
若把离散 horizon ID、horizon embedding 或 horizon-specific router 直接注入 latent path，模型可能学习
benchmark horizon shortcut，使 $H=191$ 与 $H=192$ 的内部表征出现人为跳变。

## 2. External Prior-Art Boundary

| Work | 已覆盖的关键空间 | 对本项目的约束 |
| --- | --- | --- |
| [ElasTST, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/d7aa002885ccbe68cf6880da583761b2-Paper-Conference.pdf) | placeholders、structured mask、horizon-invariant outputs、horizon reweighting | exact prefix invariance 与 uniform random-horizon harmonic weighting 都不能单独成创新 |
| [TimePerceiver, NeurIPS 2025](https://openreview.net/forum?id=RCeZ063p33) | target timestamps/queries 与 generalized target-set forecasting | “target 进入 decoder”本身不新 |
| [FlowState](https://openreview.net/forum?id=R50AT6nAsM) | functional basis decoder、dynamic horizon、arbitrary sampling resolution | 单纯 continuous basis / arbitrary-horizon sampling 与 A6 邻近且 novelty 不足 |
| [Implicit Forecaster](https://openreview.net/forum?id=gqoeQPhQcE) | frequency/amplitude/phase 生成隐式未来 waves | global structured implicit decoder 已拥挤 |
| [TransDF](https://arxiv.org/abs/2505.17847) / [QDF](https://openreview.net/forum?id=vpO8n9AqEG) | label decorrelation、task covariance/weighting | 单纯 label correlation aware weighting 也不足以形成独立边界 |

[Decision] PMFO 必须超越“用 basis 表示连续曲线”：其核心边界应是 **nested refinement algebra +
prefix-restricted computation + operator-aligned training decomposition**，而不是 coordinate MLP、固定
Fourier/Legendre basis 或 horizon query。

## 3. Contribution 1 Candidate: PMFO

### 3.1 问题定义

A6 对任意 $H\leq720$ 先计算同一个`coeff: [B,C,256]`，再使用
`learned_temporal_basis[:H]: [H,256]`直接得到H步输出。它已经domain-only、exact consistent，且
output-side matmul随H增长；旧“总先生成H720再裁剪”的表述不准确。真正缺少的是nested spaces、
refinement identity、local support，以及对Encoder多尺度information sufficiency的证据。

PMFO 要学习一个共享未来函数

$$
\hat y_X(t)=\sum_{\ell=0}^{L}\sum_k a_{\ell,k}(X)\,\phi_{\ell,k}(t),
$$

其中 $\{V_\ell\}$ 构成嵌套空间 $V_0\subset V_1\subset\cdots\subset V_L$，并具有显式 refinement
relation。requested horizon $H$ 只决定求值域 $t\in[0,H]$ 以及哪些 compact-support basis 与该域相交，
不进入 $a_{\ell,k}(X)$ 的 learned conditioning path。

### 3.2 Tensor contract

1. A6 Encoder先输出`memory: [B,C,P,D]`；若D1-B证明信息充分，PMFO从该memory构造scale views；
2. 一个共享 coefficient generator 输出多分辨率 coefficients
   `a: [B, C, N_coeff]`；
3. deterministic refinement/restriction operator 根据 query coordinates 选择有效 support，形成
   `Phi_H: [N_coeff(H), H]`；
4. `y_hat_H = einsum(a_active, Phi_H) -> [B, H, C]`。

这里的 $H$ 是 shape/domain control，不是 semantic condition。若两次调用使用相同 history，短输出必须
等于长输出的 restriction；任何 learned horizon embedding、per-horizon expert 或 benchmark-ID table
均列入禁止项。

### 3.3 为什么不是旧 STBO 或 residual route

- PMFO 直接生成 forecast function，不是 `A6 + correction`；
- 旧 STBO 使用固定 tile/local bank，缺少跨层 refinement identity，也被 rank/capacity control 阻断；
- PMFO 的核心可证伪性质是 nested-space restriction/refinement，而不是 tile specialization；
- 若收益仅来自更多 coefficients 或 local support，parameter-matched no-refinement control 应解释收益，
  则 PMFO narrative 失败。

### 3.4 Candidate 排序

| Candidate | Continuity | Novelty potential | 历史证据风险 | 决策 |
| --- | --- | --- | --- | --- |
| PMFO nested multiresolution | H 仅控制 domain | 中高，但需进一步查 wavelet/operator prior art | STBO capacity confound需严控 | 主候选，先 diagnostic |
| Parallel semigroup future evolution | 无 H embedding，可按需展开 | 中 | B13 GRU/no-transition 强负证据 | 备选，只在 PMFO problem gate 失败后审计 |
| Coordinate-query implicit decoder | 连续 coordinate | 低 | 与 TimePerceiver/FlowState 接近 | diagnostic/control only |
| Horizon embedding/router | 存在离散 shortcut | 低 | 旧证据未严格否定但用户边界不接受 | 不进入 paper-core |

## 4. Contribution 2: 从 Horizon Measure 到 PIR

### 4.1 Horizon-measure 是否有研究价值

[Fact] 若部署 horizon $H\sim\mu$，平均 prefix risk 为

$$
R_\mu=\mathbb E_{H\sim\mu}\left[\frac{1}{H}\sum_{t=1}^{H}\ell_t\right]
=\sum_t w_\mu(t)\ell_t,
\quad
w_\mu(t)=\mathbb E\left[\frac{\mathbf 1(t\leq H)}{H}\right].
$$

因此 measure alignment 是严谨且有价值的 evaluation/training protocol 概念。[Fact] 但 ElasTST 已给出
random-horizon 等价的 horizon reweighting；当前 natural baseline 又采用 single full-720 uniform-step
loss，不存在旧 B7 multi-prefix 的 `14.39x` 重复加权 pathology。

[Decision] `Horizon-measure-aligned training` 可作为论文的问题定义与 protocol，但不能以 simple
$w_\mu(t)$ 作为 Contribution 2。

### 4.2 PIR 核心思想

设 $P_\ell$ 是 PMFO 嵌套空间 $V_\ell$ 上的 projection，定义正交或近正交增量
$\Delta_\ell=P_\ell-P_{\ell-1}$。对 squared/Huber-compatible error $e$，训练风险写成

$$
R_{\mathrm{PIR}}(e)=\sum_{\ell=0}^{L}\alpha_\ell(\mu)
\left\|\Delta_\ell e\right\|_2^2.
$$

$\alpha_\ell(\mu)$ 由 deployment measure 对该 refinement scale/support 的覆盖决定。它不是重复训练多个
prefix，也不是逐 timestep 的 harmonic weighting；它把共享的 coarse trajectory 与后续 refinement
information 分开计量，使 decoder 的结构单元与 loss 的归因单元一致。

### 4.3 Novelty boundary 与风险

- 相对 ElasTST：PIR 对 function-space increments 加权，不是对 raw steps 做 harmonic weighting；
- 相对 TransDF/QDF：PIR 使用 decoder-aligned nested projection，不依赖 batch covariance inversion，也不把
  benchmark horizons 当多任务 ID；
- 相对 generic multi-task balancing：PIR 权重来自 deployment measure 与 refinement algebra，不由 task ID
  或 observed gradient norm 任意调节。

[Uncertainty] 对 L1 loss 不存在直接的 Parseval-style exact decomposition。第一阶段必须使用 L2 或平滑
Huber 做理论/数值审计，不能在未证明时声称 L1 等价。若 projection increments 不能在真实 labels/error
上形成稳定、跨 dataset 的尺度分工，PIR 应在 Step 3 关闭。

### 4.4 备选训练方向

| Candidate | 价值 | 主要风险 | 优先级 |
| --- | --- | --- | --- |
| PIR | 与 decoder 共用理论对象，叙事闭环 | 依赖 nested representation 真实成立 | 1 |
| Distributionally robust horizon risk | 对未知部署 horizon distribution 有价值 | 容易退化为 generic DRO | 2 |
| Gradient-balanced horizon regions | 可缓解短/长区间冲突 | generic multi-task optimizer prior art 强 | 3 |
| Fixed curriculum / random horizon sampling | 工程可用 | novelty 弱，易与 ElasTST 重叠 | control only |

## 5. 分阶段研究计划

### C-D0：冻结 baseline reference（已完成）

- 3 datasets × 3 seeds × 8 horizons；
- 作用：后续所有 candidate 的唯一 test reference；
- 状态：`frozen_test_reference_ready`。

### C-D1：PMFO/PIR problem-existence diagnostics（当前 Step 2-3）

只做 offline/no-forecast-training diagnostic：

1. `D1-A`：在train labels与baseline residuals上构造DCT/block/random nested projections；
2. `D1-B`：用fixed ridge probes比较full `[P,D]` memory、patch mean、per-sample shuffled memory与raw history，
   判断coarse/mid/fine coefficients是否仍可恢复；
3. `D1-C`：审计learned basis的effective rank、condition、temporal entropy、support与DCT/block overlap；
4. 对delta-720、uniform-H、log-uniform-H和benchmark-H计算raw risk与projected increment risk；
5. 在固定train batches记录encoder/coeff/basis/all gradient cosine与norm，并先验证uniform-weight Parseval invariant。

Gate：至少 2/3 datasets 显示稳定 multiresolution increment structure；PIR 必须比 raw harmonic weighting
提供额外且可解释的 gradient/risk separation。否则 PMFO/PIR 分别回滚 Step 2，不实现。

### C-D2：Step 4-6 narrative/theory gate

- 完成 wavelet/neural operator/function-space decoder 专项 prior-art audit；
- 明确 refinement identity、support rule、复杂度和 exact restriction proof；
- 预注册 parameter/FLOP controls 与 falsification conditions；
- 只有 narrative gate 通过才写模型代码。

### C-D3：最小 implementation gate

先单 dataset、单 seed、dense horizons，仅比较：

1. natural A6 baseline；
2. parameter-matched no-refinement basis control；
3. PMFO + uniform full loss；
4. PMFO + PIR（仅在 PIR 独立 gate 通过后）。

禁止加入 Encoder innovation、MoE、router、auxiliary reconstruction 或 per-horizon hyperparameter。

### C-D4：独立主效应与 joint gate

若 PMFO 和 PIR 均通过最小门，运行 `2x2` factorial：A6/PMFO × full/PIR，要求 decoder 与 training
分别有独立主效应，joint gain 不能完全由任一单项解释。

### C-D5：full matrix 与 generality

3 datasets × 3 seeds × dense horizons；随后才增加第二 backbone 与官方 native baselines。任何 test 结果
不得回流修改 natural profiles。

## 6. Self-Critique

[Speculative] PMFO+PIR 的联合叙事比原先两个独立标签更完整，但也存在“共依赖过强”的风险：若 PMFO
不成立，PIR 可能失去自然对象。为避免整条主线单点失败，C-D1 同时保留 raw-step measure/DRO 作为
training-only fallback，但它不会在没有新机制边界时被强行包装成 Contribution 2。

[Decision] 当前 rollback point 是 Step 2/3，而不是 Step 7。下一步先实现 D1 diagnostic protocol 与
offline analysis，不实现 PMFO/PIR 模型，不启动 remote training。
