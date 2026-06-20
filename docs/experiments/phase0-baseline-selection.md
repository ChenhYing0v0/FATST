# Phase 0 Baseline Selection

## 目的

[Fact] 本项目最终希望围绕三个候选创新点展开：variable-horizon decoder、
future-aware mechanism、MoE-style conditional architecture。

[Inference] Phase 0 的 baseline 不是论文中的最终 comparison baseline，而是后续
机制实验的 internal base。它必须同时满足四个条件：

1. 性能不能太弱，否则后续机制很难超过 SRSNet。
2. 架构不能太复杂，否则新机制容易与已有强归纳偏置冲突。
3. 数据流必须清晰，方便替换 decoder、接入 future-aware state、插入 MoE。
4. 与 SRSNet 的核心竞争点应有可比性，但不能直接继承 SRSNet 的 selective patching
   机制，否则论文创新边界会变得混乱。

## 结论

[Inference] Phase 0 推荐采用 `PatchTST-style channel-independent patch encoder +
FixedHead` 作为 internal base，简称 `PatchEncoderFixedHead`。

它不是直接把 PatchTST 作为最终模型，而是抽取一个可控的 base data flow：

$$
X \in \mathbb{R}^{B \times L \times C}
\rightarrow
P \in \mathbb{R}^{(B C) \times N \times P_l}
\rightarrow
E \in \mathbb{R}^{(B C) \times N \times d}
\rightarrow
Z \in \mathbb{R}^{(B C) \times N \times d}
\rightarrow
\hat{Y} \in \mathbb{R}^{B \times H \times C}.
$$

其中：

- $P_l$ 是 patch length。
- $N$ 是 patch token 数。
- channel-independent 处理把每个变量作为一条 univariate series，共享 patch embedding
  和 encoder 参数。
- `FixedHead` 是 Phase 0 的故意限制：它把 encoded patch states 映射到固定 horizon，
  用于暴露 variable-horizon 和 step-specificity 问题。

## 为什么不是只选最高分模型

[Inference] 后续机制需要接在 baseline 上继续演化，所以 baseline 的选择不是普通
leaderboard 选择。过强、过复杂、机制已经重叠的模型会造成两个问题：

- 如果新机制有效，很难证明贡献来自新机制，而不是 baseline 原有机制。
- 如果新机制无效，无法判断是机制本身错误，还是它破坏了复杂 baseline 的稳定结构。

因此，Phase 0 应选择一个“强但可拆”的基础，而不是一个“强但机制饱和”的模型。

## 候选比较

### DLinear / LTSF-Linear

[Fact] DLinear 的优势是极简、稳定、可作为 sanity floor。它把 trend / seasonal
decomposition 与 linear projection 结合，强调简单 linear model 在 LTSF benchmark 上
可以非常强。

[Inference] 但它不适合作为本项目主 base：

- 它的核心状态几乎就是 linear projection，没有足够清晰的 token-level 或
  query-level representation 供 future-aware state 和 MoE 使用。
- 在其上加入 QueryDecoder 或 MoE，容易从“线性基线”直接跳到另一个新模型，
  中间缺少可解释的 representation bridge。
- 如果目标是超过 SRSNet，DLinear 的上限风险偏高。

用途：保留为 Phase 0 sanity baseline，用来判断新模型是否至少超过简单强基线。

### iTransformer

[Fact] iTransformer 把 Transformer 作用维度反转到 variate tokens，强调 variate-centric
representation 和跨变量关系建模，是一个强 forecasting backbone。

[Inference] 但它不适合作为第一 internal base：

- 它的主要创新发生在 variate dimension，而本项目第一创新点发生在 future horizon
  dimension。
- variable-horizon decoder 要处理的是 $Q_H$、$U_H$、prefix consistency，而
  iTransformer 的主张更偏变量关系与 lookback generalization。
- 在其上加入 future-aware 和 MoE 时，routing 很容易自然落到 variate tokens，
  从而偏离本项目“future query / horizon segment routing”的主线。

用途：后续可作为 external strong baseline 或 robustness reference，不建议作为第一可改造
base。

### SRSNet

[Fact] SRSNet 的核心是 Selective Representation Space：在 patch perspective 下，通过
Selective Patching 和 Dynamic Reassembly 构造更灵活的 representation space。它本身
就是本项目的重要 comparison baseline。

[Inference] 不建议把 SRSNet 作为 internal base：

- 它已经在 input patch selection 维度引入了强 adaptive mechanism。
- 本项目的创新重点在 future horizon、future-aware state 和 conditional operators。
  如果直接基于 SRSNet 改造，input-side selective patching 会和 future-side mechanism
  纠缠，难以归因。
- SRSNet 过强且机制较新，作为 base 会压缩本项目的创新边界：审稿人可能认为改进只是
  对 SRS 的二次包装。

用途：作为必须正面对比的重点 baseline，而不是改造 base。

### ElasTST / TIMEPERCEIVER / SRP++

[Fact] 这些方法与本项目的 variable-horizon 或 future-aware 目标高度相关。

[Inference] 不建议直接作为 Phase 0 base：

- ElasTST 已经包含 placeholder、structured mask、tunable RoPE、multi-scale patch 和
  horizon reweighting，机制与 Phase 1 高度重叠。
- TIMEPERCEIVER 已经把 target query 作为核心 formulation，直接采用会削弱本项目在
  future query decoder 上的原创空间。
- SRP++ 已经把 step-specific representation 和 LoRA-style expert sharing 结合，机制
  与 Phase 1/3 都重叠。

用途：作为设计证据和 ablation reference，而不是第一 internal base。

### PatchTST-style Patch Encoder

[Fact] PatchTST 的核心设计是 patching 和 channel-independence：把 time series 分成
subseries-level patches 作为 Transformer tokens，并对每个变量独立处理、共享模型参数。

[Strong Evidence] 这个设计满足本项目 internal base 的关键条件：

- 性能基础较强：patching 保留局部语义、降低 attention 计算量，并支持更长 lookback。
- 架构可拆：encoder、patch embedding、normalization、prediction head 边界清楚。
- 复杂度适中：相比 SRSNet、ElasTST、TIMEPERCEIVER，它没有内置 future-aware 或
  variable-horizon 机制。
- 与 SRSNet 可比：SRSNet 也是 patch perspective 的强方法，选择 patch base 让最终比较
  更公平，同时避免继承 SRS 的 adaptive patch selection。

[Inference] 它最适合作为 Phase 0 主 base。

## 与三个创新点的兼容性

### 对 variable-horizon decoder

Patch encoder 产生的 $Z \in \mathbb{R}^{(B C) \times N \times d}$ 是 history-side patch
states。Phase 0 使用 `FixedHead`：

$$
\hat{Y}_{1:H} = W_H \cdot \text{Flatten}(Z).
$$

这个 head 正好暴露固定 horizon projection 的局限。Phase 1 可以替换为：

$$
U_H = A_\theta(Q_H, Z, M_H),
\quad
\hat{Y}_{1:H} = O_\theta(U_H).
$$

因此，baseline 到 variable-horizon decoder 的变化是局部的：保留 patch encoder，
只替换 prediction interface。

### 对 future-aware mechanism

future-aware teacher branch 可以对齐在 decoder-side states，而不是强行改 encoder：

$$
S_H^{student}=P_\theta(U_H),
\quad
S_H^{teacher}=T_\psi(Y_{1:H}, Q_H).
$$

Patch encoder 只负责 history representation，不承担 future-aware 机制本身。这样可以
保持归因清晰：

- Phase 0：只有 history patch encoder + fixed head。
- Phase 1：加入 future query decoder。
- Phase 2：在 decoder states 上加入 future-aware alignment。

### 对 MoE

MoE 可以先不放在 input patch tokens 上，而是放在 future query / horizon segment states：

$$
r_h = \text{Router}(U_h, S_h, q_h),
\quad
\tilde{U}_h = \sum_k r_{h,k} E_k(U_h).
$$

这避免与 SRSNet 的 input-side selective patching 混淆。Patch encoder 提供稳定
history features，MoE 专注于 future-side mechanism selection。

## 推荐 Phase 0 配置

### 主模型

`PatchEncoderFixedHead`

- Normalization: RevIN 或 project-level instance normalization，先保持与 patch-based
  forecasting 常规设置一致。
- Patching: 固定 patch length 和 stride，不使用 adaptive patch selection。
- Encoder: 小型 Transformer encoder，避免过深。
- Head: fixed horizon linear head。
- Training: single-horizon 和 longest-horizon-prefix 两种 protocol 都保留。

### 必要对照

1. `DLinear`
   - 目的：确认模型至少超过简单强 linear baseline。
2. `PatchEncoderFixedHead`
   - 目的：后续机制的 internal base。
3. `PatchEncoderFixedHead + parameter-matched wider head`
   - 目的：后续如果机制变多，排除“只是参数更多”的解释。
4. `SRSNet`
   - 目的：论文重点 comparison baseline；不作为 internal base。

## Phase 0 必做诊断

### 基础性能

- MSE / MAE by dataset and horizon。
- Parameter count。
- Training time and GPU memory。
- Seed variance，至少在核心小集合上复跑。

### Horizon 诊断

- Error-by-horizon:

$$
e_h = \frac{1}{B C}\sum_{b,c}(\hat{Y}_{b,h,c}-Y_{b,h,c})^2.
$$

- Prefix consistency:

$$
\Delta(H_1,H_2)
=
\frac{1}{B H_1 C}
\left\|
\hat{Y}_{1:H_1}^{(H_1)}
-
\hat{Y}_{1:H_1}^{(H_2)}
\right\|_2^2.
$$

- Step-specificity:
  对 decoder-side states 或 head-row induced outputs 计算 horizon position 间的
  cosine similarity / CKA。Phase 0 如果只有 fixed head，则记录 head rows 或
  per-step output sensitivity。

### 失败判据

如果 `PatchEncoderFixedHead` 明显弱于 DLinear，说明 patch encoder 配置或训练 protocol
不可靠，不能继续作为 base。

如果 `PatchEncoderFixedHead` 基础性能接近 SRSNet，但 horizon diagnostics 无明显问题，
则 variable-horizon 创新点需要重新论证。

如果 `PatchEncoderFixedHead` 基础性能合理，但 prefix consistency 或 error-by-horizon
暴露明显缺陷，则进入 Phase 1 是合理的。

## 当前决定

[Inference] 当前选择：

- 主 internal base：`PatchEncoderFixedHead`。
- sanity floor：`DLinear`。
- 重点 external comparison：`SRSNet`。
- 暂不作为 base：`iTransformer`、`ElasTST`、`TIMEPERCEIVER`、`SRP++`。

[Hypothesis] 这个选择最可能在性能、可解释性、可扩展性之间取得平衡：
patch encoder 提供足够强的基础性能和与 SRSNet 的 patch-level 可比性；
fixed head 保留清晰缺陷，给 variable-horizon decoder、future-aware state 和 MoE 留出
明确改造空间。

## 下一步

1. 建立 `PatchEncoderFixedHead` 的最小本地实现，并同步写
   `docs/code-explanation`。
2. 建立 `DLinear` sanity baseline。
3. 定义 Phase 0 dataset / horizon matrix，先用小集合在本地跑通，再迁移到
   `529_Lab-3090`。
4. 在任何远程实验前执行 `scripts/remote/check_529lab_3090_gpus.sh`。

Phase 0 具体实验协议见 `docs/experiments/phase0-experiment-protocol.md`，机器可读配置见
`configs/phase0_protocol.json`。

## 外部来源

- PatchTST: `https://arxiv.org/abs/2211.14730`，official implementation:
  `https://github.com/yuqinie98/PatchTST`。
- DLinear / LTSF-Linear: `https://arxiv.org/abs/2205.13504`，official
  implementation: `https://github.com/vivva/DLinear`。
- iTransformer: `https://arxiv.org/abs/2310.06625`，official implementation:
  `https://github.com/thuml/iTransformer`。
- SRSNet: `https://arxiv.org/abs/2510.14510`，official implementation:
  `https://github.com/decisionintelligence/SRSNet`。
