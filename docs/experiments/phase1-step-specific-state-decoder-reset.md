# Phase1-A.5 Step-Specific State Decoder Reset

## 定位

[Fact] Phase1-A.1 到 Phase1-A.4 已经验证了三条重要负结果：

1. `PatchEncoderSegmentQueryHead` 直接替换 fixed flatten head 后显著弱于
   `PatchEncoderFixedHead`。
2. `PatchEncoderFixedHeadAdapter` 保留 fixed head 后可以产生非零修正，但收益不稳定。
3. `PatchEncoderFutureAwareAdapter` 与 repair gate 证明 teacher/student alignment 和
   leakage boundary 可以成立，但 forecasting gain 仍不足以作为 paper-core。

[Decision] 因此下一轮不再继续修补 post-head affine adapter，也不直接进入 MoE。
本 reset 回到长研究模板的 step 1-6：重新定义 decoder 需要解决的问题、评估该问题是否真实、
提出更贴近底层 prediction process 的候选架构。

## Step 1: 调研分析

当前 notes 给出三个互相制约的事实：

- [Strong Evidence] ElasTST 证明 varied-horizon / horizon-invariance 是合理诊断，但其重点
  是 consistency，不等价于更强的 forecasting accuracy。
- [Strong Evidence] TimeAlign 说明 training-only future branch 可以作为 future distribution
  anchor，但它本身不解决 decoder readout 处的 representation bottleneck。
- [Strong Evidence] SRP++ 指出 multi-step forecasting 的核心风险是
  step-invariant representation：不同 future steps 或 segments 可能需要不同 hidden
  representations，而不只是不同 output rows。
- [Strong Evidence] TIMEPERCEIVER 的 target query 思想说明 target positions 应进入 decoder
  computation，而不是只作为 output dimension。
- [Strong Evidence] Seg-MoE 与 MoHETS 支持 MoE 的 routing unit 和 expert operator 应带有
  time-series inductive bias；MoE 不应在弱 decoder 后作为参数补偿层。

## Step 2: 待解决问题

当前更精确的问题不是：

> fixed head 是否缺少一个 future-side interface？

而是：

> fixed head 是否把所有 future steps 绑定到同一个 history representation，使不同 future
> segments 只能通过 output rows 区分，而不能在进入 readout 前形成 step/segment-specific
> representations？

用符号表示，当前 base 近似为：

$$
Z = E_\theta(X), \qquad
\hat{Y}_{1:H}=W_H\operatorname{Flatten}(Z).
$$

这里每个 step 的 readout row 不同，但它们共享同一个 $Z$。如果某些 future segments 需要不同
状态，例如 trend-dominant、periodic-dominant 或 regime-shift-sensitive state，那么仅改变
$W_H$ 的 row 可能不足。

## Step 3: 问题是否存在且值得研究

[Strong Evidence] Phase0 segment oracle 显示，同一个 `0-720` 区间内，局部最佳 checkpoint
在 segments 之间变化；`pred_len=720` 全局最好，但不是每个 segment 的局部最优。

[Strong Evidence] Phase1-A.1 的失败说明简单 segment query 不能牺牲 fixed head 的 readout
capacity。换言之，问题不是 “fixed head 太强所以不能动”，而是新 decoder 必须保留它的强
readout capacity。

[Strong Evidence] Phase1-A.2 到 A.4 的 partial results 说明 post-head correction 太靠后：
它能修正输出，但没有改变 readout 前的 representation，因此很难稳定解决
step-specific representation bottleneck。

[Inference] 这个问题值得研究，因为它同时满足三点：

1. 与已有论文的理论主张对齐：SRP++ 的 step-specific representation 与 TIMEPERCEIVER 的
   target query 都指向 decoder-side state。
2. 与本项目负结果对齐：直接 query readout 删除容量会失败，post-head affine correction
   只得到 partial repair。
3. 能形成后续统一框架：future-aware alignment 可以约束 segment-specific state，MoE 可以
   作为 segment-specific state transition operator。

## Step 4: Candidate Idea

候选 idea 暂命名为 `Step-Specific State Decoder`，本地实现候选名为
`PatchEncoderStepSpecificStateAdapter`。

核心改动是把 adaptation 放在 readout 之前：

$$
Z=E_\theta(X),
$$

$$
U_j=A_\theta(q_j, Z),
$$

$$
\tilde{Z}_j = T_\theta(Z, U_j),
$$

$$
\hat{Y}_{a_j:b_j}=W_{a_j:b_j}\operatorname{Flatten}(\tilde{Z}_j).
$$

其中 $j$ 是 future segment index，$q_j$ 是 segment query，$U_j$ 是 future segment state，
$T_\theta$ 是 segment-conditioned state adapter，$W_{a_j:b_j}$ 复用 fixed head 对应
segment 的 readout rows。

第一版 $T_\theta$ 不使用 MoE，只使用 zero-initialized token-channel FiLM：

$$
\tilde{Z}_j = Z \odot (1+\gamma_j) + \beta_j,
$$

$$
(\gamma_j,\beta_j)=R_\theta(U_j), \qquad
\gamma_j,\beta_j \in \mathbb{R}^{d}.
$$

这与 Phase1-A.2 的关键差异是：

- Phase1-A.2: 先 fixed head 输出 $\hat{Y}^{base}$，再在 output space 做 affine correction。
- Phase1-A.5: 先生成 segment-specific representation $\tilde{Z}_j$，再用 fixed head rows
  读出该 segment。

## Step 5: 理论可行性

[Hypothesis] 如果 fixed head 的主要优势来自 dense readout rows，那么复用 $W_{a_j:b_j}$
可以避免 SegmentQueryHead 的容量坍塌。

[Hypothesis] 如果当前瓶颈来自 step-invariant $Z$，那么在 readout 前构造
$\tilde{Z}_j$ 比 output-space affine correction 更直接；它改变的是 readout 输入，而不是
只改变 readout 结果。

[Hypothesis] zero initialization 使初始 forward 等价于 fixed head：

$$
\gamma_j=0,\ \beta_j=0
\Rightarrow
\tilde{Z}_j=Z
\Rightarrow
\hat{Y}_{a_j:b_j}=W_{a_j:b_j}\operatorname{Flatten}(Z).
$$

因此该候选的优化风险比直接替换 head 更低，且性能下界应接近 fixed head。

[Risk] 如果 $T_\theta$ 学到的只是很小的 uniform scaling，说明 step-specific state 并未产生
实质作用。该情形应回退到问题定义，而不是继续加 MoE。

## Step 6: 第一版实验方案

第一版仍使用 one-to-one horizon training，避免 mixed-horizon optimization 干扰机制判断。

对比模型：

- `PatchEncoderFixedHead`
- `PatchEncoderFixedHeadAdapter`
- `PatchEncoderStepSpecificStateAdapter`

实验矩阵：

- datasets: `ETTh2`, `ETTm1`, `Weather`
- horizons: `96`, `192`, `336`, `720`
- seed: `2021`
- segment length: `48`

第一版 pass 条件：

1. [Performance] 相比 `PatchEncoderFixedHead` 至少 `6/12` main MSE wins，且 mean relative
   MSE < 0。
2. [Stability] 不出现单个 dataset 全 horizon 系统性退化；若 Weather 继续全负，应判定失败。
3. [Mechanism] segment-conditioned $\gamma,\beta$ 或 $\tilde{Z}_j-Z$ 的统计不能完全退化；
   segment similarity 需要显示不同 segment state 有可分性。
4. [Capacity control] 参数量必须与 `PatchEncoderFixedHeadAdapter` 可比；若明显增加，需要
   加入 parameter-control。

如果该候选通过，再进入：

- Future-aware state alignment: 对齐 $U_j$ 或 $\tilde{Z}_j$，而不是对齐 post-head output adapter。
- Future-side MoE: 把 $T_\theta$ 替换为 shared dense path + routed heterogeneous operators。

如果该候选不通过，应回退到 step 2-3，重新判断 decoder-side state 是否是当前项目的主线；
不应继续把 MoE 叠到失败的 state adapter 上。
