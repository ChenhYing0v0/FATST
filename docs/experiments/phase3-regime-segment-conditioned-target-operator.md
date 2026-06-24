# Phase3-C Regime/Segment-Conditioned Target Operator

更新时间：2026-06-24

## 1. 研究位置

`current_step`: 11-step loop 的 Step 4-6。

[Fact] Phase3-A 已证明 R.3 的剩余 gaps 不来自同一输入下的 prefix prediction conflict，而是来自
short-only extra windows 与 H720 late segments。

[Fact] Phase3-B 已证明这些 failure groups 能被 prediction-before features 分离：

- `ETTm1/96` short extra windows: `history_mean` AUC `0.997619`;
- `Weather/96` short extra windows: `window_index_norm` AUC `1.000000`, `history_std` AUC `0.979425`;
- `ETTh2 337-720` high-error late segment: `window_index_norm` AUC `0.845886`,
  `history_slope_abs_mean` AUC `0.828835`。

[Decision] 下一步不是 output residual correction。Residual/error 只作为诊断标签；模型机制必须发生在
prediction 生成之前。

## 2. Problem

R.3 的 target-set decoder 已经保持 prefix consistency，但仍在两类区域弱于 fixed specialist：

1. short horizon 的额外 test windows；
2. H720 的 late target segments。

这说明同一个 target-set carrier 对不同 history regime 和 target segment 的 operator 可能过于同质。

## 3. Core Idea

引入轻量的 `Regime/Segment-Conditioned Target Operator`：

$$
z=\operatorname{RegimeMLP}(\phi(x), p),
$$

$$
g_j=\operatorname{Gate}(z, q_j),
$$

$$
\tilde{U}_j=\operatorname{TargetOperator}(U_j, q_j, g_j),
$$

$$
\hat{Y}_{a_j:b_j}=\operatorname{Readout}(\tilde{U}_j).
$$

其中：

- $x$ 是 history input window；
- $\phi(x)$ 是 history summary；
- $p$ 是 prediction-before window position；
- $q_j$ 是 target segment feature；
- $U_j$ 是 R.3 target-set state；
- $\tilde{U}_j$ 在 output readout 前生成。

## 4. Mechanism Boundary

允许：

- history/window-position 生成 regime token；
- target segment feature 控制 segment operator；
- 对 target state 或 readout 前 hidden state 做 FiLM / low-rank / gated operator；
- gate diagnostics 记录 gate 与 history statistics、window position、segment error 的关系。

不允许：

- prediction 后直接加自由 residual correction；
- 使用 future target 构造 prediction path；
- 用只针对 observed gaps 的 dataset-specific if-else；
- 在 Phase3-C gate 前启动 full MoE。

## 5. Candidate Implementation

候选名：

- `PatchEncoderRegimeSegmentTargetOperator`

建议从最小结构开始：

1. 沿用 `PatchEncoderPrefixRiskWeighted` 的 patch encoder、target features、target states 与 base readout。
2. 从 input history 计算 compact summary：mean、std、recent mean/std、slope magnitude。
3. 拼接 `window_index_norm` 得到 `regime_token`。
4. 将 `regime_token` 与每个 segment 的 `q_j` 输入小 MLP，产生 bounded gate。
5. Gate 只调制 readout 前 hidden state，例如 FiLM scale/shift 或 low-rank delta operator。
6. 初始化为 near-identity，使初始行为接近 R.3。

## 6. Gate

Primary baseline: R.3 `PatchEncoderPrefixRiskWeighted`。

Pass 条件：

1. observed gap settings 至少修复 3/5：
   `ETTm1/96`, `Weather/96`, `ETTh2 193-336`, `ETTh2 337-720`, `ETTm1 337-720`。
2. mean MSE vs R.3 不劣于 `+0.2%`。
3. non-gap mean MSE 不劣于 `+0.2%`。
4. prefix consistency mismatch 保持数值零级别。
5. gate diagnostics 显示 gate 与 prediction-before regime/segment signals 有可解释关系。

Fail 条件：

1. 改善只来自个别 dataset noise；
2. gate 退化为纯 `window_index_norm` lookup，history statistics 不贡献解释；
3. non-gap 明显退化；
4. prefix consistency 被破坏；
5. 需要 output residual correction 才能取得收益。

## 7. Experiment Plan

1. 实现最小 model candidate 与 manifest 注册。
2. 本地 smoke：forward、loss、prefix consistency、gate artifact shape。
3. 远程最小 gate：优先 `ETTm1`, `ETTh2`, `Weather`，覆盖 `96/720`。
4. 若最小 gate 通过，再扩展到完整 4 horizons。
5. 若失败，回滚到 Step 2-3：重新判断问题是否应转向 base architecture 或 external baseline selection。

## 8. Control Plan After Window-Index Concern

当前 `window_index_norm` run 只能作为 positive but confounded evidence。下一步必须拆分两个因素：

1. `history_only_h96_h720`: `USE_WINDOW_POSITION=0`, `TARGET_HORIZONS=96,720`。
   目的：判断收益是否依赖 split-position shortcut。
2. `history_only_h96_h192_h336_h720`: `USE_WINDOW_POSITION=0`,
   `TARGET_HORIZONS=96,192,336,720`。
   目的：在与 R.3 相同 horizon set 下判断结构收益。
3. 可选 `window_h96_h192_h336_h720`: `USE_WINDOW_POSITION=1`,
   `TARGET_HORIZONS=96,192,336,720`。
   目的：只隔离 horizon-set confound。

若 `history_only_h96_h720` 后收益大幅消失，则当前 positive result 应视为 split-position shortcut
证据，不应继续该机制。若 history-only 仍保留大部分收益，再进入完整 horizon-set control。
