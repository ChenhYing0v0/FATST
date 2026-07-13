# Phase4: Horizon-Agnostic Supervision Reset

[Status] 已被
`docs/experiments/phase4-horizon-supervision-scheduling-r3-reset.md`
取代。本文仅保留为历史 diagnostic evidence；其中 `Component-Space Supervision`
结论不再是当前 active Phase4 route。

`current_step`: 11-step loop 的 Step 1-3，暂不进入实现。

## Reset Decision

[Decision] 之前的 Phase1-Phase3 实验结果只作为 diagnostic evidence 使用，不作为新主线的
边界条件。当前不继续围绕 `regime_segment_operator`、QDF matrix transfer、pairwise horizon
map 或 MoE 堆叠方案推进。

[Decision] 本轮从 Step 1 重新开始，研究对象从 fixed evaluation horizons
`{96,192,336,720}` 扩展为更一般的 training supervision units：

- time-domain future positions；
- contiguous future intervals；
- randomly sampled target sets；
- decorrelated / low-rank future components；
- curriculum 或 stochastic schedule over these units。

## Step 1: 调研分析

### 文献证据

[Strong Evidence] `QDF / MetaDF` 说明 standard temporal MSE 等价于 identity weighting，
忽略 future-step autocorrelation 和 heterogeneous task weights。它支持“future steps 不应等权”
这一 objective-side premise，但其 full learned quadratic objective 不应被本项目直接照搬。

[Strong Evidence] `TransDF` 进一步把问题从 weighting 推到 supervision unit 本身：forecast
horizon 变长会造成 excessive step-wise tasks，导致 multi-task optimization difficulty；将
labels 投影到 decorrelated significant components 可以减少 task amount，并保留 inference-time
forecast form。

[Strong Evidence] `ElasTST` 支持 varied-horizon inference 与 horizon-invariance 测试。它的价值
不是让本项目复刻 placeholders，而是提醒我们：请求不同 horizon 不应任意改变已有 prefix
forecast；因此 evaluation horizons 是能力集合，而不是训练监督必须镜像的单位。

[Strong Evidence] `TIMEPERCEIVER` 把 target index set $J$ 显式输入 decoder query，说明目标
positions 可以作为模型条件，而不必绑定到固定 output head 或固定 horizon run。

[Strong Evidence] `SRP++` 指出 step-invariant representation 有 expressiveness bottleneck，
但它仍偏向 step/segment-specific adaptation。对当前项目来说，它更适合作为 diagnostic：如果
不同 future units 需要不同 representation，应先证明这种 heterogeneity 来自 supervision conflict。

### 本项目 diagnostic 证据

[Strong Evidence] `PatchEncoderFixedHead` specialist 仍然非常强；多轮轻量 architecture/operator
patch 没有稳定成为 paper-core。

[Strong Evidence] Phase3-C history-only operator 在 `h96,h720` reduced set 下 positive，但在完整
`96,192,336,720` 下 fail。这说明 operator 不是充分机制；training supervision composition
改变会强烈影响结果。

[Strong Evidence] `PatchEncoderPrefixRiskWeightedH96H720` control 显示 removing `192/336`
会改善 H720 segment gaps，但同时损伤 H96 平均表现。因此 pairwise horizon map 只能解释局部
现象，不能作为最终研究问题本身。

[Inference] 当前最稳的抽象是：我们不是在寻找哪个 evaluation horizon pair 最好，而是在寻找
什么样的 future supervision basis 能在统一模型中降低冗余、冲突和过度任务化。

## Step 2: 待解决问题

旧问题：

> 哪些 evaluation horizon 组合会互相干扰？

[Decision] 这个问题过窄。它被 benchmark horizon choices 锁死，容易把方法设计限制为
`96/192/336/720` 的组合搜索，也不够像高水平论文的核心问题。

新问题：

> In unified multi-horizon forecasting, evaluation horizons define where we
> measure forecasting ability, but they are not necessarily the right units for
> training supervision. What horizon-agnostic supervision basis can reduce
> redundant or conflicting future-step tasks while preserving full-horizon
> evaluation accuracy?

中文表述：

> 在统一 multi-horizon forecasting 中，评估 horizon 只是测试能力的位置；训练监督不应默认
> 逐 horizon 镜像这些位置。我们要研究的是：是否存在更合适的、horizon-agnostic 的 future
> supervision basis，使模型在完整评估 horizon 上更稳定、更高效地学习。

## Step 3: 问题是否真实且值得研究

[Fact] 问题真实性的外部证据来自 QDF/TransDF：direct multi-step objective 的 future-step
依赖、label autocorrelation、task amount 和 optimization conflict 已经被近期工作明确指出。

[Fact] 问题真实性的内部证据来自 Phase3 controls：同一个 carrier 在不同 training horizon set
下表现差异明显，且 reduced set 对 H720 与 H96 的影响方向不同。

[Strong Evidence] 问题值得研究，因为它把以下三件事统一起来：

1. multi-horizon one-model 的训练目标；
2. varied-horizon / prefix consistency 的接口约束；
3. objective-side future-step dependency。

[Risk] 该问题可能仍然被证明只是 objective regularization，而不是新 architecture。为避免过早
写成 architecture story，Step 4-6 应先设计 diagnostics 和 supervision protocols，而不是
直接实现复杂 model。

## Step 4 Candidate Ideas

### Candidate A: Pairwise Horizon Interference Map

[Decision] 降级为 diagnostic，不作为主线。

优点：成本低，能回答 `96/192/336/720` benchmark 内部的局部冲突。

缺点：仍以 evaluation horizons 为训练单位，不能回答 horizon-agnostic supervision 的核心问题。

### Candidate B: Random Target-Interval Supervision

[Hypothesis] 每个 batch 随机采样一个或多个 future intervals `[a,b]`，而不是固定采样
`H in {96,192,336,720}`。训练覆盖连续 future positions，评估仍在 `96,192,336,720`。

优点：实现相对接近 target-set carrier；不被固定 horizon pairs 锁死。

风险：如果 interval sampling 没有理论结构，可能只是 stochastic regularization。

### Candidate C: Component-Space Supervision

[Hypothesis] 使用 train-label SVD/PCA 或其他 orthogonal basis，将 future sequence 投影为
decorrelated components，并按 component significance 分配监督。评估仍回到 time-domain
`96,192,336,720`。

优点：最符合 QDF/TransDF 证据；horizon-agnostic；有明确理论口径：降低 label autocorrelation
和 task amount。

风险：可能偏向低频/高方差成分，损伤短期局部波动，需要 segment-level 与 high-frequency
diagnostics。

### Candidate D: Curriculum Over Supervision Basis

[Hypothesis] 先训练低秩 / 长尺度 future components，再逐步引入 full time-domain steps 或
horizon-specific constraints。

优点：可以解释为 optimization curriculum，而不是固定 loss trick。

风险：实验维度较多，应在 Candidate C 的 label-side diagnostics 通过后再做。

## Step 5: 理论可行性初评

[Decision] 当前最值得优先推进的是 Candidate C：`Component-Space Supervision`。

理由：

1. 它不按 evaluation horizons 划分，回应了 horizon-agnostic 要求。
2. 它与 QDF/TransDF 的 objective-side 证据一致，但不直接复制任何 upstream method。
3. 它允许把旧实验作为 diagnostic：pairwise/reduced-horizon results 可用于检查哪些 time-domain
   segments 被 component supervision 改善或损伤。
4. 它有清晰反例：如果 top components 只改善 H720 trend 而系统性损伤 H96，则该方向应回滚。

可行的数据流：

$$
Y \in \mathbb{R}^{B \times T \times D},\qquad \hat{Y}=f_\theta(X)
$$

将 label/prediction reshape 到 training label basis：

$$
Y' \in \mathbb{R}^{(B D) \times T},\qquad \hat{Y}' \in \mathbb{R}^{(B D) \times T}
$$

用 train split 学习 orthogonal projection：

$$
P \in \mathbb{R}^{T \times K}
$$

投影监督：

$$
Z=Y'P,\qquad \hat{Z}=\hat{Y}'P
$$

候选 loss：

$$
\mathcal{L}
=
(1-\alpha)\lVert \hat{Y}-Y\rVert_2^2
+
\alpha \lVert \hat{Z}-Z\rVert_1
$$

其中 $K$ 或 component weights 是 training supervision schedule，而不是 evaluation horizon set。

## Step 6: 下一步最小推进方案

[Decision] 下一步不直接跑远程训练。先做 label-side diagnostic，判断 component-space
supervision 是否有足够真实结构。

### Diagnostic 1: Label Basis Audit

目标：在 train split 上评估 future label sequence 的 redundancy 和 component structure。

输出：

- top component explained variance ratio；
- effective rank；
- off-diagonal step covariance / correlation；
- component-to-time energy heatmap；
- component-to-evaluation-horizon contribution；
- dataset-level stability across `ETTh2`, `ETTm1`, `Weather`。

Gate：

- 若 top components 能解释大部分 label variance，且 component-to-time energy 不是纯 identity，
  则 Candidate C 进入 Step 6 implementation design。
- 若 component basis 接近 identity 或 effective rank 接近 full rank，则回退到 Candidate B
  random interval supervision。

### Diagnostic 2: Existing-Residual Projection Audit

目标：不训练新模型，读取 R.3 / fixed head 现有 residual，判断 component-space error 是否能解释
已观察到的 horizon/segment gaps。

输出：

- time-domain MSE vs component-domain loss ranking；
- top-component residual energy；
- high-frequency / low-frequency component residual split；
- H96 / H720 segment gap 的 component attribution。

Gate：

- 若 component-domain loss 能区分 known gap cases，则进入 component-supervised training。
- 若不能区分，则 component supervision 不应成为第一实现。

## 11-Step Record

- `current_step`: Step 1-3 complete，Step 4-6 初步候选形成；Label Basis Audit 已完成，
  暂不进入远程训练。
- `problem`: evaluation horizons are not necessarily the right training supervision units.
- `existence_evidence`: QDF/TransDF external evidence；Phase3 reduced/full controls as diagnostic evidence。
- `idea`: horizon-agnostic supervision via future components, random intervals, or curriculum.
- `theory_check`: component-space supervision is currently strongest because it reduces autocorrelation and task amount.
- `design`: first run label-side and existing-residual diagnostics; no model training yet.
- `gate`: component basis must show nontrivial low-rank/decorrelated structure and explain existing residual gaps.
- `artifacts`: this document; `Papers/transdf-transformed-label-alignment.md`;
  `analysis/phase4_label_basis_audit_20260624/phase4_label_basis_report.md`。
- `decision`: Label Basis Audit 通过；下一步进入 Existing-Residual Projection Audit，而不是
  pairwise horizon training。

## Diagnostic Result: Label Basis Audit

[Fact] 已实现并运行 `scripts/analyze_phase4_label_basis_audit.py`。该脚本只读取 train split
future labels，不训练模型，不读取 validation/test labels。

主要结果：

| Dataset | Effective rank | Top16 variance | Top32 variance | Top64 variance | Mean abs offdiag corr |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ETTh2` | `11.47` | `0.873` | `0.916` | `0.948` | `0.532` |
| `ETTm1` | `9.74` | `0.881` | `0.940` | `0.967` | `0.554` |
| `Weather` | `24.18` | `0.790` | `0.833` | `0.865` | `0.411` |

[Strong Evidence] `pred_len=720` 的 train-label sequence 不是接近 identity/full-rank 的独立
step tasks。它存在强 off-diagonal correlation 和低 effective-rank structure；因此
`Component-Space Supervision` 进入下一步 diagnostic 是合理的。

[Inference] 这也解释了为什么不应优先做 pairwise horizon map：未来序列的主要结构不由
`96/192/336/720` 这些离散 evaluation horizons 定义，而是由跨整个 future sequence 的低秩
component basis 定义。

下一步：

1. 读取现有 R.3 / fixed head residual artifacts；
2. 将 residual 投影到 label-basis components；
3. 判断已知 horizon/segment gaps 是否集中在少数 components 或特定 component groups；
4. 若能解释 known gaps，再进入 component-supervised training loss 设计。

## Diagnostic Result: Existing-Residual Projection Audit

[Fact] 已实现并运行 `scripts/analyze_phase4_residual_projection_audit.py`。该脚本读取已有
R.3 `predictions_test.npz`，不训练模型，不补跑远程实验。

主要结果：

| Metric | Value |
| --- | ---: |
| specialist gap rows | `4/12` |
| gap mean residual top16 energy share | `0.789` |
| non-gap mean residual top16 energy share | `0.828` |
| gap minus non-gap top16-over-label ratio | `-0.054` |
| H720 segment gap top16 reconstruction share | `0.716` |
| H720 segment non-gap top16 reconstruction share | `0.714` |

[Decision] Component-space residual structure 真实存在，但 known gaps 并没有更集中在 dominant
top components。Top-only TransDF-style loss 不应作为第一候选。

[Decision] 下一步进入 `Component-Balanced Objective` 设计：

- design document:
  `docs/experiments/phase4-component-balanced-objective-design.md`;
- code explanation:
  `docs/code-explanation/phase4-residual-projection-audit.md`;
- report:
  `analysis/phase4_residual_projection_audit_20260624/phase4_residual_projection_report.md`。
