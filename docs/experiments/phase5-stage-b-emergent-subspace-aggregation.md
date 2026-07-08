# Phase5 StageB B11 Emergent Subspace Aggregation

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B11-ESA` |
| `current_step` | Step 7：B11-BCF local implementation smoke passed |
| `problem` | A6 不应依赖显式 stage / horizon encoding；需要利用 `learned_temporal_basis` 自发形成的 continuous future geometry，让架构更自然地聚合 history information |
| `existence_evidence` | B10-TSI-A/B 显示 basis row spaces 已有分化、coeff 在不同 subspaces 上被差异化使用；B11 进一步证明 sliding-window basis subspaces 沿未来时间轴连续变化，coeff projection 也随窗口距离变化 |
| `idea` | 用 basis-induced continuous subspace descriptors 驱动 coefficient field / history aggregation，而不是把人工 stage token 加到 coeff |
| `theory_check` | `B11-BCF` 只在 A6 primary prediction path 内工作；目标是更有效利用 basis geometry，不改变 unified/prefix-consistent 立场 |
| `design` | `B11-BCF`: Continuous Basis-conditioned Coefficient Field |
| `narrative_gate` | `passed_for_minimal_local_implementation`; only for the continuous soft field with required controls |
| `effectiveness_gate` | `local_smoke_passed`; remote small gate not evaluated |
| `artifacts` | `analysis/phase5_stage_b_b11_esa_basis_coeff_diagnostic_20260708/b11_esa_basis_coeff_report.md`; `docs/code-explanation/phase5-stage-b-b11-esa-basis-coeff-diagnostic.md`; `docs/code-explanation/phase5-stage-b-b11-bcf.md`; `artifacts/smoke_phase5_stage_b_b11_bcf_local/b11_bcf_etth2/` |
| `decision` | `local_implementation_smoke_passed`; remote small gate required before any paper-core claim |

## Motivation

B9/B10 的失败暴露了一个叙事风险：显式 `stage token` 或 `target-set head` 会让模型看起来像
horizon-conditioned / stage-conditioned forecaster，而不是 unified multi-horizon model。

B11 的问题不是：

```text
stage_id -> coeff
```

而是：

```text
learned_basis geometry -> subspace descriptors -> history aggregation -> coeff/state
```

也就是说，future-region 信息不作为外部标签输入，而是来自 A6 自己学出的 basis geometry。

## Diagnostic Result

`B11-ESA` Step 2/3 诊断读取 clean A6 checkpoints，在不训练模型的情况下测试两个问题：

1. `learned_temporal_basis[720,256]` 是否自发形成 future geometry；
2. 真实 forward 中的 `coeff[B,C,256]` 是否沿这些 subspaces 差异化使用。

### Hard Row Clustering

KMeans row clustering 的结果并不充分稳定：

| Dataset | K=4 stage NMI | Projection cosine |
| --- | ---: | ---: |
| ETTh2 | `0.5325` | `0.2708` |
| ETTm1 | `0.0057` | `0.0483` |
| Weather | `0.0068` | `0.0146` |

[Interpretation] 这说明不能把 B11 简化成“basis rows 自发聚成四个 hard stages”。ETTm1/Weather 的
cluster 不是时间局部的，继续做 hard clustering 会重新落入 stage-like 分段叙事。

### Sliding-Window Subspace Geometry

更符合 unified 叙事的是 sliding-window subspace diagnostic。用 window length `96`、stride `48`、
rank `16` 得到：

| Dataset | Adjacent overlap | Far overlap | Distance-overlap Spearman | Adjacent proj cosine | Far proj cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | `0.3900` | `0.0649` | `-0.7016` | `0.5585` | `0.1504` |
| ETTm1 | `0.4021` | `0.0811` | `-0.5472` | `0.5391` | `0.2379` |
| Weather | `0.3810` | `0.0700` | `-0.2786` | `0.4071` | `0.0484` |

[Interpretation] A6 basis 沿未来时间轴形成连续变化的 subspace geometry：相邻 windows 的 subspace
overlap 高，远距离 windows 的 overlap 低。真实 `coeff` 在这些 subspaces 上的投影方向也随时间距离降低。

## Narrative Gate Implication

B11 与 B9/B10 的区别：

| Route | Mechanism | Narrative issue |
| --- | --- | --- |
| B9 | 人工 stage token 调制 `coeff` | 过于 stage-conditioned；no-stage control 阻断 |
| B10-C/D | frozen/offline target-set readout | readout/head 病态；不是 trainable native path |
| B11 | basis-induced continuous subspace aggregation | 与 unified model 更一致；仍需 Step 4-6 gate |

## Step 4-6 Design Requirements

B11 method design 必须满足：

1. 不输入 hard `stage_id` 或 `horizon_id`；
2. basis/subspace descriptors 来自 `learned_temporal_basis` 或其 smooth window/subspace representation；
3. history aggregation 发生在 primary prediction path 内，而不是 residual correction；
4. prefix consistency 必须可检查；
5. 必须设置 controls：
   - `no-basis`: 使用 learned constant slots，不看 basis geometry；
   - `shuffled-basis`: 打乱 basis-window/order 后生成 descriptors；
   - `constant-slot`: 同参数量但不随 future basis 改变；
   - `A6 fallback`: 初始化或 gate 关闭时回到 A6。

## Method Candidate: B11-BCF

`B11-BCF` 指 Continuous Basis-conditioned Coefficient Field。它保留 A6 的 learned basis prediction
形式，但不再用一个全局 `coeff[B,C,K]` 服务整个未来时间轴，而是生成一个沿未来 basis geometry
连续变化的 coefficient field。

### A6 Baseline Path

对 clean A6-LBF-r256，令：

- `hidden: [B,C,R]`，来自 TimeAlign history encoder；
- `basis: [720,K]`，其中 `K=256`；
- `bias: [720]`；
- `coeff_base = learned_basis_coeff(hidden): [B,C,K]`。

当 requested prefix 为 `H` 时：

$$
\hat y_{b,t,c} = basis_t^\top coeff\_base_{b,c} + bias_t,\quad t < H.
$$

这个路径的优点是 prefix-native、simple、functionally stable。限制是 `coeff_base` 对未来所有
positions 共享；future position 信息主要在 `basis_t` 中，sample-wise 信息主要在 `coeff_base`
中。B11 不把这个问题解释成“缺少 stage token”，而解释成：basis 已经学出连续 future subspaces，
但 history-to-coeff path 还没有显式利用这些 subspaces 来组织 sample-wise coefficient state。

### B11-BCF Path

B11-BCF 用 `basis` 自身构造 continuous descriptors。设 window length `L=96`、stride `S=48`，
得到 `M=14` 个 overlapping basis windows。每个 window descriptor 由 window 内 basis rows
经过同一个小 projector 得到：

- `basis_window_m: [L,K]`;
- `q_m = BasisWindowProjector(basis_window_m): [Dq]`;
- `Q = [q_1,...,q_M]: [M,Dq]`。

`Q` 不是人工 stage id，也不包含 requested horizon id。它来自当前模型自己的 temporal basis，因此它表达的是
learned future coordinate geometry。

最小实现先采用 hidden-level field，避免直接改 encoder attention：

1. `state_base = StateProjector(hidden): [B,C,D]`;
2. 对每个 basis window，用 shared mixer 生成 window-specific coefficient state：
   `state_m = Mixer(state_base, q_m): [B,C,D]`;
3. `coeff_m = CoeffHead(state_m): [B,C,K]`；
4. 每个 future row `t` 对 windows 做 smooth mixing，权重来自 `basis_t` 与 window descriptors
   的相似度或固定 local kernel：
   `alpha[t,m] = softmax(score(row_descriptor_t, q_m) / tau)`；
5. `coeff_field_t = sum_m alpha[t,m] * coeff_m: [B,C,K]`；
6. 输出：

$$
\hat y_{b,t,c} = basis_t^\top coeff\_field_{b,t,c} + bias_t,\quad t < H.
$$

该路径把 readout/head 从 single coefficient vector 扩展为 continuous coefficient field。这里的
readout 可以理解为 prediction head 的核心部分，但论文叙事不应写成“加了一个更大的 head”；关键是
`basis -> descriptor -> coeff_field -> basis projection` 形成了一个 basis-conditioned primary operator。

### Function-Preserving Initialization

B11-BCF 必须可退回 A6：

- 初始时令所有 window states 共享 A6 的 `coeff_base`；
- field branch 的新增投影或 mixing contribution 使用 zero-initialized scalar/vector gate；
- gate 关闭时，任意 `t` 都满足 `coeff_field_t = coeff_base`；
- 因此 `H=96` 直接跑 B11 与先跑 `H=720` 再取 prefix 的输出必须一致。

实现上可以用 gated path 保证 fallback，但论文叙事不写成 residual correction。它是 primary coefficient
field 的 function-preserving initialization path，不是 `A6(x) + error repair`。

### Why This Is Not Stage Conditioning

B9/B10 的问题在于显式输入 `stage_id` / `target_set_id` 会让模型依赖人为分段。B11-BCF 的不同点是：

- descriptor 来自 learned basis 的 continuous geometry；
- future row 的 mixing 是 soft and smooth，不是 hard cluster；
- benchmark horizons 只是 evaluation prefixes，不决定模型内部计算路径；
- 同一个 `720` field 在所有 horizons 下共享，prefix consistency 是硬约束。

### Controls

小规模实现必须同时提供以下 controls，否则无法判断 gain 来源：

| Control | Design | What It Tests |
| --- | --- | --- |
| `a6_fallback` | gate 关闭或 field rank 为 1 | B11 是否 function-preserving；排除实现误差 |
| `no_basis` | `q_m` 换成同数量 learned constant slots | 如果它等于 B11，说明收益来自容量/多槽 head，不是 basis geometry |
| `shuffled_basis` | descriptor 保持数值分布但打乱 window order | 如果它等于 B11，说明连续 future order 不是关键 |
| `constant_slot` | 所有 `t` 使用相同 slot mixture | 如果它等于 B11，说明 row-wise continuous field 没起作用 |
| `a6_clean` | 当前 clean A6-LBF-r256 | performance anchor |

`no_basis` 是最重要的 mechanism control。B9 的失败已经说明：同参数量 no-condition control 能吃掉
微小收益时，不能把结果解释为 future-aware mechanism。

## Theory Check

[Strong Evidence] B11 诊断证明 hard clusters 不稳，但 sliding-window subspaces 有连续几何：
相邻/远距离 overlap 分别约为 `0.3911/0.0720`，projection cosine 也随 window distance 降低。

[Hypothesis] 如果 basis subspace geometry 真是 A6 已学到的 useful future coordinate system，那么让
history-to-coeff path 看到 smooth basis descriptors，应比 no-basis/constant controls 更有效地生成
future-position-aware coefficient field。

[Counter-Argument] B11 也可能只是扩大 prediction head 容量。尤其 Weather 的
distance-overlap Spearman 只有 `-0.2786`，说明 continuous geometry 较弱。若 `no_basis` 或
`constant_slot` control 持平或更好，则 B11 不能作为 paper-core method。

## Local Verification Plan

Step 7 local implementation 完成前，最小可验证版本必须通过：

1. `python -m py_compile` on touched Python files；
2. A6 fallback check：同 checkpoint/同输入下，gate 关闭时 B11 与 A6 max abs diff 为 `0.0` 或数值容差内；
3. Prefix consistency check：`H=96` output 与 `H=720` output prefix 的 max abs diff 为 `0.0` 或数值容差内；
4. ETTh2 one-batch CPU smoke：`b11_bcf`、`b11_no_basis`、`b11_shuffled_basis`、`b11_constant_slot`
   至少能 forward/backward。

## Remote Small Gate

远程前必须 commit/push，并按 GPU policy 做 `nvidia-smi` preflight。

最小 small gate 不做 full matrix。建议：

- Datasets: ETTh2, ETTm1, Weather；
- Horizons: 96, 192, 336, 720；
- Arms: `a6_clean`, `b11_bcf`, `b11_no_basis`, `b11_constant_slot`；
- Optional arm: `b11_shuffled_basis`，若 GPU 时间不足可在 `b11_bcf` 初步正向后补跑；
- Seed: 与 clean A6/B9 small gate 对齐。

Effectiveness gate：

1. `b11_bcf` 相对 `a6_clean` 的 overall mean MSE 不劣化，并最好有稳定 wins；
2. `b11_bcf` 必须优于 `no_basis` 与 `constant_slot`，否则不能 claim basis-conditioned mechanism；
3. Weather 不允许出现明显 numerical pathology 或单数据集灾难性退化；
4. 即使 MSE 正向，若 `no_basis` control 持平，则只能记录为 capacity/head effect，不进入 paper-core。

## Decision

`B11-ESA` 的 Step 4-6 narrative gate 只对 `B11-BCF` 通过。该设计与 StageA 的连接是：

- StageA 给出 learned basis unified forecast operator；
- B11 不再外加人工 stage/horizon condition；
- B11 研究如何让 learned basis 自身形成的 continuous future geometry 进入 primary coefficient field；
- 这可以构成第二个架构贡献候选，但必须先通过 no-basis / constant-slot controls。

## Step 7 Local Implementation Result

已实现 readout modes：

- `basis-conditioned-coefficient-field`;
- `basis-conditioned-coefficient-field-no-basis`;
- `basis-conditioned-coefficient-field-shuffled-basis`;
- `basis-conditioned-coefficient-field-constant-slot`。

本地验证结果：

| Check | Result |
| --- | ---: |
| `py_compile` | passed |
| A6 fallback H96 max abs | `3.695488e-06` |
| B11 H96 vs H720 prefix max abs | `0.000000e+00` |
| B11/control synthetic backward | passed |
| ETTh2 one-batch CPU smoke | passed |

Code explanation:

- `docs/code-explanation/phase5-stage-b-b11-bcf.md`。

下一步进入 remote small gate preparation。不得实现 hard cluster/stage variant；不得在 remote small gate
缺少 `no_basis` / `constant_slot` controls 时做 paper-core claim。
