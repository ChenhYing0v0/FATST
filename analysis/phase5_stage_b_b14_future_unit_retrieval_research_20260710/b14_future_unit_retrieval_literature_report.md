# Phase5 StageB B14 Future-Unit Retrieval Literature And Boundary Audit

> Status update 2026-07-10：retrieval novelty boundary仍有效。full contextual replacement已失败；exact
> A6-preserving hierarchical `P48-S24` memory已通过 equivalence gate。为保证 evidence semantics，B14 Step 3
> 基于 29 个无 padding 的 complete canonical patch supports启动，trainable retrieval仍未授权。

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B14-FURD`：Future-Unit Retrieval Demand |
| `current_step` | Step 1 literature audit and Step 2 problem definition |
| `scope` | problem/narrative boundary only；no successor model implementation |
| `decision` | target-query/retrieval is established prior art；only a carrier-specific retrieval-demand mismatch may justify further diagnostics |

## Source Status

本轮首先读取 Zotero-derived full-text notes：

- `Papers/elastst-varied-horizon.md`；
- `Papers/timeperceiver-generalized-forecasting.md`。

并复核本仓库 2026-07-07 已完成外部 primary-source verification 的记录：

- `docs/experiments/phase5-stage-b-native-future-stage-operator.md`；
- `analysis/phase5_stage_b_future_query_aligned_architecture_research_20260707/stage_b_architecture_direction_report.md`。

其中已核验的 primary / official sources 包括：

- ElasTST：<https://arxiv.org/abs/2411.01842>；
- TimePerceiver：<https://arxiv.org/abs/2512.22550>；
- TimePerceiver official repository：<https://github.com/efficient-learning-lab/TimePerceiver>；
- MQ-RNN：<https://arxiv.org/abs/1711.11053>；
- Temporal Fusion Transformer：<https://arxiv.org/abs/1912.09363>。

[Limitation] 2026-07-10 的 fresh web search 连续两次返回 network error，随后 arXiv 与 GitHub raw direct
requests 也出现 SSL connection failure。因此本轮没有把 fresh network response 写成新证据；结论依赖已读
Zotero full text 与三天前的 repo-recorded primary-source verification。论文内容本身属于低 drift evidence，
但进入 Step 4-6 前仍应恢复网络后再次核对最新版本与 official implementation。

## Prior-Art Mechanism Map

### ElasTST

[Strong Evidence] ElasTST 已经用 future placeholders、structured self-attention mask、multi-patch assembly
与 horizon reweighting 解决 varied-horizon inference 和 horizon invariance。

[Boundary] “requested horizon 决定 future placeholder 数量”不是本项目可独占的 novelty。B14 不能只把
placeholder 从 step 改成 segment/unit 后声称新架构。

### TimePerceiver

[Strong Evidence] TimePerceiver 的 target timestamp queries 通过 decoder cross-attention 从 input
representations 中 retrieve information，并支持 arbitrary target segments。

[Boundary] “future query 从 history tokens 检索信息”本身已经是直接 prior art。B14 不能把普通 query
cross-attention 写成贡献。

### MQ-RNN

[Strong Evidence] MQ-RNN 显式构造 horizon-specific contexts，并使用 shared local MLP 处理不同 horizon。

[Boundary] horizon/unit-specific context 也不是新概念。B14 必须证明标准 LTSF、无 known-future covariates、
coarse benchmark-independent units 与 active A6 carrier 之间存在新的具体矛盾。

### Temporal Fusion Transformer

[Strong Evidence] TFT 是 multi-horizon architecture，并区分 static、observed history 与 known future
inputs，通过 feature selection 与 attention 建模不同时间位置的信息需求。

[Boundary] “不同 future positions 需要不同信息”是已有共识；本项目仍需提供 A6-specific problem evidence，
不能只凭该直觉实现模块。

## Local Negative-Evidence Controls

### Phase1 SegmentQueryHead

[Fact] `PatchEncoderSegmentQueryHead` 直接用 future segment queries 替换 fixed dense head 后：main MSE
`0/12` wins、segment MSE `0/30` wins、平均退化 `+6.79%`。

[Attribution] 其 segment length `48` 较小，且 head parameter ratio 明显下降；它证明不能牺牲 strong
readout capacity，不足以否定 large-unit retrieval。

### Phase1 Step-Specific State Adapter

[Fact] pre-head segment state modulation 相对 controls 为 partial，mean MSE 仍为正，mean segment activation
cosine `0.9644`。

[Attribution] 该 route 使用 latent FiLM，并未证明不同 units 真正从不同 history evidence 检索信息。

### B8/B9/B11/B13

- B8 在 late coefficient/readout 做 future query modulation，受 DCT control 阻断；
- B9 使用 canonical hard stages，no-stage capacity control 解释收益；
- B11 使用 basis-conditioned field，no-basis/constant-slot controls 解释收益；
- B13 使用 recurrent future-unit transition，hidden-memory repair 仍被 no-transition control 阻断。

[Decision] Successor 必须同时避开 late coefficient modulation、hard benchmark stages、basis descriptor
conditioning 与 recurrent transition。

## Carrier Constraint Discovered In This Audit

A6 的 pre-coefficient hidden shape 为 `[B,C,P,D]`，但 active presets 的 `P` 不统一：

| Dataset | `patch_num=P` | `d_model=D` | Flattened width |
| --- | ---: | ---: | ---: |
| ETTh2 | `48` | `32` | `1536` |
| ETTm1 | `1` | `256` | `256` |
| Weather | `48` | `128` | `6144` |

[Strong Evidence] 若直接以 A6 hidden patches 为 retrieval memory，ETTm1 只有一个 key/value token；任何
attention distribution 恒为 `1`。这种 diagnostic 会让 ETTm1 无法表达 unit-specific retrieval，属于
`diagnostic_invalid_for_direction_rejection`。

[Superseded Decision] 原计划让跨数据集 problem gate 回到 `720` raw positions，只是在规避 inherited
carrier defect。active A6 已不受 TimeAlign future alignment constraint，因此当前先重构统一 contextual patch
memory。raw-history statistics降为 robustness evidence，不再作为 B14 main interface。

## Refined Problem Boundary

B14 不问：

> future queries 是否可行？

该问题已被 prior art 回答。B14 只问：

> 在 A6 当前 input-to-output computation 中，不同 large future units 的 error gradients 是否要求不同的
> raw-history evidence，而 A6 的 target-independent input sensitivity 是否仍然高度共享？

若存在这种 `retrieval-demand mismatch`，才说明 A6 的 full-trajectory operator 可能需要 native
future-unit-specific history retrieval。若 A6 sensitivity 已经与 unit demand 同样分化，新 retrieval 会重复
已有 computation；若 demand 本身不分化，则问题不成立。

## Novelty Decision

[Decision] `future-unit-specific history retrieval` 当前只有 problem-diagnostic novelty，没有 method novelty。
即使 Step 3 通过，Step 4-6 仍必须明确：

1. 与 TimePerceiver target-query decoder 的区别不能只是 unit size；
2. 与 ElasTST placeholders 的区别不能只是 coarse tokenization；
3. 与 MQ-RNN horizon-specific context 的区别必须落在 unified prefix-count generation 与无 known-future
   covariates 的 carrier-specific mechanism；
4. method 必须由 retrieval-demand mismatch 推导，而不是先选 cross-attention 再补叙事。

[Next] `B14-PRE-HPM` 已 exact pass。重写 `B14-FURD-A`，把 raw-position demand/sensitivity按统一
`K48-S24` valid overlapping supports聚合到 29-token axis；通过后才可设计 exact parameter-matched
retrieval-vs-no-retrieval probe。
