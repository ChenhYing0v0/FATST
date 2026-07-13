# Phase5 StageB B14 Prerequisite: Contextual Patch-Wise History Encoder

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B14-PRE-CPE` |
| `current_step` | Step 9/10 carrier decision completed；valid evidence refinement locally verified |
| `problem` | active A6 已移除 TimeAlign future alignment branch，但 history encoder 仍继承 dataset-specific `patch_num` 与 token-wise MLP；ETTm1 的 `P=1` 使 B14 不存在可检索的 patch memory，ETTh2/Weather 的 patch tokens 之间也没有 encoder-level mixing |
| `existence_evidence` | code audit：ETTm1/ETTh2/Weather 的 `P=1/48/48`；encoder FFN 独立作用于每个 token；cross-patch interaction 只在 flatten 后的 coefficient head 发生 |
| `idea` | 用 horizon-independent、channel-independent、overlapping contextual patches 统一 history representation；保留 A6 learned-basis operator 作为 downstream prediction operator |
| `theory_check` | patch embedding 提供局部语义；cross-patch attention contextualizes each token；统一 `[B,C,P,D]` memory 允许 B14 比较 future-unit-specific retrieval；requested horizon 不进入 encoder |
| `design` | PatchTST-derived post-BatchNorm residual-attention encoder；`P16-S8` source anchor 与 `P48-S24` rich-patch arm；A6-LBF-r256 head unchanged |
| `narrative_gate` | `pass_as_carrier_prerequisite`：增强 Multi-Horizon Unified architecture 的 history/future interface，但不得声称 patch encoder 是论文 novelty |
| `effectiveness_gate` | contextual replacement failed；hierarchical patch memory exact-equivalence passes `3/3` datasets |
| `artifacts` | contextual 6-run gate + HPM 3-dataset exact-equivalence complete |
| `decision` | `hierarchical_patch_memory_ready`；close full encoder replacement；release B14 Step 3 diagnostic |

## Returned Contextual-Encoder Gate

| Arm | Overall mean MSE | Wins | ETTh2 | ETTm1 | Weather | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cpe_p16s8` | `+4.135%` | `1/12` | `+0.770%` | `+7.395%` | `+3.916%` | fail |
| `cpe_p48s24` | `+4.799%` | `0/12` | `+3.344%` | `+4.066%` | `+8.040%` | fail |

[Decision] 不追加 seeds，不通过 width/head/dropout sweep 复活 full contextual replacement。

### Failure Attribution

- `hypothesis_false`：未成立。返回性能接近项目早期 PatchTST-style backbone，说明 patch encoder 本身可正常
  forecasting；失败不能否定统一 local patch memory 的必要性。
- `intervention_point_wrong`：部分成立。把标准 contextual encoder直接放在 A6 coefficient readout 前，替换了
  已验证的 carrier computation，而 B14真正需要的是额外的 local retrieval axis。
- `readout_or_head_design_wrong`：主要归因。`P16` 产生 90 tokens，ETTm1/Weather 的 flatten-to-coeff head
  达 `3.55M` parameters并明显过拟合；`P48` 降到 30 tokens/`1.58M` parameters，但 single-scale coarse
  representation在 Weather/ETTh2 丢失细节。
- `optimization_or_numeric_pathology`：否。所有 runs finite、checkpoint reload正常；best-val已隔离后期过拟合。
- `capacity_control_explains`：否。更大/更小 parameter arms在不同 datasets互有优劣，不能用单一 capacity
  解释；但新增 capacity没有恢复 A6 performance。

结果标签：

```text
full_contextual_encoder_replacement_failed
readout_or_encoder_design_wrong
broader_patch_memory_direction_not_rejected
```

## Step 5/6 Repair：Hierarchical Patch Memory

本 repair 不再让未验证 patch encoder替换 accepted forecast carrier，而是明确 history encoder 的两个输出：

```text
normalized history
  ├─ accepted A6 carrier encoder -> carrier_state -> coeff -> basis[:H]
  └─ parameter-free valid P48-S24 unfold -> local_memory [B,C,29,48]
```

关键性质：

1. forecast path、state-dict keys 与 parameter count严格等于 clean A6；
2. local memory来自同一 normalized history，没有随机 projection 或未训练 token；
3. 29 个完整 48-step overlapping patches跨 datasets产生相同 evidence contract；
4. B14只在通过 problem gate 后学习 projection/retrieval，不提前污染 A6 prediction；
5. 这是 hierarchical encoder interface，不是 `A6 + residual forecast`。

### Repair Gate

使用 clean A6 seed-2021 checkpoints，在 ETTh2、ETTm1、Weather 上同时要求：

- strict state-dict load；
- parameter count完全相等；
- first-batch `{96,192,336,720}` outputs max absolute diff `0`；
- full-test MSE/MAE absolute diff `<=1e-8`；
- retrieval memory shape `[B,C,29,48]`；
- memory等于手工 normalized-history切片，且 coverage-corrected overlap-add可 exact reconstruct history。

全部通过后，carrier命名仍为 `A6-LBF-r256`，只把 encoder contract更新为
`hierarchical_patch_memory_ready`；不需要 seeds 2022/2023，因为这是 exact functional-equivalence gate。

### Repair Gate Result

[Historical Result] 初始 padded interface在 ETTh2、ETTm1、Weather `3/3` exact pass；但 padding会复制末端
history evidence。当前 valid `P=29,K=48` refinement已通过 local state/output/manual-slice/reconstruction checker，
并作为 B14 Step 3 analyzer的唯一 main interface；远程 diagnostic会再次执行 per-batch evidence audit。

## Closed Contextual Replacement Design（Historical）

以下 sections记录已执行的 `P16-S8/P48-S24` full replacement预注册设计，用于审计，不再授权
confirmation或追加 sweep。

## Why This Replaces The Carrier Tokenization Audit

本项目不再依赖 TimeAlign 的 future reconstruction/alignment branch，因此没有理由把 upstream 的
dataset-specific patch presets 当作 architecture constraint。原计划只判断 ETTm1 `P=1` 是否有害，仍会把
TimeAlign carrier 当作默认边界；本工作改为直接定义符合当前论文目标的 history interface：

```text
history [B,L,C]
  -> normalized overlapping patches [B,C,P,patch_len]
  -> shared patch projection [B,C,P,D]
  -> cross-patch contextual encoder [B,C,P,D]
  -> flatten only for A6 coeff readout [B,C,P*D]
  -> coeff [B,C,256]
  -> learned temporal basis[:H]
  -> prediction [B,H,C]
```

`[B,C,P,D]` 同时是 B14 的 canonical history memory。B14 不再退回 raw-history positions 规避
ETTm1 single-token pathology。

## Source-Informed Design

### Adopted From PatchTST

- channel independence：每个 variable 独立形成 patch sequence，encoder weights 跨 variables 共享；
- overlapping patching 与 end replication padding；
- learnable positional encoding；
- residual self-attention；
- post-norm `BatchNorm1d` encoder blocks；
- patch tokens 在 prediction head 前保持显式 token axis。

Primary sources：

- PatchTST paper: `https://arxiv.org/abs/2211.14730`；
- official implementation: `https://github.com/yuqinie98/PatchTST`，audited commit
  `204c21efe0b39603ad6e2ca640ef5896646ab1a9`。

### Intentionally Rejected

- 不复制 PatchTST fixed-horizon flatten head；继续使用 A6 prefix-native learned-basis operator；
- 不使用 decomposition branch；
- 不把 patch encoder 当作 paper contribution；
- 不使用 TimeAlign future-label encoder、alignment loss 或 dataset-specific `patch_num`；
- 不在 encoder 内注入 benchmark horizon IDs 或 future-unit IDs。

## Tensor Contract

输入 `x: [B,720,C]`：

1. `Normalize -> x_norm: [B,720,C]`；
2. transpose：`[B,C,720]`；
3. end replication padding 后 unfold：`patches: [B,C,P,K]`；
4. shared `Linear(K,D)`：`tokens: [B,C,P,D]`；
5. reshape channels into batch：`[B*C,P,D]`；
6. positional embedding + residual-attention encoder；
7. restore `memory: [B,C,P,D]`；
8. A6-only flatten：`hidden: [B,C,P*D]`；
9. `Linear(P*D,256)` 与 `basis[:H]` 生成 prefix-native output。

patch count：

$$
P=\left\lfloor\frac{L+S-K}{S}\right\rfloor+1,
$$

其中 right end replication padding 长度为 stride $S$。`L=720` 时：

- `P16-S8`: `P=90`；
- `P48-S24`: `P=30`。

## Granularity Arms

### `cpe_p16s8`

PatchTST official scripts 在 ETTh2、ETTm1、Weather 均使用 `patch_len=16, stride=8`。该 arm 是
source-supported performance anchor，不把 16-step history patch 解释为 future generation unit。

### `cpe_p48s24`

更大的 48-step overlapping history segments，每个 patch 承载更丰富的局部信息，并把 720-step history
压缩成 30 个可解释 tokens。它响应 large information-bearing unit prior，但 future generation 仍只使用
`U180/U240`。

如果两个 arms 性能差距不超过 `0.3%` overall mean MSE，优先 `P48-S24`，因为 memory 更紧凑、B14
retrieval attribution 更稳定、attention cost 更低。否则只按 effectiveness gate 选择。

## Capacity Configuration

拓扑和 patch rule 跨 datasets 相同；width 作为 capacity hyperparameter，不允许退化为 `P=1`：

| Dataset | `D` | heads | `d_ff` | layers | dropout |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 16 | 4 | 128 | 3 | 0.3 |
| ETTm1 | 128 | 16 | 256 | 3 | 0.2 |
| Weather | 128 | 16 | 256 | 3 | 0.2 |

这些 width/dropout 来自 PatchTST official supervised scripts，而不是 TimeAlign presets。训练使用
`AdamW`、`lr=1e-4`、20 epochs cosine schedule、prediction-only multi-prefix loss、`basis_rank=256` 和
`best-val` checkpoint。

## Historical Effectiveness Gate

### Small Gate

- datasets：ETTh2、ETTm1、Weather；
- arms：`cpe_p16s8`、`cpe_p48s24`；
- seed：2021；
- target prefixes：`96/192/336/720`；
- control：accepted clean A6-LBF-r256 metrics；
- report：MSE/MAE、per-dataset mean、overall mean、wins、parameters、epoch time。

一个 arm 通过 small gate 需要同时满足：

1. overall mean MSE 相对 clean A6 不高于 `+0.5%`；
2. 至少 `6/12` MSE wins；
3. 任一 dataset mean MSE 不高于 `+1.0%`；
4. 不出现 non-finite、训练 collapse 或单 setting `>+5%` 的严重退化。

### Confirmation Gate

只给 small-gate winner 追加 seeds 2022/2023。正式替换 carrier 需要：

1. 3-seed overall mean MSE 不劣于 clean A6；
2. 每个 dataset 至少 `2/3` seeds 的 mean MSE 不劣于 clean A6 `+1%`；
3. token memory shape、attention、gradient 与 checkpoint reload 均稳定；
4. B14 的 `U180/U240` token-level diagnostic 可在三个 datasets 使用同一统计定义。

## Historical Failure Attribution And Rollback

- `effectiveness_pass`：命名为 `A6-CPE-LBF-r256`，更新 active carrier，再运行 B14；
- only one granularity passes：采用通过 arm，不继续 patch sweep；
- both within `+1%` but fail stability：`optimization_or_numeric_pathology`，只允许一次 source-config
  repair；
- both clearly regress without pathology：`readout_or_encoder_design_wrong`，回 Step 5/6，不否定
  contextual patch-memory direction；
- performance passes but token profiles collapse：可作为 carrier，但不能为 B14 retrieval 提供机制证据；
- 任何结果都不能把 PatchTST-derived encoder 提升为论文 novelty。

## B14 Boundary

原 contextual confirmation gate已关闭。B14只能在 hierarchical patch-memory exact-equivalence通过后启动：

```text
future-unit error demand aggregated over canonical local patches
vs
accepted A6 raw-input sensitivity aggregated over the same patch supports
```

任何 trainable patch projection/retrieval都属于后续 B14-B probe，必须配 exact no-retrieval controls。
