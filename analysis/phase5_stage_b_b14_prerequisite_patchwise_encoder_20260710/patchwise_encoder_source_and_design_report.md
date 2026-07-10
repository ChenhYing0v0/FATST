# B14 前置工作：Patch-Wise Encoder Source And Design Report

> Status：原 contextual full replacement 已完成并失败；本文 source audit保留，当前执行路径见文末
> `Returned Decision Update` 的 hierarchical patch-memory repair。

## 结论

[Decision] 停止 `carrier tokenization audit`。active A6 已不包含 TimeAlign future align branch，history
encoder 应按本项目的 Multi-Horizon Unified architecture 重新定义，而不是继续受 upstream
dataset-specific `patch_num` 约束。

[Decision] 采用 PatchTST-derived contextual patch encoder 作为 B14 prerequisite candidate：统一 overlapping
patch rule、channel-independent shared encoder、cross-patch residual attention 和显式 `[B,C,P,D]` memory；
A6 learned-basis operator保持不变。

## Existing Encoder Defect

当前 A6 legacy encoder：

```text
PatchEmbed -> token-wise residual MLP -> flatten -> coeff
```

[Fact] token-wise MLP 不混合 patch axis。ETTh2/Weather 的不同 patches 只在 flatten 后的 coefficient head
交互；ETTm1 `P=1` 更完全没有 patch sequence。这不会使 A6 的历史性能结论无效，但使“统一 patch-wise
history memory -> future-unit retrieval”的下一阶段问题无法在三个 datasets 上一致定义。

## Primary-Source Audit

### PatchTST

PatchTST 把 subseries-level patches 当作 Transformer tokens，并采用 channel independence。论文给出的直接
动机是保留 local semantics、降低 attention cost 并扩展可利用 lookback。official supervised scripts 在
ETTh2、ETTm1、Weather 均使用 `patch_len=16, stride=8`。

本轮检查 official repository commit：

```text
204c21efe0b39603ad6e2ca640ef5896646ab1a9
```

关键 implementation invariants：

- end `ReplicationPad1d` 后 overlapping unfold；
- shared `Linear(patch_len,d_model)`；
- learnable positional encoding；
- `res_attention=True`；
- `pre_norm=False`；
- `BatchNorm` post-norm；
- patch axis 在 flatten head 前保留。

### TSMixer / PatchMixer Boundary

TSMixer 支持显式 temporal mixing 的必要性；PatchMixer 强调 permutation-variant local/global mixing。
但本项目不需要再引入一个新的 mixer contribution。B14 需要的是稳定、已知有效、容易读取的 history
token memory，因此选择 PatchTST-derived attention encoder，而不是同时研究新的 CNN/MLP mixer。

## Adopt/Reject Matrix

| Mechanism | Decision | Reason |
| --- | --- | --- |
| overlapping patches | adopt | 减少硬边界，保留 local semantic continuity |
| channel independence | adopt | 与当前 A6 channel-wise prediction contract 一致 |
| residual attention | adopt | 提供显式 cross-patch contextualization |
| post BatchNorm | adopt | 避免 Phase0 simplified pre-LayerNorm implementation gap |
| PatchTST flatten forecast head | reject | 与 A6 prefix-native learned-basis operator 冲突 |
| future reconstruction/alignment | reject | 已被 A6 dependency audit 移除 |
| decomposition branch | reject | 非 B14 prerequisite 所需，增加归因混杂 |
| multi-scale patch bank | defer | 会使 B14 memory attribution先天含混 |
| input-side MoE/selective patching | reject | 与后续 future-unit retrieval contribution boundary 冲突 |

## Narrative Consequence

重构后 paper-level data flow可以表述为：

```text
contextual history patches
  -> unified prefix-native forecast operator
  -> future-unit-aware generation/retrieval (StageB target)
```

encoder 本身只是标准 history representation，不是 contribution。真正的 contribution boundary仍在：

- 一个模型生成多个 requested horizons；
- requested horizon 决定 future units 数量；
- future units 使用不同 history evidence，但共享生成机制；
- 不先生成 full horizon 再 clipping。

## Self-Critique

[Risk] `P16-S8` 在 `L=720` 产生 90 tokens，ETTm1/Weather 的 flattened A6 coefficient head 参数明显增加；
任何收益可能包含 capacity effect。

[Mitigation] 本阶段的目标是选择有效 carrier，不声称 encoder mechanism novelty；报告必须同时列参数量与
epoch time。`P48-S24` 提供 30-token rich-patch control；若性能相近，预注册优先更紧凑的 `P48-S24`。

[Risk] Transformer tokens contextualized 后不再是纯局部 patches，B14 attention attribution不能直接解释为
causal raw-history selection。

[Mitigation] B14 主统计定义为 contextual-token demand/sensitivity，raw-position gradient只作为 robustness
cross-check，并明确不是 causal attribution。

## Returned Decision Update

remote 6-run gate 返回：`P16-S8 +4.135%`、`P48-S24 +4.799%` overall mean MSE；两个 full
replacement arms均失败。legacy A6保持 accepted。

[Decision] 回 Step 5/6，而不是否定 patch memory。source-derived Transformer适合作为独立 forecasting
backbone，但不能无代价替换 A6 已验证 computation。repair改为 hierarchical encoder contract：保留 global
carrier state，额外暴露 parameter-free normalized local patches。这样 B14可以检验 local retrieval需求，
而 performance在任何 trainable retrieval介入前由 exact equivalence保证。
