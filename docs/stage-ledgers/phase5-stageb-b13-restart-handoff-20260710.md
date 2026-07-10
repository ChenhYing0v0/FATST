# Phase5 StageB B13 Restart Handoff 2026-07-10

本文档已同步到 C0 Encoder control 完成后的当前重启状态。新对话不应从 B12、B13-GRU 或 C0 patch
route 继续调参，而应重新选择 StageB Step 2/3 problem，或明确暂停第二贡献搜索。

## Minimal Reading Order

1. `docs/stage-ledgers/phase5-stageb-b13-restart-handoff-20260710.md`
2. `docs/stage-ledgers/phase5-timealign-interface.md`
3. `docs/paper-mainline.md`
4. `docs/research-roadmap.md`
5. `docs/experiments/phase5-stage-b-ettm1-carrier-protocol-audit.md`

关键 evidence：

- `analysis/phase5_stage_b_encoder_protocol_audit_20260710/stageb_route_encoder_protocol_audit.md`；
- `analysis/phase5_stage_b_c0_cross_patch_interaction_20260710/c0_cross_patch_interaction_report.md`；
- `analysis/phase5_stage_b_c0_ettm1_carrier_protocol_gate_20260710/c0_ettm1_encoder_control_deep_analysis.md`；
- `docs/experiments/phase5-stage-b-ettm1-carrier-protocol-audit.md`；
- `analysis/phase5_stage_b_b13_future_unit_granularity_20260710/b13_future_unit_granularity_report.md`；
- `analysis/phase5_stage_b_b13_future_unit_composition_20260710/b13_future_unit_composition_report.md`；
- `analysis/phase5_stage_b_b13_future_unit_hidden_composition_20260710/b13_future_unit_hidden_deep_analysis.md`。

## Current Cursor

| Field | Content |
| --- | --- |
| `current_stage` | Phase5 StageA accepted；StageB post-C0 rollback |
| `current_11_step` | C0 Step 9/10 completed；rollback StageB Step 2/3 |
| `active_carrier` | `A6-LBF-r256`；patch-memory interface is diagnostic-only |
| `accepted_paper_core` | `A6-LBF-r256` only |
| `closed_candidate` | current `GRU-based prefix-causal future-unit composition` |
| `open_direction` | native large future-unit/stage generation without full-horizon clipping |
| `next_problem` | C0 has closed the ETTm1 patch-defect question；select a new StageB Step 2/3 problem or pause |
| `do_not_implement_next` | no C0 extra seeds, P5 mixer, width/dropout sweep or Encoder mechanism stacking |

## C0 Encoder And Protocol Audit Update

[Fact] ETTm1 unified A6 simultaneously inherits the official H720 preset:
`patch_num=1,d_model=256,d_ff=256,dropout=0.9,official-last`. Fixed ETTm1 H96/H192 use
`d_model=128,dropout=0.2`, so the existing unified-vs-fixed comparison is source-faithful but not fully
configuration-controlled.

[Strong Evidence] `P=1` is a full-window global token, not an inactive Encoder. A frozen seed-2021 clean A6 audit
removed the residual MLP while preserving LayerNorm；ETTm1 H96/H192/H336/H720 MSE increased
`12.96%/9.32%/7.29%/5.58%`.

[Fact] flattened `C*P` positional encoding is retained across all controls and is not treated as a major standalone
risk. Frozen inclusion-exclusion evidence instead shows that the P1 global Encoder contains material cross-time-region
non-additivity in all 8 attenuation-horizon settings. Therefore P5 no-mix is not function-preserving, and any returned
degradation must be attributed cautiously. A6 also instantiates an unused official `proj_x`, which does not count as
active forecast capacity.

[Decision] C0 is a short Encoder/control blocker, not Contribution 2. The revised six-arm protocol preserves the same
flattened `C*P` semantics and separates P1 global width, `P=1` vs near-state/parameter-matched `P=5`, dropout
`0.9/0.2`, and last/best checkpoints from one training trajectory. Implementation, parameter/prefix/strict-reload checks
and one-batch dual-artifact smoke passed. The returned six-arm gate is now complete.

[Returned Result] Parameter-matched P5 is worse than P1 for every horizon under both dropout values and both
checkpoint selectors: dropout0.9 mean `+4.22%/+4.17%`, dropout0.2 `+1.92%/+2.50%`, all `0/4` wins.
Wider P1-D384 also loses `0/4` with mean `+1.34%/+1.44%`. Lowering accepted P1 dropout from 0.9 to 0.2 changes
mean MSE `+0.79%/+0.34%`, and best-val does not reverse any architecture ranking.

[Decision] `patch_num_performance_defect_not_supported`. Keep ETTm1 `P=1,D=256,d_ff=256,dropout=0.9` and
official-last as the source-faithful accepted carrier. Do not add seeds or a mixer. P5 no-mix may still be the wrong
intervention because it removes frozen P1 interactions, but repairing P5 would answer a different question rather than
show that inherited P1 was defective.

## B14 Prerequisite Encoder Reconstruction

ETTm1 `patch_num=1` 是 TimeAlign official hyperparameter，而不是用户配置错误。但 A6 已移除 future
reconstruction/alignment branch，因此本项目不再把 upstream dataset-specific tokenization 当作 active
architecture constraint。

[Decision] 原 `carrier tokenization audit` 被 `B14-PRE-CPE` 取代：实现统一的 channel-independent、
overlapping、cross-patch contextual history encoder，并显式输出 `[B,C,P,D]` memory。A6 learned-basis
operator保持不变。

该 contextual carrier prerequisite 已执行并失败；legacy A6 始终保持唯一 accepted carrier。当前只授权
hierarchical patch-memory exact-equivalence repair gate。详见：

- `docs/experiments/phase5-stage-b-b14-prerequisite-patchwise-encoder.md`；
- `analysis/phase5_stage_b_b14_prerequisite_patchwise_encoder_20260710/patchwise_encoder_source_and_design_report.md`。

### Returned Gate And Repair

full contextual replacement 已完成并失败：`P16-S8 +4.135% (1/12 wins)`，`P48-S24 +4.799%
(0/12 wins)`。结果归因为 `readout_or_encoder_design_wrong`，不追加 seeds/width sweep。

[Decision] Step 5/6 repair 改为 hierarchical encoder interface：accepted A6 carrier path严格保持，额外输出
parameter-free normalized valid `P48-S24` local memory `[B,C,29,48]`。只有 strict state/parameter/output/metric
equivalence 在三个 datasets 全部通过，才恢复 B14 diagnostic。

[Result] repair gate已 `3/3` exact pass：state keys、parameters、first-batch outputs、full-test MSE/MAE全部
max diff `0.0`。状态为 `hierarchical_patch_memory_ready`；B14 Step 3 diagnostic blocker已解除。

[Refinement] Step 3不使用 initial padded 30th patch；29 个 complete patches可 exact reconstruct normalized
history，overlap attribution使用 coverage correction，避免末端复制和中心位置重复计数。

[B14 Result] A1 current-gradient mismatch为 `0/6`；A2 model-independent label-patch repair仅 Weather-U180
通过，整体 `1/6`。没有 cross-dataset problem evidence，当前 B14-FURD关闭并回滚 Step 2/3。

## Fixed StageA Boundary

A6-LBF-r256 仍是唯一 accepted paper-core method：

```text
history -> hidden [B,C,R]
        -> coeff [B,C,256]
basis[:H] @ coeff
        -> prediction [B,H,C]
```

它是 prefix-compatible full-trajectory operator，而不是 strong target-set-native generator。这一叙事缺口仍是
StageB 可以研究的 paper problem，但不能因为叙事合理就跳过 mechanism controls。

## B13 Evidence Chain

### A. Large-Unit Problem Evidence

用户提出 small units 可能无法承载明显信息，因此 B13-A 使用 `U=120/144/180/240`，并把 `U=360` 只作
coarse control。

[Strong Evidence] ETTh2、ETTm1、Weather 在四个 main sizes 上全部通过 pre-registered gradient-pressure
gate（`12/12`）。所有 settings 的 adjacent gradient cosine 高于 far cosine，且这种 pattern 不普遍由 A6
basis overlap 解释。

[Decision] `partial_pass_large_unit_granularity_robust`。该结果证明 large future-region problem 值得研究，
但不证明 recurrent composition 是正确机制。

### B1. Coefficient-Memory Composition

Exact parameter-matched arms：

```text
parallel_no_transition
prefix_causal_composed
```

B1 只支持 `3/6` dataset/size settings；ETTh2-U180/U240 为 `+11.33%/+19.91%`，decision 为
`no_transition_control_explains`。由于 A6 coefficient 已是 full-trajectory bottleneck，允许一次 intervention
repair。

### B2. Hidden-Memory Intervention Repair

B2 将 memory 前移到 A6 encoder hidden，保持 unit sizes `180/240`、three seeds、state dimension、GRU、
decoder、loss 与 gate 不变。remote run 使用 commit `013dd35`、GPU 0、rows `4096/1024/1024`，完成
`36` runs。

| Dataset | U180 | U240 | Setting support |
| --- | ---: | ---: | --- |
| ETTh2 | `+5.1639%` | `+5.3589%` | `0/2` |
| ETTm1 | `-2.3454%` | `-16.0953%` | `2/2` |
| Weather | `-1.8434%` | `-6.4462%` | `2/2` |

整体虽有 `4/6` support，但 ETTh2 两个 sizes 都违反 `+0.25%` dataset non-degradation gate，且只有 `1/3`
seeds 获胜。

## Why The GRU Narrative Fails

unit 0 没有 previous-unit information，两个 arms 的 unit-0 computation topology 相同；unit-0 差异只来自
joint loss 训练出的 shared weights 不同。

- ETTh2 unit 0 改善，但 later-unit mean 在 U180/U240 退化约 `+9.78%/+9.90%`；
- ETTm1-U240 overall `-16.10%`，但 last unit `+7.50%`；
- Weather-U240 最大收益发生在 unit 0（`-12.64%`），后续收益随 depth 减小。

因此正向 aggregate gains 更像 shared-parameter optimization/regularization，而不是 previous latent unit 提供
progressively accumulating context。B2 的 parameter equality、prefix consistency 与 numeric validity 均通过，
所以不能用 diagnostic pathology 回避该机制负证据。

## Exact Verdict

[Decision] 关闭：

```text
A6 memory -> GRU recurrent future units -> shared segment decoder
```

状态为 `blocked_by_no_transition_control`。不得进入 Step 4-6、end-to-end implementation、full matrix，
也不得通过更换 GRU/head、增加 state dimension 或继续 seed/unit sweep 来复活。

[Boundary] 未关闭：

- large future-unit/stage generation；
- requested horizon 决定生成 unit 数量，而不是 clipping full trajectory；
- U180/U240 作为主要 granularity；
- native pre-readout future-region-specific history retrieval/state。

## Step 2 Restart Question

下一问题应从“previous future unit 是否帮助 later unit”改为：

> 对同一个 history，不同 large future regions 是否需要不同的 history evidence/retrieval state；如果需要，
> 能否用 shared, coordinate-continuous, non-recurrent unit generator 生成，并让 requested horizon 只决定
> 实例化多少 units？

优先顺序：

1. Step 1：targeted literature audit，检查 future token/segment generation、non-autoregressive latent query、
   variable-length decoder 与 multi-horizon unified forecasting 的 primary sources；
2. Step 2：精确定义 `future-unit-specific history retrieval`，避免重述 B8/B9/B11；
3. Step 3：先做 checkpoint/offline problem diagnostic，检验不同 U180/U240 regions 对 pre-readout history
   patch/features 的需求是否稳定不同；
4. 只有 Step 3 通过，才写 Step 4-6 method/narrative gate。

## Mandatory Controls For Any Successor

- no hard `{96,192,336,720}` stage/horizon IDs；
- no residual `A6 + correction` route；
- no clipping a pre-generated full-horizon latent trajectory and calling it unit generation；
- no small-unit `24/48/96` main design；
- no recurrence/head tuning revival of B13；
- include exact parameter-matched constant-coordinate/no-unit or shared-history controls；
- separate problem evidence, implementation mechanics, optimization protocol and paper claim；
- a positive diagnostic does not authorize model implementation before Step 4-6 narrative gate。

## Repository And Remote State

- B13 diagnostic code/protocol commit：`013dd350ec45525ffd625a13e87ee239b143238e`；
- remote repo：`/home/yingch/projects/FATST`；
- remote B2 output：`/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b13_future_unit_hidden_probe`；
- remote clean A6 probe inputs：
  `/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b13_probe_inputs/a6_clean`；
- local B2 result：`analysis/phase5_stage_b_b13_future_unit_hidden_composition_20260710/`。

本 handoff 只授权下一轮 Step 1-3 research/diagnostic planning，不授权直接实现 successor model。
