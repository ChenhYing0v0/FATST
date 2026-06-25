# Phase4 SRP Code Audit for Gradient-Routing HSS

## 审计目的

[Question] SRP/SRP++ 代码中是否有可吸收进 FATST Phase4 的内容，用于解释 S1/S2 失败并设计
下一轮实验？

[Decision] 有，但吸收对象不是完整 SRP 训练范式，而是 gradient routing 与
parameter isolation 的机制证据。

## 代码来源

- local reference: `SRP-7C55`
- paper note: `Papers/srp-step-specific-representation.md`
- key files:
  - `SRP-7C55/exp/exp_long_term_forecasting_finetune_group.py`
  - `SRP-7C55/tuners/lora/Layers.py`
  - `SRP-7C55/tuners/molora/Layers.py`
  - `SRP-7C55/tuners/molora/iTransformerMoLora.py`
  - `SRP-7C55/run.py`

`SRP-7C55` 是外部参考代码，不作为 FATST active source，不直接复制模块。

## 关键机制事实

[Fact] SRP finetune group 使用 `model_pred_len=args.pred_len` 与
`label_pred_len=args.label_pred_len`，并定义：

$$
N_{\text{tuner}} = \frac{\text{label\_pred\_len}}{\text{pred\_len}}.
$$

训练和测试时按 tuner group 拼接输出，而不是用一个共享 head 一次性承担所有 future steps。

[Fact] `_mark_tuner_as_trainable` 冻结 base parameters；只有当前 tuner 或 MoLora 共享参数参与更新。
这对应了我们现在想表达的：

> supervision scheduling 不只决定 supervision strength，也决定 gradient 被允许更新到哪里。

[Fact] SRP 的 validation loss 先得到 per-step MSE，再 reshape 成
`num_tuner x model_pred_len`，最后按 tuner group 汇总。这意味着 SRP 的证据单位是
future segment/group，而不是 aggregate horizon MSE。

[Fact] `LoraLinear` / `LoraConv1d` 的 base layer 保持原权重，forward 时加入
`lora_B @ lora_A` 的 low-rank delta。`lora_B` zero init，因此初始行为等价于 base path。

[Fact] MoLora 版本不是每个 group 一个完全独立 LoRA，而是每个 group 有 `lora_score`，
通过 softmax 混合共享 LoRA experts。这说明 segment-specific path 可以与 expert sharing
兼容，不必退化成完全独立 horizon models。

## 对 FATST 的可吸收部分

1. **gradient path isolation**
   - 对 low-predictability 或 conflict units，不应只在 shared dense loss 中降权；
   - 应考虑只更新 adapter / auxiliary branch，或对 shared backbone detach。

2. **segment-level evidence**
   - 后续报告必须保留 segment/group-level loss 与 gradient diagnostics；
   - aggregate MSE 只能作为最终 gate，不能作为机制解释本身。

3. **zero-init residual adapter**
   - 若实现 adapter-isolated HSS，应保持初始行为等价于 base path；
   - 这避免一开始就破坏 R.3/full-time anchor。

4. **shared experts with group-conditioned mixing**
   - 如果最小 adapter path 通过 gate，后续可考虑 group-conditioned expert mixing；
   - 但当前不直接上 MoE，避免在 Step 5/6 证据不足时堆机制。

## 不直接采用的部分

[Decision] 不直接采用 SRP 的 two-stage pretrain + group finetune 作为 FATST 主线。
原因是它会把问题重新绑定到 benchmark `pred_len`，而 FATST Phase4 已明确 training/evaluation
解耦。

[Decision] 不直接采用每个 benchmark horizon 一个 tuner 的设计。我们的 HSS unit 应该是
train-side future unit / predictability bucket / segment group，而不是 evaluation horizon。

[Decision] 不直接复制 SRP 的 upstream modules。FATST 应在现有
`PatchEncoderTargetSetDecoder` tensor contract 内实现最小 adapter或gradient-routing机制。

## 对失败结果的解释价值

[Strong Evidence] S1 和 S2 都仍然让 difficult future units 影响 shared representation。
SRP 代码证明一种可行工程路径：冻结 shared/base path，只开放 group-specific tuner 更新。

[Inference] Weather 上 high-novelty/high-variation blocks 可能不是“应该更努力学习”的信号，
而是应该被隔离到不污染 shared path 的信号。S2 的 scalar downweight 失败，可能因为它没有改变
gradient destination。

[Counter-check] 这个解释必须通过 gradient conflict diagnostic 验证。如果 Weather 的 noisy-hard
units 与 early/predictable units 没有明显 gradient conflict，那么 adapter isolation 的动机不足。

## 下一步

1. 先运行 `scripts/analyze_phase4_gradient_conflict_diagnostic.py`。
2. 若 Weather `noisy_hard` vs `early_1_96` / `predictable_easy` 在 shared parameter groups
   上出现更低 cosine 或 negative share，则实现最小 `adapter_isolated_supervision`。
3. 若不成立，回退到 predictability proxy：train-only residual stability 或 seasonal residual
   stability，而不是架构升级。
