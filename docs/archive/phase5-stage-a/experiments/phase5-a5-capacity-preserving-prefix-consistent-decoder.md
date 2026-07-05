# Phase5-A5: Capacity-Preserving Prefix-Consistent Decoder

## 2026-07-02 Narrative Gate 复评结论

| Field | Content |
| --- | --- |
| `current_step` | Step 4/5：重新提出 idea 并重做 theory/narrative check |
| `candidate_id` | `A5_pcf_prefix_consistent_function_preserving_decoder` |
| `previous_status` | `narrative_ready` |
| `revised_status` | `narrative_rejected_after_review` |
| `decision` | 撤回上一版 narrative gate pass；PCF 不能进入 Step 7 实现 |

[Decision] 上一版 `active trained dense anchor + cumulative prefix-consistent correction + explicit overlap
consistency` 方案不应被判定为 A5 paper-core candidate。它更像 A2 nested、A3D teacher preservation、
H1C residual/adapter 思路的混合，而不是一个重新设计的 unified head。

[Rule Correction] 按项目规则，paper-core 方法必须在 Step 4-6 先通过 SCI narrative gate。上一版文档
没有充分完成这个评估，直接把方案标为 `narrative_ready` 是不严谨的。该结论现在撤回。

## 为什么当时提出 active dense anchor

[Fact] 提出 active dense anchor 的直接动机来自已有 negative evidence：

- H1B 证明直接替换 dense 720 projection 会发生 capacity collapse；
- A3C 证明简单把 learned rows 转换到 nested heads 不足以 preserve learned function；
- A3D 证明 teacher preservation 能部分修复 function-preservation gap。

因此上一版设计试图保留一条 dense-capacity path，避免新 head 一开始就丢掉 TimeAlign 的强 readout
capacity。

[Self-Critique] 这个动机在工程上可以理解，但不足以成为论文级 unified head 设计。原因是：

1. 如果 final prediction 仍显著依赖 trained dense rows，那么短 horizon 的主体能力仍来自 full-head
   row capacity，而不是新的 unified decoder contract；
2. `anchor_H + alpha * corr_H` 在结构上很容易被审稿人视为 residual correction；
3. 依赖预训练 anchor 会把方法叙事推向 checkpoint transfer / distillation recipe，而不是一个
   clean unified architecture；
4. `L_cons` 可能只是 auxiliary regularizer，不能自动证明 decoder 本身具备 prefix-consistent
   generation semantics；
5. 该设计需要 controls 才能说明不是 anchor-only，但如果 paper-core 必须先靠 controls 排除
   anchor-only，本身就说明贡献边界不够干净。

## Narrative Gate 复评

| Gate Item | 复评判断 |
| --- | --- |
| Direct multi-prefix generation | 不通过。虽然输出 shape 可以是 `[B,H,C]`，但核心 anchor 仍来自 dense full-head rows；这更像 selected-row reuse，不是重新定义 head。 |
| Prefix consistency | 部分成立但不足。`L_cons` 是训练约束，不等于 decoder architecture 本身保证 consistency。 |
| Capacity preservation | 工程动机成立，但作为 paper-core 叙事过重依赖 pretrained anchor / teacher，削弱方法独立性。 |
| Target-prefix awareness | 不充分。`q_H` 只控制 correction path，而 anchor path 对 `H` 的作用仍主要是 row selection。 |
| 非旧机制混合 | 不通过。该方案实质混合了 A2 nested、A3D teacher preservation 和 H1C residual/adapter 思路。 |

[Decision] `A5_pcf_prefix_consistent_function_preserving_decoder` 不通过 narrative gate。它最多可作为
`diagnostic_only` 或 `control_only` 思路保留，用于验证 dense-capacity preservation 是否必要；不能作为
Stage A paper-core method。

## 对 A5 的修正边界

下一版 A5 必须重新设计，而不是修补 PCF。新的 design constraints 是：

1. **Architecture-first**
   先定义 unified head/decoder 的生成 contract，再考虑 capacity preservation。不能先放入 trained
   dense anchor，再用 correction 解释为新 head。

2. **No required pretrained anchor**
   paper-core 方法不应以预训练 full-head rows 作为必要组成。teacher/distillation 可以作为 control
   或 initialization diagnostic，但不能是方法成立的前提。

3. **No residual-as-core**
   如果结构形式是 `base + correction`，必须证明 base 不是旧 full-head crop/row-selection path；
   否则该路线应标为 control，而不是 paper-core。

4. **Prefix request must own the decoder contract**
   `target_prefix` / requested prefix 必须进入主生成路径，而不是只进入附属 correction、gate 或
   adapter。

5. **Capacity preservation must be intrinsic**
   capacity preservation 应来自参数化方式、shared basis/operator、regularized function class 或
   initialization-free contract，而不是依赖某个已训练 checkpoint。

## 下一步

回到 A5 Step 4/5，重新提出 first-principles unified head idea。下一步设计必须先回答：

> 能否构造一个不依赖 pretrained full-head anchor 的 prefix-native decoder，使 h96/h192/h336/h720
> 都由同一个主生成 contract 产生，并且该 contract 自身解释 prefix consistency 与 capacity？

在该问题回答之前，不应实现 PCF，也不应启动 A5 remote gate。
