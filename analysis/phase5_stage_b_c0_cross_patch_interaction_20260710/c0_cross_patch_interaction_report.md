# Phase5 StageB C0 Frozen Cross-Patch Interaction Diagnostic

## Research-loop position

| 字段 | 内容 |
| --- | --- |
| `current_step` | Step 2/3：确认 Encoder control problem 是否存在 |
| `problem` | P1 global-token Encoder 是否已经利用不同时间区域之间的非加性交互；若有，P5 token-wise no-mix control 不能被解释为功能保持分解 |
| `narrative_gate` | `diagnostic_only`；不增加 StageB 创新点 |
| `effectiveness_gate` | pair-median mean `>=0.05`，且至少 `75%` patch pairs 的 median ratio `>=0.05` |
| `rollback` | 若无 interaction，P5 no-mix 可视为更接近 additive decomposition；若有 interaction，只授权保留 capacity caveat，不直接增加 mixer |
| `decision` | `material_interaction_detected_but_mixer_not_authorized_before_c0_returned_results` |

## What was tested

使用 clean `A6-LBF-r256` ETTm1 official-last checkpoint，将 720-step normalized history 划为 5 个
non-overlap 144-step canonical patches。对两个 patch 分别 attenuation 后，在 learned-basis coefficient
space 计算 inclusion-exclusion interaction：

$$
I_{ij}=c(x)-c(x^{(-i)})-c(x^{(-j)})+c(x^{(-i,-j)}).
$$

其中 $x^{(-i)}$ 不是删除样本，而是把 normalized patch $i$ 乘以 $1-a$；$a$ 取 `0.25` 与 `0.50`。
随后用 `learned_temporal_basis[:H]` 将 coefficient delta 投影到 target horizon $H$：

$$
r_{ij,H}=\frac{\operatorname{RMS}(B_H I_{ij})}
{\frac{1}{2}[\operatorname{RMS}(B_H\Delta_i)+\operatorname{RMS}(B_H\Delta_j)]+10^{-12}}.
$$

`r` 衡量 pair interaction 相对于两个 single-patch main effects 的大小。诊断读取 test split 的前
32 batches，共覆盖所有 10 个 patch pairs；不训练、不改 checkpoint。

## Results

| Attenuation | H96 | H192 | H336 | H720 | Pairs passing |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.25 | 0.06461 | 0.06403 | 0.06357 | 0.06338 | 10/10, 9/10, 9/10, 9/10 |
| 0.50 | 0.12942 | 0.12855 | 0.12833 | 0.12816 | 10/10 at all horizons |

表中是 10 个 pair-level median ratios 的均值。8 个 `attenuation x horizon` settings 全部通过
预注册 material-interaction gate，且 ratio 随 attenuation 近似成比例增加。

## Interpretation and failure attribution

[Strong Evidence] 当前 P1 global Encoder 在 frozen checkpoint 上包含稳定的跨时间区域非加性交互，
不是单纯把五段历史的独立贡献相加。

[Boundary] 该结果不能证明 token mixer 会带来更低 forecasting error，也不能区分这些 interactions 是
有益、冗余还是过拟合。因此：

1. 当前六臂实验仍使用 P5 token-wise no-mix 作为可解释的 tokenization/capacity control；
2. 若 P5 退化，不能立即把退化归因于 patch granularity，也必须考虑 interaction capacity 被移除；
3. 只有六臂返回后满足预注册条件，才可设计独立 mixer control；
4. mixer 即便执行，也只能用于让 Encoder 更可控、合理，不可升级为 StageB paper-core mechanism。

本诊断没有失败；它识别了 design-level confound。StageB research direction 仍未被此结果验证或否定。

## Artifacts

- `cross_patch_interaction_pairs.csv`：每个 attenuation、patch pair、horizon 的 ratio 分布统计；
- `cross_patch_interaction_summary.csv`：每个 attenuation、horizon 的跨 pair gate 汇总；
- `scripts/analyze_phase5_stage_b_c0_cross_patch_interaction.py`：冻结 checkpoint 诊断实现。
