# Phase5-A8TAG Teacher-Advantage Gate Interpretation

本文档是 `Phase5-A8TAG` remote artifacts 的人工解释层。自动统计报告见
`phase5_timealign_hss_a6s_stability_gate_report.md`；本文件负责把指标转成 11-step
decision。

## Reader Path

1. 先看 `phase5_timealign_hss_a6s_summary.csv`：判断 A8TAG variant-level 是否超过
   A7DG、A6-LBF 与 best stage control。
2. 再看 `phase5_timealign_hss_a6s_dataset_summary.csv`：判断 teacher advantage 与 gate
   是否按数据集产生可解释分化。
3. 最后看 `phase5_timealign_hss_a8tag_cross_family_summary.csv` 与
   `phase5_timealign_hss_a8tag_cross_family_dataset_summary.csv`：判断 A8TAG 相对 A7DG /
   uniform A6ST / A6-LBF 的真实位置。

## Main Facts

[Fact] A8TAG 最佳 variant 是 `a8tag_advratio_w10_d0999_wu1`，12 个
dataset-horizon setting 的 mean MSE 为 `0.285779`，相对 best stage control 为
`+0.91%`，wins 为 `0/12`。

[Fact] A8TAG 最佳 variant 相对 A6-LBF-r256 为 `+0.03%`，基本退化到 A6-LBF；相对
A7DG best `0.284431` 变差约 `+0.47%`。

[Fact] Cross-family 排序为：

| Family | Variant | mean MSE | vs A6-LBF | vs best control | wins |
| --- | --- | ---: | ---: | ---: | ---: |
| A7DG | `a7dg_abs004_t001_w02_d0999_wu1` | 0.284431 | -0.40% | +0.46% | 2/12 |
| A6ST_uniform | `a6st_w02_d0999_wu1` | 0.285442 | -0.00% | +0.87% | 2/12 |
| A6-LBF | `a6_lbf_r256` | 0.285698 | +0.00% | +0.88% | 0/12 |
| A8TAG | `a8tag_advratio_w10_d0999_wu1` | 0.285779 | +0.03% | +0.91% | 0/12 |

## Mechanism Diagnosis

[Strong Evidence] A8TAG 的 gate 逻辑没有保住 A7DG 的关键正向信号。A7DG 在 ETTh2 上
相对 A6-LBF 为 `-1.95%`，而 A8TAG-ratio 在 ETTh2 上为 `-0.00%`，几乎等同于关闭
self-teacher。

[Fact] A8TAG-ratio 在 ETTm1/Weather 上确实更安全：ETTm1 相对 A6-LBF 为 `+0.05%`，
Weather 为 `+0.04%`；它比 A7DG 的 `+0.51%` / `+0.23%` 更接近 A6-LBF。但这个改善
只是损失 ETTh2 gain 后的退化安全性，不是新的整体优势。

[Strong Evidence] Binary teacher-advantage gate 不是更好的解释。它在 ETTm1/Weather
上的 gate 接近 `0.95/1.00`，但 MSE 明显差于 ratio gate。这说明“teacher 在当前
supervised prefix 上稍好”不足以证明 teacher trajectory 是有价值的 consistency
target；hard imitation 会压制 student 的可塑性。

[Inference] A7DG 的正向部分更像是 high-disagreement / raw-final drift 区域的稳定化，
而不是“EMA teacher 当前 label risk 更低”这一局部条件。A8TAG 把 teacher usefulness
定义为当前 batch label advantage，因此在 ETTh2 上关闭了最需要 stability 的路径，在
ETTm1/Weather 上又容易因小幅 advantage 触发过强 imitation。

## Gate Decision

| Field | Decision |
| --- | --- |
| `current_step` | Step 9/10/11：评估 A8TAG remote artifacts 并决定 rollback |
| `narrative_gate` | conditional pass failed：teacher-advantage 是清晰机制，但 empirical evidence 显示它不能解释 useful self-teacher |
| `effectiveness_gate` | failed：0/12 wins vs best stage control；整体弱于 A7DG；只接近 A6-LBF |
| `decision` | `failed_as_core_candidate` |
| `rollback_point` | Step 4/5：重新建模 stability 与 capacity 的冲突；不继续做 teacher-advantage threshold/weight sweep |

## Next Research Constraint

下一步不能直接把 A7DG 与 A8TAG 混合成新的 gate，也不能继续 sweep teacher threshold。
按 stage-ledger 规则，应先回溯未执行候选，包括 `A5-S`、`A5-I`、`A5-M`、`A6-QBR`
以及 A6 learned-basis 路线中尚未被充分解释的 stability/capacity conflict，再决定新的
Step 4/5 candidate 是否有 SCI narrative gate。

