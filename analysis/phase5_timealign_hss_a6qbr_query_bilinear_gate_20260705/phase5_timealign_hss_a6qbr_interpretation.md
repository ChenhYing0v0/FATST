# Phase5-A6-QBR Query-Bilinear Gate Interpretation

本文档解释 A6-QBR remote gate 的结果，并给出 11-step decision。自动生成表包括：

- `phase5_timealign_hss_a6qbr_comparison.csv`
- `phase5_timealign_hss_a6qbr_summary.csv`
- `phase5_timealign_hss_a6qbr_dataset_summary.csv`
- `phase5_timealign_hss_a6qbr_cross_family_summary.csv`
- `phase5_timealign_hss_a6qbr_cross_family_dataset_summary.csv`

## Reader Path

1. 先看 QBR 自身：`r256/r512` 是否相对 A6-LBF-r256 有改善。
2. 再看 dataset summary：失败是否只来自 ETTh2 official-last drift。
3. 最后看 cross-family summary：QBR 是否能替代 A7DG/A8TAG 或解释 A6-LBF 的剩余缺口。

## Main Facts

[Fact] A6-QBR 两个 rank 均失败：

| Arm | mean MSE | vs A6-LBF-r256 | vs best control | wins |
| --- | ---: | ---: | ---: | ---: |
| `a6qbr_r256` | 0.401095 | +35.69% | +36.78% | 0/12 |
| `a6qbr_r512` | 0.401957 | +35.97% | +37.06% | 0/12 |

[Fact] Cross-family 排序中，QBR 明显低于所有近期候选：

| Family | Arm | mean MSE | vs A6-LBF | vs best control | wins |
| --- | --- | ---: | ---: | ---: | ---: |
| A7DG | `a7dg_abs004_t001_w02_d0999_wu1` | 0.284431 | -0.40% | +0.46% | 2/12 |
| A6ST_uniform | `a6st_w02_d0999_wu1` | 0.285442 | -0.00% | +0.87% | 2/12 |
| A6-LBF | `a6_lbf_r256` | 0.285698 | +0.00% | +0.88% | 0/12 |
| A8TAG | `a8tag_advratio_w10_d0999_wu1` | 0.285779 | +0.03% | +0.91% | 0/12 |
| A6-QBR | `a6qbr_r256` | 0.401095 | +35.69% | +36.78% | 0/12 |
| A6-QBR | `a6qbr_r512` | 0.401957 | +35.97% | +37.06% | 0/12 |

## Dataset Diagnosis

[Strong Evidence] QBR 失败不是单一 ETTh2 official-last drift：

| Dataset | Best QBR arm | vs A6-LBF-r256 | vs best control | Note |
| --- | --- | ---: | ---: | --- |
| ETTh2 | `a6qbr_r256` | +11.37% | +13.75% | 有 last-vs-best drift，但 gap 远大于 A6-LBF drift |
| ETTm1 | `a6qbr_r256` | +92.60% | +93.43% | 几乎翻倍，且 best epoch 在 10，说明不是 late drift |
| Weather | `a6qbr_r256` | +3.12% | +3.17% | 稳定负向，说明 row-key path 普遍弱于 learned basis |

[Fact] 增大 rank 没有帮助：`r512` 在 ALL、ETTh2、ETTm1、Weather 上均未优于 `r256`。这说明主要瓶颈不在
`K`，而在 coordinate-generated row-key function 的表达/优化能力。

## Mechanism Interpretation

[Inference] A6-QBR 试图把 A5-Q 的 target-query semantics 接入 A6-LBF 的 bilinear capacity path，但当前实现中
`row_key_t = G(q_t)` 由低维 absolute coordinate features 生成。这个 generator 对 720 个 future rows 的自由度
约束太强，不能承接 A6-LBF 的 learned row dictionary。

[Strong Evidence] 若 QBR 只是 rank 不足，`r512` 应至少相对 `r256` 有系统改善；但实际 `r512` 更差。这支持
“row-key generator bottleneck”而不是“rank bottleneck”。

[Self-Critique] 代码中 row-key scaling 与初始化可能影响早期优化；但当前实验并非轻微负向，而是 ETTm1
`+92.60%`、ALL `+35.69%`。把它继续修成 scale/initialization sweep 会变成 engineering repair，且 narrative
将弱化为 generated row table tuning，不适合作为 paper-core 候选继续推进。

## Gate Decision

| Field | Decision |
| --- | --- |
| `current_step` | Step 9/10/11：评估 A6-QBR remote artifacts 并决定 rollback |
| `narrative_gate` | failed after evidence：target-query semantics 没有有效转化为 forecasting capacity |
| `effectiveness_gate` | failed：0/12 wins；相对 A6-LBF-r256 `+35.69%`；r512 不改善 |
| `decision` | `failed_as_core_candidate` |
| `rollback_point` | Step 2/3：重审 Stage A architecture route 是否已经接近上限 |

## Next Research Direction

[Decision] 不继续 QBR rank/scale/teacher sweep，不启动 A5-S/A5-I/A5-M 作为新的 primary head。下一步应进行
`Stage_A_architecture_exhaustion_audit`：

1. 汇总 A5-Q/A5-B/A6-LBF/A7DG/A8TAG/A6-QBR 的机制与指标边界；
2. 判断 Stage A 是否仍能作为 paper-core architecture contribution；
3. 若 Stage A 只剩 control/negative evidence，则回 Step 2/3 重新定义 paper-core，可能转向 Stage B 或重新定义
   official-last stability/capacity conflict 的贡献边界；
4. 任何新候选都必须先通过新的 SCI narrative gate，不能从失败 head 上继续堆机制。

