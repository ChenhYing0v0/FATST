# SIFF-v2 FCC-v1 Step9/10 Result and Portfolio Decision

## 1. Long-stage record

| Field | Record |
| --- | --- |
| `current_step` | Step9/10 complete；return paper portfolio decision |
| `problem` | immutable SIFF-v2的完整method package能否稳定超过A6_FULL，且ordered coupling-scale field能否超过same-objective capacity-matched independent field？ |
| `existence_evidence` | seed2021 SIFF vs A6_FULL正向；constant/permuted/Q1-wide/PCSD controls正向；independent margin不足 |
| `idea` | 不修改SIFF-v2，只以three-seed FCC分别确认package effectiveness与ordered-field attribution |
| `theory_check` | A6_FULL comparison同时改变architecture/objective，只能支持package claim；independent comparison才支持ordered-field claim |
| `design` | seed2021/2022/2023 × 5 datasets × 3 arms；45 runs；180 official-test cells；validation four-horizon selector |
| `narrative_gate` | prelaunch `conditional_pass_as_single_architecture_contribution` |
| `effectiveness_gate` | A6_FULL package pass；ordered-field attribution fail；internal health pass |
| `artifacts` | `primary/` six files；`raw_lite/` training/test protocol artifacts；remote full checkpoints/NPZ retained |
| `decision` | `performance_pass_attribution_blocked_stop_fcc_promotion` |

## 2. What was tested and why

FCC把两个不同问题严格分开：

1. `SIFF_EQUAL vs A6_FULL`：回答用户指定的完整method package performance；
2. `SIFF_EQUAL vs SIFF_INDEPENDENT_EQUAL`：在相同`equal_skill` objective和matched active parameter budget下，
   回答ordered coupling-scale field是否有独立贡献。

比较中的gain统一定义为：

$$
g=100\left(1-\frac{\mathrm{error}_{\mathrm{SIFF}}}
{\mathrm{error}_{\mathrm{reference}}}\right).
$$

$g>0$表示SIFF更好。test macro对全部`seed × dataset × horizon` cells等权平均；dataset、horizon和seed
wins分别先在对应group内平均，再统计group gain严格大于0的数量。validation split只用于checkpoint选择和
split-consistency解释，不参与FCC pass/fail。

## 3. Protocol and artifact audit

| Audit item | Result |
| --- | --- |
| `test_access_date` | `2026-07-21` |
| `user_authorization` | A6_FULL-scope 30-run training + one complete formal test |
| `candidate_version` | `SC1-SIFF-v2-FCC-v1` |
| `training/test commit` | `87bea35678475d652a9de0df2e8e969ff9bd2c70` |
| `checkpoint_retrained` | true for new seeds；historical seed2021 reused under frozen protocol |
| `test_role` | primary mechanism effectiveness and paper benchmark |
| `matrix_complete` | 45/45 runs；180/180 test cells |
| `protocol_pass` | 45/45 |
| `checkpoint_hashes` | 45/45 unique |
| `encoder_initialization` | same dataset/seed三臂paired |
| `checkpoint_nonmutation` | 30/30 new formal tests pass |
| `maximum_prefix_gap` | `3.5763e-7` |
| `test_access_count` | one complete access；no cell/dataset/horizon tuning |

30/30 new training于`2026-07-21T14:06:38+08:00`完成；formal test于
`2026-07-21T14:09:02+08:00`启动并于`14:12:05+08:00`完成。full diagnostic NPZ约928 MiB，保留在remote
output root；repo只同步result tables和约2.3 MiB protocol/raw-lite artifacts，避免提交大体积probe arrays。

## 4. Layer 1: paper-facing effectiveness

### 4.1 SIFF method package vs A6_FULL

| Metric | Macro gain | Cell wins | Dataset wins | Horizon wins | Seed wins | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Test MSE | `+1.2497%` | 47/60 | 5/5 | 4/4 | 3/3 | pass |
| Test MAE | `+0.7549%` | 46/60 | 5/5 | 4/4 | 3/3 | pass |

MSE分解：

| Group | Gain |
| --- | ---: |
| ETTh1 / ETTh2 | `+1.3538% / +1.6163%` |
| ETTm1 / ETTm2 | `+1.6962% / +0.1855%` |
| Weather | `+1.3970%` |
| H96 / H192 | `+2.1949% / +1.4394%` |
| H336 / H720 | `+0.7487% / +0.6159%` |
| seed2021 / 2022 / 2023 | `+1.6436% / +0.6680% / +1.4376%` |

validation MSE/MAE为`+2.1008%/+0.9346%`，test方向一致。因此A6_FULL package pass不是单seed或
validation/test reversal造成。

[Claim Boundary] 该结果只能支持“SIFF完整package相对A6_FULL稳定提升”。A6_FULL与SIFF同时改变readout
architecture和objective，不能把`+1.2497%`直接归因给ordered field。此外，历史seed2021的
`SIFF vs A6_MEASURE = -0.2366% MSE`必须继续报告；本FCC按用户要求没有重新训练A6_MEASURE，因此不能claim
SIFF已超过strongest known carrier training control。

## 5. Layer 2: matched mechanism attribution

### 5.1 Ordered field vs independent field

| Metric | Macro gain | Cell wins | Dataset wins | Horizon wins | Seed wins | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Test MSE | `-0.1272%` | 28/60 | 2/5 | 2/4 | 1/3 | fail |
| Test MAE | `-0.1733%` | 19/60 | 1/5 | 0/4 | 1/3 | fail |

MSE分解：

| Group | Gain |
| --- | ---: |
| ETTh1 / ETTh2 | `+0.5875% / -0.4093%` |
| ETTm1 / ETTm2 | `-0.2928% / -1.0484%` |
| Weather | `+0.5268%` |
| H96 / H192 | `+0.0590% / +0.0410%` |
| H336 / H720 | `-0.3625% / -0.2464%` |
| seed2021 / 2022 / 2023 | `+0.2580% / -0.4349% / -0.2049%` |

validation MSE/MAE为`-0.3224%/-0.5015%`；validation MSE只有seed2022轻微`+0.1362%`，seed2021/2023均为负。
因此test negative不是split reversal或单一bad seed。新confirmation seeds共同把seed2021 near-positive消解为
three-seed negative。

[Decision] immutable SIFF-v2的ordered-field必要性不成立。same-objective independent field以更低dataset-specific
rank匹配active parameter budget，却在validation和test总体不差于ordered field；capacity/function-class control
足以解释SIFF相对A6_FULL的package gain。

## 6. Layer 3: internal mechanism health

| Health statistic | Three-seed result | Gate |
| --- | ---: | --- |
| all arrays finite | 15/15 dataset-seed units | pass |
| maximum prefix gap | `3.5763e-7` | pass |
| oracle gain mean | `+6.8100%` | pass |
| pairwise arm NRMSE mean | `0.1618` | pass |
| policy normalized entropy mean | `0.8050` | pass |
| nonconstant component RMS ratio mean | `0.1526` | pass |

internal health证明SIFF arms、policy和nonconstant components均活跃，没有collapse或numeric pathology。但oracle
headroom和arm diversity不能覆盖negative matched attribution；它们只说明存在未被fusion充分利用的内部差异。

## 7. Layer 4: failure attribution

`failure_attribution=capacity_control_explains`。

- **What failed**：ordered coupling-scale field相对independent field的three-seed MSE/MAE、dataset、horizon与seed
  gates均失败；
- **What did not fail**：完整SIFF package相对A6_FULL的performance稳定，训练/test protocol、数值、initialization、
  checkpoint与internal health均正常；
- **Direction level**：该结果是immutable SIFF-v2 ordered-field claim的方向级negative，不是diagnostic design fault；
- **What remains untested**：更一般的independent/non-ordered future-field family可能成为新architecture方向，但它不是
  SIFF-v2 confirmation，也不能post-hoc继承SIFF ordered-scale novelty；
- **Rollback**：回paper portfolio decision，不做SIFF seed/rank/width/readout/router/loss rescue。

[Self-critique] parameter matching不等于完全相同function class；independent control的自由度分配方式不同。但这正是
FCC要检验的必要性问题：如果matched-budget、更低rank的non-ordered alternative在three seeds与两个splits上不差，
就没有证据证明ordered coordinate是获得性能所必需。该结论不拒绝所有scale-aware forecasting，只拒绝当前
immutable SIFF-v2的核心归因。

## 8. Portfolio decision

最终decision：

```text
performance_pass_attribution_blocked_stop_fcc_promotion
```

Consequences：

1. SIFF-v2不能晋升为`passed_core_candidate`；
2. 不启动SIFF confirmation rescue、modern-baseline performance matrix或formal ablations；
3. A6_FULL package positive保留为secondary evidence，不改写为ordered-field创新证据；
4. A6_MEASURE历史negative继续进入limitations；
5. 当前`active_method=none`，下一节点必须是显式paper portfolio decision或新Step2/4 problem—not SIFF tuning。

Primary artifacts：

- `primary/summary.json`；
- `primary/comparison_summary.csv`；
- `primary/comparison_cells.csv`；
- `primary/run_audit.csv`；
- `primary/mechanism_health.csv`；
- `primary/test_metrics_standard_horizons.csv`。
