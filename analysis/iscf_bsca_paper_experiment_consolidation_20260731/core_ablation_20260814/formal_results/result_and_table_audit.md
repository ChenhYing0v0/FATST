# ISCF-BSCA Core-Ablation Formal Result and Table Audit

日期：2026-08-14

候选版本：`ISCF-BSCA-v1-core-ablation-20260814`

Decision：`core_ablation_complete_partial_attribution_3_of_4_controls_pass`

## 1. 完整性与 provenance

- frozen matrix：5 variants × 5 datasets × 4 horizons = 100 cells；MSE/MAE；seed 2021；
- Full：复用 exact `ISCF-BSCA-v1` 的5个validation-selected checkpoints与20个test cells；
- controls：20/20 end-to-end joint-training runs、20/20 validation selectors、20个unique checkpoint hashes；
- selector：每个dataset/variant以validation `{96,192,336,720}` mean MSE选择唯一checkpoint；
- formal test：20/20新checkpoints × four horizons = 80新test cells；与20个Full cells合并后100/100完整；
- immutable training manifest SHA256：`0fb24236aaa7b1ef2fb9fe13aebfd0947428abdb3351fab0a9744c79c45a139c`；
- protocol SHA256：`36e8ea1ea45a62e7698cd122b27f076cc9492e9f0973d7560291dd8bd91b7f40`；
- execution commit：`b3526e5d49eac67c8b18a3a52d2f1f99d0b11130`；
- test access date：2026-08-14；test role=`primary-mechanism-effectiveness-and-paper-benchmark`；`test_informed=true`；
- checkpoint mutation、partial reporting、per-horizon/per-cell tuning与非有限数值均为0。

`Fixed Scope (s=144)` 的ETTm2 training由workload-steal完成；wrapper在训练后、validation前的shell hash命令失败。修复仅重放validation，重放前后checkpoint hash一致且未访问test，因此该run继续满足同一checkpoint selector与formal-test边界。

## 2. Dataset-level mean table

下表每个dataset值均为四个horizons的均值；`Avg.`为五个dataset的macro mean。

| Variant | ETTm1 | ETTm2 | ETTh1 | ETTh2 | Weather | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full ISCF-BSCA | .344/.368 | .257/.312 | .411/.435 | .313/.369 | .217/.249 | .309/.346 |
| w/o BSCA | .353/.378 | .256/.312 | .434/.450 | .319/.375 | .218/.250 | .316/.353 |
| w/o Target-Adaptive Allocation | .345/.368 | .257/.312 | .411/.435 | .313/.369 | .217/.249 | .308/.346 |
| Shared Scope Projection | .355/.376 | .257/.314 | .411/.430 | .325/.382 | .217/.250 | .313/.351 |
| Fixed Scope ($s=144$) | .351/.372 | .257/.314 | .428/.441 | .317/.373 | .217/.249 | .314/.350 |

每个单元为`MSE/MAE`。三位小数只用于展示；formal gates和best/second ranking使用未舍入值。

## 3. Pre-registered control gates

| Control | Full macro gain MSE/MAE | Dataset MSE wins | Horizon MSE wins | Cell wins MSE/MAE | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| w/o BSCA | +2.401% / +1.972% | 4/5 | 4/4 | 17/20 / 18/20 | PASS |
| w/o Target-Adaptive Allocation | -0.039% / +0.001% | 2/5 | 0/4 | 8/20 / 10/20 | FAIL |
| Shared Scope Projection | +1.416% / +1.227% | 3/5 | 4/4 | 14/20 / 15/20 | PASS |
| Fixed Scope ($s=144$) | +1.796% / +1.038% | 5/5 | 4/4 | 16/20 / 20/20 | PASS |

## 4. Four-layer mechanism evaluation

1. `paper_facing_effectiveness`：100/100 official-test cells完整，Full的macro MSE/MAE为`0.308549/0.346278`。
2. `matched_mechanism_attribution`：BSCA objective、scope-specific projections和multi-scope design三个controls通过；Target-Adaptive Allocation control失败，因此只达到3/4 matched attribution。
3. `internal_mechanism_health`：本表不提供Scope Probability、utilization或regional preference统计；这些仍属于Figure 5独立证据块，不能补救failed matched effectiveness。
4. `failure_attribution`：未发现numeric、artifact、checkpoint或selector pathology。对当前seed2021 exact setting，learned Target-Adaptive Allocation相对equal fusion的独立accuracy utility不受支持，记为`hypothesis_false_for_exact_setting`；这不等于否定全部allocation architecture family。

## 5. Paper claim boundary

[Strong Evidence] 当前结果支持：BSCA完整objective相对prefix-only训练、scope-specific projections相对capacity-matched shared projection，以及multi-scope相对preregistered fixed `s=144`具有matched性能贡献。

[Fact] 当前结果不支持：learned Target-Adaptive Allocation在exact seed2021设置下优于equal non-adaptive fusion。二者macro差异接近零，但Full的MSE方向为负，不能用rounding tie、Figure 5 active probabilities或selected trajectory改写为正向归因。

[Boundary] Introduction和Results不得写成“all core components are effective”。可写成三项通过的窄结论，同时明确allocation advantage未由当前matched scorecard建立。完整component chain状态为`performance_partial_pass`，不是`passed_core_candidate_matched_attribution`。

## 6. Canonical artifacts

- 100-cell data：`core_ablation_100_cells.csv`；
- checkpoint manifest：`core_ablation_checkpoint_manifest.csv`；
- control gates：`core_ablation_control_gates.csv`；
- result summary：`core_ablation_result_summary.json`；
- manuscript fragment：`table/table_iscf_bsca_core_ablation.tex`；
- standalone source：`table/table_iscf_bsca_core_ablation_standalone.tex`；
- review PDF：`output/pdf/iscf_bsca_core_ablation_20260814.pdf`。

下一步不自动追加seed或重设计allocation。若作者希望恢复allocation正向claim，应先回到Step 4–6定义新的机制/归因问题并重新冻结候选；若保持当前architecture，则如实报告equal-fusion近似tie，并继续独立推进Efficiency、Figure 5与Decoder-Transfer。
