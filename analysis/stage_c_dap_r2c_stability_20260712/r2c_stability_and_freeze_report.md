# SC0-DAP-R2C Stability Confirmation And Profile Freeze

## Plan And Meaning

Phase C没有继续搜索超参数。它只为Phase B选定的三个profile补跑seeds2022/2023，并复用seed2021
selection artifacts。对每个dataset和dense horizon计算三seed MSE的sample CV；mean dense CV不超过3%、
maximum dense CV不超过5%才允许冻结。

selected-only confirmation检验absolute stability，不重新比较落选arms，因此
`relative_winner_reconfirmed=false`。params与test均未进入gate。

## Artifact Integrity

6/6新增runs训练完成，连同3个复用runs共9/9 profile-seed实例、72/72 dense validation metrics完整；配置、
seed、shape和finite-training检查均通过。初次自动分析把复用artifact的旧calibration hash误要求为Phase C新
hash，造成3个false mismatch；修正provenance规则后无需重训。

## Stability Result

| Dataset | Frozen profile | Mean dense MSE CV | Max dense MSE CV | Gate |
| --- | --- | ---: | ---: | --- |
| Weather | `P12/D64/ff128` | 0.323% | 0.749% | pass |
| ETTm1 | `P24/D32/ff64` | 0.707% | 1.405% | pass |
| ETTh2 | `P12/D64/ff128` | 2.094% | 4.867% | pass, boundary-close |

ETTh2 maximum CV距离5%阈值只有0.133个百分点。因此它通过预注册gate，但后续机制实验必须报告逐seed与
逐horizon结果，不能把该carrier描述为高度稳定。

## Decision

[Decision] `SC0-DAP-R2`完成，正式冻结
`configs/stage_c_mechanism_control_natural_dataset_profiles.json`。后续同dataset的baseline、method、ablation
与matched control必须共享相同profile；禁止根据test或新mechanism重新选择。

[Failure Attribution] 本轮没有direction-level failure。初次`complete=False`是analyzer provenance check
错误，属于analysis implementation fault，修正后完整性与stability均通过。ETTh2的边界余量是protocol
noise风险，而非当前失败；它应进入后续effect-size与multi-seed判据设计。

## Next Rollback Cursor

SC0 research-instrument blocker关闭。StageC回到Step 1-3，下一步是SC1-PFO/SC2-HML的prior-art与problem
existence诊断；尚未授权任何paper-core method实现。
