# Efficiency：Accuracy + Four-Horizon Parameters + One-Epoch Cycle

日期：2026-08-17

Protocol：`ISCF-BSCA-EFFICIENCY-ACCURACY-PARAMS-EPOCH-20260817`

## 1. 论文问题

本表只回答一个系统级问题：在同时支持$H\in\{96,192,336,720\}$时，一个unified ISCF-BSCA model与四个horizon-specific models相比，accuracy、deployed parameter count与典型one-epoch training cycle分别如何变化。

## 2. 最小充分比较集合

正文比较ISCF-BSCA、TimeAlign与QDF。三者均具备完整Main I 7 datasets × 4 horizons accuracy，并有与对应官方复跑checkpoint一一匹配的parameter与native timing logs。

旧Efficiency中的DLinear-$H720$-prefix与PatchTST-$H720$-prefix是one-model prefix services，不满足本轮“baseline四个fixed-H models求和”的比较定义，故移出新主表。其余Main I systems只有published accuracy而无完整audited 7×4 local checkpoint/timing set，不混合不同配置的accuracy与cost。

## 3. 指标合同

- `Accuracy`：直接读取canonical Main I，对每个system计算28个dataset--horizon cells的MSE/MAE算术平均；
- `Params`：ISCF-BSCA每dataset统计一个unified inference checkpoint；TimeAlign/QDF分别把H96/H192/H336/H720四个native inference checkpoints参数相加；最后对七个datasets作macro mean；
- `1-Epoch Cycle`：每checkpoint从已完成native training logs提取`train + scheduled validation、no test`的epoch-cycle duration，并取所有完成epochs的中位数；baseline在每dataset内加和四个fixed-H models，ISCF-BSCA计一个unified model；最后对七个datasets作macro mean。

中位数降低首epoch初始化与偶发系统抖动影响。所有timing来自RTX 3090 native completed runs；它不是重新执行的exclusive-GPU microbenchmark，因此只解释为logged native epoch-cycle cost。

## 4. 完整性与边界

- accuracy cells：3 systems × 7 datasets × 4 horizons=`84/84`；
- parameter service units：3 × 7=`21/21`；
- timing checkpoint objects：ISCF 7 + TimeAlign 28 + QDF 28=`63/63`；
- new training=`0`；formal test access=`0`；
- 不因结果方向删除training-time列，也不将parameter reduction写成uniform compute advantage。
