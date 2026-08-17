# Efficiency：Main I Accuracy + Peak Memory + Checkpoint Storage

日期：2026-08-17

Protocol：`ISCF-BSCA-EFFICIENCY-ACCURACY-MEMORY-STORAGE-20260817`

## 1. 论文问题

本表回答：完整服务$H\in\{96,192,336,720\}$时，一个unified ISCF-BSCA model与四个horizon-specific models相比，accuracy、peak inference memory与checkpoint storage如何变化。

## 2. 比较集合

正文比较ISCF-BSCA、TimeAlign、QDF、AMD与SimpleTM。五者具备完整的Main I 7 datasets × 4 horizons accuracy，以及与对应本地复跑checkpoints匹配的7-dataset profiler evidence。

iTransformer-$H720$-prefix、DLinear-$H720$-prefix与PatchTST-$H720$-prefix当前只有每dataset一个$H=720$ checkpoint，不属于本轮“四个horizon-specific checkpoints求和”的对比角色。TVNet仅ETTh2具备author-corrected local reproduction evidence。其余Main I systems缺少完整且配置匹配的7 × 4 local checkpoint与peak-memory evidence，不混入主表。

## 3. 指标合同

- `Accuracy`：canonical Main I中28个dataset--horizon cells的MSE/MAE算术平均；
- `Peak memory`：fresh process、独占RTX 3090、FP32、batch size 1、synthetic standardized input下的`torch.cuda.max_memory_allocated()`；完整服务所需checkpoints全部resident。ISCF-BSCA执行一次$H=720$ forward并生成prefix views；TimeAlign/QDF/AMD/SimpleTM依次执行四个native fixed-H models；
- `Checkpoint storage`：实际checkpoint文件bytes。ISCF-BSCA计一个unified checkpoint；TimeAlign/QDF/AMD/SimpleTM分别将$H=96,192,336,720$四个checkpoint文件相加；
- memory与storage先在每个dataset形成完整four-horizon service，再对七个datasets取macro mean。

SimpleTM的Main I accuracy沿用冻结表中的multi-run summary；resource service不是ensemble，因此每个horizon固定使用repeat 0的一个checkpoint。该repeat选择在resource profiling前冻结，不依据memory或storage结果选择。

## 4. 复用与边界

- Main I accuracy：140/140 cells；
- cost units：5 systems × 7 datasets=`35/35`；其中既有profiler复用21 units，新增AMD/SimpleTM peak-memory measurement 14 units；
- checkpoint objects：ISCF-BSCA 7个，四个horizon-specific families各28个，共119个table-role objects；
- new training=`0`；formal test access=`0`；
- peak memory包含resident model weights与一次all-horizon service的activation peak，不等同于training memory；
- QDF peak memory低于ISCF-BSCA，必须作为负向边界完整报告。
