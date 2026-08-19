# Efficiency：Main I Accuracy + Peak Memory + Checkpoint Storage

日期：2026-08-17

Protocol：`ISCF-BSCA-EFFICIENCY-ACCURACY-MEMORY-STORAGE-20260817`

## 1. 论文问题

本表回答：完整服务$H\in\{96,192,336,720\}$时，一个unified ISCF-BSCA model与四个horizon-specific models相比，accuracy、peak inference memory与checkpoint storage如何变化。

## 2. 比较集合

正文比较ISCF-BSCA、TimeAlign、QDF、AMD、SimpleTM、DLinear、iTransformer、PatchTST与TimeMixer。九者均具备完整的Main I 7 datasets × 4 horizons accuracy。前三组复跑baseline与AMD/SimpleTM使用实际trained checkpoint文件；后四组因当前本地资产不具备完整7 × 4 trained checkpoint inventory，统一使用官方Main-I脚本配置实例化四个模型，并以标准`torch.save(state_dict)`序列化大小及四模型常驻forward测量作为architecture-equivalent resource evidence。

后四组不得描述为“实际训练checkpoint文件总量”。这一统一替代合同优于混用零散trained artifacts或将Main-II的单个$H=720$ checkpoint乘四，因为后两种做法分别破坏跨dataset一致性与Main-I服务语义。TVNet仅ETTh2具备author-corrected local reproduction evidence，仍不进入resource表。

## 3. 指标合同

- `Accuracy`：canonical Main I中28个dataset--horizon cells的MSE/MAE算术平均；
- `Peak memory`：fresh process、独占RTX 3090、FP32、batch size 1、synthetic standardized input下的`torch.cuda.max_memory_allocated()`；完整服务所需模型全部resident。ISCF-BSCA执行一次$H=720$ forward并生成prefix views；八个baseline依次执行四个native fixed-H models；
- `Checkpoint storage`：ISCF-BSCA、TimeAlign、QDF、AMD与SimpleTM使用实际checkpoint文件bytes；DLinear、iTransformer、PatchTST与TimeMixer使用官方配置模型的标准`torch.save(state_dict)`checkpoint-equivalent bytes。所有baseline均对$H=96,192,336,720$四个对象求和；
- memory与storage先在每个dataset形成完整four-horizon service，再对七个datasets取macro mean。

SimpleTM的Main I accuracy沿用冻结表中的multi-run summary；resource service不是ensemble，因此每个horizon固定使用repeat 0的一个checkpoint。该repeat选择在resource profiling前冻结，不依据memory或storage结果选择。

## 4. 复用与边界

- Main I accuracy：252/252 cells；
- cost units：9 systems × 7 datasets=`63/63`；其中既有profiler复用21 units，新增实际checkpoint measurement 14 units与official-architecture-equivalent measurement 28 units；
- table-role objects：ISCF-BSCA 7个，八个horizon-specific families各28个，共231个；其中119个为actual trained checkpoints，112个为architecture-equivalent serialized state dicts；
- new training=`0`；formal test access=`0`；
- peak memory包含resident model weights与一次all-horizon service的activation peak，不等同于training memory；
- DLinear、SimpleTM与QDF的peak memory低于ISCF-BSCA，SimpleTM与DLinear的storage也更低，必须作为负向边界完整报告。
