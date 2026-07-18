# CCSF runtime stability repair代码说明

## Failure path

`CCSFCouplingFieldReadout._true_contrast_descriptor`把每个scope group的normalized contrast汇总为mean、RMS与
endpoint descriptors。旧`group_rms=sqrt(mean(values**2))`在values全0时forward finite但backward产生NaN。
零初始化correction head使单batch smoke没有暴露该问题；head第一次更新后，梯度才进入descriptor path。

## Repair

`group_rms`现计算为`sqrt(mean(values**2) + contrast_epsilon)`。epsilon复用CCSF既有`1e-6`合同，不新增参数、
branch或hyperparameter，也不改变full-domain-before-crop projectivity。

`scripts/check_stage_c_siff_ccsf_runtime_repair.py`新增两个定向tests：

- identical arms显式制造zero contrast，检查descriptor与arm gradients全部finite；
- zero input/target下对三个temperature各执行3个AdamW updates，检查loss、gradients与parameters。

remote runner的resource smoke从1 batch增加到3 batches，并从pilot config读取external output root，避免retry与失败
artifact混放。

## Code-theory consistency

Intended theory没有变化：descriptor表示same-forward arm contrast，relative teacher监督policy。repair只把一个数学上
不光滑的RMS实现替换为标准epsilon-smoothed norm。若三batch真实Weather smoke或任一正式run仍出现non-finite，说明
当前root-cause attribution不完整，必须继续停留在Step7A runtime repair，不能启动formal evaluation。
