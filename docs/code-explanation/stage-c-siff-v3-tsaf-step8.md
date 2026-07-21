# Stage C SIFF-v3 TSAF Step8 Remote/Test Tooling Explanation

## 1. Evaluator compatibility

`expected_matrix_size`原先只读取`arms`。TSAF config把historical references与new-training arms统一放在
`effective_arms`，因此helper现按`arms -> effective_arms -> matrix.arms`顺序解析。该变化不改变已有config的行为，
只使TSAF的45-run authorization可被通用evaluator识别。

TSAF formal config补充：

- `expected_runs=45`：完整effective matrix，不是25个new runs；
- `coupling_scales`：readout invariant核对；
- `training_contracts`：只接受H720 full loss、four-horizon validation selector与`equal_skill`；
- `diagnostic_protocol.future_bins`：定义八个lead-time bins。

## 2. Runner modes

runner现在有三个互斥执行角色：

1. normal：训练25个new runs，final evaluation只用validation；
2. `RESOURCE_SMOKE=1`：两个2-batch resource checks，不产生formal result；
3. `FORMAL_TEST_ONLY=1`：要求training artifacts已完整，只读取checkpoint并生成formal-test artifacts。

formal-test mode在每个run前后计算`checkpoint.pt` SHA256；任何变化立即失败。`STATUS_ONLY=1`分别报告training与test
完成数。training与formal-test分别写`launch_record_seed*.txt`和`formal_test_launch_record_seed*.txt`，防止后者
覆盖training provenance。dry-run不触发conda、dataset或test访问。

## 3. Authorization boundary

2026-07-21 authorization只覆盖seed2021的25-run training和一次完整formal test。confirmation seeds始终false。
checker在authorized state下不在本机尝试normal runner，只验证syntax、executable、25-job dry-run、evaluator
authorization与confirmation boundary。

## 4. Code-theory consistency

- Intended protocol：validation选择checkpoint，test只评估冻结checkpoint；
- Realized code：training与formal-test mode分离，formal mode要求25/25 training先存在；
- Nonmutation evidence：test前后checkpoint SHA256必须相同；
- Remaining risk：resource use、remote dataset path和真实batch numeric behavior仍需Step8 smoke；
- Falsification：任一training/test artifact或hash contract失败时停止，不允许用partial cells判定TSAF。
