# CCSF Step7B shared-temperature pilot 预启动报告

## 1. 当前结论

[Fact] `SC1-SIFF-v2-CCSF-TEMP-PILOT-v1`已通过14/14类本地prelaunch gate。Step7B现只授权一个
validation-only shared-temperature pilot，不授权formal Phase A、official test或confirmation。

- 矩阵：5 datasets × 3 temperatures × seed2021 = 15 runs；
- 选择面：每个run报告H96/H192/H336/H720，共60个validation cells；
- 唯一pilot arm：`ccsf_relcal`；
- temperature grid：`{0.05, 0.1, 0.25}`；
- checkpoint：四horizon validation MSE平均最优的`best-val`；
- shared selection：对5 datasets × 4 horizons的validation MSE作macro mean，三者只选一个全局temperature；
- tie：在`1e-12`容差内选择更大的temperature；
- pilot checkpoint不进入formal comparison，选定temperature后所有10 arms必须from-scratch重训。

synthetic smoke故意令`0.1`与`0.25`并列，analyzer按冻结规则选择`0.25`，同时输出
`formal_phase_a_authorized=false`与`formal_test_access_authorized=false`。该结果只验证选择器，不是实际超参数结论。

## 2. 为什么先做pilot

CCSF的`relative-regret teacher`需要一个共享temperature。若直接在50-run official-test Phase A中选择temperature，
就会把ordinary hyperparameter selection与formal effectiveness混合。当前拆分保证：

1. validation只选择一个跨dataset共享的temperature；
2. official test不参与选择；
3. 选择后才生成例如`SC1-SIFF-v2-CCSF-v1-tau10`的formal candidate identity；
4. formal 10-arm matrix重新训练，不复用pilot weights或checkpoints。

该pilot不能证明CCSF有效，也不能淘汰architecture或objective；它只冻结正式候选所需的一个全局training constant。

## 3. 实现与artifact合同

`configs/stage_c_siff_ccsf_temperature_pilot_v1.json`冻结Step7A/profile hashes、矩阵、选择规则、授权与rollback。
`scripts/remote/run_stage_c_siff_ccsf_v1.sh`以Weather优先的dataset-major顺序在3 GPUs上执行15 runs，训练后调用
`scripts/analyze_stage_c_siff_ccsf_temperature_pilot.py`。runner强制：

- `final_evaluation_split=val`；
- `checkpoint_policy=best-val`；
- `official_test_mode=false`；
- `save_predictions=false`；
- 5 datasets × 4 horizons完整后才允许选择。

analyzer输出：

- `temperature_cell_metrics.csv`：temperature、dataset、target horizon、MSE/MAE、split、checkpoint policy/hash；
- `temperature_summary.csv`：每个temperature的20-cell macro MSE/MAE与完整性统计；
- `selected_temperature.json`：唯一选择、formal candidate name、config/profile hashes与禁止复用/禁止test标记。

本地证据位于`local_gate/`：`prelaunch_gate.json`为14/14，`pilot_jobs.csv`为15 jobs，
`selection_contract.csv`为3×20 cells。

## 4. 远程只读preflight

检查时间：2026-07-18 15:17 +08:00；host=`star3090.iai.zju.edu.cn`。

- GPU0/1/2均为RTX 3090，memory used=15 MiB、free=24110 MiB、utilization=0%；
- 无活跃`train_repo.py`或CCSF训练；
- `/home/yingch/dataset`存在；
- Python 3.12.13、PyTorch 2.9.0+cu128、CUDA 12.8，CUDA可用；
- remote repo为`main`，HEAD=`c4c4730`，尚未包含本轮代码；
- remote worktree已有3个与本轮无关的historical analysis CSV修改，必须保留，后续`git pull`前需确认不会冲突。

本轮没有remote pull、resource smoke、pilot launch或test access。

## 5. 11-step与决策边界

| Field | Record |
| --- | --- |
| `current_step` | Step7B validation-pilot prelaunch complete |
| `problem` | CCSF formal candidate缺少一个未用test选择的shared calibration temperature |
| `existence_evidence` | Step7A 18/18；contrast identifiability 5/5；不等于performance evidence |
| `idea` | 五dataset共同validation score选择一个shared temperature |
| `theory_check` | selection/test separation、tie determinism、no checkpoint reuse通过 |
| `design` | 15 runs / 60 validation cells / one arm / three temperatures |
| `narrative_gate` | 继承Step6 conditional pass；pilot本身不是contribution |
| `effectiveness_gate` | not started；formal official-test Phase A仍未授权 |
| `artifacts` | config、runner、analyzer、checker、local gate与本报告 |
| `decision` | `step7b_temperature_pilot_prelaunch_pass`；只允许后续启动pilot |

Rollback：若pilot不完整则只补缺失的冻结run；若出现numeric pathology则回Step7A修复；若完整则选择一个shared
temperature并返回post-pilot formal-candidate prelaunch audit。不得从pilot validation直接宣称机制成功，也不得在
temperature未选定前启动50-run/test矩阵。
