# CCSF Step7B pilot工具代码说明

## 1. 配置与边界

`configs/stage_c_siff_ccsf_temperature_pilot_v1.json`不是formal method config，而是validation-only超参数选择协议。
它锁定Step7A与five-profile hashes、`{0.05,0.1,0.25}`、seed2021、15-run矩阵和60-cell完整性，并分开记录
`pilot_remote_training_authorized=true`、`formal_phase_a_authorized=false`与
`formal_test_access_authorized=false`。因此pilot授权不能被解释为formal test授权。

## 2. Remote runner

`scripts/remote/run_stage_c_siff_ccsf_v1.sh`当前执行temperature pilot：

1. 从config和profile生成`dataset × temperature`的15条job；
2. 以Weather、ETTm1、ETTm2、ETTh1、ETTh2的dataset-major顺序展开，三worker跨GPU分配；
3. 每条job以`hidden -> CCSF arms/contrast/policy -> full T forecast -> validation prefixes`训练和评估；
4. 强制`best-val`、四horizon validation selector、`final_evaluation_split=val`、no official test和no predictions；
5. 全部完成后才调用shared-temperature analyzer。

`DRY_RUN=1`只打印15条合同；`RESOURCE_SMOKE=1`运行Weather/tau0.1的三batch smoke；`STATUS_ONLY=1`只统计
完整run。正式运行可恢复已完成job，但完整性由四个核心artifact共同判断。

三batch要求来自首次runtime failure：单batch只验证初始forward/backward，无法覆盖correction head更新后梯度进入
contrast descriptor的第二步路径。runner同时从pilot config读取`remote_root`，使失败attempt与retry artifacts隔离。

## 3. Shared-temperature analyzer

`scripts/analyze_stage_c_siff_ccsf_temperature_pilot.py`读取每个run的`metrics_by_target_horizon.csv`与
`checkpoint.pt`。每个temperature必须恰有5 datasets × 4 horizons=20 cells，且每个cell必须是`val/best-val`和
finite MSE/MAE。选择分数为：

$$
S(\tau)=\frac{1}{5\times4}\sum_{d}\sum_{H\in\{96,192,336,720\}}
\operatorname{MSE}_{val}(d,H;\tau).
$$

选择最小$S(\tau)$；容差内并列时取更大的$\tau$。输出列的来源与含义如下：

- `temperature_cell_metrics.csv`：原始validation cell，加上checkpoint SHA256和artifact path；
- `temperature_summary.csv::macro_mse/macro_mae`：20个完整cells的算术平均；
- `dataset_count/horizon_count/validation_cells`：防止partial matrix误选；
- `selected_temperature.json`：选择结果及formal候选命名，但显式保持Phase A/test未授权。

## 4. Prelaunch checker

`scripts/check_stage_c_siff_ccsf_step7b.py`检查Step7A config/local-gate hashes、profile hash、temperature grid、
15 jobs、workload order、15次CLI parse、validation-only选择边界、runtime repair hash、授权隔离、runner dry-run、
synthetic tie和external output root。它生成`pilot_jobs.csv`、`selection_contract.csv`与`prelaunch_gate.json`。

synthetic smoke中tau0.1与tau0.25并列且必须选tau0.25；这里只证明tie-break实现与配置一致，不表示实际pilot应选
tau0.25。

## 5. Code-theory consistency

Intended theory是先用不含test label的共同validation evidence固定teacher geometry，再以一个formal candidate
检验architecture/objective归因。代码通过完整矩阵、shared selection、no-test flags和no-checkpoint-reuse实现该边界。

仍未被证明的部分：哪一个temperature最优、CCSF能否超过v1/A6_MEASURE/independent controls，以及contrast和
RELCAL是否形成非冗余interaction。这些必须在实际pilot与随后重新冻结的formal Phase A中回答。

Falsifiers包括：partial matrix仍产生选择、per-dataset temperature、test split进入pilot、pilot checkpoint被复用、
formal Phase A在temperature选择前启动，或actual runner与config hashes不一致。
