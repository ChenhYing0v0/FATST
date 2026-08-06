# QDF Main I `L=336` 全矩阵复现：设计与 launch gate（2026-08-06）

## 1. Cursor 与授权边界

- `current_step=Step 6 protocol freeze -> Step 8 remote execution authorized`。
- 用户明确要求停止使用当前 `L=96` mixed-source QDF block，改为 `seq_len=336` 下对 Main I 八个数据集全部本地复跑；本轮授权涵盖训练结束后的完整 formal test。
- ISCF-BSCA HPO、architecture、objective、selected profiles、TimeAlign artifacts 和其他 baselines 均保持冻结。
- 远程 queue 启动后不驻守；只有用户通知完成后才同步 artifacts、执行 32/32 audit 并重建 Main I。

## 2. Exact reproduction rule

QDF source 固定为官方仓库 commit `eb0693a962928e229417fd80b401c37b0dac6a67`，backbone=`TQNet`，seed=`2023`。官方六个 scripts 均使用 `seq_len=96`；本轮只把 `seq_len` 改为 `336`，保留各 dataset × horizon released optimization profile、`label_len=48`、30 epochs、patience 5、RevIN、QDF loss、gradient clipping 和 validation early stopping。每个 dataset-horizon 是独立 fixed-H system，训练结束后访问一次 test。

远程曾因 `num_workers=10` 触发 file-descriptor limit，因此本轮统一固定 `num_workers=0`。它改变 host-side loading 并避免已知资源故障，不改变 split、sample order、model、objective 或 checkpoint selector。

## 3. Dataset roles

| Dataset | Profile role | 必要 adaptation |
|---|---|---|
| ETTh1/ETTh2/ETTm1/ETTm2/Weather/ECL | `official_released_profile` | 仅 `seq_len: 96 -> 336` |
| Solar | `source_informed_ecl_profile` | official Solar loader、137 channels、10-minute daily `cycle=144`；optimization profile 来自 ECL |
| Exchange | `source_informed_etth1_profile` | custom loader、8 channels、保守非季节 `cycle=1`；optimization profile 来自 ETTh1 |

QDF release 不含 Solar 和 Exchange scripts。因此这两个数据集是 official-code source-informed reproduction，不得冒充 upstream official preset；它们也不构成 matched mechanism attribution。

## 4. Frozen matrix、test role 与 table replacement

- Matrix：`8 datasets × {96,192,336,720} × seed2023 = 32 fixed-H systems`。
- Metrics：每 cell 完整保存 MSE/MAE，不允许 per-cell rescue、seed selection、metric selection 或选择性删除。
- Checkpoint：upstream validation-loss early stopping；test 不选择 epoch/checkpoint。
- QDF hyperparameters：不使用 test 调优；role=`official-code local single-seed L336 reproduction`。
- Main I replacement gate：只有 32/32 checkpoints、learned QDF loss、effective configs、test metrics、logs 和 hashes 全部通过，才一次性替换当前 QDF L96 block；partial L336 values 禁止混入。
- Dense table 继续使用七个共同 datasets；Exchange companion 从 ISCF/TimeAlign 两系统扩为 ISCF/TimeAlign/QDF 三系统。其他 baseline 缺失的 Exchange cells 不插值。

## 5. Resource smoke、调度与存储

Preflight：remote GPU 0--2 均约 18 MiB、utilization 0%；quota 约 176 GiB / 200 GiB soft、220 GiB hard。QDF 不保存 prediction arrays，预计主要存储为每个 system 的 checkpoint、learned loss、metrics、config 与 logs。

Resource smoke 只运行八个 dataset 的 H720，每个至多两个 train/validation batches、`final_evaluation_split=none`，验证 data path、shape、numeric 和显存，不产生 paper evidence。Formal schedule 为 workload-aware 三队列：

- GPU0：ECL four-H -> ETTh1 four-H；
- GPU1：Solar four-H -> Exchange four-H -> ETTh2 four-H；
- GPU2：Weather four-H -> ETTm1 four-H -> ETTm2 four-H。

各 dataset 内先运行 H720/H336，再运行 H192/H96。该安排优先展开高 channel 或长序列数据，并避免 fast/slow paired barrier。

## 6. Gates 与 rollback

1. `prelaunch_pass`：JSON parse、source hashes、32-row dry-run、shell syntax、Python compile 和 table-builder regression 全部通过。
2. `resource_smoke_pass`：8/8 H720 smoke 均生成 checkpoint 与 learned loss、无 test metrics、无 OOM/NaN/Traceback。
3. `formal_pass`：32/32 unique cells 的 checkpoint/A/config/metrics/log 完整，effective config 固定 `seq_len=336`、seed2023、正式 test；artifacts 通过 hash/provenance/numeric audit。
4. `rollback`：任一 cell 不完整时保留当前 L96 QDF table，不混用 partial L336 results；先做 exact failure attribution，再决定是否只重跑失败 cell。

## 7. Prelaunch decision

`PASS_AND_COMPLETE`。用户已授权的精确32-cell block已在8/8 bounded smoke后完整执行；32/32 artifact/config/numeric gates通过，Main I replacement完成。正式结果见同目录`result_and_table_audit.md`。

## 8. Remote launch provenance

- exact experiment commit：`6eb8605d4edbd754c918a622f3c4e2d24aa6590b`
- remote host/repo：`529_Lab-3090:/home/yingch/projects/FATST`
- environment：conda `moe`，GPU 0--2=`NVIDIA GeForce RTX 3090`
- output：`/home/yingch/exp_outputs/r-2026-fatst/qdf_main_i_seq336_20260806`
- prelaunch resource：GPU 0--2 均为 18 MiB/0%；quota=`176/200 GiB soft, 220 GiB hard`
- resource smoke：2026-08-06 19:30:54--19:31:28 +08:00；8/8 checkpoint、8/8 `A.pth`、0 test metrics、0 failure markers
- formal launch：2026-08-06 19:31:57 +08:00；PID=`3885616`
- initial active jobs：GPU0 ECL-H720、GPU1 Solar-H720、GPU2 Weather-H720；三个 `run.py` processes 均存活并进入 data/meta-training path
- user-requested monitoring policy：不驻守；等待用户通知完成后再同步、audit 和重建表格

## 9. Completion

- formal completion：2026-08-06 20:42:52 +08:00
- required artifacts：160/160；checkpoint/A/metrics/config/stdout均32/32
- checkpoint hashes：32 unique；learned-loss hashes：32 unique
- result：`QDF_L336_32_of_32_COMPLETE_MAIN_I_REPLACEMENT_PASS`
- canonical result：`analysis/iscf_bsca_paper_experiment_consolidation_20260731/qdf_main_i_l336_20260806/result_and_table_audit.md`
