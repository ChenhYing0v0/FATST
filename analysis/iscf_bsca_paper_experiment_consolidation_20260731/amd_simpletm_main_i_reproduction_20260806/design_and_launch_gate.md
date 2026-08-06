# AMD / SimpleTM Main I 官方复现：设计与 launch gate（2026-08-06）

## 1. 当前步骤与授权边界

- `current_step=Step 6 protocol freeze -> Step 7 adapter implementation -> Step 8 remote launch pending resource gate`。
- 用户已授权 AMD、SimpleTM 的 baseline evidence，包括官方源码、官方训练 profiles、remote training 与 formal test。
- 本轮只覆盖 Main I 的 7 个 dense datasets：`ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `Weather`, `ECL`, `Solar`，以及 `H={96,192,336,720}`。
- `ISCF-BSCA HPO=false`；TimePerceiver、SRSNet 和其他 baseline 不在本次授权内。
- 两个 baseline 都属于 `native_horizon_specific_accuracy_context`，不得用作 ISCF-BSCA 的 matched mechanism attribution。

## 2. Source 与 license audit

### 2.1 AMD

- official repository：`https://github.com/TROUBADOUR000/AMD`；
- frozen commit：`000d377a1ed8946aa817ff357cdf1de64b99abb9`；
- license：MIT；
- 7 个目标 dataset 均有 released shell script；
- official contract：`seq_len=512`、`seed=2024`、每个 horizon 一次训练；
- upstream `main.py` 在最后一个 epoch 无条件覆盖 `best_model`，因此有效 selector 是 `official-last`，不是严格 validation-best；formal test 只在训练结束后调用一次。

本地不重新调参，也不把 AMD 改成 `L=96/336`。结果必须标记为 local official-native reproduction。

### 2.2 SimpleTM

- official repository：`https://github.com/vsingh-group/SimpleTM`；
- frozen commit：`3c77d820837b726afb03c943235ea95bc924243d`；
- upstream repository 未提供 LICENSE；因此不把源码 vendoring 或提交到 FATST，只在 repo-external remote output 下 clone exact commit，按 research-only execution 使用；
- 7 个目标 dataset 均有 released script；
- official contract：`seq_len=96`、`fix_seed=2025`、validation early stopping `patience=3`；
- scripts 原生使用 `itr=3`，但 Solar H192/H336 使用 `itr=2`，故完整矩阵共有 82 个 checkpoint repetitions。

SimpleTM upstream training loop每个 epoch 计算 test loss。为遵守项目的 validation-only epoch/checkpoint selection 边界，本轮 runtime adapter 只移除该 epoch-level test pass；保留官方超参数、训练 objective、validation early stopping 与最终 evaluation。每个 validation-selected checkpoint 在训练结束后只执行一次 formal test。另将 `num_workers` 固定为 0，避免远程 file-descriptor failure；该变更不参与模型选择。

首次resource smoke在训练step前暴露NumPy 2 compatibility failure：upstream `np.Inf`在remote NumPy 2.4中已移除。Compatibility adapter只执行单次、hash-frozen的`np.Inf -> np.inf`别名替换，不改变数值、objective或selector。失败的ECL/Solar/Weather smoke logs必须保留为protocol provenance。

Compatibility修复后的三个jobs均完成checkpoint且test=0，但初版health checker把upstream正常消息`Validation loss decreased (inf --> value)`误判为numeric Inf。Checker只对白名单化这一条initialization message；其余独立NaN/Inf、OOM、Traceback与file-descriptor检测保留。对应三份false-positive logs也保留，不复用其checkpoint进入最终14-job smoke manifest。

## 3. Frozen matrix

| Baseline | Datasets | Horizons | Native repeats | Checkpoints | Table cells |
|---|---:|---:|---:|---:|---:|
| AMD | 7 | 4 | 1/cell | 28 | 28 |
| SimpleTM | 7 | 4 | 3/cell，Solar H192/H336=2 | 82 | 28 |
| Total | — | — | — | 110 | 56 |

Main I cell value定义：AMD 使用单次 native run；SimpleTM 使用同一 official command 的 native `itr` repetitions 的 arithmetic mean。禁止按 repeat、metric、dataset 或 horizon 选择性保留结果。

## 4. Dataset 与 metric contract

- 数据文件与当前 TimeAlign/QDF/ISCF Main I 使用同一 remote dataset identity，并在执行前逐文件 SHA256 验证；
- ETT 使用官方固定 12/4/4 months split；Weather/ECL/Solar 使用 upstream 70/10/20 split；
- `features=M`，标准化器只拟合 train split；
- formal metrics 为 standardized-scale MSE/MAE；
- SimpleTM official test loader 的 `drop_last=True` 保留并披露；AMD test loader 的 `drop_last=False` 保留；因此两者是 source-native accuracy context，不是 matched protocol comparison。

## 5. Artifact schema

每个 `baseline × dataset` unit 必须保留：

1. exact upstream repository/commit 与 source hashes；
2. official script path/hash；
3. runtime patch hashes；
4. 完整 `run.log`；
5. 每个 repetition 的 checkpoint/hash；
6. 每个 repetition 的 MSE/MAE 与 test role；
7. `complete.json`；
8. 后处理生成的 56-cell table surface 与全量 artifact manifest。

只有 AMD 28/28 与 SimpleTM 82/82 raw rows 同时通过 hash/provenance/numeric-health audit 后，才允许一次性替换 Main I 中这两个 baseline；partial rows 不进入论文表。

## 6. Resource、调度与 storage gate

- remote host：`529_Lab-3090`；conda env：`moe`；GPU 0/1/2；
- output root：`/home/yingch/exp_outputs/r-2026-fatst/amd_simpletm_main_i_7dataset_20260806`；
- 当前 user quota 为 177 GiB used / 200 GiB soft / 220 GiB hard，最大剩余约 43 GiB；
- 本轮 storage budget 冻结为 35 GiB；resource smoke 后按 checkpoint size 外推，若预计超过预算则不启动 formal queue；
- 先并行运行 SimpleTM ECL/Solar/Weather，后续由 dynamic queue 填充，避免 fast GPU 等待固定配对；
- resource smoke 为 14 个 `baseline × dataset` H720 jobs，每个只训练 1 epoch/1 repeat，不访问 test，并必须生成一个 checkpoint。

## 7. Gates 与 rollback

### Prelaunch gate

- exact commits、source/script hashes、dataset hashes已冻结；
- 14 个 dataset units、56 table cells、110 training repetitions可由 config 机械重建；
- JSON、Python compile、shell syntax、runtime patch exact-match 与 dry-run 必须全部通过。

### Resource gate

- 14/14 smokes完成；
- 14 checkpoints；0 formal test rows；
- 无 OOM、NaN、Traceback、file-descriptor failure；
- 预计总 storage 不超过 35 GiB。

### Formal gate

- AMD 28 raw metrics/checkpoints；SimpleTM 82 raw metrics/checkpoints；
- 110 unique checkpoint hashes；
- 56/56 table cells，全部 MSE/MAE，无选择性缺失；
- test access/selector/source provenance完整。

### Rollback

任一 gate 失败时，当前 Main I 中 AMD/SimpleTM 的 published values 保持不变；不得混合 partial local rows。OOM 或 compatibility failure只归因于 exact source/protocol execution，不作为 baseline 方法效果结论。

## 8. Prelaunch decision

`prelaunch=conditional_pass`：用户授权、source、matrix、adapter 与 rollback 边界已冻结；在完成本地最小验证、commit/push、remote pull、GPU复核和 14/14 resource smoke 前，不启动 formal queue。

Remote GitHub HTTPS checkout在训练前连续两次超时。允许的source-transport fallback为：把本地已经exact commit/hash audit、且保留`.git` metadata的两个official checkout同步到repo-external `_upstream`目录；remote runner重新验证commit、clean status与全部source hashes。该fallback只改变source transport，不改变executed source或experiment contract。
