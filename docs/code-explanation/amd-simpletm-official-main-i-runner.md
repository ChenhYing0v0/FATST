# AMD / SimpleTM official Main I runner 代码说明

## 1. 目标与边界

本模块在不导入旧项目代码的前提下，从 AMD 与 SimpleTM 官方仓库的固定 commit 复现 Main I 七数据集结果。它不修改 ISCF-BSCA model，也不提供新的 HPO 接口。

核心文件：

- `configs/amd_simpletm_official_main_i_reproduction.json`：source、dataset、matrix、authorization 和 gate 的 machine-readable contract；
- `scripts/remote/run_amd_simpletm_official_main_i_reproduction.sh`：remote checkout、GPU-aware queue、resource/formal gate；
- `scripts/run_amd_simpletm_official_dataset.py`：单个 `baseline × dataset` 执行、runtime adapter、artifact 物化；
- `scripts/analyze_amd_simpletm_official_main_i_reproduction.py`：110 raw repetition rows 到 56 table cells 的完整性与 hash 审计。

## 2. Remote orchestrator

Shell runner支持 `dry-run / resource-smoke / run / status` 四种模式。

1. `dry-run` 只展开 14 个 dataset units，并断言 56 cells、110 repetitions；
2. `resource-smoke` 在 exact upstream checkout 的临时副本上执行 H720 one-epoch/no-test smoke；
3. `run` 首先检查 14/14 smoke completion records，再由 GPU 0/1/2 动态领取 formal units；
4. `status` 只读取 `complete.json`，不启动或修改实验。

每个 GPU worker一次独占一个 dataset unit；同一 unit 内按 official script 顺序执行 horizons/repetitions，因此不会出现多个进程写同一 checkpoint 目录。

## 3. AMD execution path

输入为 official dataset script与数据文件软链接。formal mode直接执行 released shell script：

`[batch, L=512, C] -> AMD -> [batch, H, C] -> MSE objective`

AMD upstream在每个 epoch计算 validation MSE，但最后一个 epoch无条件写入 `best.pt`；因此 artifact 记录为 `official-last`。训练结束后，runner从 stdout解析四个 horizon 的 test MSE/MAE，并将 `checkpoints/{name,name2,name3,name4}/best.pt` 依次映射到 H96/H192/H336/H720。

resource-smoke 的副本只把 horizon缩为720、epoch缩为1、train/validation各限制为2 batches，并在 final test前返回。该 patch只用于执行健康检查，不产生 paper-facing metric；formal mode不设置batch限制。

## 4. SimpleTM execution path

Runner读取 official shell script，先合并 shell line continuation，再解析四条 `python -u run.py` command。所有 released hyperparameters与 native `itr` 次数保留。

运行时只做 protocol 与 artifact adapters：

1. 从 `train()` 删除 epoch-level test loader/evaluation，checkpoint仍只由 validation early stopping选择；
2. formal mode在每个 selected checkpoint后执行一次 `test()`；resource-smoke通过环境 gate跳过 test；
3. CLI末尾追加 `--num_workers 0`，避免 remote file-descriptor failure，不改变数据样本、objective或selector。
4. 将upstream `utils/tools.py`中的`np.Inf`替换为NumPy 2等价别名`np.inf`；该compatibility patch发生在early-stopping初值，不改变训练语义。
5. upstream training `setting` 的format string遗漏最后一个`ii`占位符，使同一command的native `itr` checkpoints覆盖到同一路径；runner在已格式化的`setting`末尾追加`_{ii}`，只恢复repeat artifact identity，不改变seed推进、训练、validation selection或test数值。

对应张量主路径为：

`batch_x [B,L,C] -> wavelet/GeomAttention encoder -> output [B,H,C] -> MSE + l1_weight * attention_regularizer`。

每条 command的 `result_long_term_forecast.txt` 增量段必须产生与 `itr` 相同数量的 MSE/MAE rows；修复后checkpoint目录末尾的repeat index与 row一一对应。若只有metrics而checkpoint被覆盖，整个dataset unit必须失败，不能用其partial results构表。

## 5. Artifact 与统计定义

`metrics.csv` 每行字段：

- `baseline/dataset/horizon/repeat`：raw repetition identity；
- `seed_contract`：official seed语义；
- `mse/mae`：该 checkpoint 的一次 formal test；
- `checkpoint/checkpoint_sha256`：远程 artifact与immutable identity；
- `test_role`：selector/test access边界。

Analyzer逐 checkpoint重新计算 SHA256，拒绝重复 hash、缺失 repetition 或缺失 cell。SimpleTM table cell等于 official native repetitions的 arithmetic mean；AMD cell等于其单次 official run。最终必须严格得到 56 cells。

## 6. Code-protocol consistency

- intended protocol：官方源码与官方 profiles，validation-only epoch/checkpoint selection，完整 MSE/MAE surface；
- code realization：exact commit/hash verification、SimpleTM test-access patch、source-native repeats与complete-matrix analyzer；
- remaining proxy：两仓库的 native split/loader细节并未改成与 ISCF 完全 matched，且 SimpleTM upstream缺少 license；
- falsification：任何 source hash mismatch、test in smoke、110 checkpoint/metric不完整、重复 checkpoint hash、numeric/runtime failure或 storage超预算都会阻止 table replacement。
