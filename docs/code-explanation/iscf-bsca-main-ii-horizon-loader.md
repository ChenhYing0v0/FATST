# Main II horizon-specific loader 重算代码说明

## 1. 目标与边界

本轮不训练新模型，只复用已冻结 external baseline H720 checkpoints。代码修正的是 evaluation surface：每个 H 都从该 baseline 的 official H-specific script/data provider 构造 test loader，再将 history 输入 H720 model，取 output 前 H steps 与该 loader 的 H-step target 比较。

## 2. 数据与 tensor flow

对 batch size $B$、history length $L$、channel count $C$、requested horizon $H$：

- loader 产生 `batch_x: [B, L, C]` 与包含 H-step target 的 `batch_y`；
- H720 model 固定输出 `outputs: [B, 720, C]`；
- evaluator 形成 `prediction = outputs[:, :H, :]`；
- evaluator 形成 `target = batch_y[:, -H:, :]`；
- MSE/MAE 在 `[B, H, C]` 的所有元素上累计。

如果 official model API 要求 decoder input，则 `label_len` 部分来自已知历史，future 720 slots 全部为零。代码不会把 `batch_y[:, -H:, :]` 或其扩展传入预测分支。

## 3. evaluator 分工

- `evaluate_main_ii_timealign_checkpoint.py`：复用 TimeAlign effective config 与 H720 checkpoint，loader 的 `pred_len` 改为 H，并用 `is_training=False` 的 history-only forecast path；
- `evaluate_main_ii_qdf_checkpoint.py`：复用 QDF H720 `TQNet`，以 fixed-H QDF data provider 提供 history 与 cycle index；
- `evaluate_main_ii_amd_simpletm_checkpoint.py`：从 official logs/scripts 分别捕获 H720 model command 与 H-specific loader command；
- `evaluate_main_ii_horizon_loader_upstream_checkpoint.py`：为 iTransformer、PatchTST、DLinear 复用 frozen H720 training command/checkpoint，同时 monkeypatch official test entrypoint 只替换 loader horizon 与 prefix comparison；
- `build_main_ii_horizon_loader_job_manifest.py`：验证 63 unique checkpoint hashes 并展开 252 formal jobs；
- `run_main_ii_horizon_loader_formal_tests.sh`：按 dataset-major workload 顺序在 GPU 0/1/2 并行执行，并逐 job 检查 `drop_last`、input-only 标记和 artifact 完整性。

## 4. code-theory consistency

理论要求是比较“同一个 H720 forecasting function 在各 official fixed-H test surface 上的 prefix 风险”，而不是“同一组 H720 origins 上的多个 prefix 风险”。新代码通过固定 model horizon、改变 loader horizon实现这一点。仍属 source-native system comparison：不同 baseline 的 lookback、optimizer、scaler与 `drop_last` 保留各自 official semantics，因此不能用于 matched mechanism attribution。

可证伪条件包括：实际 loader `drop_last` 与冻结 source audit 不同、origin count 不随 H 合理变化、H720 无法复现同 checkpoint anchor、checkpoint hash 变化、或 future label 进入模型输入。任一条件出现都会阻断表格生成。
