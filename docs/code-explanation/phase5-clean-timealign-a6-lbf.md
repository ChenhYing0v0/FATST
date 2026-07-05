# Phase5 Clean TimeAlign + A6-LBF-r256 Code Explanation

本文档记录 StageA 固化后的干净代码状态。旧 StageA 变体、teacher/self-teacher、EMA、QBR、target-query、
nested/residual adapter、validation-prefix diagnostic export 等代码已从主训练入口移除。当前主线只保留：

1. 原始 `official` TimeAlign dense head，用于 fixed/unified baseline；
2. `learned-basis-forecast-operator`，即 A6-LBF-r256 unified carrier。

## Forward Path

入口文件：

- `baselines/timealign_official/models/TimeAlign.py`
- `baselines/timealign_official/train_repo.py`

`TimeAlign.Model.forward(x, y, is_training, target_prefix)` 的主路径：

1. `x: [B, seq_len, C]` 经 `Normalize` 后 reshape 为 `[B, C, seq_len]`，进入 `PatchEmbed`；
2. history encoder 多层更新 `x`；
3. 训练时 future label `y: [B, pred_len, C]` 进入 future autoencoder path；
4. `align_loss = align(ffn(x), y.detach())`，即 alignment pressure 仍主要约束 history path；
5. encoder output reshape 为 `hidden: [B, C, patch_num * d_model]`；
6. readout：
   - `official`: `proj_x(hidden) -> [B, pred_len, C]`;
   - `learned-basis-forecast-operator`: `coeff = learned_basis_coeff(hidden)`，
     `output[h] = learned_temporal_basis[h] @ coeff + learned_temporal_bias[h]`。

## A6-LBF Tensor Contract

对于 A6-LBF-r256：

- `hidden`: `[B, C, R]`，其中 `R = patch_num * d_model`;
- `learned_basis_coeff`: `R -> 256`;
- `learned_temporal_basis`: `[720, 256]`;
- `target_prefix = H` 时只取 `learned_temporal_basis[:H]` 与 `learned_temporal_bias[:H]`;
- 输出为 `[B, H, C]`，训练/评估时按 requested horizon 比较 prefix。

这个实现保留了 StageA 的主要结论：A6-LBF-r256 是 anchor-free、非 residual、prefix-native 的 unified
forecast operator，并且在 fixed-horizon per-horizon TimeAlign 对照上达到 `9/12` MSE wins、overall
MSE `-4.82%`。

## Training Contract

`train_repo.py` 当前只保留必要训练项：

- `pred_loss`：`full` 或 `multi-prefix`;
- `recon_loss`：official TimeAlign future reconstruction pressure；
- `alignment_loss`：official TimeAlign history/future alignment pressure；
- `loss = pred_loss + w_recon * recon_loss + w_align * alignment_loss`。

被删除的旧 StageA 项：

- warm-start checkpoint copying；
- teacher / self-teacher / EMA；
- target-query、QBR、nested、residual adapter、generated row modes；
- basis smoothness / coeff L2 diagnostic regularizers；
- validation-prefix diagnostic export hooks。

## Verification

清理后已通过：

- `python -m py_compile baselines/timealign_official/train_repo.py baselines/timealign_official/models/TimeAlign.py`;
- ETTh2 CPU smoke：`readout_mode=learned-basis-forecast-operator`，`basis_rank=256`，
  `pred_loss_mode=multi-prefix`;
- ETTh2 CPU smoke：`readout_mode=official`，`pred_loss_mode=full`。
