# Phase5 Clean TimeAlign + A6-LBF-r256 Code Explanation

本文档记录 StageA 固化后的干净代码状态。旧 StageA 变体、teacher/self-teacher、EMA、QBR、target-query、
nested/residual adapter、validation-prefix diagnostic export 等代码已从主训练入口移除。当前主线只保留：

1. 原始 `official` TimeAlign dense head，用于 fixed/unified baseline；
2. `learned-basis-forecast-operator`，即 A6-LBF-r256 pure unified carrier。

## Forward Path

入口文件：

- `baselines/timealign_official/models/TimeAlign.py`
- `baselines/timealign_official/train_repo.py`

`TimeAlign.Model.forward(x, y, is_training, target_prefix)` 的主路径：

1. `x: [B, seq_len, C]` 经 `Normalize` 后 reshape 为 `[B, C, seq_len]`，进入 `PatchEmbed`；
2. history encoder 多层更新 `x`；
3. encoder output reshape 为 `hidden: [B, C, patch_num * d_model]`；
4. readout：
   - `official`: `proj_x(hidden) -> [B, pred_len, C]`;
   - `learned-basis-forecast-operator`: `coeff = learned_basis_coeff(hidden)`，
     `output[h] = learned_temporal_basis[h] @ coeff + learned_temporal_bias[h]`。

`readout_mode=official` 仍保留 TimeAlign 的 future reconstruction/alignment branch，用于复现 official
baseline。`readout_mode=learned-basis-forecast-operator` 不再实例化 `patch_emb_y`、`autoencoder`、
`proj_y`、`ffn` 或 `align`，因此 A6-LBF 的 forward/training path 不再读取 future label branch。

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

`train_repo.py` 当前按 readout mode 分开训练：

- `official`: 保留 TimeAlign 原始 `pred_loss + w_recon * recon_loss + w_align * alignment_loss`;
- `learned-basis-forecast-operator`: 强制 `w_recon=0.0`、`w_align=0.0`，只训练 `pred_loss`；
- A6-LBF 默认使用 `pred_loss_mode=multi-prefix`，使同一个 720-step model 接受
  `96/192/336/720` prefix supervision。

B4 dependency ablation 给出的代码-理论依据是：`no_align_no_recon` 相对 current A6-LBF mean MSE 仅
`+0.07%`，且有 `7/12` wins。因此 future reconstruction/alignment branch 不再作为 A6-LBF 的 active
mechanism 保留。

被删除的旧 StageA 项：

- warm-start checkpoint copying；
- teacher / self-teacher / EMA；
- target-query、QBR、nested、residual adapter、generated row modes；
- basis smoothness / coeff L2 diagnostic regularizers；
- validation-prefix diagnostic export hooks。
- A6-LBF future reconstruction/alignment branch and its auxiliary losses。

## Verification

清理后已通过：

- `conda run -n r2026-fsa python -m py_compile baselines/timealign_official/train_repo.py baselines/timealign_official/models/TimeAlign.py`;
- ETTh2 CPU smoke：`readout_mode=learned-basis-forecast-operator`，`basis_rank=256`，
  `pred_loss_mode=multi-prefix`，effective `w_recon=w_align=0.0`，training log 中 weighted recon/align
  均为 `0.0`;
- ETTh2 CPU smoke：`readout_mode=official`，`pred_loss_mode=full`，仍保留 non-zero recon/align loss；
- 结构检查：A6 模型实例不再含 `patch_emb_y/autoencoder/proj_y/ffn/align/normalization_y`，official
  模型实例仍包含这些属性。
