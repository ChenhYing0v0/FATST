# C0 ETTm1 Encoder Control Gate 代码说明

## Scope

本次更新只让 clean A6 Encoder 的结构参数与 checkpoint evaluation 更可控。它不是 StageB method update，
也不改变 accepted learned-basis forecast operator 的定义。

## Training adapter

`baselines/timealign_official/train_repo.py` 新增四个 legacy Encoder overrides：

- `--legacy-patch-num` -> `official_args.patch_num`；
- `--legacy-d-model` -> `official_args.d_model`；
- `--legacy-d-ff` -> `official_args.d_ff`；
- `--legacy-dropout` -> `official_args.dropout`。

这些 overrides 只允许用于 token-MLP Encoder 与 exact `learned-basis-forecast-operator`，防止误改 official
baseline 或其他 StageB method。`patch_num` 必须整除 `seq_len`，`d_model` 必须为正偶数。

实际 tensor path 保持：

```text
x [B,720,C]
 -> flatten channels [B,C*720]
 -> PatchEmbed [B,C*P,D]
 -> two residual token MLPs [B,C*P,D]
 -> memory [B,C,P,D]
 -> hidden [B,C,P*D]
 -> coeff [B,C,256]
 -> prefix prediction [B,H,C]
```

`model_diagnostics.json` 新增：

- `active_forward_parameters`：clean A6 forward 使用的 PatchEmbed、Encoder、LayerNorm、basis coefficient
  与 temporal basis/bias 参数；
- `unused_proj_x_parameters`：实例中存在但 clean A6 forward 不调用的 dense projection；
- `inactive_or_other_parameters`：`total - active`，用于审计而非 capacity matching。

## Dual checkpoint evaluation

`--evaluate-dual-checkpoints` 在同一次 optimization trajectory 结束后保留：

- `last_state`：最后一个 epoch 更新后的 state；
- `best_state`：validation mean MSE 最低 epoch 的 state。

两者分别 strict load 到同一 model 后执行相同 test loader，并写出
`metrics_last_*` 与 `metrics_best_val_*`。主 `checkpoint.pt/metrics_by_target_horizon.csv` 仍遵从
`--checkpoint-policy`，保证现有 downstream scripts 兼容。

## Runner and analyzer

`scripts/remote/run_phase5_stage_b_c0_ettm1_carrier_protocol_gate.sh` 用 `wait -n` 在可用 GPUs 上动态填充
六个 ETTm1 arms，默认 seed 2021、10 epochs。每个 arm 记录 effective config、environment、training log、
dual metrics 与 diagnostics。

`scripts/analyze_phase5_stage_b_c0_ettm1_carrier_protocol_gate.py` 分别读取 last/best-val：

1. wider P1 vs accepted P1，检查 global-state bottleneck；
2. low-capacity P5 vs P1，检查 capacity collapse；
3. parameter-matched P5 vs P1，在 dropout 0.9/0.2 下检查 patch effect。

`relative_mse_pct=(candidate/baseline-1)*100`，负值更好。单个 selector/dropout preliminary gate 要求
mean `<=-0.5%`、至少 3/4 wins、最大 regression `<=+1.0%`。

## Code-theory consistency

理论目标是把 `patch_num`、global state width、active parameter capacity、dropout 与 selector 分离。代码
通过固定 readout/loss/data/seed，并只覆盖四个 Encoder scalars，实现了最小控制矩阵；dual selector 使用同一
训练轨迹，避免 seed/CUDA trajectory confound。

仍未实现功能等价控制：P1 与 P5 的 interaction topology 本来不同，parameter matching 也不等于 function
matching。冻结 interaction diagnostic 已明确记录这一边界。若效果只出现在一个 dropout 或 selector，当前
patch-defect 解释即被 falsify。
