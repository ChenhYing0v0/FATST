# Phase5-A6S Official-Last Stability Code Explanation

本文档说明 `baselines/timealign_official/train_repo.py` 中新增的 A6S stability 训练选项。
这些选项默认关闭，不改变既有 TimeAlign / A6 行为。

## Forward And Optimization Flow

### EMA final weights

当 `--ema-decay > 0` 时，训练开始后创建 `ema_state`：

```text
ema_state[name] = model.state_dict()[name].clone()
```

每次 `optimizer.step()` 后更新：

```text
ema_state = decay * ema_state + (1 - decay) * current_state
```

若同时设置 `--ema-eval`，训练结束后加载 `ema_state`，再保存 `checkpoint.pt` 并运行 test
evaluation。因此输出仍然来自 fixed training schedule 的 final weights path，不使用 validation-best
selection。

### Learned-basis operator smoothness

该正则只允许用于 `readout_mode == learned-basis-forecast-operator`。A6-LBF 的输出路径为：

```text
hidden: [B, C, R]
learned_basis_coeff.weight: [K, R]
learned_temporal_basis: [720, K]
operator = learned_temporal_basis @ learned_basis_coeff.weight: [720, R]
coeff = learned_basis_coeff(hidden): [B, C, K]
output = einsum("hk,bck->bch", basis[:H], coeff): [B, C, H]
```

新增 `--basis-operator-smoothness-weight` 后，训练 loss 追加：

```text
mean((operator[t+1] - operator[t])^2)
+ mean((learned_temporal_bias[t+1] - learned_temporal_bias[t])^2)
```

新增 `--basis-coeff-l2-weight` 后，训练 loss 追加：

```text
mean(learned_basis_coeff.weight^2) + mean(learned_basis_coeff.bias^2)
```

两者的原始 loss 值会写入 `training_log.csv`：

- `train_basis_operator_smoothness_loss`
- `train_basis_coeff_l2_loss`
- `basis_operator_smoothness_weight`
- `basis_coeff_l2_weight`
- `ema_decay`
- `ema_eval`

## Code-Theory Consistency

[Fact] EMA 不改变模型结构，也不读取 validation-best checkpoint。它检验 final checkpoint 的 weight
trajectory variance 是否是 A6 drift 的主要来源。

[Fact] operator smoothness 直接约束 A6-LBF 的 induced dense-equivalent operator，而不是添加 residual
prediction path 或 pretrained anchor。因此它仍保持 prefix-native learned operator 的机制边界。

[Proxy] operator smoothness 只是 stability proxy。它不能证明 learned basis 的真实泛化机制，且可能损害
需要 sharp future-row dictionary 的数据集。

[Falsification] 若 `A6S-EMA` 改善明显而 `A6S-HeadStability` 无效，则问题更像 generic optimization
variance；若二者均无效，则 A6-LBF 的剩余 gap 不是简单 final-weight stability 或 operator smoothness
可修复，需要回 Step 4/5 设计更强 stability mechanism。

## Analyzer And Remote Wrappers

`scripts/analyze_phase5_timealign_hss_a6s_stability_gate.py` 读取每个 run 的
`metrics_by_target_horizon.csv` 与 `training_log.csv`，并和 A6 ETTh2 reference/control 对齐。新增的
diagnostic columns 包括：

- `last_train_loss`
- `last_weighted_basis_operator_smoothness_loss`
- `last_weighted_smoothness_to_train_loss`

这些列用于判断 smoothness regularizer 是否真的进入优化，而不是只看 flag 是否开启。

`scripts/remote/run_phase5_timealign_hss_a6s2_stability_calibration_gate.sh` 复用 A6S 的模型实现，只改变
diagnostic strength：

- `ema_decay=0.995/0.999`
- `basis_operator_smoothness_weight=10.0/100.0`

该 wrapper 仍强制 `CHECKPOINT_POLICY=official-last`，默认只跑 ETTh2，输出到
`/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_a6s2_stability_calibration_gate`。

## A6S Minimal Gate Interpretation

[Fact] A6S minimal gate 的最佳 variant 仍相对 ETTh2 best stage control 差 `+2.00%`，wins `0/4`。

[Fact] `smooth1e-3` 的 `weighted_smoothness / train_loss` 最大只有 `4.86e-07`，几乎不改变优化。

[Decision] 该结果不支持把 EMA 或当前 smoothness setting 作为 paper-core，但也不能把
operator-level stability 机制完全判死。下一步 A6S2 只做 strength calibration diagnostic。

## A6S2 Calibration Interpretation

[Fact] A6S2 复用同一个 analyzer 与 remote wrapper family。`lbf_r256_ema0999` 相对 A6-LBF-r256
平均 MSE 改善 `-1.46%`，相对 ETTh2 best stage control 仍差 `+0.67%`，wins `1/4`。

[Fact] `smooth10/smooth100` 没有改善，且 stronger smoothness 使 validation drift 变大。该结果支持
暂停简单 temporal smoothness regularization route。

[Decision] `EMA-0.999` 是 control signal，不是 model contribution。若继续，应把它转化为
training-time self-teacher / consistency mechanism，让 raw final checkpoint 学到 trajectory-averaged
prediction behavior。

## A6ST Self-Teacher Training Flow

`--self-teacher-loss-weight > 0` 时，训练开始后复制当前 `model` 得到 `self_teacher_model`：

```text
self_teacher_model = deepcopy(model)
self_teacher_model.eval()
```

每个 training batch 内：

```text
student_output = model(..., target_prefix=H)
teacher_output = self_teacher_model(..., target_prefix=H).detach()
self_teacher_loss = L1(student_output[:, :H], teacher_output[:, :H])
loss += self_teacher_loss_weight * self_teacher_loss
```

`optimizer.step()` 后更新 teacher：

```text
teacher = decay * teacher + (1 - decay) * student
```

[Fact] A6ST 最终保存和评估的是 raw student weights；它不设置 `--ema-eval`，也不把 EMA teacher
checkpoint 用作 test-time model。

[Proxy] 该机制仍有 generic mean-teacher/KD 风险。它只有在 raw official-last checkpoint 接近
A6S2 `ema0999` control，且保持 A6-LBF prefix-native operator 贡献边界时，才有继续设计价值。

## A7DG Disagreement-Gated Self-Teacher

A6ST cross-dataset sanity 显示 uniform self-teacher consistency 对 ETTh2 有益，但会伤害
ETTm1/Weather。A7DG 在同一 self-teacher path 上新增一个默认关闭的 detached gate：

```text
self_teacher_loss: scalar
pred_loss: scalar
signal = self_teacher_loss                  # absolute mode
signal = self_teacher_loss / pred_loss      # ratio mode
gate = sigmoid((signal - threshold) / temperature)
weighted_self_teacher_loss = gate * self_teacher_loss
loss += self_teacher_loss_weight * weighted_self_teacher_loss
```

[Fact] `gate` 来自 detached scalar signal，不把梯度传回 threshold path。它只调节 self-teacher
loss 的有效强度，不改变 A6-LBF 的 forward output graph。

[Fact] `self_teacher_gate_mode=none` 时 `gate=1`，因此旧 A6ST 行为保持不变。

新增训练日志字段：

- `self_teacher_gate_mode`
- `self_teacher_gate_threshold`
- `self_teacher_gate_temperature`
- `train_self_teacher_gate`
- `train_weighted_self_teacher_l1`

[Theory] 如果 ETTh2 的主要问题是 raw student 与 slow EMA trajectory 分离，而 ETTm1/Weather
已经处于低 drift 区间，则 disagreement gate 应在 ETTh2 上保持较高 self-teacher force，在
ETTm1/Weather 上自动降权。

[Proxy] 该 gate 仍可能变成 threshold tuning。只有当 `train_self_teacher_gate` 与跨数据集 metric
同时证明“高 drift 激活、低 drift 退火”时，它才有继续作为 method candidate 的价值。

[Falsification] 若 A7DG 保不住 ETTh2 gain，或 ETTm1/Weather 仍系统性负向，则 self-teacher
stability route 应停止，不能继续堆 threshold/schedule。

## A8TAG Teacher-Advantage Gate

A8TAG 进一步把 A7DG 的 disagreement threshold 改成 supervised teacher advantage。训练时同时计算：

```text
pred_loss = L1(student_output, target_y)
self_teacher_target_loss = L1(teacher_output, target_y)
self_teacher_loss = L1(student_output, teacher_output)
```

Binary gate：

```text
gate = 1 if self_teacher_target_loss < pred_loss else 0
```

Relative advantage gate：

```text
gate = clamp((pred_loss - self_teacher_target_loss) / pred_loss, 0, 1)
```

最终仍然只改变 self-teacher consistency 的有效强度：

```text
weighted_self_teacher_loss = gate * self_teacher_loss
loss += self_teacher_loss_weight * weighted_self_teacher_loss
```

[Fact] Gate 使用 detached `pred_loss` 与 `self_teacher_target_loss`，因此它是 supervised risk
comparison，而不是一个可被主模型直接优化的可微捷径。

[Fact] A8TAG 不使用 `ema_eval`，最终评估仍是 raw `official-last` student weights。

新增训练日志字段：

- `train_self_teacher_target_l1`
- `train_self_teacher_advantage_l1`

[Theory] 如果 EMA teacher 只是 trajectory-smoother，它只有在当前 prefix 上更接近 label 时才值得
被 student imitation。该 gate 比 A7DG 的 absolute/ratio threshold 更有机制边界。

[Falsification] 若 `train_self_teacher_advantage_l1` 多数为负或接近零，说明 EMA teacher 并不是
可靠 target；若 teacher advantage 存在但 metrics 仍不改善，则 self-teacher route 需要停止，
回到新的 capacity-preserving unified head 设计。
