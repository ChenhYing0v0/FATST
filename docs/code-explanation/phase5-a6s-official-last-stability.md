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
