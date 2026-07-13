# Phase0 Prefix Consistency Analysis

## 目的

Phase0 roadmap 要验证固定 horizon head 是否在 variable-horizon 或 prefix consistency 上暴露
可量化问题。`scripts/analyze_phase0_prefix_consistency.py` 用已有
`predictions_test.npz` 完成这个分析，不重新训练模型。

## 对齐方式

对每个 dataset，读取同一 model、同一 seed 下的 horizon-specific checkpoints：

```text
h96, h192, h336, h720
```

由于 test dataset 按时间顺序生成 sliding windows，`h720` 的可用窗口数最少，且对应短
horizon test windows 的前缀。因此，对 prefix horizon $H$，脚本使用：

```text
fixed(H).pred[:N720, :H, :]
max(720).pred[:N720, :H, :]
```

其中 `N720` 是 `h720` test windows 数量。

## 指标

`fixed_mse` / `fixed_mae`：

```text
horizon-specific fixed head 的 aligned prefix test error
```

`max_prefix_mse` / `max_prefix_mae`：

```text
h720 fixed head 在同一 prefix 上的 test error
```

`relative_mse_change`：

```text
(max_prefix_mse - fixed_mse) / fixed_mse
```

`fixed_vs_max_pred_mse`：

```text
horizon-specific prediction 与 h720 prefix prediction 之间的预测差异
```

`truth_alignment_mse`：

```text
短 horizon ground truth 与 h720 prefix ground truth 的对齐误差
```

该值应接近 0；否则说明 sliding-window 对齐假设错误。

## 解释边界

[Fact] 这个分析仍然是 Phase0 fixed-head diagnostic，不是 Phase1 strict same-checkpoint
variable-horizon inference。

[Strong Evidence] 如果 `h720` prefix 在短 horizon 上显著劣于 horizon-specific fixed head，
说明 fixed direct head 学到的是 horizon-specific projection，不能自然支持 variable-horizon
request。

[Strong Evidence] 如果 `fixed_vs_max_pred_mse` 明显非零，说明不同 fixed horizon heads 即使在同一
input window 和同一 prefix target 上，也会产生不同预测。这支持 Phase1 需要显式
variable-horizon decoder 或 prefix-consistent decoder。
