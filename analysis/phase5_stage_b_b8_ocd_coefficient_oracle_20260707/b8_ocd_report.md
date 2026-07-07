# Phase5 StageB B8-OCD Coefficient-Space Oracle Diagnostic

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B8-FQA` |
| `diagnostic_id` | `B8-OCD` |
| `current_step` | Step 3：problem-existence diagnostic |
| `problem` | A6-LBF 的 `coeff[b,c]` 对 future positions 不变，可能限制不同 future segments 的 sample-specific 表示 |
| `idea_under_test` | 固定 clean A6 prediction 与 learned temporal basis，测试 segment-specific coefficient correction 是否能显著降低 residual |
| `decision` | 见文末，当前仅为 oracle diagnostic，不是 method result |

## 诊断定义

本诊断使用 clean A6-LBF-r256 的 `predictions_test.npz` 和 `checkpoint.pt`。对每个 dataset，先计算 denorm-space residual：

```text
residual[b, t, c] = true[b, t, c] - pred[b, t, c]
```

然后将 residual reshape 为 `[sample * channel, 720]`。诊断比较两种 oracle correction：

- `global_oracle`：整段 720 steps 共用一组 correction coefficients；
- `segment_oracle`：四个 segments `[0,96), [96,192), [192,336), [336,720)` 分别求 correction coefficients。

为了避免 256 维 coefficient 在短 segment 上平凡拟合，本报告只使用 ranks `{8,16,32,64}`，并加入 `dct` control。

## Rank 32 Summary

| dataset | basis | rank | base_mse | global_reduction_pct | segment_reduction_pct | segment_minus_global_reduction_pct |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | learned_a6 | 32 | 0.395028 | 56.81% | 71.37% | 14.56% |
| ETTh2 | dct | 32 | 0.395028 | 80.44% | 84.49% | 4.06% |
| ETTm1 | learned_a6 | 32 | 0.408370 | 40.12% | 64.41% | 24.29% |
| ETTm1 | dct | 32 | 0.408370 | 83.94% | 89.50% | 5.56% |
| Weather | learned_a6 | 32 | 0.303046 | 30.76% | 44.54% | 13.78% |
| Weather | dct | 32 | 0.303046 | 86.10% | 89.50% | 3.40% |

## Rank 64 Summary

| dataset | basis | rank | base_mse | global_reduction_pct | segment_reduction_pct | segment_minus_global_reduction_pct |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | learned_a6 | 64 | 0.395028 | 62.20% | 79.05% | 16.85% |
| ETTh2 | dct | 64 | 0.395028 | 86.24% | 87.61% | 1.37% |
| ETTm1 | learned_a6 | 64 | 0.408370 | 44.58% | 72.77% | 28.19% |
| ETTm1 | dct | 64 | 0.408370 | 90.98% | 91.85% | 0.87% |
| Weather | learned_a6 | 64 | 0.303046 | 39.89% | 61.91% | 22.01% |
| Weather | dct | 64 | 0.303046 | 90.08% | 91.18% | 1.10% |

## Learned Basis Rank 64 Segment Detail

| dataset | basis | rank | segment_start | segment_end | effective_rank | base_mse | oracle_mse | relative_mse_reduction_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | learned_a6 | 64 | 0 | 96 | 64 | 0.244162 | 0.054333 | 77.75% |
| ETTh2 | learned_a6 | 64 | 96 | 192 | 64 | 0.328225 | 0.029589 | 90.99% |
| ETTh2 | learned_a6 | 64 | 192 | 336 | 64 | 0.351929 | 0.052328 | 85.13% |
| ETTh2 | learned_a6 | 64 | 336 | 720 | 64 | 0.465608 | 0.114565 | 75.39% |
| ETTm1 | learned_a6 | 64 | 0 | 96 | 64 | 0.272766 | 0.021727 | 92.03% |
| ETTm1 | learned_a6 | 64 | 96 | 192 | 64 | 0.345527 | 0.019495 | 94.36% |
| ETTm1 | learned_a6 | 64 | 192 | 336 | 64 | 0.397309 | 0.065375 | 83.55% |
| ETTm1 | learned_a6 | 64 | 336 | 720 | 64 | 0.462129 | 0.173697 | 62.41% |
| Weather | learned_a6 | 64 | 0 | 96 | 64 | 0.141368 | 0.083206 | 41.14% |
| Weather | learned_a6 | 64 | 96 | 192 | 64 | 0.223432 | 0.105177 | 52.93% |
| Weather | learned_a6 | 64 | 192 | 336 | 64 | 0.297317 | 0.126087 | 57.59% |
| Weather | learned_a6 | 64 | 336 | 720 | 64 | 0.365517 | 0.122064 | 66.61% |

## 初步判定

[Fact] `segment_oracle` 在所有 dataset 上都应优于或等于 `global_oracle`，因为它放宽了 coefficient sharing；关键不是是否有提升，而是提升是否足够大、是否超过 DCT control。

- ETTh2: learned segment-minus-global gain = `16.85%`, DCT control = `1.37%`; learned segment reduction = `79.05%`, DCT segment reduction = `87.61%`.
- ETTm1: learned segment-minus-global gain = `28.19%`, DCT control = `0.87%`; learned segment reduction = `72.77%`, DCT segment reduction = `91.85%`.
- Weather: learned segment-minus-global gain = `22.01%`, DCT control = `1.10%`; learned segment reduction = `61.91%`, DCT segment reduction = `91.18%`.

[Decision] `B8-OCD` 未通过 problem-existence gate：当前 evidence 不足以说明 B8-FQA 的 learned-basis coefficient interface 有强于 generic DCT/low-rank control 的 segment-specific headroom。

[Rollback] 不实现 B8-FQA。StageB 应回到 Step 2/3，重新寻找 architecture-level problem，或将 B7-UPO 仅作为 small contribution candidate 保留。
