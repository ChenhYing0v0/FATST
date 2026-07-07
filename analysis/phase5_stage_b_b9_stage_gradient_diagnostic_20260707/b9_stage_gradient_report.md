# Phase5 StageB B9-SGC Stage Gradient Diagnostic

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B9-FSN` |
| `diagnostic_id` | `B9-SGC` |
| `current_step` | Step 2/3：native future-stage-aware problem diagnostic |
| `problem` | A6-LBF 用一个 `coeff[b,c]` 服务所有 future stages，可能造成不同 stage 的训练信号冲突 |
| `residual_policy` | 不拟合 residual，不设计 residual correction；只分析 primary prediction path 的 stage gradients |

## 诊断定义

对 clean A6 checkpoint，在 train split 上取若干 batch，手动执行 A6 forward 到：

```text
coeff = learned_basis_coeff(hidden)  # [B, C, 256]
prediction = learned_temporal_basis @ coeff
```

然后分别计算四个 non-overlap future stages 的 MSE loss，并求每个 stage loss 对同一个 `coeff` 的梯度：

```text
stages = [0,96), [96,192), [192,336), [336,720)
g_s = d loss_s / d coeff
```

若不同 `g_s` 的 cosine 较低或为负，说明一个共享 coefficient 同时服务所有 future stages 存在 native stage pressure；这支持 B9-FSN 的问题存在。

## Summary

| dataset | batches | mean_pairwise_cosine | min_pairwise_cosine_mean | negative_pair_rate_mean | early_tail_cosine_mean | max_min_grad_norm_ratio_mean |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 8 | 0.072295 | 0.008138 | 0.083333 | 0.041227 | 1.987591 |
| ETTm1 | 8 | 0.170591 | 0.069761 | 0.041667 | 0.111646 | 1.417532 |
| Weather | 8 | 0.047600 | 0.009524 | 0.083333 | 0.013533 | 1.416012 |

## Decision

[Fact] 本诊断没有拟合 residual，也没有设计 correction module；它只看训练信号中不同 future stages 对同一个 A6 coefficient 的梯度方向是否一致。

- ETTh2: mean pairwise cosine = `0.072`, early-tail cosine = `0.041`, negative pair rate = `0.083`.
- ETTm1: mean pairwise cosine = `0.171`, early-tail cosine = `0.112`, negative pair rate = `0.042`.
- Weather: mean pairwise cosine = `0.048`, early-tail cosine = `0.014`, negative pair rate = `0.083`.

[Decision] `B9-SGC` 暂定通过 problem-candidate gate：至少两个 dataset 显示不同 future stages 对共享 coefficient 的梯度方向不够一致，支持 native future-stage-aware representation/operator 的问题存在。

[Next] 进入 Step 4-6 前仍需设计 narrative gate：方法必须是 primary prediction path，不得是 residual correction；并需要定义 capacity-preserving initialization。
