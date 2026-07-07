# Phase5 StageB B9-SGC Stage Gradient Diagnostic

本文档解释 `scripts/analyze_phase5_stage_b_b9_stage_gradient_diagnostic.py` 的诊断逻辑、输入 artifacts、输出列含义和 code-theory consistency。

## 诊断目的

`B9-SGC` 检验 native future-stage-aware architecture 的问题基础：

> A6-LBF-r256 使用同一个 `coeff[b,c]` 服务所有 future stages。若不同 stage losses 对该 coefficient 的梯度方向不一致，则说明 primary prediction path 中存在 stage-native pressure。

该诊断不拟合 residual，不设计 residual correction，也不训练新模型。

## 输入 Artifacts

默认 checkpoint root：

```text
analysis/phase5_stage_b_b8_ocd_coefficient_oracle_20260707/raw/
  TimeAlignOfficialUnified720_A6LBF_r256_main_official-last/
```

每个 dataset 读取：

```text
{dataset}/mixed_h96_h192_h336_h720/seed2021/checkpoint.pt
```

本地 dataset root 默认：

```text
/Users/river/PaperResearch/Project/datasets
```

脚本通过 `baselines/timealign_official/train_repo.py` 的 `OFFICIAL_PRESETS` 和 `build_official_args` 复用 clean A6 的 dataset/model 配置。

## 主要 Tensor Flow

脚本手动执行 clean A6 forward path，使 `coeff` 可以作为 autograd target：

```text
batch_x: [B, 720, C]
x_norm = Normalize(batch_x)
x_tokens = PatchEmbed(x_norm)                # [B, C*patch_num, d_model]
x_tokens = encoder(x_tokens)
hidden = reshape/flatten(x_tokens)           # [B, C, R]
coeff = learned_basis_coeff(hidden)          # [B, C, 256]
prediction = learned_temporal_basis @ coeff  # [B, 720, C]
prediction = Normalize.denorm(prediction)
```

然后取 train target：

```text
target = batch_y[:, -720:, :]                # [B, 720, C]
```

## Stage Gradient

脚本使用四个 non-overlap future stages：

```text
early_0_96
mid_96_192
late_192_336
tail_336_720
```

对每个 stage：

```text
loss_s = MSE(prediction[:, start:end, :], target[:, start:end, :])
g_s = d loss_s / d coeff
```

然后计算所有 stage-pair 的 cosine：

```text
cos(g_i, g_j)
```

若 cosine 接近 1，说明不同 stage 对共享 coefficient 的训练方向一致；若 cosine 接近 0 或为负，说明不同 stage 对同一个 coefficient 的需求不同。

## CSV 输出

### `b9_stage_gradient_batches.csv`

每行是一个 dataset/batch：

- `loss_{stage}`：该 stage 的 MSE loss；
- `grad_norm_{stage}`：该 stage loss 对 `coeff` 的 gradient norm；
- `cos_{stage_i}_vs_{stage_j}`：两个 stage gradients 的 cosine；
- `mean_pairwise_cosine`：所有 stage-pair cosine 的均值；
- `min_pairwise_cosine`：该 batch 最低 stage-pair cosine；
- `negative_pair_rate`：cosine 小于 0 的 pair 比例；
- `early_tail_cosine`：early 与 tail 的 cosine；
- `max_min_grad_norm_ratio`：最大/最小 stage gradient norm 比值。

### `b9_stage_gradient_summary.csv`

按 dataset 聚合 batch rows：

- `mean_pairwise_cosine`；
- `min_pairwise_cosine_mean`；
- `negative_pair_rate_mean`；
- `early_tail_cosine_mean`；
- `max_min_grad_norm_ratio_mean`；
- 各 stage 的 mean loss 和 mean gradient norm。

## Report 输出

`b9_stage_gradient_report.md` 汇总诊断定义、dataset summary 和 gate decision。

当前 gate：

- 若至少两个 dataset 的 `mean_pairwise_cosine < 0.5` 或 `early_tail_cosine_mean < 0.35`，则通过 problem-candidate gate；
- 若所有 dataset 的 stage gradients 高度同向，则不支持 native stage-specific architecture。

## Code-Theory Consistency

理论意图：

```text
验证 single coefficient state 是否同时承受不同 future stages 的不同 primary prediction pressures。
```

代码实现：

```text
固定 clean A6 checkpoint；
在 train split 上计算四个 stage losses；
直接求 stage losses 对 A6 coefficient 的梯度方向。
```

这不是 residual route：

- 没有使用 `true - pred` 拟合 correction；
- 没有构造 `y = A6(x) + delta`；
- 没有把 oracle residual gains 当作 method evidence。

仍只是 proxy 的部分：

- 只取默认 `8` 个 train batches；
- 只分析 checkpoint-local gradients，不等于完整训练动态；
- 只证明 stage pressure 存在，不证明某个具体 B9 architecture 会提升效果。

下一步若进入 Step 4-6，必须设计 primary prediction path，并保留 A6 function-preserving fallback。
