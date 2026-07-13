# Phase5 StageB B13-FUCO-A Future-Unit Granularity Diagnostic

对应文件：

- `scripts/analyze_phase5_stage_b_b13_future_unit_granularity.py`

## Purpose

该 analyzer 执行 B13-FUCO 的 Step 2/3 Diagnostic A：

> B9 在 canonical horizon stages 上观察到的 shared-coefficient gradient pressure，是否能在较大的、
> benchmark-independent future unit sizes 上稳定复现？

本脚本不训练模型、不拟合 residual、不实现 future-unit generator。它只能决定是否进入下一步
parameter-matched Diagnostic B。

## Inputs

默认 A6 checkpoint root：

```text
analysis/phase5_stage_b_b8_ocd_coefficient_oracle_20260707/raw/
  TimeAlignOfficialUnified720_A6LBF_r256_main_official-last/
```

每个 dataset 读取：

```text
{dataset}/mixed_h96_h192_h336_h720/seed2021/checkpoint.pt
```

默认 datasets：

- `ETTh2`；
- `ETTm1`；
- `Weather`。

数据来自本地 train split，默认 dataset root：

```text
/Users/river/PaperResearch/Project/datasets
```

## Unit Sizes

```text
main:          120, 144, 180, 240
coarse control: 360
```

所有 sizes 均整除 `pred_len=720`。`360` 只有两个 units，因此只提供 first/last coarse pressure，
不参与 `3/4 main sizes` gate。

## Tensor Flow

脚本以 `model.eval()` 执行 deterministic checkpoint-local forward，但保留 autograd：

```text
batch_x: [B,720,C]
  -> Normalize + PatchEmbed + encoder
hidden: [B,C,R]
  -> learned_basis_coeff
coeff: [B,C,256]
  -> learned_temporal_basis @ coeff
prediction: [B,720,C]
```

对 unit size `U`：

```text
unit_m = [m*U, (m+1)*U)
loss_m = MSE(prediction[unit_m], target[unit_m])
grad_m = d loss_m / d coeff
```

同一个 batch forward 被所有 unit sizes 复用；脚本只重复构造不同 unit losses 和 coefficient gradients。

## Gradient Statistics

### Pairwise cosine

```text
gradient_cosine(i,j) = cosine(grad_i, grad_j)
```

pair type：

- `adjacent`: unit distance `1`；
- `far`: distance `>= ceil(unit_count/2)`；
- `middle`: 其余 pairs。

batch-level columns：

- `mean_pairwise_cosine`：所有 unit pairs 的均值；
- `min_pairwise_cosine`：最低 pair cosine；
- `adjacent_cosine` / `far_cosine`；
- `adjacent_minus_far_cosine`；
- `first_last_cosine`；
- `negative_pair_rate`；
- `max_min_grad_norm_ratio`。

### Shared alignment efficiency

定义：

$$
\eta_{\mathrm{shared}}
=
\frac{\left\|\sum_m g_m\right\|_2^2}
{M\sum_m\left\|g_m\right\|_2^2}.
$$

输出列为 `shared_alignment_efficiency`：

- identical equal-norm gradients 时为 `1`；
- orthogonal equal-norm gradients 时约为 `1/M`；
- 它描述共享 coefficient 的方向复用程度，不等价于可实现模型的 gain。

## Basis Geometry Control

对每个 A6 basis unit：

```text
basis_unit = learned_temporal_basis[m*U:(m+1)*U]  # [U,256]
```

通过 SVD 取得 rank-32 row subspace `Q_m: [256,32]`，pair overlap 定义为：

$$
\operatorname{overlap}(i,j)
=
\frac{\left\|Q_i^TQ_j\right\|_F^2}{r}.
$$

输出：

- `basis_adjacent_overlap`；
- `basis_far_overlap`；
- `basis_distance_overlap_spearman`；
- `gradient_basis_pair_spearman`：同一 unit pairs 的 mean gradient cosine 与 basis overlap 的 Spearman。

若最后一项很高，gradient relation 可能主要继承 A6 basis geometry；Diagnostic A 不能将它直接解释为
compositional generator 的必要性。

## Bootstrap

对每个 dataset/unit size 的 batch rows 做 deterministic bootstrap，默认 `1000` iterations：

- `mean_pairwise_cosine`；
- `first_last_cosine`；
- `adjacent_minus_far_cosine`；
- `shared_alignment_efficiency`。

每项输出 observed mean 与 `p05/p50/p95`。

## Output Artifacts

默认目录：

```text
analysis/phase5_stage_b_b13_future_unit_granularity_20260710/
```

| File | Meaning |
| --- | --- |
| `b13_future_unit_gradient_batches.csv` | 每个 dataset/unit size/batch 的 gradient statistics |
| `b13_future_unit_gradient_pairs.csv` | 每个 batch 的 unit-pair gradient cosine |
| `b13_future_unit_basis_pairs.csv` | A6 basis unit-pair subspace geometry |
| `b13_future_unit_summary.csv` | dataset/unit-size aggregate 与 gate columns |
| `b13_future_unit_bootstrap.csv` | bootstrap intervals |
| `b13_future_unit_granularity_report.md` | reader-facing Step 2/3 decision report |

## Gate Logic

单个 main dataset/unit size 为 `robust_support`，当：

```text
bootstrap p95(mean_pairwise_cosine) < 0.50
and
bootstrap p95(first_last_cosine) < 0.35
```

若至少两个 datasets 各有至少 `3/4` main sizes 通过：

```text
partial_pass_large_unit_granularity_robust
```

该状态只允许进入 Diagnostic B；它不是 `narrative_ready`，也不能触发 model implementation。

## Code-Theory Consistency

Intended theory：

- future-unit problem 不应依赖 canonical horizon boundaries；
- 较大的 units 应承载足够 future information，避免 B12 small-unit capacity confound；
- A6 shared coefficient 若持续收到不同方向的 large-unit gradients，则存在需要进一步诊断的 shared-state pressure。

Code realization：

- 使用 `120/144/180/240` main unit sizes 与 `360` coarse control；
- 直接求 unit losses 对 A6 `coeff` 的 gradients；
- 用 bootstrap 检查 batch stability；
- 用 A6 basis subspace overlap 标记 geometry confound。

Proxy limitations：

- 仍是 single checkpoint / seed2021；
- checkpoint-local gradient 不等于完整训练动态；
- gradient conflict 可能来自 basis geometry；
- 没有测试 unit states 是否应 independent、no-transition 或 prefix-causal composition；
- 没有性能证据。

Falsification boundary：

- large-unit gate 普遍失败时，可拒绝“B9 canonical pressure 能跨 granularity 支撑 B13”的限定命题；
- 不能据此拒绝所有 future-unit architecture；
- gate 通过时也只能进入 Diagnostic B，不能直接实现或远程训练 B13。
