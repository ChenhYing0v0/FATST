# Phase5 StageB B8-OCD Coefficient Oracle Analyzer

本文档解释 `scripts/analyze_phase5_stage_b_b8_ocd_coefficient_oracle.py` 的诊断逻辑、输入 artifacts、输出列含义和 code-theory consistency。

## 诊断目的

`B8-OCD` 检验 `B8-FQA` 的核心前提：

> A6-LBF-r256 的 `coeff[b,c]` 对所有 future positions 共享；如果允许同一 learned temporal basis 下的 future-segment-specific coefficient correction，是否能显著降低 A6 residual？

该脚本只做 oracle diagnostic，不是方法实现，也不训练新模型。

## 输入 Artifacts

默认输入根目录：

```text
analysis/phase5_stage_b_b8_ocd_coefficient_oracle_20260707/raw/
  TimeAlignOfficialUnified720_A6LBF_r256_main_official-last/
```

每个 dataset 需要：

```text
{dataset}/mixed_h96_h192_h336_h720/seed2021/checkpoint.pt
{dataset}/mixed_h96_h192_h336_h720/seed2021/predictions_test.npz
```

其中：

- `checkpoint.pt` 提供 `learned_temporal_basis: [720, 256]`；
- `predictions_test.npz` 提供 `pred` 和 `true`，shape 为 `[N, 720, C]`。

## 主要 Tensor Flow

对每个 dataset：

```text
pred, true: [N, 720, C]
residual = true - pred             # [N, 720, C]
residual_rows = moveaxis(...).reshape(N*C, 720)
```

例如 ETTh2：

```text
pred/true: [2161, 720, 7]
residual_rows: [15127, 720]
```

然后从 checkpoint 读取 A6 basis：

```text
learned_temporal_basis: [720, 256]
left_space = svd(learned_temporal_basis).U  # [720, 256]
```

脚本使用 left singular vectors 作为 learned basis span 的有序正交表示，避免 raw basis column scale 影响 projection oracle。

## Oracle 定义

脚本比较两种 correction：

1. `global_oracle`：在整段 720 steps 上使用一组 correction coefficients。
2. `segment_oracle`：在四个 segments 上分别使用 correction coefficients：

```text
[0,96), [96,192), [192,336), [336,720)
```

为了避免 256 维 coefficient 在短 segment 上平凡拟合，脚本只报告：

```text
rank = 8, 16, 32, 64
```

并加入 `dct` control。DCT control 使用相同 rank 与相同 global/segment oracle 流程，用于判断结果是否只是 generic low-frequency residual structure。

## CSV 输出

### `b8_ocd_summary.csv`

每行是一个 dataset/basis/rank 的整体结果：

- `dataset`：数据集。
- `basis`：`learned_a6` 或 `dct`。
- `rank`：projection rank。
- `samples`：test samples 数。
- `channels`：变量数。
- `residual_rows`：`samples * channels`。
- `base_mse`：A6 原始 residual MSE。
- `global_oracle_mse`：global correction 后 residual MSE。
- `segment_oracle_mse`：segment-specific correction 后 residual MSE。
- `global_reduction_pct`：global correction 消除的 residual energy 比例。
- `segment_reduction_pct`：segment correction 消除的 residual energy 比例。
- `segment_minus_global_reduction_pct`：segment correction 相对 global correction 的额外 residual energy reduction。

### `b8_ocd_segment_detail.csv`

每行是一个 dataset/basis/rank/segment 的局部结果：

- `segment_start`, `segment_end`：segment 区间。
- `effective_rank`：该 segment 内 QR 后保留的有效 rank。
- `base_mse`：该 segment 原始 residual MSE。
- `oracle_mse`：该 segment oracle correction 后 residual MSE。
- `relative_mse_reduction_pct`：该 segment residual energy reduction。

## Report 输出

`b8_ocd_report.md` 汇总 rank 32、rank 64 与 learned basis rank 64 segment detail，并给出 gate decision。

当前 gate 规则：

- 只看 `segment_oracle` 是否优于 `global_oracle` 不够，因为 segment oracle 是更宽松的拟合问题；
- 必须进一步比较 DCT control；
- 若 learned basis 的 segment-specific headroom 不能在绝对 residual reduction 上超过 DCT control，则不能证明 B8-FQA 的 learned-basis coefficient interface 是强问题。

## Code-Theory Consistency

理论意图：

```text
测试 future-position-invariant coefficient 是否是 A6 residual 的关键瓶颈。
```

代码实现：

```text
固定 clean A6 prediction；
固定 checkpoint 中的 learned_temporal_basis span；
只在 residual space 中比较 global vs segment-specific coefficient correction。
```

仍只是 proxy 的部分：

- `predictions_test.npz` 是 denorm-space output；脚本没有重放 RevIN normalization，因此 oracle 是 denorm-space residual correction。
- oracle coefficients 来自 true residual，不是可学习 query module 的真实输出。
- DCT control 只排除 generic low-frequency confounder，不能证明其他 architecture route 无效。

可证伪 B8-FQA 的 evidence：

- segment-specific learned correction 没有显著优于 global correction；
- 或 segment-specific learned correction 的优势被 DCT control 更好解释。

本次结果属于第二种，因此不应直接实现 B8-FQA。
