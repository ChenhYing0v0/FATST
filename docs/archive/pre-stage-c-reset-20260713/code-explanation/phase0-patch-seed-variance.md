# Phase0 PatchEncoderFixedHead Seed Variance

## 目的

Phase0 已选择 `PatchEncoderFixedHead` 作为 internal base。进入 Phase1
Variable-Horizon Decoder 前，需要确认该 base 在核心设置上没有明显 seed instability。

## Runner

远程入口：

```text
scripts/remote/run_phase0_patch_seed_variance.sh
```

默认输出：

```text
/home/yingch/exp_outputs/r-2026-fatst/phase0_patch_seed_variance
```

## Matrix

```text
Model: PatchEncoderFixedHead
Datasets: ETTh2, ETTm1, Weather
Horizons: 96, 720
Seeds: 2021, 2022, 2023
```

总计 18 runs。

## 数据流影响

[Fact] 该 runner 不改变 `PatchEncoderFixedHead` 的模型结构、训练目标、dataset split 或
artifact contract。

[Fact] 它只改变 `seed`，并把输出组织到 repo 外部实验目录。

## 判据

[Strong Evidence] 如果三个 seed 的 MSE/MAE 标准差较小，且均值仍维持 Phase0 gate 中
`PatchEncoderFixedHead` 的优势，则 Phase1 可以基于该 internal base 开始 decoder 机制实验。

[Speculative] 如果某个 dataset/horizon 的 seed variance 很大，则 Phase1 的所有机制实验必须
在该 setting 上至少使用 3 seeds，否则单 seed 改善不能作为机制证据。
