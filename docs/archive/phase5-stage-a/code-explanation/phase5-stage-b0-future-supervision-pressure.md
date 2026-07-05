# Phase5-B0 Future Supervision Pressure Code Explanation

本文档说明 B0 diagnostic 对 `baselines/timealign_official/train_repo.py` 的最小改动。

## Purpose

B0 只审计 TimeAlign future branch pressure，不实现 routing method。新增代码用于控制和记录：

- reconstruction pressure: `w_recon * recon_loss`;
- alignment pressure: `w_align * alignment_loss`。

## Effective Config Flow

`build_official_args` 中：

```text
official_args.w_recon = args.w_recon
official_args.w_align = preset.w_align if args.w_align_override is None else args.w_align_override
```

因此默认行为不变；只有显式传入 `--w-align-override` 时才覆盖 dataset preset。

## Training Loss Flow

训练 loss 仍是：

```text
loss = pred_loss
     + teacher/self-teacher terms
     + official_args.w_recon * recon_loss
     + official_args.w_align * alignment_loss
     + optional regularizers
```

B0 不改变 gradient path，只增加诊断可控性。

## New Log Columns

`training_log.csv` 新增：

- `train_weighted_reconstruction_l1 = official_args.w_recon * train_reconstruction_l1`;
- `train_weighted_alignment_loss = official_args.w_align * train_alignment_loss`。

这些列用于判断 future branch pressure 是否真的进入优化，以及不同 ablation 下 pressure 是否被关闭。

## Code-Theory Consistency

[Intended theory] 如果 future supervision pressure 是 useful/harmful 的关键变量，那么关闭或降低
`w_recon/w_align` 应改变 prediction behavior、validation drift 或 cross-dataset gap。

[Code realization] 当前只允许控制 scalar loss weights，不改变 future branch architecture，也不改变
prediction head。它是 diagnostic control，不是 method。

[Proxy boundary] 仅看 scalar loss ablation 不能证明 reliability-aware routing 成立；若 B0 有正向结果，
还需要后续 Step 4/5 method narrative gate 区分 loss-only weighting 与 gradient-path routing。

