# Baselines

Baseline 工作先记录 upstream evidence，再决定是否写本地 wrapper。

## Phase 0 Candidates

Phase 0 当前包含三个独立候选：

- `dlinear/`
  - sanity floor
  - 参考 DLinear official implementation
- `patch_encoder_fixed_head/`
  - clean PatchTST-style patch encoder + fixed horizon head
  - 参考 PatchTST official implementation
- `segtsft_dense_fixed_head/`
  - Seg-MoE-inspired dense TSFT backbone + fixed horizon head
  - 移除 MoE、router loss 和 autoregressive forecast
  - 参考 Seg-MoE official implementation

每个候选目录都有自己的 `dataset.py`、`model.py`、`train.py`，避免后续复刻时由共享
训练代码引入隐性差异。

## Phase 1 Candidates

Phase 1 当前新增一个 decoder-side candidate：

- `patch_encoder_segment_query_head/`
  - 保留 Phase0 patch encoder
  - 用 future segment queries + cross-attention 替换 fixed flatten head
  - one-to-one horizon training，用于 `Future-Segment Decoder Gate`
  - 不包含 future teacher branch 或 MoE

当前重点 comparison baseline：

- SRSNet

旧 `R_2026_FSA` 中的 SRSNet 性能数据不得默认复制到本仓库。需要使用时，先确认
具体来源、用途和落点。
