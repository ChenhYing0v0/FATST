# SegTSFTDenseFixedHead Phase 0 Baseline

This folder contains an independent dense TSFT-style baseline inspired by
Seg-MoE, but with the mechanisms that overlap later research stages removed.

Source reference:

- Official Seg-MoE repository: `https://github.com/evortigosa/segmoe_forecast`

Removed for Phase 0:

- segment-wise MoE
- token-wise MoE fallback
- router probabilities and imbalance/load-balance loss
- autoregressive forecasting loop

Kept:

- channel-independent convolutional patch embedding
- online instance normalization
- RMSNorm
- RoPE
- GQA-style attention
- dense FFN
- direct fixed horizon head
