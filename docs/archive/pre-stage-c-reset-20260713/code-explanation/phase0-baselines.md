# Phase 0 Baselines Code Explanation

## Scope

[Fact] Phase 0 now contains three independent baseline folders:

- `baselines/dlinear`
- `baselines/patch_encoder_fixed_head`
- `baselines/segtsft_dense_fixed_head`

Each folder contains its own `dataset.py`, `model.py`, and `train.py`. This deliberately
duplicates simple training and data-loading logic so that future reproduction or debugging
can compare each candidate without hidden shared abstractions.

## Common Data Contract

All three baselines use:

$$
X \in \mathbb{R}^{B \times L \times C},
\quad
Y \in \mathbb{R}^{B \times H \times C}.
$$

For Phase 0:

- $L=336$
- $H \in \{96,192,336,720\}$
- datasets: ETTh2, ETTm1, Weather

The dataset loaders reproduce the usual LTSF chronological split style:

- ETTh2: ETT-hour fixed month boundaries.
- ETTm1: ETT-minute fixed month boundaries.
- Weather: chronological 0.7/0.1/0.2 split.

All loaders scale by fitting mean/std on the training split only.

## DLinear

Source reference: official DLinear `models/DLinear.py`.

Forward path:

1. Input:

   $$
   x \in \mathbb{R}^{B \times L \times C}
   $$

2. Moving average decomposition:

   $$
   x = x_{seasonal} + x_{trend}
   $$

3. Permute each component to channel-major:

   $$
   x_{seasonal}, x_{trend}
   \in \mathbb{R}^{B \times C \times L}
   $$

4. Apply linear projections along the time dimension:

   $$
   \hat{Y}_{seasonal}, \hat{Y}_{trend}
   \in \mathbb{R}^{B \times C \times H}
   $$

5. Add and return:

   $$
   \hat{Y}
   =
   (\hat{Y}_{seasonal}+\hat{Y}_{trend})^\top
   \in \mathbb{R}^{B \times H \times C}
   $$

Role:

- sanity floor
- detects broken data split, scaling, or metric logic

## PatchEncoderFixedHead

Source reference: official PatchTST supervised scripts and patching design.

Forward path:

1. Input:

   $$
   x \in \mathbb{R}^{B \times L \times C}
   $$

2. Optional RevIN normalization along time:

   $$
   x' = \frac{x-\mu(x)}{\sigma(x)}
   $$

3. Channel-independent reshape:

   $$
   x' \rightarrow \mathbb{R}^{(B C) \times 1 \times L}
   $$

4. End padding and overlapping patch extraction:

   $$
   P \in \mathbb{R}^{(B C) \times N \times P_l}
   $$

   where $P_l=16$ and stride is 8.

5. Linear patch embedding and positional embedding:

   $$
   E \in \mathbb{R}^{(B C) \times N \times d}
   $$

6. Transformer encoder:

   $$
   Z = Encoder(E)
   \in \mathbb{R}^{(B C) \times N \times d}
   $$

7. Flatten fixed head:

   $$
   \hat{Y}
   =
   Linear(Flatten(Z))
   \in \mathbb{R}^{B \times H \times C}
   $$

8. RevIN denormalization restores the sample scale.

Role:

- clean patch-based internal-base candidate
- intentionally uses a fixed direct head to expose variable-horizon limitations

## SegTSFTDenseFixedHead

Source reference: official Seg-MoE TSFT implementation.

Removed mechanisms:

- segment-wise MoE
- token-wise MoE fallback
- router probability outputs
- imbalance/load-balance loss
- autoregressive forecast loop

Kept dense mechanisms:

- channel-independent convolutional patch embedding
- online instance normalization
- RMSNorm
- RoPE
- GQA-style attention
- DropPath
- dense FFN
- direct fixed horizon head

Forward path:

1. Input:

   $$
   x \in \mathbb{R}^{B \times L \times C}
   $$

2. Convert to Seg-MoE TSFT convention:

   $$
   x^\top \in \mathbb{R}^{B \times C \times L}
   $$

3. Online instance normalization along time:

   $$
   \tilde{x}_{b,c,:}
   =
   \frac{x_{b,c,:}-\mu_{b,c}}{\sigma_{b,c}}
   $$

4. Channel-independent Conv1d patch embedding with `patch_width=8`:

   $$
   E \in \mathbb{R}^{(B C) \times N \times d}
   $$

5. Dense TSFT blocks:

   $$
   Z_{\ell+1}
   =
   Z_\ell
   +
   Attention(RMSNorm(Z_\ell))
   +
   FFN(RMSNorm(\cdot))
   $$

   Attention uses RoPE and grouped query/key/value heads. FFN is dense MLP, not MoE.

6. Final RMSNorm and fixed head:

   $$
   \hat{Y}
   =
   Linear(Flatten(Z))
   \in \mathbb{R}^{B \times H \times C}
   $$

7. Online instance denormalization restores the sample scale.

Role:

- modern dense baseline candidate
- tests whether PatchTST-style base is too old or weak
- provides a cleaner future bridge to MoE experiments without using MoE in Phase 0

## Artifact Contract

All `train.py` files write the same minimum artifacts:

- `checkpoint.pt`
- `predictions_test.npz`
- `metrics.json`
- `metrics_by_horizon.csv`
- `training_log.csv`
- `effective_config.json`
- `environment.json`

[Fact] These scripts are Phase 0 scaffolding. They do not yet implement cross-run summary
aggregation, prefix consistency between independently trained horizon checkpoints, or
head-step similarity export. Those should be added before paper-facing Phase 0 experiments.
