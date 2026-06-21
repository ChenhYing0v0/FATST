# Phase 0 Experiment Protocol

## 目的

[Fact] Phase 0 的目标不是提出新模型，而是在三个候选基础之间做 gate：
`DLinear`、`PatchEncoderFixedHead`、`SegTSFTDenseFixedHead`。后续
variable-horizon decoder、future-aware mechanism 和 MoE-style conditional
architecture 都应先相对最终选中的 base 证明贡献。

[Inference] 本协议优先保证三件事：

1. 与常见 LTSF 论文 benchmark 设置对齐。
2. 保留足够强的 patch-based base，使后续有机会正面对比 SRSNet。
3. 记录 horizon-level diagnostics，避免只看平均 MSE/MAE。

## 数据集矩阵

### 数据根目录

- Local root: `/Users/river/PaperResearch/Project/datasets`
- Remote root: `/home/yingch/dataset`

### Canonical datasets

| Group | Dataset | Local relative path | Remote relative path | Channels | Rows | Status |
| --- | --- | --- | --- | ---: | ---: | --- |
| ETT | ETTh1 | `ETT-small/ETTh1.csv` | `ETT-small/ETTh1.csv` | 7 | 17420 | available_later |
| ETT | ETTh2 | `ETT-small/ETTh2.csv` | `ETT-small/ETTh2.csv` | 7 | 17420 | active_phase0_gate |
| ETT | ETTm1 | `ETT-small/ETTm1.csv` | `ETT-small/ETTm1.csv` | 7 | 69680 | active_phase0_gate |
| ETT | ETTm2 | `ETT-small/ETTm2.csv` | `ETT-small/ETTm2.csv` | 7 | 69680 | available_later |
| Weather | Weather | `weather/weather.csv` | `weather/weather.csv` | 21 | 52696 | active_phase0_gate |
| ECL | ECL | `electricity/electricity.csv` | `electricity/electricity.csv` | 321 | 26304 | available_later |
| Solar | Solar | `Solar/solar_AL.txt` | `Solar/solar_AL.txt` | 137 | 52560 | available_later |
| Traffic | Traffic | `traffic/traffic.csv` | `traffic/traffic.csv` | 862 | 17544 | tentative |

Notes:

- ETT rows/channels count excludes header.
- Weather/ECL/Traffic channels exclude `date`.
- Solar uses `solar_AL.txt` as canonical input because it is already a wide LTSF-style matrix.
  `Solar/Solar.csv` is long format and is not used in Phase 0 unless we later add an explicit
  preprocessing step.
- Phase 0 gate only runs ETTh2、ETTm1、Weather. Other rows remain documented as available
  datasets for later expansion.

## Horizon Matrix

[Fact] Phase 0 uses the standard long-term forecasting horizons:

$$
H \in \{96, 192, 336, 720\}.
$$

The default lookback length is:

$$
L = 336.
$$

This follows the common PatchTST/DLinear long forecasting scripts, which use `seq_len=336`
and `pred_len={96,192,336,720}` for the corresponding multivariate benchmarks.

## Data Split

### ETT

Use the standard ETT chronological split:

- ETTh1 / ETTh2:
  - train: first 12 months
  - validation: next 4 months
  - test: final 4 months
- ETTm1 / ETTm2:
  - train: first 12 months
  - validation: next 4 months
  - test: final 4 months

[Implementation note] The runner should encode these as the conventional boundary indices
rather than random ratios.

### Weather / ECL / Solar / Traffic

Use chronological ratio split:

$$
\text{train}:\text{val}:\text{test}=0.7:0.1:0.2.
$$

No random shuffling is allowed across time.

## Models In Scope

### Candidate gate

Phase 0 compares three candidates:

1. `DLinear`: sanity floor and data/protocol check.
2. `PatchEncoderFixedHead`: clean PatchTST-style patch encoder with fixed direct head.
3. `SegTSFTDenseFixedHead`: Seg-MoE-inspired modern dense TSFT backbone with MoE and
   autoregressive forecasting removed.

### Main internal base: `PatchEncoderFixedHead`

Data flow:

$$
X \in \mathbb{R}^{B \times L \times C}
\rightarrow
P \in \mathbb{R}^{(B C) \times N \times P_l}
\rightarrow
Z \in \mathbb{R}^{(B C) \times N \times d}
\rightarrow
\hat{Y} \in \mathbb{R}^{B \times H \times C}.
$$

Minimum configuration:

| Field | Value |
| --- | --- |
| `seq_len` | 336 |
| `patch_len` | 16 |
| `stride` | 8 |
| `padding_patch` | `end` |
| `revin` | true |
| `revin_affine` | false |
| `subtract_last` | false |
| `encoder_layers` | 3 |
| `d_model` | 128 |
| `n_heads` | 16 |
| `d_ff` | 256 |
| `dropout` | 0.2 |
| `fc_dropout` | 0.2 |
| `head_dropout` | 0.0 |
| `activation` | `gelu` |
| `head` | flatten + linear fixed horizon head |

[Inference] Although official PatchTST uses smaller dimensions for ETTh in some scripts,
Phase 0 should start with a unified configuration so that dataset-level differences are not
confounded with architecture changes. If ETTh overfits or the base is unstable, we can add a
documented ETTh-small ablation later.

### Sanity floor: `DLinear`

Purpose:

- Check whether the patch base is genuinely stronger than a simple linear baseline.
- Detect broken data loading, scaling, or training settings.

Default:

- same `seq_len=336`
- same horizons
- same dataset split
- MSE training loss

### Modern dense candidate: `SegTSFTDenseFixedHead`

[Inference] This candidate tests the user's concern that PatchTST may be too old and weak
as the internal base. It keeps Seg-MoE's modern dense TSFT-style ingredients but removes
the mechanisms that would overlap with later innovations:

- remove segment-wise MoE
- remove token-wise MoE fallback
- remove router probabilities and imbalance/load-balance losses
- remove autoregressive forecast
- use a direct fixed horizon head

Minimum configuration:

| Field | Value |
| --- | --- |
| `seq_len` | 336 |
| `patch_width` | 8 |
| `width_factor` | 4 |
| `encoder_layers` | 4 |
| `d_model` | 128 |
| `n_heads` | 4 |
| `n_kv_heads` | 2 |
| `d_ff` | 256 |
| `dropout` | 0.2 |
| `drop_path` | 0.1 |
| `norm_type` | `rms` |
| `rope_theta` | 10000.0 |
| `use_input_norm` | true |
| `head` | flatten + linear fixed horizon head |

### Parameter control

After `PatchEncoderFixedHead` is implemented, create a `PatchEncoderFixedHeadWideHead`
control with similar parameter count to the first mechanism model that adds decoder or MoE
parameters. This control is not required for the first smoke run, but it is required before
claiming a mechanism gain.

## Training Protocol

### Default optimization

| Field | Value |
| --- | --- |
| optimizer | `Adam` |
| learning rate | `1e-4` |
| loss | MSE |
| max epochs | 100 |
| early stopping patience | 10 |
| batch size ETT | 128 |
| batch size Weather | 128 |
| batch size ECL | 32 |
| batch size Solar | 64 |
| batch size Traffic | 24 |
| AMP | false by default; may be enabled only after numerical sanity check |

The batch sizes follow the memory pressure implied by channel count. Traffic may require
further reduction before full runs.

### Seeds

Primary seed:

$$
2021
$$

Seed variance set:

$$
\{2021, 2022, 2023\}.
$$

Phase 0 should first run seed `2021` across the full matrix. Seed variance is required for
selected core comparisons before any paper-facing claim.

### Reproducibility

At run start record:

- Python version
- torch version
- CUDA version
- GPU model
- selected GPU id
- dataset path and file hash if practical
- effective config JSON
- git status or commit
- random seeds for `random`, `numpy`, `torch`, `torch.cuda`, and `PYTHONHASHSEED`

## Evaluation Protocol

### Primary metrics

Report MSE and MAE:

$$
\text{MSE}=\frac{1}{BHC}\sum_{b,h,c}(\hat{Y}_{b,h,c}-Y_{b,h,c})^2,
$$

$$
\text{MAE}=\frac{1}{BHC}\sum_{b,h,c}|\hat{Y}_{b,h,c}-Y_{b,h,c}|.
$$

### Horizon diagnostics

Error-by-horizon:

$$
e_h^{mse}=\frac{1}{BC}\sum_{b,c}(\hat{Y}_{b,h,c}-Y_{b,h,c})^2,
$$

$$
e_h^{mae}=\frac{1}{BC}\sum_{b,c}|\hat{Y}_{b,h,c}-Y_{b,h,c}|.
$$

Segment metrics:

- `1-96`
- `97-192`
- `193-336`
- `337-720`

For shorter horizons, only valid prefix segments are reported.

### Prefix consistency

For Phase 0 fixed-head models, strict same-checkpoint variable-horizon consistency is not
available. Therefore record two related quantities:

1. `fixed_vs_max_prefix_mse`:

$$
\Delta_{fixed,max}(H)
=
\frac{1}{BHC}
\left\|
\hat{Y}_{1:H}^{fixed(H)}
-
\hat{Y}_{1:H}^{max(720)}
\right\|_2^2.
$$

2. `max_prefix_error`:

Evaluate the $H=720$ checkpoint on prefixes `{96,192,336,720}` against ground truth.

[Inference] If Phase 0 already shows large mismatch between horizon-specific fixed heads and
the max-horizon prefix model, it supports Phase 1's variable-horizon decoder direction. In
Phase 1, this diagnostic will become strict same-checkpoint prefix consistency:

$$
\hat{Y}^{request(H_1)}_{1:H_1}
\approx
\hat{Y}^{request(H_2)}_{1:H_1}.
$$

### Step-specificity

For `PatchEncoderFixedHead` and `SegTSFTDenseFixedHead`, record:

- cosine similarity between fixed head rows assigned to different future steps
- optional CKA / cosine similarity among per-step induced outputs on a validation batch

[Inference] If all future steps behave nearly identically while error-by-horizon differs
strongly, it suggests the fixed head lacks useful step-specific representation.

## Run Matrix

### Smoke matrix

Purpose: check data loading, tensor shapes, metrics, artifact writing.

| Model | Dataset | Horizon | Epochs |
| --- | --- | ---: | ---: |
| `DLinear` | ETTh2 | 96 | 1 |
| `PatchEncoderFixedHead` | ETTh2 | 96 | 1 |
| `SegTSFTDenseFixedHead` | ETTh2 | 96 | 1 |

Run locally first.

### Core matrix

Purpose: establish the base before mechanism work.

Models:

- `DLinear`
- `PatchEncoderFixedHead`
- `SegTSFTDenseFixedHead`

Datasets:

- ETTh2
- ETTm1
- Weather

Horizons:

- 96
- 192
- 336
- 720

This gives:

$$
3 \times 3 \times 4 = 36
$$

training runs for seed `2021`.

### Seed-variance matrix

After the core matrix has returned:

- use ETTh2, ETTm1, Weather
- run seeds `{2021,2022,2023}`
- include horizons `{96,720}` first

This controls variance without exploding the run count too early.

## Artifact Contract

Each run must write:

```text
artifacts/runs/phase0/<model>/<dataset>/h<horizon>/seed<seed>/
```

Required files:

- `effective_config.json`
- `environment.json`
- `metrics.json`
- `metrics_by_horizon.csv`
- `metrics_by_segment.csv`
- `training_log.csv`
- `checkpoint.pt`
- `predictions_test.npz`

Optional but preferred:

- `head_step_similarity.csv`
- `prefix_consistency.json`
- `memory_snapshot.txt`

Global summary:

```text
artifacts/reports/phase0/summary_metrics.csv
artifacts/reports/phase0/summary_by_horizon.csv
artifacts/reports/phase0/summary_by_segment.csv
```

## Remote Execution Rules

Before any remote run:

```bash
scripts/remote/check_529lab_3090_gpus.sh
```

Use `529_Lab-3090`. Prefer GPU 1 or GPU 2 when memory is safe. Avoid GPU 0 unless the user
explicitly accepts the risk.

[Fact] Remote conda env is `moe`; local conda env is `r2026-fsa`.

## Gate Criteria

## Phase0 Decision Status

[Fact] After the Phase0 gate, Weather-720 rerun, and targeted controls,
`PatchEncoderFixedHead` is selected as the canonical internal base for Phase1.

[Fact] `PatchEncoderFixedHead` is a clean PatchTST-style internal base, not an exact PatchTST
paper reproduction. Paper-facing PatchTST and DLinear baselines should be reproduced from native
upstream code before final comparison claims.

[Fact] The selected-base seed-variance lite matrix is complete:
`PatchEncoderFixedHead × {ETTh2, ETTm1, Weather} × {96,720} × {2021,2022,2023}`.

[Strong Evidence] The largest observed MSE CV is `2.47%` on `Weather / 720`, so Phase0 is finalized
and Phase1 Variable-Horizon Decoder experiments can start from `PatchEncoderFixedHead`.

[Strong Evidence] Prefix consistency diagnostic is complete. The h720 prefix is up to `+4.79%` MSE
worse than the horizon-specific fixed head on `Weather / 96`, and direct fixed-head prediction
mismatch reaches `0.044742` MSE on `ETTm1 / 192`. This is a measurable fixed-head
variable-horizon problem, not a data alignment issue (`truth_alignment_mse = 0.0`).

[Strong Evidence] Segment-wise checkpoint oracle diagnostic is complete. Over the shared `0-720`
forecast interval split into 48-step segments, the `pred_len=720` checkpoint has the best average
MSE on ETTh2, ETTm1, and Weather, but it is not segment-wise dominant. Winner counts are:
ETTh2 `h192=4, h336=1, h720=10`; ETTm1 `h96=2, h336=5, h720=8`; Weather
`h96=3, h192=2, h720=10`. This supports a Phase1 decoder that can adapt readout behavior by
requested horizon or forecast segment, instead of relying on a single fixed direct head.

### Continue to Phase 1 if:

- `PatchEncoderFixedHead` is consistently stronger than or competitive with `DLinear`.
- `SegTSFTDenseFixedHead` either improves on `PatchEncoderFixedHead` or exposes why the
  older clean patch base is still preferable.
- Horizon diagnostics show meaningful error drift or prefix instability.
- Training is stable on ETTh2, ETTm1, and Weather.

### Revisit baseline if:

- `PatchEncoderFixedHead` is weaker than `DLinear` on most core datasets.
- `SegTSFTDenseFixedHead` is unstable, too expensive, or its extra dense architecture makes
  later mechanism attribution unclear.
- Training collapses or shows large seed variance.

### Do not claim:

- SRSNet comparison success until SRSNet is reproduced or imported with explicit approval.
- Reproducibility until artifact paths and seeds are complete.
- Variable-horizon capability from Phase 0 fixed-head results alone.

## Source Defaults Used

- PatchTST official supervised scripts use `seq_len=336`, horizons `{96,192,336,720}`,
  `patch_len=16`, `stride=8`, `RevIN=1`, and patch-based Transformer encoder settings.
- DLinear official long-forecasting scripts also use `seq_len=336` and horizons
  `{96,192,336,720}`.
- Seg-MoE official code provides the modern TSFT backbone and segment-wise MoE design used
  as reference for `SegTSFTDenseFixedHead`, but Phase 0 removes MoE, routing loss, and
  autoregressive forecasting before using it as a candidate base.
