# Phase5 StageB B11-ESA Basis/Coeff Diagnostic Code Explanation

## 诊断位置

| 字段 | 内容 |
| --- | --- |
| `script` | `scripts/analyze_phase5_stage_b_b11_esa_basis_coeff_diagnostic.py` |
| `candidate_id` | `B11-ESA` |
| `current_step` | StageB Step 2/3：emergent basis-subspace problem diagnostic |
| `input` | clean A6 checkpoint、test split、`learned_temporal_basis`、真实 forward `coeff` |
| `output` | `analysis/phase5_stage_b_b11_esa_basis_coeff_diagnostic_20260708/` |
| `scope` | Frozen checkpoint diagnostic；不训练新模型；不做 method success claim |

## Reader Path

B9 和 B10 的失败说明：直接把 `stage_id` 或 `target_set_id` 编码进 `coeff` / readout，容易变成
horizon-conditioned 分段模型，也容易被 no-stage / pooled controls 解释。

B11 的诊断问题改为：

> A6 的 `learned_temporal_basis` 是否已经形成可利用的 future geometry？真实 `coeff` 是否沿这个 geometry
> 被差异化使用？

因此脚本不输入 stage token，也不按 horizon 强制分段。

## Tensor Flow

脚本读取 clean A6 checkpoint：

```text
learned_temporal_basis       # [720, 256]
```

随后在 test split 上复用 A6 forward path：

```text
batch_x                      # [B, 720, C]
x_norm                       # [B, 720, C]
patch_emb_x                  # [B, C * patch_num, D]
encoder layers               # [B, C * patch_num, D]
hidden                       # [B, C, patch_num * D]
coeff = learned_basis_coeff  # [B, C, 256]
coeff_rows                   # [B*C, 256]
```

默认每个 dataset 收集 `6000` 个 `coeff_rows`。

## Diagnostic A: Hard Basis-Row Clustering

第一组诊断对 normalized basis rows 做 deterministic KMeans：

```text
normalize(learned_temporal_basis[t]) -> row direction
KMeans(row directions) -> cluster_id[t]
```

然后事后报告：

- `cluster_stage_nmi`: cluster label 与 benchmark future regions 的 normalized mutual information；
- `mean_cluster_locality`: cluster 是否在时间轴上局部连续；
- `projection_pair_cosine_mean`: 同一个 `coeff` 投影到不同 cluster row subspaces 后的平均 cosine；
- `output_energy_entropy_mean`: 输出能量是否集中在少数 clusters。

[Boundary] `cluster_stage_nmi` 不是训练输入，也不是聚类目标。它只帮助判断自发 clusters 是否近似
benchmark stage。

## Diagnostic B: Sliding-Window Subspace Geometry

Hard clustering 对 ETTm1/Weather 不稳定，因此脚本又加入更符合 unified 叙事的连续 subspace 诊断。

默认设置：

```text
window_len = 96
stride = 48
rank = 16
```

对每个 sliding window：

```text
basis_w = learned_temporal_basis[start:end]     # [96, 256]
Q_w = top-r right singular vectors              # [256, 16]
projected_coeff_w = coeff_rows @ Q_w @ Q_w.T    # [N, 256]
```

报告：

- `adjacent_subspace_overlap_mean`: 相邻 windows 的 basis row-space overlap；
- `far_subspace_overlap_mean`: 距离至少 240 steps 的 windows overlap；
- `distance_subspace_overlap_spearman`: window 距离与 row-space overlap 的 Spearman；
- `adjacent_projection_cosine_mean`: coeff 在相邻 windows subspaces 上的投影方向 cosine；
- `far_projection_cosine_mean`: coeff 在远距离 windows subspaces 上的投影方向 cosine；
- `distance_projection_cosine_spearman`: window 距离与 coeff projection cosine 的 Spearman。

## Current Result

Hard KMeans：

| Dataset | K=4 stage NMI | Projection cosine | Interpretation |
| --- | ---: | ---: | --- |
| ETTh2 | `0.5325` | `0.2708` | temporal clusters are visible |
| ETTm1 | `0.0057` | `0.0483` | clusters are not stage-local |
| Weather | `0.0068` | `0.0146` | clusters are not stage-local |

Sliding-window geometry：

| Dataset | Adjacent overlap | Far overlap | Distance-overlap Spearman | Adjacent proj cosine | Far proj cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | `0.3900` | `0.0649` | `-0.7016` | `0.5585` | `0.1504` |
| ETTm1 | `0.4021` | `0.0811` | `-0.5472` | `0.5391` | `0.2379` |
| Weather | `0.3810` | `0.0700` | `-0.2786` | `0.4071` | `0.0484` |

## Code-Theory Consistency

[Intended theory] A6 learned basis may form a continuous future geometry. A second-stage architecture should exploit
this geometry without explicit horizon/stage conditioning.

[Code realization] The script tests both hard cluster and sliding-window row-space views. Hard clustering checks
whether basis rows form discrete self-organized groups; sliding windows check whether subspaces vary continuously
along the future axis.

[Observed boundary] Hard clusters are not robust across datasets, so B11 should not become a hard cluster/stage
method. Sliding-window subspaces are robust enough to support Step 4-6 design of continuous basis-conditioned
aggregation.

## Decision

`B11-ESA` passes Step 2/3 problem diagnostic and may enter Step 4-6 design gate.

The next design must be continuous and basis-conditioned, with no hard `stage_id` / `horizon_id`. Required controls:
`no-basis`, `shuffled-basis`, `constant-slot`, and `A6 fallback`.
