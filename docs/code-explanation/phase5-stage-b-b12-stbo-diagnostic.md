# Phase5 StageB B12-STBO Diagnostic Code Explanation

对应脚本：

- `scripts/analyze_phase5_stage_b_b12_stbo_diagnostic.py`

## Purpose

`B12-STBO` 指 `Subspace-Tiled Basis Operator`。该诊断不训练模型，也不修改 TimeAlign/A6 代码。
它只检查一个 Step 2/3 问题：

> A6 的 `learned_temporal_basis[720,K]` 是否可以被更原生的 tile-local / subspace-tiled basis operator
> 解释，从而避免当前 full-720 step basis + prefix slicing 的设计？

## Inputs

脚本默认读取：

| Input | Default |
| --- | --- |
| A6 checkpoints | `analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707/raw/official-last/TimeAlignOfficialUnified720_a6_clean_official-last/<dataset>/mixed_h96_h192_h336_h720/seed2021/checkpoint.pt` |
| Dataset root | `/Users/river/PaperResearch/Project/datasets` |
| Datasets | `ETTh2`, `ETTm1`, `Weather` |
| Tile length | `48` |
| Gate rank | `16` |

这些 A6 checkpoints 是 clean `learned-basis-forecast-operator` 路径，用于读取：

- `learned_temporal_basis: [720,K]`;
- A6 model forward 中的真实 `coeff: [B,C,K]`。

## Dataflow 1: A6 Basis Tile Factorization

脚本先读取：

```text
basis: [720,K]
```

在默认 `tile_len=48` 下 reshape 为：

```text
basis_tiles: [M,L,K]
M = 15
L = 48
```

然后比较四种 local-basis 解释方式。

### shared local basis

将所有 tile 沿 atom/channel 维拼接：

```text
concat = [basis_tile_0, ..., basis_tile_14]: [L, M*K]
```

对 `concat` 做 SVD，取 top-r left singular vectors：

```text
U_shared: [L,r]
```

`shared_local_energy` 表示单套 `U_shared` 对全部 A6 basis tiles 的重构能量占比。

### independent tile basis

对每个 `basis_tile_m: [L,K]` 单独做 SVD，取 top-r left singular vectors。

`independent_tile_energy` 是 upper bound。它不能单独作为方法正证据，因为每个 tile 独立 basis 容易退化成
segmented Direct head。

### basis bank

脚本对 flattened basis tiles 做 deterministic KMeans，默认比较 `Q=2` 和 `Q=4` banks。
每个 bank 内的 tiles 共享一套 local basis：

```text
U_q: [L,r]
```

`bank4_energy` 是 B12-B `Subspace Basis Bank` 的主要 feasibility proxy。

### local DCT control

`local_dct_energy` 使用固定 DCT basis `DCT[L,r]`。若 shared/bank 与 DCT 接近，则 B12 可能只是 generic
smooth local basis，而不是 A6-specific learned subspace operator。

## Dataflow 2: Train-Label Tile-Basis Audit

脚本只使用 train split，并按当前 repo 的 dataset 规则做 normalization：

```text
train_values: [T,C]
future_matrix: [N*C,720]
```

其中每一行是一个 normalized future label trajectory。再切成：

```text
label_tile_m: [N*C,L]
```

然后用与 A6 basis audit 对称的方式计算：

- `shared_local_energy`;
- `bank4_energy`;
- `independent_tile_energy`;
- `local_dct_energy`;
- gap to independent；
- difference vs DCT。

该步骤检查 tile-local basis 是否也符合真实 label distribution，而不只是 checkpoint 参数的结构。

## Dataflow 3: Coeff Projection Audit

脚本加载 A6 model，并在指定 split 上收集真实 forward coefficient：

```text
x: [B,seq_len,C]
history encoder -> hidden: [B,C,R]
learned_basis_coeff(hidden) -> coeff: [B,C,K]
coeff_rows: [B*C,K]
```

对每个 `basis_tile_m: [L,K]` 做 SVD，取 coefficient-space row subspace：

```text
Q_m: [K,r]
```

然后把 `coeff_rows` 投影到每个 `Q_m`：

```text
projection_m = coeff_rows @ Q_m @ Q_m.T
```

脚本记录：

| Metric | Meaning |
| --- | --- |
| `adjacent_projection_cosine` | 相邻 tile projection direction 的平均 cosine |
| `far_projection_cosine` | 距离至少 240 steps 的 tile projection direction cosine |
| `distance_projection_cosine_spearman` | tile 距离与 projection cosine 的 Spearman |
| `projection_entropy_mean` | `coeff` 在 tile subspaces 上的 normalized projection entropy |
| `output_entropy_mean` | `coeff @ basis_tile.T` 的 output energy entropy |

若 adjacent cosine 明显高于 far cosine，说明 A6 coeff 的使用方向沿 tile/subspace 有结构变化。
若 entropy 过低，则说明 `coeff` 只服务少数 tiles；若 entropy 较高，则说明同一个 coeff 同时服务多个 tile
subspaces，这会支持 B12 的 stage/tile-local coefficient design 需求。

## Gate Logic

默认 gate rank 为 `16`。脚本输出的 `decision` 只用于 Step 2/3，不等于方法有效性。

进入 Step 4-6 的强条件是：

1. 至少两个数据集满足 basis-side shared/bank feasibility；
2. 至少两个数据集满足 label-side shared/bank feasibility；
3. 至少两个数据集满足 coeff projection structure；
4. `local_dct` 不能解释 label-side shared/bank gain。

如果只看到 `independent_tile_energy` 很高，而 shared/bank 明显落后，则报告会倾向
`independent_tile_only_no_method`，因为该结果更像分段 Direct head。

如果 basis-side 正向但 label 或 DCT control 风险存在，则报告会倾向
`partial_support_basis_operator_but_label_or_dct_control_risk`，要求继续诊断，而不是直接实现模型。

## Outputs

脚本输出到 `analysis/phase5_stage_b_b12_stbo_diagnostic_20260708/`：

| File | Content |
| --- | --- |
| `b12_stbo_basis_factorization.csv` | A6 basis tile factorization metrics |
| `b12_stbo_label_factorization.csv` | train-label tile basis metrics |
| `b12_stbo_coeff_projection.csv` | A6 coeff projection into tile subspaces |
| `b12_stbo_gate_summary.json` | decision and gate pass/fail lists |
| `b12_stbo_report.md` | human-readable diagnostic report |

## Code-Theory Consistency

Intended theory:

- B12 should be considered only if A6's full-720 basis can be represented as shared/bank local tile bases;
- the same tile-local structure should appear in train labels;
- `coeff` usage should show tile/subspace differentiation.

Code realization:

- A6 basis side is tested by SVD reconstruction of `basis_tiles`;
- label side is tested by train-only normalized future label matrix;
- coeff side is tested by real A6 forward coefficients projected into tile row-spaces.

Proxy limitation:

- Reconstruction energy does not prove that a trainable STBO model will improve MSE;
- KMeans banks are diagnostic proxies, not the final routing design;
- DCT is only one generic local-basis control.

Falsification:

- If shared/bank basis does not approach independent tile basis, B12 lacks compression feasibility;
- if DCT explains train-label tile structure, B12 is not distinct enough from generic local spectral bases;
- if coeff projection has no adjacent/far structure, stage/tile-local coeff generation is not yet supported.

