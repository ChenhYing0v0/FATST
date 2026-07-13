# Phase5 StageB B6 Prefix-Native Objective Diagnostic Code Explanation

本文档解释 `scripts/analyze_phase5_stage_b_b6_prefix_objective_diagnostic.py`。该脚本是只读离线诊断，
不训练模型，不实现新 objective。

## Inputs

| Input | Role |
| --- | --- |
| train split raw dataset | 构造 train-only future label matrix |
| A6-LBF-r256 `predictions_test.npz` | 构造历史 A6 residual matrix |
| optional no-align/no-recon checkpoints | 读取 `learned_temporal_basis`，评估 A6 learned basis coverage |

默认 A6 prediction artifact 来自：

```text
analysis/phase5_timealign_hss_a6_capacity_native_gate_20260703/raw/official-last/TimeAlignOfficialUnified720_A6_a6_lbf_r256_official-last
```

checkpoint 不写入 repo。当前报告使用远程 B4 `no_align_no_recon` checkpoints 的临时本地副本抽取
`learned_temporal_basis`。

## Tensor And Statistic Flow

1. 读取 ETTh2 / ETTm1 / Weather 的 train split，并按 official TimeAlign split 规则用 train split
   mean/std 标准化。
2. 抽样最多 `4096` 个 train windows，构造 future label matrix：

```text
future_labels: [num_windows, 720, C]
label_matrix: [num_windows * C, 720]
```

3. 对 `label_matrix` 做 temporal covariance eigendecomposition，得到 train-only label PCA basis。
4. 构造 DCT basis 作为 generic low-frequency control。
5. 若提供 checkpoint，读取：

```text
learned_temporal_basis: [720, 256]
```

并对其做 SVD，使用 left singular vectors 作为 A6 learned temporal span 的有序正交 basis。
6. 读取 A6 prediction residual：

```text
residual = pred - true
residual_matrix: [test_samples * C, 720]
```

抽样最多 `30000` 行后，投影到 label PCA / DCT / A6 learned basis。

## Output Columns

| Column family | Meaning |
| --- | --- |
| `label_pca_top{k}_energy` | train labels 被 top-k train-only PCA components 捕捉的能量比例 |
| `label_dct_top{k}_energy` | train labels 被 top-k DCT low-frequency basis 捕捉的能量比例 |
| `label_a6_basis_top{k}_energy` | train labels 被 A6 learned temporal basis span 捕捉的能量比例 |
| `*_minus_dct_top{k}` | 相对 generic DCT control 的增益；用于判断是否只是 frequency/smoothness effect |
| `full_to_prefix_subspace_overlap` | H720 basis 限制到 prefix 后，与 prefix-only PCA basis 的 subspace overlap |
| `residual_*_top{k}_energy` | A6 residual 被对应 basis 捕捉的能量比例 |
| `residual_step_spearman` | residual step-wise energy 与 forecast step index 的 Spearman 相关 |

## Code-Theory Consistency Evaluation

[Intended Theory] B6-PLO 只有在 label/residual structure 不能被 generic low-frequency basis 解释、且与 A6
learned basis 有明确连接时，才有资格进入 Step 4-6 method design。

[Code Realization] 脚本同时比较 train-label PCA、DCT control、A6 learned temporal basis，并把 residual
投到这些 basis 上。

[Observed Boundary] 当前结果显示 PCA 与 DCT 的 top32 coverage 几乎相同，A6 learned basis top32 还弱于
DCT。因此 B6 不能直接升级为 learned-basis objective。

[Falsification] 若后续 clean A6 rerun 或更强 coefficient-space diagnostic 显示 A6 learned basis/residual
有稳定且明显优于 DCT 的结构，B6 才能重新进入 Step 4-6。
