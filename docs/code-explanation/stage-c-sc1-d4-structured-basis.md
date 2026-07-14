# SC1-D4 Structured-Basis Diagnostic Code Explanation

## Functional Boundary

SC1-D4不修改forecast model。它从frozen A6提取`memory [B,C,P,D]`，将其reshape为
`features [B*C,P*D]`，以相同GroupedNonlinearHead分别预测七种orthogonal bases的
`coefficients [B*C,720]`，再通过`coefficients @ basis`生成full future。

## Basis Construction

- `balanced_interval`：复用D2 recursive midpoint basis；
- `identity`：`eye(720)`；
- `dct2`：orthonormal DCT-II rows；
- `pca_fit`：只用fit-target covariance做`eigh`，eigenvectors作为rows；
- `permuted_interval`：对balanced basis的time columns做seeded permutation；
- `random_interval_tree`：每个interval在25%-75%范围seeded split，使用general Haar contrast；
- `random_orthogonal`：复用D2 QR Gaussian control。

所有basis执行`basis @ basis.T` orthogonality check。所有head使用同一random group seed，从而family之间只有
basis变化。PCA不读取inner-holdout、official validation或test targets。

## Training And Horizon Evaluation

训练复用D2 `train_head`，full coefficient target为`target @ basis.T`。由于basis正交，full H720
evaluation-space MSE与time-space MSE等价。final validation先重建`prediction [N,720]`，再裁剪八个prefix：
`48/96/144/192/288/336/512/720`，分别累计MSE/MAE。

这不是variable-H training：optimization仍只使用H720。八horizon metrics回答同一full-domain readout裁剪后
是否具有一致收益。

## Geometry Statistics

`d4_basis_geometry.csv`每个arm定义：

- `covariance_offdiag_ratio`：basis-space fit-target covariance的off-diagonal Frobenius ratio；
- `variance_capture_topK`：按coefficient variance排序后的top-K energy share；
- `mean_atom_support_fraction`：每个atom非零time coordinates的平均比例；
- `active_atoms_hH`：在prefix前H positions上有非零support的atom数。

Analyzer先平均三个grouping seeds，再形成15个dataset-checkpoint primary units；所有effect使用
`log(control_error / balanced_error)`，positive表示balanced更好。

## Artifacts

每dataset worker输出：`d4_probe_metrics.csv`、`d4_training_history.csv`、
`d4_basis_geometry.csv`、`d4_metadata.json`。Analyzer输出checkpoint/dataset/macro horizon comparisons、
geometry summary、`d4_summary.json`与human-readable report。

## Code-Theory Consistency

| Intended distinction | Code realization | Falsification |
| --- | --- | --- |
| standard structure | identity/DCT/PCA same head | balanced global noninferiority fail |
| contiguous locality | column-permuted balanced atoms | balanced vs permuted fail |
| exact balancing | random local interval tree | balanced vs random tree fail |
| D3 replication | QR random orthogonal | H720 replication fail |
| unified horizon relevance | same full forecast evaluated at eight prefixes | horizon consistency fail |

任何pass只授权Step 5；D4不证明end-to-end decoder effectiveness或paper novelty。
