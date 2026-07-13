# StageC D1 Offline Diagnostic Code Explanation

## Functional Flow

入口：

- `scripts/run_stage_c_d1_offline_diagnostic.py`；
- `scripts/analyze_stage_c_d1_offline_diagnostic.py`；
- `scripts/remote/run_stage_c_d1_offline_diagnostic.sh`；
- `scripts/sync_stage_c_d1_offline_diagnostic_results.sh`。

每个dataset worker复用frozen natural contract和3个validation-selected checkpoints，不读取test、不更新
forecast model weights。

## Tensor And Artifact Flow

### Source collection

对train/validation batch：

1. `batch_x: [B,720,C]`按history mean/std标准化；
2. frozen A6输出`prediction: [B,720,C]`；
3. 构造`label/residual: [B,C,720] -> [B*C,720]`；
4. `encode_history`输出`memory: [B,C,P,D]`；
5. probe features为`full_hidden [B*C,768]`、`patch_mean [B*C,D]`、
   `patch_shuffled [B*C,768]`与`raw_history [B*C,720]`。

### D1-A projections

`dct_basis [720,256]`和fixed random orthogonal basis把source映射到nested coefficient groups；localized
block projection按90/30/10/5/1步的nested partitions计算increments。输出
`d1_structure_metrics.csv`。

### D1-B probes

所有label/residual DCT-256与block-144 targets一次拼接。每种feature用train statistics标准化，并求解

$$
W=(X^TX/N+\lambda I)^{-1}X^TY/N,\qquad\lambda=0.01.
$$

validation只计算fixed probe的level-wise R2/NRMSE，输出`d1_probe_metrics.csv`。这是closed-form diagnostic，
不是forecast-model training。

### D1-C basis and gradients

对`learned_temporal_basis [720,256]`做SVD/QR，输出rank、condition、entropy、support与subspace overlap到
`d1_basis_geometry.csv`。

四种horizon measures先转换为exact step weights
$w(t)=\sum_{H\ge t}p(H)/H$。raw risk直接对error加权；projected risk先用nested block projections得到
orthogonal increments，再去掉cross-scale terms。对encoder/coeff/basis/all参数分别导出gradient norm与
cosine到`d1_gradient_metrics.csv`。

## Analyzer Gates

Analyzer要求Weather、ETTm1、ETTh2四类CSV和9条metadata完整；任何`uses_test_split=true`或
`trains_forecast_model=true`直接报错。它输出：

- `d1_dataset_gate.csv`；
- `d1_summary.json`；
- `d1_diagnostic_report.md`。

## Code-Theory Consistency

- intended theory：先分离future structure、Encoder information与basis geometry，再决定PMFO架构边界；
- code realization：D1-A/B/C分别有独立source、control和gate，PIR先通过uniform Parseval invariant；
- proxy boundary：DCT/block是fixed diagnostic spaces，不是预定PMFO basis；linear ridge只证明线性可恢复性，
  不能证明最终nonlinear decoder一定有效；
- falsification：Parseval失败使PIR诊断无效；structured basis不优于random、full memory不优于shuffle，或
  gradient separation不跨dataset时，对应problem candidate不得进入实现。
