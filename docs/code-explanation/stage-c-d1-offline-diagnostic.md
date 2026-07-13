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

1. `batch_x: [B,720,C]`计算history mean/std，Encoder仍按A6 RevIN contract工作；
2. frozen A6输出evaluation-space `prediction: [B,720,C]`；
3. 构造`future_deviation=y-history_mean`和`residual=y-prediction`，再变换为
   `[B,C,720] -> [B*C,720]`；
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

同一frozen LBF head还读取original、patch-shuffled和patch-mean-collapsed memory。输出
`d1_frozen_decoder_metrics.csv`，其中`model_r2_vs_zero_deviation`判断当前memory/head path是否解释future
deviation，两个relative SSE increase判断有序patch content是否实际参与预测。该counterfactual不能单独证明
memory具备完备multiresolution semantics。

该CSV各量均来自validation：`model_mse/zero_deviation_mse/shuffled_mse/collapsed_mse`分别是对应prediction
的elementwise SSE除以元素数；`model_r2_vs_zero_deviation=1-model_sse/zero_sse`；两个
`*_relative_sse_increase=counterfactual_sse/model_sse-1`；`forward_reconstruction_max_abs`验证direct memory
decode与正式forward一致。

### D1-C basis and gradients

对`learned_temporal_basis [720,256]`做SVD/QR，输出rank、condition、entropy、support与subspace overlap到
`d1_basis_geometry.csv`。

四种horizon measures先转换为exact step weights
$w(t)=\sum_{H\ge t}p(H)/H$。raw risk直接对error加权；projected risk先用nested block projections得到
orthogonal increments，再去掉cross-scale terms。对encoder/coeff/basis/all参数分别导出gradient norm与
cosine到`d1_gradient_metrics.csv`。v2在evaluation-space error上计算risk，使梯度与benchmark MSE尺度一致。
Analyzer中的`raw_*_gradient_separation`定义为same measure raw gradient相对delta-720 raw gradient的
`1-cos`；`projected_*_gradient_separation`定义为projected gradient相对same-measure raw gradient的`1-cos`。
aggregate列对uniform/log-uniform/benchmark三种measure取均值，per-measure列必须同时报告以防单一measure
主导结论。

## Analyzer Gates

Analyzer要求Weather、ETTm1、ETTh2五类CSV和9条metadata完整；任何`uses_test_split=true`或
`trains_forecast_model=true`直接报错。它输出：

- `d1_dataset_gate.csv`；
- `d1_summary.json`；
- `d1_diagnostic_report.md`。

## Code-Theory Consistency

- intended theory：先分离future structure、Encoder information与basis geometry，再决定PMFO架构边界；
- code realization：D1-A/B/C分别有独立source、control和gate，PIR先通过uniform Parseval invariant；
- proxy boundary：DCT/block是fixed diagnostic spaces，不是预定PMFO basis；linear ridge只证明线性可恢复性；
  frozen counterfactual可能同时包含memory information与现有head位置依赖，不能证明最终PMFO一定有效；
- falsification：Parseval失败使PIR诊断无效；structured basis不优于random、full memory不优于shuffle，或
  gradient separation不跨dataset时，对应problem candidate不得进入实现。
