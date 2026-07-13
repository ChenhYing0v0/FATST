# StageC SC1-D2 Diagnostic Pipeline Code Explanation

## Scope

本次只新增diagnostic pipeline，不修改`TimeAlign.Model`、Encoder、decoder或训练主入口。核心文件：

- `scripts/run_stage_c_sc1_d2_diagnostic.py`：冻结checkpoint、提取memory、训练head-only probes；
- `scripts/analyze_stage_c_sc1_d2_diagnostic.py`：完成paired attribution与formal/core3 gate；
- `scripts/remote/run_stage_c_sc1_d2_diagnostic.sh`：三dataset GPU workers；
- `scripts/sync_stage_c_sc1_d2_diagnostic_results.sh`：同步raw artifacts并本地重算。

## Forward Data Flow

1. `batch_x: [B,720,C]`按A6相同history mean/std归一化；
2. frozen `_encode_normalized_history`产生`memory: [B,C,P,D]`；
3. flatten为`features: [B*C,P*D]`；five-dataset contract下实际width为768/1536/3072；
4. `batch_y[:,-720:,:]`用同一history mean/std变为`target: [B*C,720]`；
5. time heads直接输出`[N,720]`；coefficient heads输出`alpha: [N,720]`；
6. coefficient training使用`target @ Q.T`，validation通过`alpha @ Q`还原time output；
7. 每个row的history std为scalar，故orthogonal transform前后的evaluation-space squared error相同。

## Probe Modules

`LowRankLinearHead`是无activation的`P*D -> 256 -> 720`；`DenseNonlinearHead`提供parameter-matched与
same-total-units两个controls；`GroupedNonlinearHead`为每个group建立独立`P*D -> 32 -> n_l` block，并把
outputs写回其coefficient indices。parameter-matched dense hidden按实际input width动态计算；768/1536/3072
分别为197/250/291，matched parameter gap均低于0.5%。random-group只改变indices assignment，random-basis
只改变fixed orthogonal matrix；模块大小和optimizer保持一致。

## Split And Optimization Safety

train rows按sample ID切分，避免同一window不同channels同时落入fit与inner holdout。input z-score只用fit
statistics。best epoch只看inner holdout；official validation只在训练结束后评估一次。forecast checkpoint全部
`requires_grad_(False)`，optimizer只接收新probe head parameters。

## Artifact Definitions

### `d2_probe_metrics.csv`

- `final_fit_mse_eval`：最后执行epoch的fit evaluation-space MSE；
- `best_holdout_mse_eval`：保存probe state对应的最低inner-holdout MSE；
- `val_mse_norm/val_mae_norm`：instance-normalized full-H720 validation error；
- `val_mse_eval/val_mae_eval`：乘回history std并加回mean后的validation error；
- `parameters`：当前probe trainable parameter count；仅用于capacity audit。

### `d2_training_history.csv`

每个dataset/checkpoint seed/arm/epoch的fit与inner-holdout evaluation-space MSE，用于判断optimizer是否未收敛。

### `d2_metadata.json`

记录checkpoint path、profile、row counts、seeds、group sizes、basis/Parseval gaps、torch/CUDA与
`uses_test_split=false`、`forecast_model_updated=false`等invariants。

### Analyzer outputs

- `d2_pairwise_metrics.csv`：同dataset/checkpoint seed内的rank、nonlinearity、dense、random-group与
  random-basis归因gain；random两类不得合并后代替各自gate；
- `d2_dataset_summary.csv`：三seed mean/std与positive-seed count；
- `d2_summary.json`：completeness、invariants、hard gates与decision；
- `d2_diagnostic_report.md`：面向研究决策的中文解释。

## Code-Theory Consistency

 intended theory是检验“scale-specific nonlinear function allocation”是否优于generic nonlinear/capacity与
任意grouping。代码通过冻结同一memory、两个dense controls和六个random controls实现该归因。它仍只是
proxy：固定11 groups、hidden=32和head-only training不能穷尽所有scale mechanisms。若true scale失败，只能
否定当前final-head problem formulation；若成功，也只证明问题值得返回Step 4，不证明某个paper method成立。
