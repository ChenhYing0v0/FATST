# StageC D12-A Predictable-Frame Diagnostic 代码说明

## 1. 目的与边界

D12-A只诊断一个问题：在A6的rank-256 future synthesis coordinates中，history可预测的future
variation是否形成稳定、且不同于raw-label PCA的subspace。它不会实现PRISM/CAPE forecast method，不读取
validation/test，也不会把diagnostic pilot权重交给最终模型。

v1使用uniform normalized rows，remote evidence暴露其与raw forecast risk不一致；active v2改用
`history_std_squared` weights，并复用v1 pilot checkpoints。v1证据保留作failure-attribution record。

入口包括：

- `scripts/run_stage_c_d12_predictable_frame_feasibility.py`：单dataset cross-fit worker；
- `scripts/analyze_stage_c_d12_predictable_frame_feasibility.py`：五dataset冻结gate；
- `scripts/remote/run_stage_c_d12_predictable_frame_feasibility.sh`：三GPU workload-aware runner；
- `scripts/sync_stage_c_d12_predictable_frame_feasibility_results.sh`：只同步小型统计与日志。

## 2. Worker 数据与shape流

### 2.1 Forward folds

train dataset包含长度720的history和长度720的future。window index $i$对应raw interval
$[i,i+1439]$。`fold_ranges()`令pilot train最后一个window与OOF第一个window相隔1439个index，保证二者
raw intervals不重叠。每个OOF block确定性抽取512个windows。

### 2.2 RevIN-normalized coordinates

一个batch的输入为`batch_x[B,720,C]`、`batch_y[B,720,C]`。`normalized_rows()`使用每个
window/channel的history mean与variance，得到：

- `normalized_history[B,720,C]`；
- `future_rows[B*C,720]`；
- `history_rows[B*C,720]`。

统计位于A6 basis实际合成预测的normalized future coordinates；不混入最终RevIN denormalization。

### 2.3 A6 primary pilot

每个fold从同一seed 2021初始化一个完整A6，使用该dataset冻结natural profile、full-720 L1、AdamW与
cosine schedule训练固定20 epochs。没有early stopping，也不使用OOF选择checkpoint。

`normalized_a6_prediction()`按真实forward路径执行：

1. `normalized_history[B,720,C] -> memory[B,C,P,D]`；
2. `flatten -> hidden[B,C,P*D]`；
3. `learned_basis_coeff -> coeff[B,C,256]`；
4. `learned_temporal_basis[720,256] @ coeff -> prediction[B,720,C]`。

worker同时比较手工normalized path经RevIN还原后的结果与official `Model.forward()`，将最大差记录为
`forward_reconstruction_max_abs`。

### 2.4 DCT-ridge robustness pilot

`history_rows[N,720]`先投影到前128个orthonormal DCT coordinates，得到`features[N,128]`；ridge
回归直接预测`future_rows[N,720]`。它只检验predictable subspace是否完全依赖A6 model bias，不是paper
baseline或method candidate。

### 2.5 Covariance sufficient statistics

`MomentAccumulator`不保存所有prediction，而是对以下`[N,720]` rows以float64累加sum与outer product：

- `label`；
- `a6`；
- `ridge`；
- `a6_residual=a6-label`；
- `ridge_residual=ridge-label`。

v2对row $n$使用$w_n=s_{x,n}^2$，covariance由
$[\sum_nw_nx_nx_n^T-(\sum_nw_nx_n)(\sum_nw_nx_n)^T/\sum_nw_n]/\sum_nw_n$恢复。
OOF SSE和fold-centered zero SST都按`weight_sum*720`归一化，因此
`oof_r2=1-oof_mse/zero_mse`与raw-space MSE一致。

## 3. 输出字段定义

### 3.1 `fold_metrics.csv`

- `row_count`：OOF sampled windows乘channel数；
- `a6_oof_mse` / `ridge_oof_mse`：normalized future逐元素MSE；
- `zero_mse`：使用该fold label mean的constant predictor MSE；
- `*_oof_r2`：相对上述constant predictor的解释率；
- `*_trace`：对应covariance总variation；
- `*_effective_rank`：covariance非负eigenvalue分布的entropy effective rank；
- `*_minimum_eigenvalue` / `*_symmetry_max_abs`：PSD与对称性invariant；
- `*_predictable_trace_fraction`：prediction covariance trace除以label covariance trace。
- `weight_effective_sample_fraction`：weighted ESS除以row count；
- `weight_max_share`：单row最大weight占总weight的比例。

### 3.2 `subspace_metrics.csv`

- `optimal_predictable_capture`：pilot covariance自身top-r eigenvectors捕获的energy比例；
- `raw_label_basis_predictable_capture`：raw-label top-r eigenvectors捕获pilot covariance的比例；
- `raw_relative_capture_gap`：前两者差值除以optimal capture；
- `label_pilot_subspace_overlap`：$\|U_y^TU_p\|_F^2/r$；
- `subspace_overlap`：两个forward folds的同pilot top-r overlap。

### 3.3 `dataset_summary.json`与五dataset gate

每个dataset的primary rank固定为256；rank32/64仅解释geometry。dataset同时通过predictability、trace、
fold stability、rank256 headroom、A6/ridge robustness及numeric invariants才计为support。五dataset至少3个
support才允许D12-B；无论结果如何，D12-A均不直接授权method implementation或test。

## 4. Remote与同步

remote runner按`Weather`、`ETTm1+ETTh2`、`ETTm2+ETTh1`分配三张GPU，并支持检测
`dataset_summary.json`后断点跳过。pilot checkpoints与`720x720` moment archives保留在repo外remote
output root；v2从v1 root读取pilot checkpoint，只重算ridge、OOF与moments。sync脚本只取复核结论所需的
CSV、JSON、training history与logs。

## 5. Code-Theory Consistency

- intended theory：若$\operatorname{Cov}(E[y\mid x])$在rank256下相对raw-label PCA存在稳定headroom，
  CAPE问题才值得进入D12-B；
- code realization：用train-only purged forward OOF A6 predictions估计predictable covariance，并用独立
  DCT-ridge pilot复核subspace；
- proxy：finite-epoch A6/DCT-ridge prediction covariance只是conditional-mean covariance的biased proxy；
- falsification：OOF predictability退化、fold不稳定、pilot不一致，或raw-label rank256 capture gap低于0.5%，
  均阻止CAPE前进；
- fairness：所有结果是problem diagnostic，不是A6与新decoder的effectiveness comparison，也没有
  frozen-component replacement结论。
