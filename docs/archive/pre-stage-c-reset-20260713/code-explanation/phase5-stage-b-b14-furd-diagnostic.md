# Phase5 StageB B14-FURD Step 3 Diagnostic Code Explanation

## Purpose And Boundary

`scripts/analyze_phase5_stage_b_b14_future_unit_retrieval_demand.py`只读取 frozen A6 checkpoint，判断不同
future units的 model-independent label-patch dependence是否比 A6 existing sensitivity更 unit-specific。它不训练模型、
不让 patch side path进入 prediction，也不能单独验证 retrieval method。

## Forward And Gradient Flow

```text
batch_x [B,720,C]
  -> Normalize -> x_norm [B,720,C] (autograd leaf)
  ├─ accepted A6 encoder/readout -> coeff [B,C,256] -> prediction [B,720,C]
  └─ valid unfold(K=48,S=24) -> evidence memory [B,C,29,48]

unit loss / Hutchinson scalar
  -> gradient w.r.t. the same x_norm [B,720,C]
  -> mean over B,C -> position profile [720]
  -> coverage-corrected aggregation -> patch profile [29]

patch memory / future unit
  -> fixed rank-8 DCT descriptors
  -> centered linear CKA over batch*channels observations
  -> label-patch dependence profile [29]
```

模型 parameters全部 frozen；`x_norm`保留 autograd。prediction path与 clean A6一致，side path只定义相同的
history evidence supports。

## Evidence Contract

每个 batch写入 `b14_history_patch_evidence_audit.csv`：

- `manual_patch_max_abs_diff`：旁路 memory与手工 normalized-history slices的最大逐元素差；
- `reconstruction_max_abs_diff`：patch overlap-add除以 position coverage后与 `x_norm`的最大差；
- `forecast_max_abs_diff`：显式 normalized-leaf forward与 model forward的最大差；
- `min/max_position_coverage`：valid patches对每个 raw position的覆盖次数，应为 `1/2`。

任一 contract失败时，结果只能标记 `diagnostic_invalid_for_direction_rejection`。

## Coverage-Corrected Aggregation

令 $q(t)$ 为 normalized 720-position attribution，$c(t)$ 为覆盖位置 $t$ 的 patch数。patch $p$ 的 mass为：

$$
Q_p=\sum_{t\in p}\frac{q(t)}{c(t)}.
$$

因此 $\sum_p Q_p=\sum_t q(t)$。这避免 overlap让中间 history positions被计算两次；valid unfold也避免
right padding重复末端 value。

## Main Statistics

- `error_conditioned_demand`：future-unit MSE对 `x_norm` 的 absolute gradient profile；
- `target_independent_sensitivity`：4-draw Hutchinson output-Jacobian RMS profile；
- `label_patch_dependence`：不经过 A6 Jacobian的 DCT-8 linear-CKA profile；
- `mean_label_shuffle_gap`：true CKA减 4-draw shuffled-target CKA；
- `delta_cosine`：mean sensitivity pair cosine减 mean demand pair cosine；
- `delta_js`：mean demand pair JS减 mean sensitivity pair JS；
- `coeff_gradient_cosine`：unit loss对 A6 coefficient的 signed-gradient control；
- `pair_matrix_spearman`：demand与sensitivity unit-pair结构的一致性；
- entropy/centroid：profile concentration与 history-time位置摘要。

`b14_future_unit_retrieval_profiles.csv`中的每行是一个 batch、unit、profile type和 patch的 normalized mass；
`normalized_history_energy`只是输入 evidence分布的描述，不进入 gate。

## Gate And Falsification

单 setting需要 bootstrap `p05(delta_label_cosine)>0.05`、`p05(delta_label_js)>0.01`、
`p05(mean_label_shuffle_gap)>0`、mean sensitivity cosine `>=0.80`。整体至少两个 datasets的 U180/U240
同时支持，才进入 B14-B parameter-matched probe。原 error-conditioned demand保留为 A1 control，不再拥有
方向否定权。

以下情况不允许否定 broader direction：non-finite/zero profile、Hutchinson失败、evidence contract失败或 mass
conservation error `>1e-6`。这些只说明 diagnostic无效。

## Runner And Aggregation

- remote runner：`scripts/remote/run_phase5_stage_b_b14_future_unit_retrieval_demand.sh`；
- cross-dataset aggregation：`scripts/summarize_phase5_stage_b_b14_future_unit_retrieval_demand.py`；
- remote默认每 dataset 8 train batches、batch size 16、4 Hutchinson draws、1000 bootstrap iterations。
