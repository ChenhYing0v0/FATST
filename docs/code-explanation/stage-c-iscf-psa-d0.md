# SC-ISCF-PSA-D0 分析代码说明

## 1. 作用与边界

`scripts/analyze_stage_c_iscf_psa_d0.py`只读取existing EQUAL validation replay，检验frozen direct policy向
uniform收缩时是否存在稳定finite-capacity frontier。它不训练model、不修改checkpoint、不读取official test，也不实现
paper method。

配置由`configs/stage_c_iscf_psa_d0.json`冻结。normal execution前显式检查
`existing_validation_artifact_analysis_authorized=true`与`formal_test_access_authorized=false`。

## 2. Input tensors与shape

每个dataset/seed读取：

- `probe_arms: [R,S,T]=[256,5,720]`；five ISCF scope-arm forecasts；
- `probe_direct_policy: [R,T,S]=[256,720,5]`；source EQUAL direct policy；
- `probe_targets: [R,T]=[256,720]`；validation targets；
- `scales: [S]=[5]`；必须逐值等于`[1,48,144,360,720]`。

代码验证arms/policy转置shape、targets shape、finite values、simplex normalization与scope order。所有dataset按
channel-group boundary把256 rows切为147 fit rows和109 evaluation rows，避免同一source sample的channels跨split。

## 3. Policy transformations

### 3.1 Convex uniform

$$
p_\alpha=(1-\alpha)p+\alpha/S.
$$

输入与输出shape均为`[R,T,S]`。该family是primary diagnostic。

### 3.2 Convex scope marginal

scope prior只从LODO source datasets的147 fit rows求`[S]` marginal，再广播到held-out data：

$$
p_{\alpha,m}=(1-\alpha)p+\alpha\bar p_{\mathrm{source}}.
$$

它区分“必须向uniform收缩”与“只需向跨样本scope frequency收缩”。

### 3.3 Temperature

$$
p_{\tau,s}=\operatorname{softmax}_s(\log p_s/\tau).
$$

该control与convex shrinkage同样趋向uniform，但路径不同，只作generic smoothing对照。

### 3.4 Fusion

`probe_arms`先转为`[R,T,S]`，再与policy逐scope相乘并求和：

$$
\hat y_{r,t}=\sum_s p_{r,t,s}a_{r,t,s},
$$

输出`[R,T]`，与targets计算L1和MSE。arms始终不变，因此该probe只改变frozen readout weights。

## 4. Leave-one-dataset-out selection

对每个held-out dataset：

1. 使用另外four datasets × three seeds的147 fit rows；
2. 对预定义grid逐值计算每run相对source policy的L1/MSE gain；
3. 以macro L1 gain选择唯一global value，tie时选grid中更小值；
4. 将该value固定到held-out dataset × three seeds的109 evaluation rows；
5. 不按held-out dataset、seed、position bin或MSE二次选择。

因此five folds恰好生成primary convex family的15个held-out run metrics。

## 5. Output columns

### `selection_curves.csv`

- `family/value`：transformation与grid value；
- `heldout_dataset`：本fold不参与选择的dataset；
- `source_run_count`：选择数据中的run数，固定12；
- `fit_macro_gain_l1_percent`、`fit_macro_gain_mse_percent`：12个source runs的macro gain；
- `selected`：该fold是否选中此value。

### `selected_run_metrics.csv`

- `selected_value`：只由其他four datasets fit rows选择；
- `baseline_l1/mse`、`candidate_l1/mse`：held-out 109 rows上的absolute risk；
- `gain_l1/mse_percent`：source policy到selected transformation的relative gain；
- `policy_entropy_before/after`：按$\log S$归一化的mean entropy；
- `policy_mean_l1_movement`：policy tensor的mean absolute change。

### `dataset_summary.csv`

- `selected_alpha`：convex-uniform fold value；
- `macro_gain_l1/mse_percent`：held-out three-seed macro；
- `joint_positive`：两项macro gain是否都大于0。

### `evaluation_curves.csv`

保存所有families、all grid values在held-out evaluation rows上的risk curve；不得用它重新选择value。

### `position_bin_metrics.csv`

把selected value固定后，在`[0,96)`、`[96,192)`、`[192,336)`、`[336,720)`四个disjoint future-position
segments上报告同一组metrics。它只解释frontier位置结构，不是checkpoint/method selection surface。

### `decision.json`

记录15-run macro gains、positive runs/datasets/folds、selected alphas、entropy-gain rank correlation、five machine
checks与最终decision。只有convex-uniform family参与primary decision；其他families是diagnostic controls。

## 6. Code-theory consistency

Intended theory是：在不增加information时，降低conditional fusion-weight freedom可能改善finite-sample generalization。
代码精确实现frozen one-dimensional frontier，并通过LODO阻止held-out dataset调参。

仍然只是proxy的部分：post-hoc policy shrinkage不模拟route loss对arms与policy的joint-training co-adaptation，也不能排除
historical EQUAL与new controls之间的run drift。因此negative只能拒绝`H1 inference_weight_overfit`，positive也只能建立
problem evidence；两者都不直接授权paper method或formal test。

Falsification：若convex-uniform在LODO evaluation上macro L1/MSE不同时为正，或缺少4/5 datasets、12/15 runs与
4/5 nonzero selections的稳定性，则frozen inference shrinkage不受支持；不得以此拒绝joint-training方向。
