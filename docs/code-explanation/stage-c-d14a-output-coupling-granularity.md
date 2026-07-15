# StageC D14-A0 Code Explanation

## 1. Scope

D14-A0是`diagnostic_only`，不更新A6或任何forecast model。实现只验证：在fixed past、统一输出720步的
条件下，future outputs的sharing scope是否存在稳定的局部最优尺度。test split、D14-B和paper method均未启用。

## 2. Forward Data Flow

入口是official dataset window：

1. `history: [B, 720, C]`，`future: [B, 720, C]`；
2. 每个window/channel用history statistics作instance normalization；
3. 转置并展平为`history_rows/future_rows: [B*C, 720]`；
4. 只在当前fold的fit rows上拟合PCA，得到`components: [720, 64]`；
5. 所有split经同一transform得到`X: [N, 64]`；
6. full affine fit得到`W_ols: [64, 720]`；
7. 对future partition中的每个block `B_j`，将`W_ols[:, B_j]`投影到rank-$r_s$的output subspace；
8. `prediction: [N, 720] = (X-x_mean) @ W_s + y_mean`。

因此不同arm共享data rows、normalization、PCA、full affine solution和intercept，唯一变化是future output
coordinates在哪个scope内共享low-rank directions。

## 3. Arms And Parameter Contract

- `canonical_s1/s48/s144/s360/s720`：从future origin开始连续分块；
- `shifted_s48/s144/s360`：循环平移半个block，检查边界偶然性；
- `random_s48/s144/s360`：相同block size、rank与parameter count，打散future order；
- `train_selected_best`：只按train calibration MSE选择单一canonical scale；
- `equal_canonical`、`train_mean`、`persistence`：context controls。

五个canonical arms的factor parameter counts为46800、47040、46800、46640、47040，最大relative gap
为0.513%。`canonical_s1`在代码中必须与unconstrained `W_ols`逐元素一致。

## 4. Fold And Artifact Semantics

每个dataset使用三个chronological folds。每fold保存：

- `fold_metrics.csv`：calibration/validation上的full、3 bins与8 dense-prefix MSE/MAE；
- `validation_bin_losses_fold*.npz`：每个arm、每个row、每个future bin的MSE/MAE；
- `parameter_budget.csv`：factor count与observed block rank；
- `metadata.json`：indices、split gap、PCA orthogonality、condition number、selected scale与environment。

不保存full predictions，避免把不必要的大型diagnostic artifacts带入repo。

## 5. Analyzer Statistics

`analyze_stage_c_d14a_output_coupling_granularity.py`从row-level bin losses计算：

- `carrier_skill_relative_gain`：train mean到calibration-selected fixed scale的validation改善；
- `canonical_oracle_relative_gain`：sample × bin oracle相对fixed selected scale的headroom；
- `canonical_vs_random_oracle_relative_gain`：有序连续partition是否优于任意partition；
- `stable_crossing`：同一scale pair是否在至少2/3 folds中对short/mid/long发生方向反转。

只有carrier、numeric invariants和三项problem gates同时通过，才返回Step 4-6设计paper method。

## 6. Code-Theory Consistency

### Intended theory

如果统一multi-horizon prediction既不能完全独立生成，也不应全局共享同一output subspace，那么局部future
domains应呈现可重复的risk crossing，并留下fixed-scale无法兑现的oracle headroom。

### Code realization

blockwise rank constraint直接改变future outputs共享latent directions的scope；random/shifted controls分别排除
generic grouping与fixed-boundary解释；closed-form fit降低neural optimization confound。

### Remaining proxy

PCA64 + linear reduced-rank operator只是低confound probe，不等于最终nonlinear decoder，也不能验证可学习router。

### Falsification boundary

若carrier skill或numeric invariant失败，诊断无效；若诊断有效但problem gates失败，只能否定当前linear RRR
证据。正结果也只证明问题值得进入Step 4-6，不证明paper method有效。
