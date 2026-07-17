# StageC SIFF/MCCA Step 9 分析代码说明

## 1. 模块目的

`scripts/analyze_stage_c_post_pcc_step7b_deep_dive.py`读取已经通过protocol audit的55个new validation runs、
25个matched references与Step7B aggregate gate，不重新训练模型，也不读取test。

其输出用于回答：

1. architecture、training与joint effect分别落在哪些datasets/horizon regions；
2. ETTm2 negative是否是全面fit失败，还是short-prefix局部pathology；
3. exact SIFF/MCCA应关闭到什么边界。

## 2. 输入与路径解析

- `--raw-root`：11个new arms的lightweight artifacts；
- `--pcsd-reference-root`：A6、PCSD direct与dense matched references；
- `--pcc-reference-root`：equal-skill与PCC transport references；
- `--run-summary`：Step7B analyzer确认过的aggregate table；
- `--gate`：frozen gate result；
- `--seed`：本阶段固定为2021。

每个run从
`<root>/<arm>/<dataset>/h720_full/seed<seed>/metrics_by_target_horizon.csv`
读取$H=1,\ldots,720$的prefix MSE/MAE；new arms另读取`training_log.csv`。

## 3. Forward artifact flow

```text
metrics_by_target_horizon.csv
  -> metric_curve[720]
  -> arm_scoreboard.csv
  -> paired comparison
  -> horizon_bin_effects.csv

training_log.csv
  -> best/last H720 validation
  -> finite and early-stop audit
  -> training_stability.csv

run_summary.csv + gate.json
  -> architecture/MCCA/joint main effects
  -> leave-one-dataset-out localization
  -> step9_attribution.json
```

## 4. 输出字段定义

### 4.1 `arm_scoreboard.csv`

- `dense_mse_auc` / `dense_mae_auc`：720个prefix metrics的算术平均；
- `gain_over_a6_percent`：$100(1-L_{\mathrm{arm}}/L_{\mathrm{A6}})$；
- `h1_mse`、`h48_mse`、`h96_mse`、`h192_mse`、`h336_mse`、`h720_mse`：
  对应CSV中的exact prefix metric。

### 4.2 `horizon_bin_effects.csv`

- `horizon_bin`：requested prefix horizon区间，不是target-step block；
- `mse_gain_percent`：先在区间内平均candidate/reference prefix MSE，再计算relative gain；
- `dataset=macro`：先得到五个dataset gain，再做未加权平均。

区间固定为1–48、49–96、97–192、193–336与337–720。

### 4.3 `training_stability.csv`

- `best_h720_val_mse`：training log内最小`val_mean_mse`；
- `last_h720_val_mse`：early-stop触发时最后一轮的validation MSE；
- `last_over_best_fraction`：`last / best - 1`；
- `best_epoch`：trainer记录的`best_epoch_so_far`；
- `all_finite`：所有epoch-level H720 validation MSE均为有限值。

该表只能诊断H720 checkpoint path，不能替代dense-horizon evaluation。

### 4.4 `leave_one_dataset_out.csv`

逐次移除一个dataset后重算macro gain与wins，仅用于判断单一dataset是否主导aggregate；任何leave-one-out结果均
不得修改冻结的five-dataset gate。

### 4.5 `step9_attribution.json`

除formal gate外，记录：

- architecture/MCCA/joint macro effect；
- joint移除ETTm2后的定位统计；
- ETTm2 SIFF-vs-PCSD H1/H720 paired gain；
- dense AUC隐含target weights；
- coupling training target measure/error norm、checkpoint measure与screening measure的显式区分；
- 五类failure attribution与research rollback。

## 5. Dense AUC target-weight identity

脚本的`harmonic_target_weights()`计算

$$
w_t=\frac1T\sum_{H=t}^{T}\frac1H.
$$

它来自

$$
\frac1T\sum_{H=1}^{T}\frac1H\sum_{t=1}^{H}e_t
=\sum_{t=1}^{T}w_t e_t.
$$

该函数只用于解释evaluation measure，不改变任何模型输出或结果。

## 6. Code-theory consistency

- intended theory：从完整factorial matrix定位architecture、training与measure interaction；
- code realization：所有comparison均来自同dataset、same-seed paired artifacts，horizon bins在gate后固定分析；
- proxy boundary：single-seed validation只能定位failure，不能证明new method effectiveness；
- falsification：若输入缺少任意dense horizon、出现non-finite metric或summary row不完整，脚本直接失败；
- prohibited claim：leave-one-out、H1 pathology或harmonic identity均不能自行升级为Contribution 2。
