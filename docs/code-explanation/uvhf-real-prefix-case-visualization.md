# UVHF 真实前缀对照图的数据路径

本轮为 validation-only case study，不变更模型代码、论文冻结内容或 test 结果。

## 数据与对齐

`analysis/uvhf_real_prefix_case_20260905/select_case.py` 读取四个 horizon 的 DLinear arrays：`pred/true [N,H,7]`、`history [N,720,7]`、`origin_index [N]`、train scaler。按最短可共同使用的 H720 验证窗口固定 N=2161。四个模型的 origin、history、对应 true prefixes 和 train scaler 必须逐元素相同。

UVHF 使用 frozen Main-I/II profile 的完整 validation export，channel 0/6 各 `[2161,720]`，按 `validation_window_index` 重排。`raw_forecast_origin = 8639 + validation_window_index`；future step 1 是下一 raw row。对齐 ground truth 的 scaled tolerance 为 5e-6。最后再检查选中案例与原始 CSV 的真实值。

## 统计量与选择

每个 origin-channel cell 计算四个完整 horizon 的 MSE，以及统一共同前缀 96 步上的 MSE。MSE 是对应 target indices 上 squared error 的均值；MAE 为 absolute error 均值。scaled 数值采用 train-split 标准差，raw 数值使用原始单位。

`mean_nchpd` 是六个 horizon pair 各自短 horizon overlap 内的 mean absolute difference，再对六个 pair 等权平均。`disagreement_percentile` 是该变量全部 origins 的 average-tie percentile rank。`min_gain` 是四 horizon 相对 MSE 改善的最小值。`uvhf_corr720` 是完整720步 Pearson correlation；`target_std_scaled` 是真实未来的 train-scale 标准差。

Eligibility 与 score 在 figure contract 中先于新评分冻结；`all_candidate_scores.csv` 保留全部候选及布尔门槛，`eligible_candidates.csv` 保留排序。`source_data.csv` 的 `step<=0` 是720步 history，`step>=1` 是720步 future；各 `dlinear_hH` 超出其定义长度时留空，不是缺失预测或填零。

## 证据限制与最小验证

CHPC 是统一轨迹的 prefix identity，不由较小 MSE 推导。数值 request check 单独从冻结 checkpoint 执行。旧 L96 baseline 比 UVHF 少看624步，因此本轮拒绝把它作为 matched-history accuracy evidence，补齐 L720 control。控制仍沿用原 source-audited DLinear 实现与验证早停协议，不冒充 Main-I published baseline。

验证由 Python syntax、JSON parse、真实数组的 origin/history/scaler/target assertions、原始 CSV 复核、数值重放与最终图像 QA 组成；不宣称这是一轮新的完整 test benchmark 或 mechanism-effectiveness gate。
