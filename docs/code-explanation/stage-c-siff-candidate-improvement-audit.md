# StageC SIFF candidate improvement audit 代码说明

## 功能边界

`scripts/analyze_stage_c_siff_candidate_improvement.py`只读取Step9已经保存的diagnostic artifacts，不加载模型、
不训练、不重新访问remote test split。它回答两个诊断问题：当前policy是否把权重分给更熟练的arm，以及当前
convex fusion geometry是否限制可实现性能。

## 输入 tensor 与计算流

对`SIFF_EQUAL`和`SIFF_INDEPENDENT_EQUAL`的五个datasets，脚本读取：

- `arm_row_bin_mse [N,8,5]`：每条series/channel row、八个future bins、五个scope arms的MSE；
- `policy_row_bin_usage [N,8,5]`：相同cell上的平均policy weights；
- `probe_arms [256,5,720]`：256条probe rows的五个完整预测；
- `probe_fused [256,720]`与`probe_targets [256,720]`。

Routing路径在最后一个arm维度上比较argmax weight与argmin loss，并计算centered cosine alignment。这里的
policy-weighted arm MSE是allocation proxy；它不等于先混合预测再计算的fused MSE。

Fusion路径把256 rows切成两个固定fold。训练半的`[128,5,720]`通过moveaxis与reshape变成
`[128*720,5]`，用SLSQP分别拟合nonnegative simplex weights和`[-1,1]` bounded affine weights；再在另一半
`[128*720,5]`上计算MSE，随后交换fold。weights始终sum-to-one，因此只改变跨arm组合，不引入intercept。

## 输出定义

- `routing_calibration.csv`：逐arm、逐dataset的skill alignment与allocation statistics；
- `probe_fusion_capacity.csv`：逐arm、逐dataset、逐fold的learned/uniform/convex/affine/best-fixed MSE及weights；
- `summary.json`：macro结果、完整统计定义与diagnostic-only边界。

## Code-theory consistency

理论问题是“现有conditional fusion是否缺少relative arm competence信息”。脚本以实际arm loss与policy usage的
对齐度直接测试该问题，并用cross-fit static fusion判断当前learned policy是否低于一个更弱的readout基准。

它仍只是proxy：probe只有每dataset 256 rows；static stacking使用test-derived probe；row-bin usage不是逐target
policy；任何离线拟合weights都不能部署或作为新模型效果。可证伪证据是policy已高对齐且learned fusion系统性优于
cross-fit static baselines；本次结果与该反证相反，但仍需新的end-to-end candidate验证改进机制。
