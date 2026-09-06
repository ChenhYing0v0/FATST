# ETTm1案例图确认前审计

日期：2026-09-06。作者侧审计结论：**HUFL / origin2295（review_case_1）通过，本轮推荐交付。** 保留此前ETTh1结果，不修改论文或专利正文。通过对象为selected validation illustration，不是完整benchmark或外部期刊接收。

## 数据与模型

ETTm1原始CSV本地/3090 SHA256相同，详见protocol。预测起点2017-07-19 21:30:00（raw row36854，zero-based）；future step1从下一15min观测开始，完整720步覆盖180h/7.5天。图中显示最后48步历史，全部720步future和完整96步inset，无平滑、移位、外推短H或删除后程。

UVHF沿用冻结ETTm1__h2_table5_capacity/seed2021，完整配置和SHA256固定。TimeMixer按native ETTm1 unify配置补齐H96/192/336/720四模型，全部完成10epochs，checkpoint分别由minimum validation mean-batch MSE选择，未访问test。配置、所有epoch learning curves与负向epoch结果保留在baseline_provenance.json/training_logs。硬件调度覆盖问题已记录，不影响选样规则。

## 候选筛选与失败归因

第一轮使用历史按UVHF视觉fidelity预筛的256个HUFL validation候选；181个通过四H/common96精度条件，235个通过后程fit条件，57个同时通过上述条件和visibility96>=.075。按visible_net、min_gain、visibility96降序，origin间隔至少96步，共得到4个待审图。该选择是显式post-hoc，不代表任意样本的平均表现。

| 候选origin | full R2 | tail R2 | last192 R2 | H720 MSE改善 | 审阅决定 |
|---|---:|---:|---:|---:|---|
| 6657 | 0.607 | 0.557 | 0.565 | 25.5% | 不推荐：后程漏掉较深谷值，不能只凭visible_net首位通过 |
| **2295** | **0.910** | **0.910** | **0.948** | **56.7%** | **推荐：主图峰谷/相位贴合，后程baseline偏移清楚，inset分歧保留** |
| 2392 | 0.902 | 0.886 | 0.892 | 54.5% | 次选：H192 MSE仅改善0.7%，四H优势不如2295稳健 |
| 10703 | 0.620 | 0.576 | 0.512 | 21.3% | 不推荐：后程深谷失配，重复此前ETTh1被用户指出的问题 |

failure_attribution：6657虽然在显示代理分数排名第一，但后程绝对贴合仍不足；代理不能代替完整图复审。回退到同一合格候选集内的第二个分离候选2295，不降低任何硬门槛，不改变模型参数或checkpoint。既有256池已找到满足要求的样本，未继续在全75607个cells中挑更大gain；全部UVHF重放缓存仅用于核验与备用。

## 选中案例的数值审计

| Horizon | UVHF MSE（train scale） | TimeMixer MSE（train scale） | MSE降低 |
|---|---:|---:|---:|
| 96 | 0.089582 | 0.129060 | 30.6% |
| 192 | 0.130973 | 0.223422 | 41.4% |
| 336 | 0.249590 | 0.369772 | 32.5% |
| 720 | 0.210883 | 0.486976 | 56.7% |

四H MAE也均较低，原尺度和train scale数值完整保存在selected_metrics.csv。full R2=.910106，tail R2=.910155，tail corr=.965231，tail amplitude ratio=.948124，tail bias/std=.145682；last192 R2=.948460。R2以各窗口GT均值为参照，不能将跨样本R2差作为总体性能估计。

common96 visibility=.084255，超过固定.075门槛；第67步四个TimeMixer预测max−min约4.97原始单位。mean cross-horizon disagreement约1.29单位，按六个pair各自完整overlap的平均绝对差再等权平均，不是confidence interval。UVHF独立请求96/192/336/720与H720前缀的max gap均为0；该一致性结论来自独立请求验证，不是仅把H720数组裁剪四次。

## 视觉复审

- 相较此前ETTh1每720步约30个日周期，ETTm1的720步覆盖7.5天；主图周期明显展开，GT与UVHF的峰谷、相位和TimeMixer的后程偏高容易区分。
- H336之后紫色TimeMixer轨迹存在持续的偏高，UVHF更接近GT；这里无需依赖inset才能辨认差异。
- inset保留完整96步，下降段与谷底回升段的四条horizon-specific预测明显不同；GT深谷与UVHF的贴合也可见。
- 仍存在失配：主图约192–336步UVHF低估部分峰值，未逐点跟随高频波动。图不宣称每一步都更好，不证明prefix一致性导致精度改善。

## 最终QA和交付

独立CPU checkpoint重放、history/GT原始CSV对齐、四H TimeMixer source核验、prefix identity检查通过；详见review_case_1/numeric_audit.json。图形QA验证全部inset点、完整主轨迹无遮挡、标签位置、指标复算、矢量文字及导出分辨率；详见review_case_1/figure_qa.json。183×135mm、SVG/PDF矢量、PNG300dpi、TIFF1000dpi。静态宽度检测WARN由实际尺寸核验解决。

本轮decision=selected_case_passed，current_step=9–10（可视化诊断）；paper-core effectiveness和mechanism attribution状态不变。

## 建议英文图注

**Fig. X | Consistent prefixes and improved forecasting on ETTm1.** Ground truth,
four horizon-specific TimeMixer forecasts and one UVHF trajectory are shown for
the same forecast origin in the HUFL series. Each step represents 15 minutes;
the full 720-step forecast covers 7.5 days. The inset enlarges the complete
96-step shared prefix. At step 67, TimeMixer forecasts differ by 4.97
original-scale units, while separate UVHF horizon requests yield identical
overlapping predictions. MSE reductions use each complete requested horizon.
Mean cross-horizon disagreement averages absolute differences over all six
horizon pairs and their full overlaps. This validation example was selected
for forecast fidelity, visible prefix disagreement and full-trajectory
readability; it illustrates model behavior rather than population performance.
