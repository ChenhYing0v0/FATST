# 共同96步可见分歧：样本重选协议

本轮由用户明确要求重新选择更明显的真实样本，是在旧图审核失败后的post-hoc illustrative selection，不是预注册benchmark或随机代表性样本。当前Step 9：重新分析已有validation artifacts。仅选择可视化案例，不改变方法或performance gate。

失败归因：旧规则以各pair完整overlap CHPD percentile、MSE gain与corr加权，不能保证共同96步窗口中曲线持续明显分开。返回可视化样本选择步骤。

第一轮范围沿用ETTh2的2161 origins × HUFL/OT，共4322 cells；L720 DLinear四独立checkpoint和冻结UVHF均不变。
保留accuracy条件：UVHF四个完整H MSE全部低于对应DLinear，且共同96步MSE低于四个DLinear各自前96步。原有target_std_scaled>=0.25保留；不再要求完整overlap CHPD percentile。

新的可见性量仅用共同96步真实预测计算：
- `mean_range96`：四条DLinear在每一步的max-min，再对96步均值。
- `q25_range96`：pointwise range的25th percentile，避免只奖励单个尖峰。
- `visibility96`：mean_range96除以六条曲线（GT、UVHF、四baseline）共同96步的整体max-min；跨变量可比，衡量真实绘图区间中的分离宽度。
- `persistent_fraction96`：range超过该整体max-min的10%的步数占比。
- `mean_sorted_gap96`：每步四预测排序后相邻三间距均值，再对96步均值；它等于mean_range96/3，仅解释四线可见性，不当作独立证据。

在accuracy eligible内先按visibility96降序，再按persistent_fraction96、min_gain降序确定首选，并保留所有候选表。审核前5个时间上相距至少96步的候选，避免五个相邻窗口重复；若首选视觉审核不通过，记录具体原因再处理。若全部不足，才扩展已有UVHF其他变量或数据集/TimeMixer artifact。所有扩展需记录，不能隐藏失败。

最终图保留完整720步和共同96步放大，不平滑、偏移或夸大数值，坐标随真实数据范围确定。图注披露挑选过程，结论只用于实例展示。

## 第一轮失败与范围扩展

首选origin1048/OT的visibility96=.11567（旧例.05516），mean range=2.56949°C（旧例1.27534°C），但四条线仍近似平行贴近，H192仅2.0% gain。前5个分离origin的96步preview已检查；未达到用户希望的直观分歧。归因为候选变量范围偏窄，暂不通过最终视觉gate。
下一轮扩展ETTh2全部7变量，共15127 cells；从同一UVHF checkpoint进行CPU validation-only replay，不训练、不访问test。选择规则与accuracy条件保持不变。原2变量候选结果保留在`round1/`。
