# 新样本图注与审稿自评

## 建议英文图注

**Fig. X | Shared-prefix disagreement and unified forecasting on real data.**
Ground truth, four independently trained horizon-specific DLinear forecasts,
and one UVHF trajectory are shown for the LUFL variable of ETTh2. Both systems
receive the same 720-step history; the final 48 observed steps are displayed.
The inset repeats all six curves for the complete 96-step shared prefix,
without smoothing, vertical offsets, or omitted points. Its axes use the
original dataset scale. At step 51, DLinear predictions span 1.37 original-scale
units; this is a descriptive range, not an uncertainty interval. Separate
UVHF requests at H = 96, 192, 336 and 720 give identical overlapping values.
Percentages report MSE reductions over each complete requested horizon;
mean cross-horizon disagreement averages six horizon pairs over their full
overlapping prefixes. The validation example (origin 144, LUFL; last observed
time 1 July 2017 at 23:00) was selected post hoc from 15,127 origin–variable
pairs for visible shared-prefix disagreement subject to lower UVHF MSE at all
four horizons and on the shared prefix. This selected illustration does not
estimate population performance or establish a causal accuracy benefit of
prefix consistency. Source data, complete candidate scores, and replay audits
accompany the figure.

## 筛选经过与失败归因

1. 旧OT/origin1239按完整overlap CHPD、MSE改善和corr排序，未充分针对inset的共同96步，属于sample-selection criterion与展示窗口不匹配。
2. 第一轮在已有HUFL/OT的4322 cells中，949个满足accuracy条件。首选OT/origin1048的前缀相对分离宽度提高，但H192仅2.0% MSE改善，且四线仍接近；前5个时间分离候选审核后暂不采用。结果完整保存在round1。
3. 扩展到全部7变量后，15127 cells中2378个满足相同accuracy条件。按未改变的visibility96规则首选LUFL/origin144；对前5个时间分离候选再次人工审视后保留首选。
4. 未更换baseline、未训练或访问test。全变量UVHF inference只用历史输入和全零future占位。新样本独立request check的四H prefix max gap均为0。

## 为什么比旧样本更适合

`visibility96`为四条DLinear逐时刻max-min的96步均值，除以六条曲线在共同96步的整体数据max-min。旧例5.52%，新例14.18%，约2.57倍。这是相对于各自数据范围的可见性比较，不是不同变量原始单位上的绝对分歧比较。

`persistent_fraction96`采用同一数据范围10%的描述性阈值；旧例0/96步、新例81/96步。该阈值仅用于视觉筛选，不是统计显著性或正式机制metric。`q25_range96`为0.71985原尺度单位，说明差异不限于单个尖峰。

新例四H MSE降低26.1%、75.5%、78.1%、70.7%；四H MAE也均较低。原尺度mean six-pair CHPD约0.46。新旧例变量不同，不能拿0.46与旧0.70°C直接比较。

## Reviewer自审及限制

- 前96步可见短H与长H预测的分离，尤其前段和约44–61步；完整720步图能显示baseline长期偏离GT和UVHF的误差语境。并非四条线在每一时刻都大幅分开。
- 最初自动标注最大range落在step6；为避免用起始局部过渡主导解读，改标共同前缀内部24–80步中range最大处step51（1.374872 units），并保留完整96步。该标注位置是post-hoc display choice，不影响选择分数或误差。
- 初版连接线经过H192标注与inset ylabel，改接窗口左下角；H720标签移到空白处。原始曲线不遮挡，坐标随LUFL真实范围调整；颜色、线宽、主图+inset结构沿用原版。
- **自审通过作为前缀不一致的真实案例图；比旧例更适合当前展示目的。** 强度属于持续可见的差异，不能称为四条完全分离的轨迹，不能以单例替代全矩阵效果证据。若要求示意图式的大幅分叉，当前已检索DLinear样本不能保证这种视觉效果，不能靠人为偏移实现。
- UVHF仍有未捕获的波动与长期偏差，完整保留；prefix identity不等于逐点更准确，也不证明精度提升由CHPC因果产生。
- 这是作者侧自审，不是外部reviewer接受或期刊质量保证。

## 重现顺序

1. `replay_all_channels.py`：使用r2026-fsa Python，CPU重放2161×720×7的冻结UVHF prediction。
2. `select_visible_case.py`：使用numpy/pandas计算全部候选、筛选并导出source_data、metrics与selection audit。
3. 父目录`check_prefix_requests.py --selection-dir analysis/uvhf_real_prefix_case_20260905/reselection`：使用r2026-fsa验证所选origin的独立请求。
4. 父目录`plot_single_panel.py --zoom --output analysis/uvhf_real_prefix_case_20260905/reselection`：沿用版式，按`figure_settings.json`使用LUFL原尺度、显示范围与step51标注。

首轮历史分数使用既有GPU channel0/6 arrays；第二轮使用全变量CPU replay，其与旧GPU数组数值误差另记QA。图形的数值来源始终为source_data.csv。完整raw arrays/checkpoints沿用父目录ignore边界，不重复提交。
