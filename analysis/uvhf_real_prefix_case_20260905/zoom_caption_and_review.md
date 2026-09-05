# 总图内嵌前缀放大：契约、图注与审核

## 本轮契约

用户要求放大前部，以凸显prefix inconsistency，继续保留一张大总图。
沿用Python与现有single-panel绘图代码（structural adaptation），新增可选`--zoom`。
核心结论：在已选validation案例中，四个horizon-specific DLinear在相同未来时刻给出不同预测；UVHF请求共享统一前缀，且该例四个完整H的MSE较低。
图型为quantitative comparison：一个主图保留完整720步轨迹，一个嵌入窗口重复全部共同96步，既保留长期误差语境，也使局部差异可读。

字段直接映射`source_data.csv`的`step`、`ground_truth`、`uvhf`、`dlinear_h{96,192,336,720}`；温度单位°C。放大窗口仅重复step 1–96的96×6个原始预测/真实值；不平滑、不偏移、不重采样。两轴均线性；窗口范围x=[1,96]、y=[26,53]，完整包含六条前缀的全部值。原样本、checkpoint、selection与指标不变。
主图矩形标明完全相同的范围，虚线连接放大窗口；窗口置于真实曲线上方，不能遮蔽主图数据。新增子Axes是局部放大窗，不将图拆成并列多面板。尺寸183×135mm，editable PDF/SVG、300dpi PNG、1000dpi TIFF。

## 建议英文图注

**Fig. X | Unified forecasts and horizon-specific prefix disagreement on real data.**
Ground truth, four independently trained horizon-specific DLinear forecasts,
and one UVHF trajectory are shown for ETTh2 oil temperature using the same
720-step input history; only the final 48 observed steps are displayed.
The outlined region is enlarged in the inset, showing all 96 shared future
steps and all six original curves on explicit linear temperature axes.
At step 68, the four DLinear predictions span 2.13°C; this range is descriptive,
not an uncertainty interval. UVHF prefix identity is supported by separate
96-, 192-, 336- and 720-step request checks. MSE reductions are evaluated over
each complete requested horizon; mean cross-horizon disagreement averages
absolute differences over all six horizon pairs and their full overlaps.
This previously selected validation example (origin 1239, OT; forecast origin
16 August 2017 at 14:00) illustrates the behavior, not population performance
or a causal effect of prefix consistency on accuracy. No curves are smoothed
or displaced. Source data and selection/checkpoint audits accompany the figure.

## Reviewer自审

- 失败原因定位：上一版720步横轴压缩共同96步，是展现尺度问题；无需重新选择sample、dataset或baseline。
- 初版inset中H336主图标注靠近inset的30°C刻度；先调整高度仍靠近temperature label，最终将H336标签左移24步对应距离后复核。
- 最终六条原始曲线在窗口中完整保留，step68以竖虚线和原值range bracket定位；图例与主图共用颜色和marker，UVHF与GT均保留可见误差。
- 自动preflight仅作为代码检查；额外核验窗口数组与CSV逐元素相等、原始source/metrics哈希不变、窗口不遮挡主图曲线、SVG文本与尺寸、PNG/TIFF分辨率及PDF字体嵌入。
- 审核结论：**通过作为selected-case解释性图的自审**。局部前缀分歧比无放大版本清晰，同时保留完整轨迹比较。该结论不是外部审稿通过，也不能替代完整test benchmark或证明CHPC导致精度提升。

## 重现

`python analysis/uvhf_real_prefix_case_20260905/plot_single_panel.py --zoom`

输出`uvhf_real_prefix_zoom.{pdf,svg,png,tiff}`；无参数仍生成历史单Axes版本。
历史`single_panel_qa.json`记录上一commit的无放大版本；本轮以`zoom_qa.json`为准。
