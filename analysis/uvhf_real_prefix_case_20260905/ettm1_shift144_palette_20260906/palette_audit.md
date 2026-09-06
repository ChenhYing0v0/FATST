# Origin8896曲线配色修订

合同：仅为已认可样本建立GT > UVHF > H720 > H336 > H192 > H96视觉层次。Python复用单主图+完整prefix inset；数据、比例、范围、线宽、标记与预测不变。非新实验。

GT #000000；UVHF #C44E28（朱红）；H720 #28689A（深蓝）；H336 #8874A5（柔紫）；H192 #629C95（青绿）；H96 #BEA35C（赭金）。主比较使用暖/冷对照，短horizon采用柔和颜色。四种基线标记保留，避免仅凭颜色识别；不声称所有色觉条件下完全可区分。重要性是视觉设计意图，并非期刊规定某算法必须对应某颜色。

plot_single_panel.py新增可选figure_settings.palette映射，仅在当前目录配置；默认旧配色不变。所有线、图例、端点标签和UVHF指标强调同步使用新颜色。复现：python analysis/uvhf_real_prefix_case_20260905/plot_single_panel.py --zoom --output analysis/uvhf_real_prefix_case_20260905/ettm1_shift144_palette_20260906

图像检查完成，纯黑GT明确、朱红UVHF与蓝色H720区分清晰，短horizon仍可读。CSV与上一版逐字节一致。实际绘图数组、指标、标签和导出检查通过；183×135mm、可编辑SVG/PDF、PNG300dpi、TIFF1000dpi。静态13PASS/1WARN/0FAIL；静态宽度提示已由导出实测解决。未指定具体期刊，不把配色视为外审或所有期刊规范的保证。

样本选择与checkpoint证据边界沿用ettm1_shift144_20260906/reviewer_audit.md；复制的numeric_audit为同一数据的既有重放证据，本次没有重新训练或预测。
