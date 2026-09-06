# 用户指定配色预览

沿用origin8896数据与布局，按用户给定HEX映射：GT #000000，UVHF #B22222，TimeMixer720 #1E5AA8，336 #2E7D32，192 #D27D2D，96 #9E9E9E。仅替换figure_settings.palette，所有数据CSV与前版逐字节相同。旧图保留。

复现：python analysis/uvhf_real_prefix_case_20260905/plot_single_panel.py --zoom --output analysis/uvhf_real_prefix_case_20260905/ettm1_shift144_reference_palette_20260906

Python渲染后已目视检查主图与完整96步inset。check_figure_exports.py完成实际数组、指标、标签边界、无主轨迹遮挡、183×135mm、可编辑SVG/PDF与PNG300dpi/TIFF1000dpi检查。数值重放记录沿用相同数据的既有numeric_audit，不是新实验。

参考图中的“100%兼容黑白打印”不作为已验证结论；颜色在灰度或不同色觉条件下可能接近，现有标记辅助辨识。当前交付为用户指定彩色效果预览。样本证据边界沿用ettm1_shift144_20260906/reviewer_audit.md。
