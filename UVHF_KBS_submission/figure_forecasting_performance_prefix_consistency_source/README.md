# 预测性能与前缀一致性图：暂定冻结

用户于2026-09-06确认当前可视化为论文可用图，仅入submission，插入位置与正文表述待后续讨论。

正式图文件在上级目录：figure_forecasting_performance_prefix_consistency.pdf（矢量）、.svg（编辑）、.png（预览）、.tiff（高分辨率）。不预分配Figure编号，未改动TeX、参考文献或已有图文件。

数据：ETTm1 / LUFL / validation origin8896；由origin8752后移144步。单个定向选择案例，不是总体估计。TimeMixer为独立visualization checkpoint，不是Main-I表来源。用户接受的呈现状态不改变这些证据边界。完整来源说明见analysis/uvhf_real_prefix_case_20260905/ettm1_shift144_20260906/reviewer_audit.md。

当前文件是固定快照，不随analysis目录后续改图自动更新。未来变更应产生明确新修订。

复现（使用含numpy/pandas/matplotlib及Arial字体的Python环境）：
`python plot_refined.py --zoom --output .`
运行会在本目录生成uvhf_real_prefix_zoom.*，不覆盖上级冻结图。

源数据与绘图配置保留；numeric_audit记录此前独立checkpoint重放。figure_qa中的旧文件名/hash对应analysis源目录快照，本包当前校验以freeze_manifest.json为准。此目录为源文件与审计支持资料，之后正式上传期刊时按需要选择文件。
