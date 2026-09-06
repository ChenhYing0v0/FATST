# 图形细节六项修订

按用户要求：中文总标题“预测性能与前缀一致性”（SimHei字体）；副标题ETTm1 · LUFL；UVHF图例添加同色实心圆点；主图的分歧竖线和长文本移除，在完整prefix inset保留极差竖线并以“Prefix inconsistency / Δ = 0.73”简写引线标注；删除mean cross-horizon disagreement整行；删除最底部统计说明。

删除页脚后主轴下边距从0.235调整到0.16，MSE行移至0.055，保留183×135mm。没有改变数据范围、sample、标尺数值、曲线配色、预测与指标。采用Python本地副本plot_refined.py，旧版本不覆盖。

数据CSV与reference_palette版逐字节相同。复用check_figure_exports.py并将其绘图模块引用定向至当前plot_refined.py：实际96点inset数据、全部指标、主图不被inset遮挡、标签边界、183×135mm、字体和分辨率检查通过。静态预检使用当前绘图源。视觉检查中文无缺字，圆点可见，分歧标注位于inset右上空白区域，两条脚注均已删除。

数据仍为origin8896的定向选择validation案例，TimeMixer非Main-I checkpoint，选择限制和统计定义保留在此前ettm1_shift144_20260906/reviewer_audit.md以及随附数据中。图面按用户要求简化不改变证据等级；正式论文caption仍需说明案例来源与指标定义。

复现：python analysis/uvhf_real_prefix_case_20260905/ettm1_detail_revision_20260906/plot_refined.py --zoom --output analysis/uvhf_real_prefix_case_20260905/ettm1_detail_revision_20260906

后续按用户确认，将inset标注改为两行Prefix inconsistency和Δ = 0.73。Δ是第69步四个TimeMixer预测的极差，不是完整prefix最大分歧。已重新检查导出与文字位置。

最新修订：总标题改为Forecasting Performance and Prefix Consistency；恢复Mean cross-horizon disagreement整行。inset标注采用深蓝灰粗体、白底浅边框与加粗极差括线；上界扩至4.05为标注留白，不裁剪数据。主轴下边距0.195，MSE与分歧页脚分别0.087/0.038。首轮标注框接近曲线，增加顶部留白后再次目视检查并通过导出检查。此为内部呈现审计，非期刊接受保证。
