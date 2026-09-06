# UVHF 优势优先的 ETTm1 案例审计

## 结论与用途

推荐 `review_case_0`：ETTm1、LUFL、validation origin 8752。用途是同一真实未来窗口上精度与跨 horizon 前缀分歧的说明性证据。当前使用经过定向选择的validation窗口，不是总体效果估计，也不是Main-I的checkpoint重放。不能从该图推断前缀一致性因果性地导致精度提升。

此前MUFL3738数值优势存在，但全程视觉差距过弱。失败归因是样本选择目标过度偏向“温和差距”，而不是绘图缺陷或模型机制失败。本轮提高独立分段增益要求，放宽baseline能力下限，详细见protocol.md及select_render.py；不更换任何checkpoint。

## 选择范围与失败记录

从10801个validation origin × 7变量，共75607个已有候选中筛出46个。eligible.csv记录全部通过候选。按最小分段MSE改善与前缀可见性的乘积排序，同变量96步内不重复取候选，审阅前三个：

| 候选 | 数值与图像判断 | 决定 |
| --- | --- | --- |
| LUFL8752 | 重复低谷在全程可区分，UVHF尾程保持拟合，前缀既有分歧又有准确性优势 | 推荐 |
| HUFL5381 | TimeMixer分歧清晰，但UVHF对多个峰谷的幅值也有明显低估 | 不选 |
| LUFL8932 | 优势存在，但中段和末端的局部谷形更不均匀 | 不选 |

LUFL8752和历史LUFL8897窗口高度重叠，属于同一时间段的不同forecast origin，不能算独立新证据。新的上下文对应新的原始模型预测，未移动或拼接已有曲线。

## 精度与baseline能力

| 完整预测长度 | UVHF MSE降低 | TimeMixer R2 |
| --- | ---: | ---: |
| 96 | 85.53% | 0.375 |
| 192 | 76.48% | 0.513 |
| 336 | 81.83% | 0.329 |
| 720 | 61.55% | 0.503 |

UVHF完整720步R2=0.809，337–720步R2=0.751，最后192步R2=0.755；相应TimeMixer最后192步R2=0.487。TimeMixer有周期预测能力，但短期相对误差差距仍大；不能将这版称为无选择偏差或差距居中的样本。

1–96、97–192、193–336、337–720四个互不重叠区间，对所有覆盖该区间的TimeMixer模型逐一比较，共10项，UVHF的最小MSE改善54.13%，最小MAE改善33.38%。另检查529–720，MSE及MAE均未反转。segment_metrics.csv保留全部11项明细。

对于主图持续到720的TimeMixer曲线，四区间MSE分别改善83.46%、60.83%、62.26%、54.13%。因此全程优势并非仅由开头累计指标贡献。

## 前缀与可视性

完整1–96 inset没有裁剪失败区间。TimeMixer在相同输入上下文、相同未来位置上给出不同值；第75步四预测极差约0.51原始单位。前96步平均四预测极差除以GT范围为0.1575；83/96步的极差超过该范围的10%。六对horizon在各完整重叠区间平均绝对差再取平均，约0.16原始单位；不是置信区间。

UVHF以四个target_prefix独立请求重放，公共前缀与完整720预测的最大绝对差全部为0，见numeric_audit.json。这个独立检查区别于直接截取同一数组。

图像审阅：约80–96、175–192、270–305及后半段多次低谷，UVHF更贴近GT，TimeMixer偏高；不是依靠模糊的重叠曲线声称优势。开头约75步后，UVHF跟随下降而TimeMixer多条曲线分离后回升，精度和前缀问题同时可见。真实尖峰以及约400–435步的UVHF失误保持可见，不主张逐点胜出。

## 完整性、可复核性与导出

select_render.py仅合并已有指标筛选，调用既有build_cases.py导出原始CSV并复用plot_single_panel.py。GT对应原始数据行34560+8752开始的720步，输入历史来源前720行；训练集scaler逆变换。所有720点和完整96点均保留，无平滑或错位。source_data.csv列由history、ground_truth、uvhf以及四个timemixer_hH组成；超出各H区域留空。

MSE/MAE基于同变量原始尺度计算，gain=1-error_UVHF/error_TimeMixer；R2=1-MSE/GT方差。没有种子/重复实验置信区间，本图n=1选定窗口、单变量，seed2021。训练checkpoint与baseline溯源沿用ettm1_20260906/baseline_provenance.json。Main-I TimeMixer是文献表格值，本图为独立原生baseline实验。

check_final_case.py在CPU独立加载冻结UVHF checkpoint，校验SHA256、GT和history对齐、cache对重放误差、四个baseline缓存与CSV一致。audit_selected.py重算分段与R2；复用check_figure_exports.py核对实际绘图数组、无inset遮挡主轨迹、标签边界、SVG/PDF字体和导出尺寸。静态预检13 PASS、1 WARN、0 FAIL；宽度静态解析警告由实际183×135mm导出检查解决。SVG/PDF可编辑，PNG300dpi，TIFF1000dpi。

审计判断：作为“性能优势+前缀分歧”的选定案例可通过本轮内部审阅；保留上述选择偏差、同时间区间复用和非Main-I来源边界，不能称为期刊外审通过。
