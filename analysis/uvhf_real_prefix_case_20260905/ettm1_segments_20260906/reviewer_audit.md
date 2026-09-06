# 分段性能与前缀可见性复审

2026-09-06。作者侧结论：推荐ETTm1 LUFL/origin8897（review_case_1），撤回此前HUFL/origin2295的推荐判断；旧文件保留。当前步骤=9–10 selected-case诊断，无新训练/test访问，冻结论文与专利不变。

## 旧图失败原因

旧图193–336步相对TimeMixer H720的MSE改善为−257.7%、MAE改善−111.9%，相对H336的MAE也略差。全H336 MSE改善掩盖局部劣势。旧prefix visibility=.0843，只有41.7%的时刻分歧超过prefix数据range的10%。之前的确认没有覆盖用户关心的分段条件，属于样本筛选与审阅设计缺陷，应回退选样，不能归咎于用户视觉判断。

## 新筛选和审阅

全10801×7=75607个validation cells，使用已有UVHF/四TimeMixer冻结预测。新条件先写protocol：四不重叠区间的MSE/MAE均优于该区间有效的所有baselines；prefix最强baseline MSE改善>=20%；visibility>=.12、持续分歧比例>=.5；full/tail/last192 R2>=.7/.65/.5，原有fit条件保持。零方差区间R2记NaN并排除，不以无限值通过。

347例通过，按prefix visibility与最小分段MSE改善排序，间隔96步取5例完整图审阅。HUFL3419分歧最强，但前缀深谷/突变贴合弱于LUFL8897；LUFL8801为次选，prefix拟合与分段优势弱于8897；LUFL9233和MUFL3488后程深谷/峰值失配较明显，不推荐。选择排名第二的8897，未调模型或为它放宽门槛。变量名按原CSV为LUFL（此前进度消息写作LULL是笔误）。

## 对两个问题的直接证据

- 193–336：UVHF MSE比TimeMixer H336低84.97%、比H720低64.41%；MAE分别低65.04%、42.30%。四个不重叠区间上，MSE和MAE均优于每个有效TimeMixer，不再只审累计H。
- prefix96：相对四baseline中MSE最低者，UVHF仍降低92.72%。visibility=.244813，是旧例约2.9倍；96/96个时刻四预测range超过该prefix全部六曲线range的10%。step48范围约.62原始单位，mean六pair完整overlap disagreement约.23。注意变量尺度不同，不能用原始range与HUFL旧例直接比大小。
- full/tail/last192 R2=.810907/.804392/.793728；四H MSE改善约96.7%/82.9%/90.9%/75.5%。该案例全程绝对fit低于旧例.91，但分段优势、前缀精度及可见分歧更符合本次要求。

## 视觉审计与限制

主图193–336段UVHF更能跟随GT谷底，蓝色H336明显偏高；后程紫色H720的谷底偏高也持续可见。inset中TimeMixer四条轨迹在前半段与谷底分开，UVHF贴近真实低谷，无需仅靠单点括号理解分歧。保持一个总图和完整前96步inset，全部720点无平滑、重采样、移位或删除。末尾约690–720及个别尖峰仍有失配；分段均优不等于每一步都优，不能宣称全时刻占优。

本轮仍为post-hoc selected validation illustration，不支持普遍性能或一致性导致精度收益的因果推断。全75607行结果和失败标志保留，model/checkpoint/seed不变。

## 核验与交付

独立CPU冻结checkpoint重放通过，四独立H请求prefix max gap均0；GT/history与原始CSV相符，四baseline曲线与native导出相符。导出图由Python/matplotlib生成，183×135mm PDF/SVG、300dpi PNG、1000dpi TIFF。技术QA复算四H及每个区间指标、检查inset逐点source一致、主曲线不被inset覆盖。最终图位于review_case_1，数字和图形检查记录见numeric_audit.json及figure_qa.json。
