# ETTm1分段及前缀复审

用户否决origin2295：193–336步局部劣势，inset优势/分歧不足。旧图保存但撤回推荐结论。current_step=9→样本选择回退，非模型方向否定。本轮用已有10801×7全量validation预测，无训练/test访问；Python nature-figure，沿用总图+96步inset。

原有四H MSE/common96优势、full/tail/last192 fit gate保持。新增先验显示筛选：所有不相交区间1–96、97–192、193–336、337–720上，UVHF MSE和MAE均低于在该区间有效的每一个TimeMixer H；共同96前缀对最强baseline MSE至少降低20%；visibility96>=.12；前缀四baseline range超过全部六曲线prefix range的10%的时刻比例>=.5。full R2>=.7、tail R2>=.65、last192 R2>=.5用于保留比旧硬gate更高的绝对贴合。若空集，公开缺少哪项，不静默降门槛。

排名：prefix visibility96降序，最小分段MSE改善降序；同变量origin间隔至少96，选最多5例全图审阅。保留75607行全量结果和失败标志。上述均post-hoc案例显示诊断，不作population估计。

结果：full/tail强化fit通过5653，分段通过10856，prefix通过7603，交集347。审阅5个分离候选后选LUFL8897；其四区间全部有效baseline比较MSE/MAE均优，193–336对H336/H720 MSE分别低84.97%/64.41%，prefix最强baseline MSE改善92.72%。独立请求gap0、source/metric/export QA通过。decision=selected_case_passed；仍保留尖峰和末段失配，不声称逐时刻占优。失败归因和旧图推荐撤回见reviewer_audit.md。
