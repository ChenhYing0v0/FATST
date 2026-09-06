# 双方合理拟合条件下的案例复审

2026-09-06。推荐ETTm1/MUFL origin3738（review_case_3），替代极端优势导向的LUFL8897。保留历次结果，不删除负面审计；不新训练、不访问test、不变更checkpoint或稿件。

## 选择规则及其边界

在既有75607个cells基础上，要求TimeMixer每个完整H R2>=.5、四common96 R2>=.3、H720后程和last192 R2>=.4；四H MSE改善均不超过60%，共同96改善不超过70%。保留UVHF full/tail/last192 R2>=.7/.65/.5及四不相交区间所有有效baseline的MSE/MAE正改善。

与上一轮相比，prefix visibility门槛从.12调至.10、最小common96 MSE改善从20%调至10%，持续分歧>=.5保留。此修订在评分前写入protocol，目的是响应用户的平衡要求；不能声称新图通过上轮更强的prefix门槛。

8例通过，不按最大差距排序，按四H gain、visibility、双方H720 R2的合格集百分位到.5的L1距离排名。首批分离候选3760、813、5381；3760虽最居中，但last192 UVHF R2=.670低于TimeMixer .736，因此追加检查下一未审、last192无反转的合格例3738（距离1.5，与813并列，和3760相距22步，不把它算独立证据）。该追加规则已写回脚本，所有8例保留。

## TimeMixer并未失效

选中3738的四H TimeMixer R2=.681646/.777647/.804138/.660357；四common96 R2=.681646/.674099/.748416/.515798。它能跟随主要周期、相位与峰谷，不是近乎平线或明显发散的对照。

同一MUFL变量全部10801个validation origins，TimeMixer H720 R2小于等于本例的比例为69.29%；按这一指标，该例高于同变量中位表现，不是TimeMixer最差尾部样本。该百分位不能推广到每个H、每个指标或所有模型表现，也不能证明完全消除了选择偏差。

## 精度与局部要求

| H | UVHF相对TimeMixer MSE降低 |
|---|---:|
|96|36.9%|
|192|37.6%|
|336|24.3%|
|720|28.1%|

UVHF full/tail/last192 R2=.755645/.650942/.666351；TimeMixer last192 R2=.653866。193–336段对H336/H720的MSE降低23.63%/13.23%，MAE降低14.75%/6.27%；四个不重叠区间均保留正MSE/MAE优势，但不是每个时刻都优。

prefix visibility=.108607，52.08%时刻分歧超过prefix六曲线range的10%。相对最强common96 baseline的MSE降低20.15%。第30步四TimeMixer跨度5.61原始单位，mean六pair完整overlap disagreement约1.52单位；UVHF独立H请求prefix max gap=0。

## 视觉审计与取舍

3738：两模型总体周期预测都合理；inset前部紫色H720的峰值较高，四个H在下降段和谷底附近分开，UVHF更接近真实前部平台；后部尖锐突变则两者都未准确跟随。完整主图双方曲线更接近，UVHF优势温和，符合本轮不夸大差距的要求。

3760：last192反转，不推荐。813：baseline能力很好但主图重叠更严重、后程深谷失配明显，不能充分兼顾此前可读性要求。5381：部分峰值持续低估、prefix深谷失配，而且H96 MSE差距接近60%上限，平衡程度弱于3738。

新图的分歧显著性和绝对贴合度并非历次最优；末尾下降、部分深谷与高频尖峰仍不准确。结论为“更平衡的selected validation illustration”，不宣称代表性随机样本，更不保证审稿人不会质疑选样。

## 来源与核验

TimeMixer沿用本轮独立训练的L96可视化对照，非产生Main-I引用数值的checkpoint；不得把这张图表述为Main-I原始run的预测。独立checkpoint重放、原始GT/history对齐、四H导出对应及prefix请求检查通过。source/指标/导出检查见review_case_3/numeric_audit.json、figure_qa.json及segment_metrics.csv。全720步和完整96步inset无平滑、移位、删点；183×135mm，15min/step，PDF/SVG/PNG/TIFF均保留。

current_step=9–10 case诊断；decision=推荐平衡案例3738；failure_attribution=以最大差距/分歧选样会偏向对照弱例，回退双方skill gate与合格集中间位置选择，未重训模型。总体准确率与泛化仍需完整独立实验，单例不能补足该证据。
