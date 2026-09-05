# 完整轨迹可读性再评估（2026-09-06）

current_step=9–10，scope=已有validation预测的selected-case展示评估。当前MUFL/origin947结果按用户要求保留，文件hash见preserved_result.json；不覆盖、不重训练、不改变checkpoint、不访问新的test结果。

problem：720步中的约24步周期使曲线密集重叠，主图的精度优势弱于96步inset。
idea：区分周期密度问题与误差可见性问题，使用已有TimeMixer全部15127个ETTh1 origin–variable cells；先保留原accuracy、fidelity、prefix visibility硬门槛，再比较262个合格样本。不降低先前门槛。

新增统计在排序前定义，均由720步原始GT、UVHF、TimeMixer H720计算。R为全部六条未来曲线的max−min（各短H仅取其有效区间），坐标没有平滑或变换：
- dominant_period：GT去均值FFT最大非零频率对应的步数，作频谱描述，不保证严格周期。
- prominent_peaks：GT上prominence>=0.3R且distance>=8的峰个数，用于近似主图的密集程度；不更改绘图数据。
- high_frequency_share：GT谱中period<48步的power比例（不含DC），越大越多快速变化。
- normalized_mae_advantage：mean(abs(TimeMixer−GT)−abs(UVHF−GT))/R，是净绝对误差优势，负值也保留。
- visible_win_fraction：|UVHF−GT|<=0.08R且|TimeMixer−GT|−|UVHF−GT|>=0.04R的步数占比。visible_loss_fraction交换两模型定义。visible_net=win−loss。阈值为本轮回顾性显示诊断，不是新性能指标或显著性检验。
- tail_visible_net：同样阈值在337–720步计算；使用同一完整R，不另行放大tail。

两个候选方向分开判断：
1. less_dense：prominent_peaks<=保留样本的75%，且visible_net、gain_h720不低于保留样本；满足原硬门槛后才可被认为兼顾密度与优势。
2. clearer_same_density：visible_net至少增加0.05，gain_h720至少增加0.10，full/tail R2不比保留样本下降超过0.05；同时报告最后192步表现，最后人工审图判断，不能仅以分数确认。

分别按峰数升序、visible_net降序、tail_visible_net降序取候选；审阅候选与保留版本使用相同183×135mm总图/inset模板及相对留白比例，绘制全部原始点。最终只做保留/候选建议，不自动覆盖用户接受图。

narrative_gate：仅说明selected-case的可读性；不能暗示population性能、样本代表性或prefix一致性与精度改善的因果关系。若两个方向都不能明显改善，报告当前预测pool内未找到，而不是无限筛选后宣称普遍优势。

## 本轮评估结果

existence_evidence：262个合格cases的dominant period均24步，支持周期密集是当前候选池的共同限制。less_dense统计筛选4个通过、clearer_same_density筛选25个通过；按同变量origin至少间隔96步规则取1个少峰、2个更清楚候选审阅。少峰HUFL/1427视觉不通过替换判断；HUFL/869推荐作主图优势优先备选，HUFL/655为次选。

effectiveness_gate：三候选均通过既有fidelity与数值重放，独立prefix请求gap=0；技术导出QA通过。decision：原图保留，869只作为备选，不自动替换。failure_attribution：当前pool无法提供更长主导周期；prominence峰数降低不等同周期变长。若要求真正稀疏的720步轨迹，rollback到dataset/variable选择并补齐对应真实四H baseline exports；不降低当前门槛或平滑绘图。详细四例比较和限制见reviewer_assessment.md。
