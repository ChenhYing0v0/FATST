# ETTm1双方合理拟合的案例选择

用户指出LUFL8897过于偏向UVHF，回退样本选择。原图保留，停止以分歧/gain最大排序。current_step=9–10诊断，checkpoint不变、新训练/test访问=0。

在评分前固定本轮规则：继续使用先前75607全量审计，不改变模型。保留四不相交区间所有有效TimeMixer对照的MSE/MAE正改善，以及UVHF full/tail/last192 fit条件。新增TimeMixer每个完整H R2>=.5、每个common96 R2>=.3、H720 tail/last192 R2>=.4；四H relative MSE gain均<=.6，避免选择baseline明显失效案例。prefix保留visibility>=.10、persistent_fraction>=.5、min_prefix_gain>=.10，但四baseline common96最大MSE改善<=.7。该显示门槛相对上轮适当降低并公开，是用户要求平衡baseline表现后的明确设计修订。

统计：R2=1−MSE/同区间GT方差；best/worst按各H取min/max，不选择checkpoint。checkpoint来源仍为本轮独立训练的TimeMixer可视化对照，不是Main-I引用表的checkpoint。

若有合格样本，在合格集合内计算[四H gain、visibility、TimeMixer full720 R2、UVHF full720 R2]每列百分位，选择到各列中位位置绝对距离之和最小者；这仅使展示在合格集合内居中，不意味着全数据集随机代表性。次选按同变量origin间隔96取最多3例，进行完整图与inset审阅。所有不通过条件保留，不静默降低门槛。

执行结果：8例通过。首选3760最后192步相对baseline发生反转，追加下一未审、last192 R2不低于baseline的合格例3738；该追加是公开的visual-review revision，取消对此单个追加例的96步分离限制，不把相关窗口当独立重复。最终3738四H改善24.3%–37.6%，TimeMixer四H R2均>.66，H720在同变量全origin中第69.29百分位，仍是selected case而非无偏代表。所有失败与限制见reviewer_audit.md。
