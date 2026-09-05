# 后程贴合度优先的样本审计协议

用户否决LUFL/origin144：虽相对DLinear误差更低，但UVHF后程不贴合GT。上一轮自审只检查相对优胜，没有给后程绝对拟合设gate；本轮将该自审结论撤回，历史结果保留。当前Step9为validation-only illustration audit，不是方法effectiveness判定。

本轮先限定现有ETTh2全部2161 origins ×7变量，冻结模型和baseline不变。若不存在同时满足条件的案例，报告空集并记录扩展路径，不悄悄降低标准或仅凭相对优势通过。

计算全720步、tail337–720、last192=529–720的拟合度：
R2 = 1 − sum((u−y)^2)/sum((y−mean(y))^2)，为该窗口内真实均值作参照的描述性拟合度；Pearson corr衡量同步变化；amplitude_ratio=std(u)/std(y)；bias_sigma=abs(mean(u−y))/std(y)。这些使用各自窗口的GT中心化尺度，仅用于回顾性样本审计，不输入预测模型。

第一轮硬条件：
- 保留上一轮accuracy_eligible（完整四H及共同96步MSE全部更低；full target std_scaled>=.25）。
- full R2>=.35；tail R2>=.25且corr>=.70；last192 R2>=0。
- tail amplitude_ratio∈[.5,1.5]，bias_sigma<=.35；避免近常数输出或整体偏移被相关性掩盖。
- visibility96>=.075，保证共同96步分歧可见性至少优于最初OT案例(.055)。
在全部gate通过者中按visibility96降序、tail_R2降序排序；审查前5个同变量origin间隔>=96的候选的完整720步，且单独看后程，不只看前缀。结果需数据审计、数值request check、最终实际图形审阅均通过后才确认。不得把筛选个例作为总体效果或CHPC精度因果证明。

所有候选及失败布尔值保留。画法保持主图+局部放大，图内完整720步，禁止平滑、纵向偏移或省略后程。显示范围由原值确定。继续nature-figure Python轨道与183mm矢量输出。

## ETTh2审计结果与扩展

15127 cells的后程硬条件交集为空；旧LUFL/origin144 full R2=-1.1447、tail R2=-3.4684、last192 R2=-4.6162、tail bias=1.9014 sigma，被明确拒绝。不能以其26%–78% relative gain通过。

扫描已有七数据集每集256个UVHF validation candidates的拟合gate，ECL、Solar、Weather各256个通过；ETTh1通过221、ETTm1通过235，ETTh2/ETTm2均0。该pool历史已按UVHF visual-fidelity筛选，后续图必须披露两阶段post-hoc selection。
优先Weather：已有冻结profile为L608，channel18，实际lookback从effective_config核实；21变量raw dataset和source-audited DLinear均可直接支持。补充4个L608 DLinear matched-history visualization controls，沿用先前协议（H96/192/336/720、seed2021、Adam1e-4、batch128、max50epochs、patience8、pytorch_default init、full-H validation最小MSE checkpoint、skip_test）。不训练UVHF。GPU运行前先核验空闲显存并同步已commit代码。
同样的后程gate和prefix visibility gate适用Weather256候选；完整H及common96 MSE优势条件保持。原本ETTh2的target_std_scaled>=.25也保留。仅审计candidate pool已包含的channel18，若空集如实记录，不临时挑其他指标。

## Weather channel18结果与全变量扩展

4/4匹配DLinear controls已完成。既有256候选中247满足accuracy条件，但最大的visibility96仅.03493，无一达到.075；因此不通过确认，不降低visibility gate。
扩大到Weather全部21变量×4551个validation origin。沿用同一UVHF和四DLinear checkpoints，仅本地CPU重放；后程、accuracy、visibility gate全保持不变。选择需审阅完整720步，而不是仅radiation preselected pool。

数值对齐说明：Weather全validation检查发现float32 numpy scaler与官方float64 scaler导致最大输入差7.6294e-6（channel14），超出ETT沿用的5e-6阈值。两loader数据shape均[5878,21]且raw split一致；将数值容差设为1e-5并记录实际gap，这不是变更样本通过标准。零方差窗口的中心化R2/amp/bias定义无效，记NaN并拒绝fit gate，不能以零除得到高分。

## Weather全量结论与ETTh1扩展

Weather95571 cells中7063个通过fit+accuracy，但最大visibility96=.05720，小于.075，继续拒绝。ETTh1冻结UVHF确认L720，原pool256例有221例通过fit；新增4个ETTh1/L720 DLinear对照，训练参数与Weather相同，仅dataset及lookback变更。后程、accuracy及visibility门槛全部保持。

## 第二次失败归因：共同前缀观察窗口

ETTh1全7变量也未通过96步visibility gate。接下来回退到展示窗口设计：比较前192步的真实重叠，H96只显示前96步，H192/H336/H720覆盖192步。该变化公开记录为post-hoc design revision，不能说旧96步gate通过。
在已通过全部fit与accuracy gate的Weather/ETTh1 cells内计算`visibility192`：每一步至少3条有效DLinear预测的max-min，再对192步均值，并除以同窗口所有有效DLinear、GT、UVHF值的整体range。前96步有4条，97–192步有3条，不补零或外推H96。仍要求visibility>=.075；如果通过，需审阅全程和局部，图注明确97步后只有3条baseline仍在比较。

## TimeMixer对照协议与来源

前192步窗口扩展亦0通过；因此回退baseline选择，使用用户初始许可的TimeMixer。
上游官方仓库https://github.com/kwuking/TimeMixer（2026-09-05检索），服务器native checkout=e24610583b36fdd8c76cc17a8df4e65759a5f460。models/TimeMixer.py为native PDM+FMM实现，predict_layers的输出长度依赖pred_len，各H独立checkpoint不保证前缀相同。先读模型、run.py、exp训练函数、data_factory与ETTh1官方unify script。

新4个ETTh1 TimeMixer与冻结UVHF同为L720；原脚本L96，因此明确为matched-history visualization control。其余参数沿用原脚本：d_model16、d_ff32、e_layers2、downsampling3层avg/window2、dropout.1、channel_independence1、use_norm1、use_future_temporal_feature0、label_len0、Adam maxlr.01、OneCycleLR/TST pct_start.2、batch128、10epochs/patience10、seed2021。

`train_timemixer.py`使用原仓库Exp训练函数；仅动态移除test loader和逐epoch test evaluation，禁止任何_get_data('test')调用；validation改为完整有序不drop，仍沿用native mean-batch validation loss checkpoint selector（末尾小batch等权，此差别披露）。不调用native run的final test。模型定义、训练loss、初始化与LR路径不变。原repo源文件不修改，保存派生训练函数和实际args便于审计。此前已有L96 TimeMixer artifacts不用于这次matched comparison。

TimeMixer筛选恢复原共同96步visibility>=.075，全部后程及accuracy条件不变，需独立全程视觉审阅。训练和evaluation均validation-only；不存在新test access。
