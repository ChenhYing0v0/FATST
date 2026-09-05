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

TimeMixer首发在EarlyStopping构造前因numpy2移除np.Inf失败，没有epoch训练。添加进程内`np.Inf=np.inf`兼容别名后重启，不更改native源文件、数值含义或共享环境版本。

## TimeMixer优化失败与稳定性重跑

lr=.01的H192/336/720在L720下发生数值发散，H192最后validation loss约1.18e10，H720约7.08e8。这些checkpoints不进入样本选择，也不作为UVHF优势证据；未启动H96。归因optimization_or_numeric_pathology，不能据此否定TimeMixer。回退优化配置：仅lr改.001、其余保持，重新完整4H训练，输出独立`uvhf_prefix_tail_timemixer_lr1e3_20260905`。此次在validation上诊断数值稳定性，无test调参。

## 用户明确更新seq_len边界

用户明确：本论文seq_len是可调超参数，允许直接用L96 TimeMixer对L720 UVHF，图中无需额外说明。按此授权，停止新增matched-length训练；已有在运行的稳定性试验若完成则保留记录，不作为继续任务的依赖。最终优先既有官方ETTh1/L96 TimeMixer四H checkpoints。

`export_native96.py`从既有etth1.log通过AST literal parsing恢复配置，读取唯一对应H的checkpoint，在ordered validation前2161个共同origins上导出。使用相同forecast origin与原始GT，TimeMixer只取该origin前96步、UVHF取前720步。新图不再写identical input history；不额外强调lookback差异。内部provenance保存真实长度与用户授权，这是标准可调超参数模型对比，不作matched architecture/mechanism attribution。后程、accuracy与96步visibility gate保持。

## 最终选择与交付决定（2026-09-05）

current_step=9–10（仅可视化诊断），decision=通过selected validation illustration；不更新paper-core effectiveness状态。使用native L96 TimeMixer后，15127个ETTh1 origin–variable cells中262个同时通过原accuracy、visibility与full/tail gate。按既定visibility96、tail R2降序审阅前5个间隔候选，保留首选MUFL/origin947。完整结果、失败候选和排序均保留在timemixer_all_candidate_audit.csv、timemixer_eligible.csv、timemixer_review_candidates.csv。

最终图及确认前审计在review_case_0/。full720 R2=.6611、tail337–720 R2=.6618、last192 R2=.4553；四H MSE改善57.0%、7.8%、13.1%、26.5%。H336 MAE略差，不能写所有metric均优。UVHF独立四H请求prefix gap全为0；TimeMixer平均六对完整overlap disagreement=1.70原始单位。新增L720/lr.001 TimeMixer已完成但不进入最终选择，数值发散的lr.01结果被明确排除。

失败归因：旧样本失败源于selection/readout审计缺失（只看relative gain），DLinear扩展失败源于在固定fidelity gate下prefix visibility不足；因此回退sample/dataset/baseline选择，保留用户接受的单张总图+局部放大形式。最终仅调整留白和标签位置，未改变曲线数据。该图可支撑所选实例的现象说明；不能替代完整official-test benchmark，也不能建立prefix consistency导致accuracy gain的因果归因。冻结论文和专利稿件未修改。
