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
