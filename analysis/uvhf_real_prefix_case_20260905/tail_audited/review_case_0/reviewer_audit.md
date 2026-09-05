# 确认前审计：ETTh1 MUFL / origin947 / native TimeMixer

## 结论与本轮失败归因

**2026-09-05 作者侧审计通过，确认此样本作为可视化交付。** 通过的是selected validation illustration，不是外部审稿接收或完整performance证明。

上一LUFL/origin144未检验后程绝对贴合，尽管相对baseline有大幅改善，tail R2=-3.468，具有明显偏移。因此撤回上一轮视觉通过结论。此次先设定full/tail/last192 R2、tail corr、amplitude和bias gate，再检查完整图；未因候选空集而降低这些标准。

DLinear在ETTh2、Weather、ETTh1的共同96步组合gate均失败；Weather/ETTh1前192步观察扩展也失败，未采用。新增L720 TimeMixer在lr=.01发生数值发散，明确排除；lr=.001重跑完成后，用户明确允许L96，因此最终直接使用既有L96 TimeMixer，未用发散模型制造优势。所有失败表、协议、checkpoint出处保留。

## 数据与预测核验

- 数据：ETTh1、MUFL、validation origin947；同一forecast origin与未来720步真实值。图只显示最后48步history。
- 15127个origin–variable cells按共同96步可见分歧及全程/后程硬条件筛选，262个通过。按visibility96、tail R2降序，首选origin947/MUFL。审阅前5个同变量origin至少相差96步的完整720步轨迹，最终保留首选，没有更换成未记录样本。
- 图中所有曲线直接来自source_data.csv，完整720步显示，H96/192/336分别在各自终点停止，无平滑、位移、截断后程或外推短H。
- 原始CSV逐元素核验history和GT；UVHF从冻结checkpoint独立CPU重放，分别请求96/192/336/720的prefix max gap全为0，future输入占位为零。具体误差容差见numeric_audit.json。
- 四个TimeMixer prediction export与图中曲线逐元素核验，checkpoint hash与原始配置完整保留。seq_len按用户授权作为可调超参数，图中不声称输入长度相同。

## 后程审计（原始单位重算）

| 检查 | 硬条件 | 实测 |
|---|---:|---:|
| full720 R2 | >=0.35 | 0.6611 |
| tail337–720 R2 | >=0.25 | 0.6618 |
| tail Pearson corr | >=0.70 | 0.8198 |
| last192 R2 | >=0 | 0.4553 |
| tail std(pred)/std(GT) | 0.5–1.5 | 0.8361 |
| abs(tail bias)/std(GT) | <=0.35 | 0.0999 |

R2以该评估窗口GT均值作参照；bias与amplitude的分母均为同窗口GT std。这些是回顾性案例诊断，不进入模型输入，也不是总体benchmark。

## 精度与前缀证据

四完整H MSE相对TimeMixer降低57.0%、7.8%、13.1%、26.5%；共同96步UVHF MSE也低于四个TimeMixer各自前缀。MAE不作为筛选目标，数值完整保留；不能把MSE优势描述成所有误差指标都更好。

visibility96=0.12578（逐时刻四预测max-min的96步均值/全部六曲线前96步数据range），超过预设0.075。第41步的range=6.39 units，属于窗口内部24–80步中较明显的分歧标注，不是置信区间。mean CHPD约1.70 units，是六对horizon各自完整overlap上的平均绝对差，与单点range不同。

## 视觉复审

1. 完整720步主图：UVHF跟随主要周期与峰值，后程无旧LUFL样本的大幅整体偏移；末段保留真实平台和模型失配，未裁剪隐藏。
2. inset前96步：四TimeMixer的峰高、谷深存在清晰差异；UVHF是一条连续预测，真实GT保持可见。
3. 初稿inset xlabel与主图峰值过近，个别H标签触及inset ticks/connector；增大主图上方留白并移动H标签，原数据范围和点值不变。
4. **限制仍然存在**：UVHF漏掉若干深谷与非周期突变，最后192步R2约0.46。因此不能称“每一步精确预测”或“全程完美跟踪”。该图可同时说明实例中的精度改善与prefix identity，不能证明两者的因果关系。

## 建议英文图注

**Fig. X | Consistent prefixes and improved forecast accuracy on real data.**
Ground truth, four horizon-specific TimeMixer forecasts, and one UVHF trajectory
are shown for ETTh1 MUFL from a common forecast origin. The inset enlarges the
complete 96-step shared prefix. At step 41, TimeMixer predictions span 6.39
original-scale units, whereas separate UVHF horizon requests yield identical
overlapping predictions. MSE reductions are evaluated over each complete
requested horizon. Mean cross-horizon disagreement averages absolute differences
over all six horizon pairs and each pair's full shared prefix. The selected
validation example was screened for both visible prefix disagreement and
forecast fidelity across the full and late portions of the trajectory. No
curves are smoothed or displaced. This case illustrates the behavior and does
not estimate population performance or establish a causal accuracy benefit of
prefix consistency. Source data and selection audits accompany the figure.
