# Introduction Concept Figure 绘图说明

## 1. Purpose

`scripts/plot_intro_concept_figure.py`
生成Introduction使用的双面板conceptual illustration。它与Section 3的真实数据图
严格分工：

- Introduction图只帮助读者理解两个问题；
- Section 3图才报告baseline predictions、NCHPD与matched sharing-risk evidence。

脚本不读取dataset、model checkpoint或experiment artifact。

## 2. Constructed prefix-disagreement panel

`constructed_forecasts()`返回：

```text
history_x/history
future_x
short
medium
long
```

history与三条forecast curves均由固定sinusoidal/trend函数构造。三条forecast在
$\tau=0$附近连续，但随后产生可见偏离，并分别在$H_1=42$、$H_2=70$与
$H_3=100$终止。`plot_prefix_panel()`在共同future step
$\tau^\star=26$处标出三个不同预测值。该差异是人为构造的视觉语义，不对应任何
dataset metric。

## 3. Constructed sharing-demand panel

`constructed_risks()`返回fine、intermediate与broad三条解析risk curves：

```text
fine: early region较低，随后上升
medium: middle region较低
broad: 随future position下降，在late region较低
```

`plot_sharing_panel()`将future domain划分为early/middle/late三个区域，用浅色
背景、winner marker与bottom ribbon标出各区最低的constructed curve。这里的
`risk`没有数值单位，也不作为实验统计量使用。

## 4. Code--paper consistency

Intended theory是：同一future step的跨horizon disagreement与region-dependent
sharing preference是两个需要在Section 3中正式检验的问题。代码仅通过解析曲线
表达这两个定义，没有把synthetic pattern当作方向证据，也没有预演ISCF/BSCA的
效果。

该设计会被以下情况证伪：

- caption或正文把constructed curves称为empirical result；
- panel b被解释为short horizon必然对应fine sharing、long horizon必然对应
  broad sharing；
- Introduction图被用于替代Section 3的matched real-data evidence。
