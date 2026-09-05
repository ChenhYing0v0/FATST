# 单一坐标图修订契约

用户于2026-09-05要求把多子图融合为一张大总图。本轮继续nature-figure的Python轨道，仅改变视觉表达，沿用ETTh2 OT / origin1239和已经审核的全部原始预测、MSE与CHPD。

- 一个Matplotlib Axes；没有inset、第二坐标轴、broken/nonlinear axis或分面。
- 完整720步预测叠加，GT为深灰、UVHF为teal，四条DLinear线用颜色与marker形状区分。每条baseline在对应H结束，UVHF对应H的endpoint落在同一曲线上。
- 原始温度坐标；共同前缀淡色背景，少量直接标注提醒读者比较同一时刻的预测。不能为了分开线条而人为偏移、平滑、放大数值。
- 采用step68作为标注位置：完整共同96步中一个具有可见分歧的内部时刻；不取最前面两步的edge transient。其点差为描述性标注，不替代six-pair mean CHPD。该时刻选择是在原样本已冻结后进行的视觉标注选择，不改变误差统计或样本选择。
- 以紧凑文字同时给出四H MSE降低、mean CHPD与UVHF prefix identity；文字不是额外统计子图。
- 183mm宽、约110mm高；沿用editable SVG/PDF、PNG preview和1000dpi TIFF。单图版本另存，保留已交付multi-panel作历史对照。
- Reviewer gate：预测曲线重叠可分辨，标注不遮挡关键数据，原始axis linear、统计范围清楚，selected validation example披露保留。若完整720步使前缀差异过密，先修改线宽、标记和布局，不换样本。
