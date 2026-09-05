# 单张总图：图注与审核

## 建议英文图注

**Fig. X | Forecast accuracy and prefix agreement from one shared history.**
Ground truth, four independently trained horizon-specific DLinear forecasts
and a unified UVHF trajectory for ETTh2 oil temperature are overlaid in one
coordinate system. Both systems receive the same 720-step history; the last
48 observed steps are shown in grey. Blue shading marks the 96 future steps
shared by all four requests. DLinear forecasts terminate at their respective
horizons; teal circles mark the corresponding endpoints on the single UVHF
trajectory. At the annotated shared step, the four DLinear predictions span
2.13°C. This descriptive pointwise range is distinct from mean CHPD, which
averages absolute disagreement over all six horizon pairs and each pair's
complete shared prefix. Percentages report MSE reductions relative to the
corresponding DLinear model, evaluated over each complete requested horizon.
The example is the previously selected ETTh2 validation origin 1239, OT,
forecast origin 16 August 2017 at 14:00; the original selection covered 4,322
origin–variable pairs across HUFL and OT. It illustrates a selected case,
not population performance or a causal effect of CHPC on accuracy. No curves
are smoothed or vertically displaced. Source data and checkpoint/selection
audits accompany the figure.

## 修订与审核

- 用户要求：一个大的总图，融合展现，取消多子图。
- 最终仅一个Axes，线性x/y轴，完整720步预测，未使用inset、第二坐标轴或broken axis。将准确度、prefix conflict和one-trajectory endpoints融合为曲线与文字标注。
- 与前一版使用完全相同的source_data、sample、metrics、frozen checkpoint，未运行训练或修改选择规则。原multi-panel输出保留，新交付名为 `uvhf_real_prefix_single.*`。
- 第一版single-panel preview发现Shared prefix文字靠近真实峰值，区间标线端帽过宽；最终将文字移至数据上方空白、端帽改成精确的±3-step宽，不覆盖相邻轨迹。
- Step68是原样本中的post-hoc视觉标注位置。四条DLinear值分别为34.740152、33.206829、35.341331、34.475143°C，range=2.134502°C；UVHF=38.622177°C，GT=43.935501°C。该点仍存在预测误差，因此不能称为精准命中GT。
- 图中完整H的MSE改善为51.8%、68.0%、60.8%、61.4%，mean six-pair CHPD=0.699261°C，UVHFprefix identity沿用已完成的数值request check。
- 尺寸183×110mm；Arial、最小文字6.2pt；PDF/SVG vector、PNG300dpi、TIFF1000dpi；source preflight以及最终PNG视觉审核通过，细节见single_panel_qa.json。
- Reviewer verdict：**单图融合表达通过**。同图能对照GT与两系统的完整轨迹，短时刻标注清楚指向同一步不同预测，底部统计明确采用完整H。相较多子图版，精细prefix差异的可读性有所降低；用原尺度range标注弥补，未人为放大/偏移曲线。结论仍限selected validation case，不将单图称为完整性能证明。

## 重现

在该目录运行 `python plot_single_panel.py`。依赖 `source_data.csv`、`selected_metrics.csv`、`selected_pair_disagreement.csv`；selection与prefix request核验沿用前次审计。该脚本没有模型推理或训练路径。
