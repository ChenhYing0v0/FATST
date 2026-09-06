# 指定样本向后移动192步

用户指定在LUFL8752基础上将0点置于旧图future step192。严格采用origin8944=8752+192，没有重新排名或更换变量、checkpoint。

## 时间与数据映射

旧图future step192成为新图最后一个已知history点（step0），旧图193成为新图step1。新图0点2017-09-27 03:45:00；预测从2017-09-27 04:00:00到2017-10-04 15:45:00，共720个15分钟步。UVHF取新起点之前720点，TimeMixer取之前96点。图中history延续既有展示宽度，source_data.csv保留完整720步历史。

build_shifted.py断言新history[-191:0]等于旧GT[1:192]，新GT[1:528]等于旧GT[193:720]。从已有全validation预测缓存读取origin8944对应的全新预测，不截取旧预测或移动曲线。没有训练或新test访问。保留旧图。

## 结果

完整H96/H192/H336/H720的UVHF MSE较各对应TimeMixer分别降低79.34%、59.9%、63.3%、68.5%（图中保留一位小数）。10个独立分段与有效horizon组合MSE和MAE均改善；最小MSE改善34.79%、MAE改善19.23%。相较TimeMixer H720四个独立区间MSE降低74.67%、51.68%、63.61%、70.72%；最后192步降低73.65%。全部见segment_metrics.csv，不以累计误差掩盖区间反转。

UVHF完整720步R2=0.7723，337–720步R2=0.7585，最后192步R2=0.7122。独立加载冻结checkpoint，在四个target_prefix请求下验证公共前缀最大差异均为0；CSV与独立重放以及baseline缓存相符。见numeric_audit.json。

## 视觉审阅与边界

保持单主图+完整1–96放大图、原始尺度、不平滑、全720点。新前缀约60步后下降段，UVHF跟随GT而TimeMixer预测偏高；四条TimeMixer前缀存在明显差异。中后段多次低谷优势仍清晰。尖峰、约230–250步局部谷形和接近720的末端偏差保留，不能称为逐点胜出。

数值与导出审计通过：183×135mm，PNG300dpi、TIFF1000dpi、PDF嵌入字体、SVG编辑文本、inset未遮挡主轨迹，指标复算通过。复用既有plot_single_panel.py与check_figure_exports.py；后者静态宽度提示由实际尺寸核对解决。

此例来自此前定向选择案例的用户指定后移窗口，与旧窗口重叠，不是独立重复或总体估计。TimeMixer仍为visualization专用checkpoint，非Main-I来源。内部审阅通过仅表示数据及呈现可信，不代表期刊外审结论。
