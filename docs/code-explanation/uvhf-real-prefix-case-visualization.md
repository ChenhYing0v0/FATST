# UVHF 真实前缀对照图的数据路径

本轮为 validation-only case study，不变更模型代码、论文冻结内容或 test 结果。

## 数据与对齐

`analysis/uvhf_real_prefix_case_20260905/select_case.py` 读取四个 horizon 的 DLinear arrays：`pred/true [N,H,7]`、`history [N,720,7]`、`origin_index [N]`、train scaler。按最短可共同使用的 H720 验证窗口固定 N=2161。四个模型的 origin、history、对应 true prefixes 和 train scaler 必须逐元素相同。

UVHF 使用 frozen Main-I/II profile 的完整 validation export，channel 0/6 各 `[2161,720]`，按 `validation_window_index` 重排。`raw_forecast_origin = 8639 + validation_window_index`；future step 1 是下一 raw row。对齐 ground truth 的 scaled tolerance 为 5e-6。最后再检查选中案例与原始 CSV 的真实值。

## 统计量与选择

每个 origin-channel cell 计算四个完整 horizon 的 MSE，以及统一共同前缀 96 步上的 MSE。MSE 是对应 target indices 上 squared error 的均值；MAE 为 absolute error 均值。scaled 数值采用 train-split 标准差，raw 数值使用原始单位。

`mean_nchpd` 是六个 horizon pair 各自短 horizon overlap 内的 mean absolute difference，再对六个 pair 等权平均。`disagreement_percentile` 是该变量全部 origins 的 average-tie percentile rank。`min_gain` 是四 horizon 相对 MSE 改善的最小值。`uvhf_corr720` 是完整720步 Pearson correlation；`target_std_scaled` 是真实未来的 train-scale 标准差。

Eligibility 与 score 在 figure contract 中先于新评分冻结；`all_candidate_scores.csv` 保留全部候选及布尔门槛，`eligible_candidates.csv` 保留排序。`source_data.csv` 的 `step<=0` 是720步 history，`step>=1` 是720步 future；各 `dlinear_hH` 超出其定义长度时留空，不是缺失预测或填零。

## 证据限制与最小验证

CHPC 是统一轨迹的 prefix identity，不由较小 MSE 推导。数值 request check 单独从冻结 checkpoint 执行。旧 L96 baseline 比 UVHF 少看624步，因此本轮拒绝把它作为 matched-history accuracy evidence，补齐 L720 control。控制仍沿用原 source-audited DLinear 实现与验证早停协议，不冒充 Main-I published baseline。

验证由 Python syntax、JSON parse、真实数组的 origin/history/scaler/target assertions、原始 CSV 复核、数值重放与最终图像 QA 组成；不宣称这是一轮新的完整 test benchmark 或 mechanism-effectiveness gate。

## 重放与绘图模块

`replay_baselines.py` 在本地 CPU 载入四个已训练 checkpoint，复用项目 DLinear model/dataset，不重新训练。每个 H 重放全部 validation windows，验证全 split MSE 与 GPU 保存值差异小于1e-5；仅保存共同2161个 origin 用于对齐。`check_prefix_requests.py` 使用 current selected origin 以及首尾 origin，输入完整720步历史、全零占位 future tensor，分别请求四个 H，检查与 H720 前缀的差值；同时检查 baseline/UVHF 的输入 history 一致以及 cached GPU prediction 与 CPU replay 一致。

`plot_figure.py` 只读取 source CSV、selection audit、metric CSV 和 horizon-pair CSV。上排两图共享 x/y 轴；显示 history 的最后96步但两系统实际输入均为720步。下排放大全部共同96步、显示四H train-standardized MSE。shaded envelope 是四个 DLinear 输出的 pointwise min–max，不是 uncertainty。底部的 mean CHPD 对六对完整 overlap 的 absolute difference 等权平均；不把 NCHPD 与 raw CHPD 混用。所有 forecast point 均保留，无 smoothing。

最终c面板还包含有独立纵轴的 signed within-system difference：每个DLinear短H预测减去DLinear H720在共同96步的prefix；UVHF请求减去UVHF H720的prefix为0。该轴不比较DLinear与UVHF之间的预测差，而比较各自系统内部的request agreement。metric图不用第二y-axis。导出183×157mm PDF/SVG、300dpi PNG和1000dpi TIFF。

## 用户指定的单张总图版本

`plot_single_panel.py`新增一个Axes的总图，不替换历史multi-panel脚本。相同720步future curves全部叠加；history仅显示最后48步，两模型实际input仍为720步。DLinear各H曲线终止于H；UVHF四个endpoints从同一个720步数组读取。图中标注step68处四条DLinear的max−min原尺度range，不是confidence interval或任意放大。底部用文字保留完整四H MSE改善和six-pair mean CHPD，没有额外统计轴。新输出为183×110mm的`uvhf_real_prefix_single.*`；代码断言`len(fig.axes)==1`，不改变sample、checkpoint或已有量化结果。

## 总图内嵌前缀放大

用户进一步要求放大前部后，`plot_single_panel.py --zoom`在主Axes内新增一个child Axes，直接重复`source_data.csv`中step 1–96的全部六条曲线。主图矩形与连接线标明窗口来源，inset采用明确的线性x/y刻度，不改变原始数值、sample或指标。step68的range沿用原值2.134502°C。画布增加到183×135mm，以在实际轨迹上方容纳inset；完整720步仍显示。无参数保留原版输出；本轮输出`uvhf_real_prefix_zoom.*`，完整契约、图注与审核见`zoom_caption_and_review.md`。注意`fig.axes`仅统计主Axes，inset由`ax.child_axes`检查；放大版本实际有一个主坐标轴和一个内嵌坐标轴。
