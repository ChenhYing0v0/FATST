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

## 按共同前缀可见性重选案例

`reselection/replay_all_channels.py`读取同一冻结UVHF模型，对2161个validation origins批量32重放全部7变量，输出`prediction_scaled [2161,720,7]`，输入为历史与全零future占位。`select_visible_case.py`计算每个origin/channel的共同96步pointwise max−min、均值、25th percentile，以及相对于六条曲线整体数据range的`visibility96`与超过10% range的步数比例；前者用于排序，后者衡量持续性。四H与common96 accuracy条件保留，全部15127候选均记录，最终选LUFL/origin144。完整字段与rollback见reselection/protocol.md及caption_and_review.md。

`plot_single_panel.py --output`复用既有图形模块：从指定目录读取source_data/metrics以及可选figure_settings，后者仅提供真实变量标签、线性坐标范围、局部标注位置和connector位置。`check_prefix_requests.py --selection-dir`复用旧checkpoints，新增变量从全channel replay缓存读取对照，保持独立四H请求验证。旧图导出文件保持原样；新的selection、request与QA记录单独存放。

## 后程绝对贴合度审计

用户否决LUFL/origin144后，`tail_audited/audit_candidates.py`对现有15127 cells增加full720、tail337–720、last192的R2、Pearson corr、std ratio、bias/sigma，使用各窗口GT仅作回顾性评估。保留accuracy与prefix visibility条件；硬条件和空集结果记录于protocol.md/gate_counts.json。旧审阅结论撤回；四H relative gain不再单独支持视觉通过。扩展到Weather须使用同样后程gate，并匹配冻结UVHF的608步history；只新增validation-only DLinear controls，不改变模型或论文冻结结果。

后程gate扩展先后否决ETTh2、Weather/ETTh1的DLinear 96步方案及192步观察窗口。`train_timemixer.py`因此使用上游native Exp/Model做L720匹配对照：从native train函数删除test-only调用，保持优化路径；完整有序validation仍使用native batch-mean checkpoint规则。每H保存配置、派生函数、checkpoint hash与完整validation预测，原始TimeMixer仓库不修改。相关source audit、控制偏差与失败结果见tail_audited/protocol.md。

## 最终native L96 TimeMixer案例

用户允许seq_len作为可调超参数后，`export_native96.py`从native etth1.log恢复四H配置并导出validation2161 origins，不训练或访问test。`audit_timemixer.py`使用ETTh1冻结UVHF全变量数组、原始GT和TimeMixer预测重算四H/common96优势及visibility；合并既有绝对fit gate，共262/15127 cells通过。`build_final_case.py`从排名第一的审阅候选导出原始单位CSV，最终为origin947/MUFL；5例完整轨迹人工检查保留首选。

`check_final_case.py`独立载入UVHF checkpoint，验证原始GT/history、四H请求prefix identity、TimeMixer缓存与source逐元素一致、原尺度重新计算fit gate。最终输出目录`tail_audited/review_case_0`，最终布局由同目录figure_settings.json控制；该settings为完成视觉审计后的权威布局，重建source后应保留它。`plot_single_panel.py`增加可选baseline名称及CSV前缀，仅为使用TimeMixer标签，不改变历史DLinear默认行为。native source参考https://github.com/kwuking/TimeMixer，冻结版本及hash在numeric_audit.json中。

`check_figure_exports.py`复用绘图函数构造画布，不改写导出文件；检查六条inset曲线与source逐点一致、主曲线未被inset覆盖、H标签与inset轴标签不相交、文本不越界，并重算各H MSE/MAE和各pair disagreement。读取SVG/PDF的可编辑文字/字体证据及PNG/TIFF尺寸、DPI，输出figure_qa.json和交付文件SHA-256。`build_final_case.py`重建source时保留已有figure_settings，但重置selection为待复审，避免把重新构建的数据静默视为已审计。

## 完整720步可读性再评估（2026-09-06）

`main_readability_20260906/evaluate_readability.py`保持既有三个gate，针对262个合格cases计算GT的FFT dominant period、prominence峰数与各时刻绝对误差优势；所有新列的输入、计算和意义定义在同目录protocol.md。结果区分“周期更稀疏”和“相同密度但误差优势更明显”，输出完整262行与候选排序。

`render_candidates.py`结构复用既有builder/plot，明确将输出root路由至新目录并保持原BASE数据源，原版本不写入。相同相对数据range/留白映射保障全程展示尺度可比。可选`endpoint_label_y`只移动右端文字，并用细线接回真实endpoint，不变更曲线。`check_final_case.py --case-dir`为同一独立checkpoint重放新增输出目录参数；默认旧路径不变。preserved_result.json锁定此前交付文件hash；本轮仅给出备选，不变更冻结论文及模型。

`audit_candidates.py`将旧export checker的CASE显式路由至新候选，复用数据/导出检查，并新增右端两个直接标签的bounding-box不重叠检查。读取三个独立numeric audit后，分别记录不推荐/推荐备选/次选的审阅状态；技术QA通过与是否推荐替换分开。最后重新计算旧版本所有文件hash，写preservation_check.json。复现顺序为evaluate_readability → render_candidates → check_final_case对各候选 → audit_candidates；渲染重建后需要重新审计。
