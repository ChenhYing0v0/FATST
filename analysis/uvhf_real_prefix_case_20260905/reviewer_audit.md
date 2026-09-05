# 真实轨迹图审稿式审核

## 结论

[Strong Evidence] 最终图通过本轮 **selected real-data case-study figure** 审核：数据真实、预测起点与历史匹配、完整 horizon 误差核对通过、结构性 prefix identity 有独立请求数值核验、展示和导出满足期刊图的基本质量要求。它有力补充 Figure 1 的构造示意与 Figure 2 的问题证据，并把它们连接到 UVHF 的真实预测结果。

该结论是作者侧审稿式自查，不是外部独立同行评审，不保证期刊接收。本图不替代 Main-I/II 的完整 official-test benchmark，也不能独立承担 mechanism effectiveness 或因果归因。

## 审核迭代记录

| 轮次 | 发现 | 归因 | 处理 |
| --- | --- | --- | --- |
| 数据准备 | 最初仅查到论文目录的256个/数据集候选；网络超时 | artifact access | 网络恢复后定位专利图工作中的完整2变量 × 2161 validation origins |
| 第一轮证据审核 | 既有专利 DLinear L96 与 frozen UVHF L720 不同 | comparison protocol / input-history confounding | 拒绝把其精度优势当作 matched-history evidence；保留 `rejected_l96/` 全部评分，补齐四个 L720 DLinear control |
| Control execution | 首次执行使用了不存在的环境路径；GitHub pull未完成 | environment / synchronization | 首次没有训练发生；确认 conda moe 路径，用已推送 commit 的 Git bundle 完成 fast-forward pull 后重启 |
| 第二轮数据审核 | 四H baseline 完成；匹配历史后有127个 eligible cells | source alignment passed | 沿冻结评分选择 origin1239 / channel6；后续不再换样本 |
| 图 v1 | 图例与上排标题重叠、页脚靠近x标签；baseline原始曲线的差异不够醒目 | presentation | 保留样本，调整布局，新增有独立单位的 within-system difference 小轴；旧preview保留 |
| 最终图 v2 | 无重叠，准确度和prefix disagreement可分别读取 | illustrative figure gate passed | 输出单张复合图、caption、source data 与审计 |

## 数据与数值结果

- Dataset=`ETTh2`，variable=`OT`，validation window index=`1239`，raw last-history index=`9878`，forecast origin=`2017-08-16 14:00:00`。图上 step1 对应下一小时。索引均为0-based。
- 两系统实际输入均为最近720步，只把最后96步history画出以留出预测空间；四个DLinear模型相互间的 history / targets / scaler逐元素对齐。
- baseline/UVHF实际模型输入的最大scaled差为 `4.76837158203125e-7`，来自scaler浮点精度；所有共同targets的最大scaled gap为 `2.384185791015625e-7`。
- UVHF=`ISCF-BSCA-MAIN-v1` frozen `h2_lr5e4`，checkpoint SHA-256=`bcfbc9955754a9825d1dd33015610a049551e3441dabd0ad98982c0fc2285d3e`。
- DLinear为source-audited本地实现，四个独立训练的L720模型，seed2021；50epoch上限、patience8、Adam lr1e-4、batch128、pytorch_default初始化，沿原prefix可视化协议。实际epochs为45/50/30/33；best-val epochs为37/45/22/25。不是smoke，也不冒充Main-I published three-run mean baseline。
- GPU结果经CPU重放：四个完整validation MSE最大差 `1.63e-8`。本轮没有UVHF训练、hyperparameter search或新的test evaluation。

| H | DLinear MSE | UVHF MSE | MSE降低 | DLinear MAE | UVHF MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 96 | 0.325580 | 0.157047 | 51.8% | 0.496739 | 0.351213 |
| 192 | 0.529718 | 0.169421 | 68.0% | 0.653057 | 0.356531 |
| 336 | 0.431496 | 0.168951 | 60.8% | 0.571810 | 0.348526 |
| 720 | 0.378086 | 0.145876 | 61.4% | 0.541472 | 0.312283 |

这些是**单个选中origin/variable、完整对应H、train-standardized**的误差；不是dataset平均值或seed平均值。图中MSE percentages逐H对比对应DLinear模型，不对多个模型误差取均值后择优报告。MAE未放入主图以避免重复，全部保留于source table。

六对CHPD raw分别为0.598754、0.572453、0.562813、1.108982、0.867941、0.484623°C，等权均值0.699261°C。每一对使用其shorter H的完整overlap；这和c下方只展示共同96步的signed difference不同，caption已分别定义。

UVHF数值核验覆盖origin0、1239、2160及全部7个变量；分别请求96/192/336/720时，相对H720的最大absolute difference均为0。输入future tensor全为0，排除了用真实future values构造预测；该路径下GPU缓存与CPU重放最大差4.77e-7。全部domain的CHPC主张仍来自架构的horizon-independent mapping及论文理论，不能把3个样本的数值检查称为全域数学证明。

## 选择透明度

1. 搜索全部2161个共同validation origins，但变量范围限现有完整UVHF export的HUFL/OT；总计4322 cells，未声称扫描7个变量或所有数据集。
2. eligibility：四个完整H均有MSE改善；共同96步上UVHF优于四个DLinear；six-pair disagreement在该变量至少75百分位；720步真实波动至少0.25 train std。
3. 127个通过。score为0.50 disagreement percentile + 0.30 worst-horizon relative improvement + 0.20 transformed 720-step correlation；按降序选择，规则在L720结果返回前已经冻结。
4. 选中案例disagreement约88.7百分位、720-step Pearson r约0.576；该图不是“完美预测”，仍保留UVHF漏掉峰值和局部变化的事实。
5. `all_candidate_scores.csv`保留所有4322候选与失败门槛，避免只留正结果。邻近起点高度重叠，不能把127/4322解释为独立样本上的普遍成功率。
6. 当前freeze UVHF profile本身已test-tuned；本次没有新访问test。不要把validation case包装为没有历史test exposure的确认性证据。

## 视觉与导出QA

- Python/matplotlib独占绘图。Build anew，只继承已存在图的克制配色、字体和marker习惯，未复用构造曲线。
- 最终183 × 157 mm双栏图；PDF一页，MediaBox约518.740 × 445.039 pt；SVG可编辑text节点72；PDF含TrueType font embedding；最小文字6.5pt。
- PNG preview为2161 × 1854 px / 300dpi；TIFF为7204 × 6181 px / 1000dpi，LZW压缩；PDF/SVG为首选线图交付。
- static preflight=13 PASS / 1 WARN / 0 FAIL。唯一WARN为validator无法静态解析width变量；已由PDF MediaBox与SVG viewBox确认183mm。
- 上排共享坐标轴，全部future点绘出；a的每条baseline线在其H终止，b由单轨迹标出endpoints。c的zoom固定为完整共同96步，不事后裁短；lower difference轴以°C显示，不以任意缩放夸大。
- 冗余编码：四个DLinear颜色配合不同marker形状，UVHF较粗teal线，ground truth深灰。相同模型的配色跨panel保持一致；误差图以方形/圆形区分系统。
- 已逐图检查PNG导出；最终版修正标题/legend/页脚重叠，曲线无平滑、插值增密、人工位移或局部删点。
- 出版社通用格式已核验；KBS guide页面403，所以不声称核验该页全部special requirements。参考[Elsevier artwork overview](https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-overview)和[artwork sizing](https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing)。

## 四层证据与最终decision

- `paper_facing_effectiveness`：本轮未执行official-test matrix，沿用稿件既有主实验；本图不可独立通过该gate。
- `matched_mechanism_attribution`：仅匹配输入历史与预测起点；架构、优化与目标不同，不能归因CHPC、MSD或BCA的单独贡献。
- `internal_mechanism_health`：仅验证prefix请求数值identity和frozen checkpoint replay；未新增router或gradient diagnostics。
- `failure_attribution`：第一轮失败是`comparison_protocol_wrong`（input-history mismatch），不是hypothesis_false；第二轮是presentation；没有用任何失败否定模型方向。
- `current_step`：Step9 case-study analysis / figure review；`decision=passed_illustrative_case_figure_not_new_core_effectiveness_gate`。冻结论文与专利正文未修改。建议后续作为实验章节中的定性实例，并与总体test表结合引用。

## 重现入口

绘图只依赖同目录四个小文件：`source_data.csv`、`selection_audit.json`、`selected_metrics.csv`、`selected_pair_disagreement.csv`，运行 `python plot_figure.py` 即可。

完整重选需baseline raw arrays和两变量完整UVHF arrays；其hash、远程training根目录和来源记录分别在selection audit、baseline training audit和provenance中。checkpoint与大型raw arrays本地保留并Git-ignore，原专利工作目录不作改动。同步及新增对照command见remote_launch.json。
