# UVHF 真实轨迹与前缀一致性对照图

## 目标与证据边界

- 日期：2026-09-05；backend：Python / matplotlib（沿用已保存偏好）。
- 用户授权：基于真实实验结果，自选 DLinear 或 TimeMixer、dataset、sample，形成一张可视化图，并迭代审稿式审核。
- 一句话目标：在同一 history、forecast origin、variable 和真实未来下，直接展示 horizon-specific 模型的 overlapping-prefix disagreement，以及 UVHF 单一轨迹在该案例中的预测误差。
- 角色：事后筛选的真实 validation case study；是否精度占优由数据决定。本图不替代官方 test 完整矩阵，不构成 CHPC 导致精度提升的因果证据。
- 本轮独立输出候选图，保留冻结论文与专利稿件。

## 拟定视觉布局

采用 quantitative grid，突出两条共享坐标的完整轨迹面板。

1. **a，Horizon-specific DLinear**：同一个起点的四个独立 checkpoint，分别画到 96、192、336、720；ground truth 为深灰，四个 horizon 用克制的色彩与稀疏形状标记编码。
2. **b，Unified UVHF**：与 a 完全相同的 x/y 范围、history、ground truth；teal 单轨迹标出四个 endpoint。不得用四条人为错开的线表示完全重合的 prefixes。
3. **c，Shared first 96 steps**：固定展示完整共同前缀，不依赖事后裁出最有利的局部；直接叠加 baseline 四条轨迹、UVHF 与 ground truth，必要时以明确标注的 baseline min–max envelope 辅助识别分歧。该 envelope 不是 uncertainty band。
4. **d，Accuracy and agreement**：展示该案例四个 horizon 的 MSE 配对结果；另列 six-pair mean CHPD（完整 pair-specific overlap）和统一轨迹的 prefix identity。所有 metric 同源计算，无双 y-axis。

最终宽度 183 mm；主要输出 editable SVG、嵌入字体的 PDF 和 PNG preview。若交付 TIFF，line-art 使用 1000 dpi。使用 Arial，最终文字不低于 6 pt，正文约 7 pt。坐标不截断数据，不平滑曲线。

## 数据与选择规则

优先复用 Figure 2 已有 DLinear 四 horizon checkpoint 与当前 UVHF Main-I/II selected-profile checkpoint。以 ETTh2 为首个候选数据集，因为它已承担正文的真实 prefix-disagreement evidence；若不能清楚同时展示两项事实，再比较其余已存在的 ETT / Weather artifacts。不得使用 smoke checkpoint 或历史非 UVHF 模型冒充当前方法。

在基于当前 UVHF frozen profile 的 validation candidate pool 上匹配 DLinear predictions，并明确保留该 pool 已经按 UVHF visual fidelity 预筛选的事实。若取得完整 frozen-profile predictions，则优先用完整共同-origin候选空间，另外记录实际搜索数量。

候选筛选前先核验 dataset bytes / ground-truth、raw origin、channel、train scaler、history length、seed 和 checkpoint 身份。baseline 四个 horizon 必须来自分别训练的四个模型；不得把同一个 H720 baseline 截断后称为 horizon-specific。

选择使用四 horizon 综合精度改善与 six-pair disagreement 的联合条件；保留所有候选评分与失败案例。单例只允许声称该案例表现；若只有部分 horizon 改善，必须照实显示。不得不断修改评分标准，直到获得预定结论。

## 审稿式审核与迭代

### 数据源恢复后的具体选择规则（计算新评分之前固定）

网络恢复后发现 2026-09-04 已有专利 Figure 5 的 ETTh2 四个独立 DLinear 模型，以及 frozen UVHF 在 channel 0 / 6 的**全部 2161 个 validation origins**，本地专利工作目录亦保留了对应 raw arrays。本轮复用这组完整数据，搜索空间固定为 4322 origin-channel cells，不再使用最初发现的 256 预筛选 pool。旧专利图只展示前 48 步，本图重新评估完整 96/192/336/720，独立选样。

- eligible：UVHF 在四个完整 horizon 上的 MSE 均低于对应 DLinear；在完整共同 96 步上均低于四个 DLinear；六 horizon-pair 按各自完整 overlap 计算的 mean absolute disagreement 位于本变量全部 origins 的至少 75 百分位；720 步真实轨迹标准差至少为 train 标准差的 0.25。
- score = 0.50 × disagreement percentile + 0.30 × 四个 horizon 中最小相对 MSE 改善 + 0.20 × (720-step Pearson correlation + 1) / 2。
- 选最高分；排序并保留全部 cells，包括未通过者。若无 eligible 则记录失败，不能以缩短评价区间挽救。
- baseline 为原论文 prefix-disagreement 协议的 source-audited DLinear，look-back=96；不冒充 Main-I 表中 published/native L336 baseline reproduction。
- 若同一案例布局审核不通过，先保留样本修改表达；只有明确属于样本可读性问题时才按固定排序检查下一案例。

**第一轮审计不通过**：读取 frozen UVHF effective config 后确认其 `seq_len=720`，旧 DLinear 为96。即使旧数据在origin1147 / OT显示四H改善，也因历史长度混杂拒绝作为 matched-history accuracy evidence；初次4322-cell评分保留于 `rejected_l96/`。补齐四个 L720 DLinear control，协议为 `matched_history_protocol.json`。选择规则保持不变，新的分数独立计算。这里只匹配输入历史，模型架构、训练目标与各自已冻结的优化协议并非 matched mechanism ablation。

1. **真实性硬门槛**：同一 raw origin/channel/target 与正确 frozen checkpoint。失败归因 `source_alignment_or_artifact_missing`，不能靠美化补救。
2. **表达门槛**：缩至 183 mm 后，5 秒内能识别“四模型互相冲突”与“一个 UVHF 轨迹”；完整周期与共同前缀均可追踪。失败归因 `presentation`。
3. **案例门槛**：UVHF 对真实变化有可见贴合，baseline 分歧持续存在，误差数字与视觉方向一致。失败归因 `sample_dataset_or_baseline_choice`，按同一规则考察下一候选。
4. **结论门槛**：注明 selected validation example；CHPC 为结构性质，sample MSE 为案例精度；不声称代表性、总体显著性或零分歧是 UVHF 独有。H720 truncation 也能提供 prefix consistency，本文优势需结合已有 Main-II accuracy evidence。
5. **输出门槛**：source-data CSV、选择审计、checkpoint / input hashes、绘图源码与导出文件可对应；source preflight 和最终尺寸视觉检查通过。

审核结果允许为未通过；不把自查等同于真实期刊同行评审或录用保证。缺失证据时明确保留阻塞项。

## 外部格式核验

查询日期：2026-09-05。KBS Guide for Authors 页面返回 403；改查 Elsevier 官方 artwork 指南。官方建议 PDF/vector、字体嵌入、Arial 等标准字体，以及 line-art TIFF 1000 dpi：

- https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-overview
- https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing

此核验覆盖出版社通用 artwork 要求，不声称已核验被拒绝访问的 KBS-specific 页面全部规定。
