# 完整720步主图可读性：候选复审

日期：2026-09-06。当前结果保留于上一轮review_case_0，commit 7888b99c；preserved_result.json逐文件hash锁定。本文是作者侧案例筛选审计，不是外部期刊审稿意见。

## 结论

**有完整轨迹精度优势更清楚的备选，但在现有合格预测池中没有发现主导周期更长的样本。** 推荐将HUFL/origin869（本目录review_case_2）作为“主图优势优先”的备选；保留用户已基本接受的MUFL/origin947，不自动替换。

筛选入口是已审计的15127个ETTh1 origin–variable cells，保留全部旧门槛后剩262个：HUFL160、MUFL102。所有262个的GT dominant period都是24步，720步约30个周期，因此仅更换这些样本不能根治紧凑重叠。服务器native TimeMixer目录目前只有ETTh1的四H checkpoint，本轮没有扩大训练或读取新的test预测；结论不覆盖尚未导出的其他数据集/模型组合。

## 比较结果

| 项目 | 保留 MUFL/947 | 备选 HUFL/869 | 次选 HUFL/655 | 少峰候选 HUFL/1427 |
|---|---:|---:|---:|---:|
| H720 MSE降低 | 26.5% | **57.3%** | 38.5% | 31.7% |
| full720 R2 | **0.661** | 0.634 | 0.631 | 0.540 |
| tail337–720 R2 | 0.662 | **0.707** | 0.627 | 0.615 |
| last192 R2 | 0.455 | **0.731** | 0.646 | 0.609 |
| prefix visibility96 | **0.126** | 0.093 | 0.107 | 0.092 |
| visible win−loss fraction | 2.4% | **24.9%** | 22.4% | 21.4% |
| GT dominant period | 24 | 24 | 24 | 24 |
| GT prominent peaks | 29 | 24 | 26 | 19 |

MSE改善分母为对应样本TimeMixer H720 MSE；R2按各窗口GT均值作为参照，跨变量的原始MSE不能直接比较。visibility96归一化与上一轮相同；后两个新增显示诊断的精确定义在protocol.md，不是统计显著性或新的模型benchmark。

## 视觉审阅与失败归因

1. **HUFL/869：推荐备选。** 主图后半段TimeMixer H720紫线在多个周期中持续高估峰值，与GT和UVHF形成可辨的竖向间隔；UVHF更贴近真实峰值。末尾192步的贴合也强于保留样本。四H MSE改善依次28.5%、42.1%、30.6%、57.3%，四H MAE亦均较低。前96步分歧仍可见，step29跨度6.62原始单位；但prefix visibility96较保留样本降低约26%，UVHF仍漏掉数个深谷，全程R2也略低。不能宣称该图每项表现都更好。
2. **HUFL/655：次选。** 前缀的峰高差较HUFL/869更清楚，主图TimeMixer持续高估亦可见；但后程与完整H720优势弱于869，故不作主推荐。更偏重inset时可保留比较。
3. **HUFL/1427：不推荐替换。** 峰数由29降至19，但这不是周期变长，而是部分GT峰值振幅变小、平台化，未跨过prominence阈值。预测曲线仍约24步振荡，主图没有真正变稀疏；全程R2降至0.54，inset精度差异也弱。failure_attribution=显示密度代理不充分，不能把“少峰”当作解决周期密集的证据。
4. 维持同一183×135mm主图+inset模板；未来数据range按相同相对比例映射留白，完整720步全部绘制，无平滑、重采样、移位或删点。仅当右端标签过近时移开文字并加短引线，不移动预测点。

## 决定与边界

- preservation：保留已接受版本及全部原始导出、source和审计，未覆盖。
- effectiveness_gate：仅case fidelity和已有数值预测审计通过，不作新的paper-core mechanism结论。
- reviewer_decision：HUFL/869为主图可读性更好的候选，仍有24步周期密集和深谷失配；尚非用户确认替换版本。
- rollback：想要真正减少720步内的周期数量，需要转向具有慢变化结构的变量/数据集，并补齐它们的同origin四H baseline预测；继续在当前262个样本中追求更长周期，已有证据不支持。
- 对当前任务的建议：先对比保留图与HUFL/869；若仍要求显著更稀疏，再扩展数据集的现有checkpoint预测。不能用曲线平滑或截掉后程伪造可读性提升。

## 数值与导出验证

三个候选均独立从同一UVHF冻结checkpoint重放，history/GT与原始CSV对齐，TimeMixer图中值与native export一致；UVHF四个独立H请求prefix max gap全为0。numeric_audit.json保留逐项结果及checkpoint hash。

导出QA复用既有检查：inset六曲线与source逐点一致、完整主轨迹未被inset覆盖、标签不越界、MSE/MAE和CHPD复算、SVG/PDF字体证据、PNG/TIFF尺寸与DPI。主候选为183×135mm矢量PDF/SVG、300dpi PNG、1000dpi TIFF。技术导出通过不等于每个候选都被推荐。
