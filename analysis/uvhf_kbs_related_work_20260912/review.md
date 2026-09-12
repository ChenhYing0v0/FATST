# KBS 相关工作引用核验

- 日期：2026-09-12；范围：prediction-oriented representation、forecast generation、multi-scale compression/decompression。
- 决策：新增 PDT 与 TCDNet，保留原有文献，引用库由 39 条增至 41 条。不以目标期刊引用数量为选择标准。
- PDT：作者提供的正式书目信息为 Chenhao Ying、Jiangang Lu，Knowledge-Based Systems，2026，116981。机制依据作者原稿：prediction-oriented domain transformation、dominant-component linear route、complex-dynamics encoder route。出版社/Crossref 元数据访问未成功，故不推测卷号或 DOI。Zotero 收录状态未核查。
- TCDNet：外部发现；出版社页面 https://www.sciencedirect.com/science/article/pii/S0950705126007471 。出版字段：Peng-Cheng Li、Jiang-Wen Xiao、Yan-Wu Wang；Knowledge-Based Systems 344 (2026) 116021；DOI 10.1016/j.knosys.2026.116021。页面搜索索引提供摘要、引言和方法概要；直接页面访问返回 403，未取得完整 PDF，因此只引用已核对的高层机制，不转述其性能或作穷尽性机制比较。Zotero 状态未核查。
- 备选：MDCNet（2024，111986）与 LightFreq（2026，115556）。本次不加入，以免重复分解/频域表示介绍。
- 位置：PDT 加入 Section 2.3；TCDNet 加入 Section 2.4。两者均为 Related Work，不属于本稿已评估 baseline，不改变实验表格或性能结论。
- 同步收紧旧概括：不将所有既有方法概括为只做输入端建模；区别改为 temporal resolution/frequency/prediction branches 与 explicit future-step state reuse。
