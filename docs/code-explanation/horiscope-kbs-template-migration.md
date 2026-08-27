# HoriScope KBS 模板迁移说明

## 目标与边界

本次工作将冻结的 HoriScope 手稿迁移到作者提供的 PDT `Knowledge-Based Systems` LaTeX 模板中。迁移仅改变投稿容器、作者信息、声明模块、highlights 与必要的浮动体控制，不改动论文的科学内容、实验数值、图片数据或参考文献条目。

## 文件组织

- `Elsevier_template/elsarticle-template-num.tex`：KBS `final,3p,times` 主入口；
- `Elsevier_template/sections/`：Sections 1--7 与 Appendix A--C；
- `Elsevier_template/tables/`：主文和附录表格；
- `Elsevier_template/figures/`：论文使用的八个冻结 figure assets；
- `Elsevier_template/ref.bib`：冻结 bibliography database；
- `Elsevier_template/highlights.tex`：嵌入模板 front matter 的 highlights；
- `Elsevier_template/highlights.txt`：投稿系统使用的可编辑纯文本版本；
- `Elsevier_template/HoriScope_KBS_submission.pdf`：经审计的作者审阅版 PDF。

## 迁移流程

`scripts/migrate_horiscope_to_kbs_template.py` 对作者提供的 KBS 主模板执行结构化注入：保留 `elsarticle` class 与 KBS 期刊配置，分别替换 preamble、front matter、main text inputs 和 back matter。Sections、tables、figures 与 `ref.bib` 作为独立文件复制到模板库，避免通过一次性全文替换破坏模板结构。

迁移稿使用 `xurl` 改善长 DOI/URL 的断行，并在 front matter 期间关闭 PDF page anchors，避免 highlights 页与正文首页产生重复锚点。Appendix A 末尾增加一个 `\FloatBarrier`，保证 Table A.3 在 Appendix B 标题之前排出；这是唯一作用于冻结 section 文件的排版差异。

## 一致性与排版审计

`scripts/audit_horiscope_kbs_submission.py` 检查以下约束：

1. KBS class、journal metadata、title、abstract 与 keywords；
2. Sections 1--7、tables、figures 与 `ref.bib` 的 byte identity；
3. Appendix 仅包含已声明的 `\FloatBarrier` 差异；
4. citation keys、figure/table paths 与 highlights 长度；
5. PDT 遗留 assets 是否已从投稿库移除；
6. LaTeX log 是否存在 overfull box、oversized float、undefined citation/reference 或编译错误；
7. 根目录作者审阅 PDF 是否与审计 build 完全一致。

当前审计结果记录于 `analysis/horiscope_kbs_template_migration_20260827/audit_report.md`。最终 PDF 为 A4、24 页；主文、Figure 4、主实验图表、Appendix A/B 表格及 Appendix C 可视化均已完成逐页视觉检查。

## 投稿前作者确认

作者姓名、单位、CRediT、基金、Acknowledgments、Competing Interest 与 Data Availability 来自作者提供的 PDT KBS 原稿。它们不属于本次科学内容迁移的一致性证明范围，正式投稿前仍需由作者逐项确认。
