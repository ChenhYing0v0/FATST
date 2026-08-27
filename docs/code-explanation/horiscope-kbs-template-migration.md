# HoriScope PDT-exact flat KBS 投稿源说明

## 目标与边界

本次工作以作者指定的 PDT 成品源文件为唯一版式参考，将冻结的 HoriScope 手稿重新组装为平铺、单主文件的 `Knowledge-Based Systems` 投稿源。迁移不改动论文的科学内容、实验数值、图片数据或参考文献条目。

## 文件组织

- `HoriScope_KBS_submission/elsarticle-template-num.tex`：唯一 manuscript `.tex`；Sections 1--7、Appendices A--C 与所有表格均直接内联；
- `HoriScope_KBS_submission/figure_*.{pdf,png}`：与主 `.tex` 同级的八个冻结 figure assets；
- `HoriScope_KBS_submission/ref.bib`：与主 `.tex` 同级的 bibliography database；
- `HoriScope_KBS_submission/.latexmkrc`、`elsarticle-num.bst`、`math_utils.tex`：从指定 PDT `source_latex/` byte-for-byte 复制的模板依赖；
- `output/pdf/HoriScope_KBS_submission.pdf`：经编译和逐页视觉审计的作者审阅版 PDF。

最终投稿源目录只包含 13 个文件，不包含 `sections/`、`tables/`、`figures/` 或 `build/` 子目录。`math_utils.tex` 是唯一额外 `.tex` 依赖；保留它是为了使 `\journal{Knowledge-Based Systems}` 之前的 PDT preamble 完全一致，论文正文与表格并未拆分到该文件。

## 迁移流程

`scripts/build_horiscope_kbs_flat_submission.py` 直接以指定 PDT `elsarticle-template-num.tex` 为 scaffold。脚本保留 PDT front matter 与 bibliography 结构，将 HoriScope 的 title、abstract、keywords 和 highlights 注入对应环境，再按论文顺序内联 Sections 1--7、声明和 Appendices A--C。表格 `\input` 在构建时展开为完整 LaTeX，figure paths 统一改为当前目录文件名。

脚本不会重写 PDT preamble，也不会加入 `xurl`、额外 page-anchor 设置或 Appendix float barrier。因此，从文件起始至 `\journal{Knowledge-Based Systems}` 之前的内容与参考源完全相同，Appendix 的组织顺序也沿用 PDT 的 inline layout。

## 一致性与排版审计

`scripts/audit_horiscope_kbs_flat_submission.py` 检查以下约束：

1. PDT journal marker 之前的 exact byte identity；
2. flat source inventory 为 13 files/0 subdirectories；
3. main text、Appendix 与八张表已确定性内联；
4. figure assets、`ref.bib` 与 PDT support files 的 byte identity；
5. citation keys、flat figure paths、figure/table environment 数量；
6. PDF page count 及 LaTeX fatal diagnostics；
7. front matter、主图表、Appendix A/B/C 与 references 的逐页视觉检查。

当前审计结果记录于 `analysis/horiscope_kbs_flat_submission_20260827/audit_report.md`。最终 PDF 为 A4、25 页。编译不存在 LaTeX error、undefined control、oversized float 或 undefined citation/reference；PDT 原 scaffold 产生的非致命 duplicate-destination warnings 与 bibliography metadata warnings 已在 audit 中单独披露。

旧的 `Elsevier_template/` 分文件版本保留为历史迁移记录，已不再是 canonical 投稿源。

## 投稿前作者确认

作者姓名、单位、CRediT、基金、Acknowledgments、Competing Interest 与 Data Availability 来自作者提供的 PDT KBS 原稿。它们不属于本次科学内容迁移的一致性证明范围，正式投稿前仍需由作者逐项确认。
