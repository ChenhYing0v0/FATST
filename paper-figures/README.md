# Paper Figures

本目录存放已经通过当前视觉审阅、可直接进入论文草稿的确定性图片资产。
探索过程、source data、绘图脚本和完整QA记录仍保留在`analysis/`与`scripts/`
中，不在本目录重复维护。

## Current assets

| Figure | Manuscript role | Status | Evidence boundary |
| --- | --- | --- | --- |
| `figure_intro_prefix_disagreement` | Introduction：baseline prefix disagreement | `approved_for_manuscript_draft` | validation-only illustrative evidence |
| `figure_intro_sharing_heterogeneity` | Introduction：future-region sharing-demand heterogeneity | `approved_for_manuscript_draft` | validation-only illustrative evidence |

每张图提供以下格式：

- `SVG`：首选可编辑矢量源；
- `PDF`：论文排版与投稿用矢量版本；
- `PNG`：快速预览与文档插图；
- `TIFF`：600 dpi高分辨率投稿版本。

## Maintenance rule

本目录中的文件是论文侧稳定副本，不应直接手工修改。后续若图件内容、数据、
caption contract或视觉编码发生变化，应先在原绘图脚本中完成修改和QA，再经审阅
后同步覆盖本目录。当前canonical生成脚本为
`scripts/plot_intro_problem_evidence_final.py`，完整来源与边界记录见
`analysis/iscf_bsca_intro_evidence_full_search_20260730/`。
