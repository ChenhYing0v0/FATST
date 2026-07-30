# Introduction Concept Figure Contract and QA

## Status

| Field | Content |
| --- | --- |
| `figure_id` | `figure_intro_conceptual_problem` |
| `status` | `draft_pending_visual_review` |
| `manuscript_role` | Introduction概念说明，不承担empirical evidence |
| `backend` | Python/matplotlib only |
| `data_role` | deterministic constructed curves |
| `empirical_data_used` | false |
| `test_accessed` | false |

## Figure contract

**Core conclusion.** Varied-horizon forecasting包含两个不同但相连的问题：
horizon-specific systems可能对同一future step给出不一致预测，而finite-capacity
decoder在不同future regions的合适cross-step sharing extent也可能不同。

**Archetype.** Two-panel schematic-led composite。

**Panel map.**

- panel a：三条constructed horizon-specific trajectories在共同future step
  $\tau^\star$给出不同值，并分别终止于$H_1,H_2,H_3$；
- panel b：fine、intermediate与broad sharing的constructed risk curves发生
  crossing，early/middle/late regions的最低曲线依次改变。

**Evidence hierarchy.** 两个panels只建立直观定义。正式problem formulation、
statistics、controls与real-data evidence全部放在Section 3。

**Reviewer risk.** 最大风险是constructed curves被误读成experiment result。图底
显式写明`Conceptual illustration with constructed curves; not empirical data`，
caption再次声明两panel均不使用empirical measurements。

## Construction

所有曲线由
`scripts/plot_intro_concept_figure.py`
中的固定解析函数生成，不使用dataset、checkpoint、prediction artifact、随机数、
sampling或label selection。两个panel使用彼此独立的restrained color families：

- panel a使用muted rose--indigo--teal区分$H_1,H_2,H_3$；
- panel b使用rust--blue--green区分fine/intermediate/broad sharing。

颜色不跨panel建立映射，避免暗示forecast horizon与sharing extent之间存在
一一对应关系。

## Export contract

- exact width：183 mm；
- SVG/PDF：editable text；
- PNG：300 dpi；
- TIFF：600 dpi，LZW；
- final visual status：pending user review；
- source data file：not applicable，because the figure is a declared schematic。

## QA result

| Check | Result |
| --- | --- |
| Nature static preflight | 13 PASS / 1 WARN / 0 FAIL |
| Remaining warning | static parser cannot resolve the width constant；PDF media box verifies 183.000 mm |
| PNG | 2161 × 870，300 dpi |
| TIFF | 4322 × 1740，600 dpi，LZW |
| SVG editable text | pass |
| PDF pages | 1 |
| empirical data access | false |
| random/sampling path | none |
| final-size visual inspection | pass for draft delivery |

Figure 1仍标记为`draft_pending_visual_review`。只有用户确认视觉版本后，才会复制
到`paper-figures/`并改为`approved_for_manuscript_draft`。
