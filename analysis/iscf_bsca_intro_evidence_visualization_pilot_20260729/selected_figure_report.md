# Introduction Problem-Evidence Selected Figure Report

## 1. Selection result

| Figure | Selected dataset/model | Role | Status |
| --- | --- | --- | --- |
| Prefix disagreement | Weather / DLinear / seed2021 | P2后的system-level inconsistency illustration | selected |
| Sharing-demand heterogeneity | ETTm1 / neutral matched scales / seed2021 | P4后的output-side problem illustration | selected |

两张图均为validation-only illustrative evidence，不承担formal prevalence、
cross-seed stability或ISCF-BSCA effectiveness gate。Figure selection不用于否定或
修改fixed ISCF-BSCA architecture与论文逻辑。

## 2. Figure 1: cross-horizon prefix disagreement

Selected artifacts：

- `selected_figures/prefix_weather/prefix_disagreement_overlay.svg/png`；
- `selected_figures/prefix_weather/prefix_disagreement_heatmap.svg/png`；
- `selected_figures/prefix_weather/pair_metrics.csv`。

展示设计：

1. full forecast context展示相同history下四个独立horizon-specific forecasts；
2. 下半panel固定以H720 forecast为参考，显式绘制共同96-step prefix上的prediction
   differences；
3. heatmap汇总六个horizon pairs的NCHPD。

origin与channel均使用约85% disagreement quantile，不是maximum或top-1%样本。
该图支持“independently optimized horizon-specific models can disagree on the same
future steps”，不声称所有model families或datasets上的disagreement都同样明显。

Caption draft：

> Independently optimized horizon-specific forecasters can assign different
> values to the same future steps. The lower panel magnifies prediction
> differences relative to the $H=720$ forecast for a purposefully selected,
> non-extreme 85th-percentile example; the heatmap summarizes normalized
> disagreement over all validation origins.

## 3. Figure 2: future-region sharing-demand heterogeneity

Selected artifacts：

- `selected_figures/sharing_ettm1/sharing_demand_visualization.svg/png`；
- `selected_figures/sharing_ettm1/region_risk.csv`；
- `selected_figures/sharing_ettm1/step_risk.csv`。

Matched controls：

- scales=`{1,8,32,128,720}`；
- all variants=111,312 parameters；
- same raw-history encoder、candidate-state generator、step-specific synthesis；
- only parameter-free latent-state pooling extent changes；
- uniform full-domain pointwise MSE；
- same seed、optimizer、checkpoint rule与validation origins；
- test accessed=false。

Selected ETTm1 pattern：

- global best fixed scale=`s128`；
- region 1的descriptive best scale=`s1`；
- most middle/long regions由`s128`占优；
- margin-qualified crossing pairs：
  `s1_vs_s720`与`s8_vs_s720`；
- s1相对s720的region risk由region 1的`-2.281%`变为region 3的`+0.698%`；
- s8相对s720由region 1的`-1.565%`变为region 3的`+0.509%`；
- multiple region winners与qualified crossover共同满足
  `crossover_visualization_candidate=true`。

Figure panels：

1. all five sharing extents的region-wise relative-risk landscape；
2. first qualified crossing pair与best-fixed scale的step-wise smoothed risk；
3. displayed scales相对best-fixed scale的region-wise risk contrast。

Caption draft：

> The preferred latent-state sharing extent changes across the future domain
> even in a capacity-matched neutral decoder family. Fine sharing is favored
> in the earliest ETTm1 region, whereas the broader $s=128$ setting dominates
> most subsequent regions. The curves and region-wise contrasts expose
> margin-qualified ordering reversals without using the proposed ISCF or BSCA
> mechanisms.

## 4. Paper placement and claim boundary

- Figure 1插在Introduction P2与CHPC formulation P3之间；
- Figure 2紧跟P4的future-region sharing-demand hypothesis；
- Problem Formulation章节给出NCHPD、region risk与matched neutral family定义；
- Experiments章节不重复“问题是否存在”，而是展示ISCF-BSCA缓解程度。

允许的Introduction claim：

1. horizon-specific systems can produce inconsistent overlapping prefixes；
2. a fixed output-side sharing extent need not be uniformly preferred across
   the forecast domain；
3. these observations motivate a horizon-agnostic decoder with adaptive
   output-side sharing.

禁止把两张illustrative figures写成cross-dataset prevalence、formal causal proof
或最终method effectiveness evidence。

Decision=`two_introduction_problem_evidence_figures_selected`。
