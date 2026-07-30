# Introduction Evidence Full Search Result and Figure Selection

## 1. Result status

| Field | Content |
| --- | --- |
| `protocol` | `SC-UVHF-INTRO-EVIDENCE-FULL-SEARCH-v1` |
| `date` | `2026-07-30` |
| `matrix` | 25/25 neutral sharing runs + 20/20 DLinear horizon-specific runs |
| `datasets` | ETTh1, ETTh2, ETTm1, ETTm2, Weather |
| `seed` | 2021 |
| `split` | validation |
| `test_accessed` | false |
| `prefix_selection` | ETTh2, origin 805, channel 0 |
| `sharing_selection` | ETTm2, origin 4177, all channels |
| `paper_role` | Introduction illustrative problem evidence |

全部same-origin history/target alignment gaps为0；五个neutral variants参数量均为
111,312。retry1从`13:27:46`运行至`13:36:56`，在约9分10秒内完成其
restart-safe remaining matrix并生成全部analysis。global queue没有再形成GPU 0
critical path。

## 2. Prefix-disagreement audit

### 2.1 Five-dataset ranking

| Dataset | Maximum joint score | Macro NCHPD | Macro RDA | Search cells |
| --- | ---: | ---: | ---: | ---: |
| Weather | 0.263774 | 0.009375 | 0.025092 | 95,571 |
| ETTh2 | 0.143870 | **0.026681** | 0.071739 | 15,127 |
| ETTm2 | 0.143173 | 0.019185 | **0.074535** | 75,607 |
| ETTh1 | 0.079118 | 0.017926 | 0.028430 | 15,127 |
| ETTm1 | 0.070621 | 0.016164 | 0.033066 | 75,607 |

Weather按maximum joint cell score排名第一，但其selected channel的ground-truth
尺度与forecast形态使full overlay不够自然。ETTh2在三个互补标准上更均衡：

1. maximum joint score排名第二，且与ETTm2接近；
2. macro NCHPD为五dataset最高；
3. macro RDA排名第二，且selected raw-value differences清晰。

因此最终illustrative Figure 1选择ETTh2，而不是机械使用maximum-cell ranking
第一的Weather。该选择属于预先声明的跨dataset visual audit；不是formal metric
gate，也不隐藏Weather的排名。

### 2.2 Selected ETTh2 evidence

选中origin=805、channel=0，在15,127个origin-channel candidates中aggregate
six-pair score最大。共同96-step prefix中，相对H720：

| Requested horizon | Mean absolute raw-value difference |
| --- | ---: |
| H96 | 2.51 |
| H192 | 2.16 |
| H336 | 2.40 |

全validation origins/channels上的NCHPD：

| Pair | NCHPD |
| --- | ---: |
| H96 vs H192 | 0.0166 |
| H96 vs H336 | 0.0148 |
| H96 vs H720 | 0.0406 |
| H192 vs H336 | 0.0150 |
| H192 vs H720 | 0.0365 |
| H336 vs H720 | 0.0366 |

[Fact] 该图足以证明independently optimized horizon-specific DLinear systems可对
同一history、同一future steps给出明显不同的predictions。

[Boundary] maximum example不证明这种幅度具有population prevalence；heatmap只
补充all-validation average disagreement，不把DLinear结论泛化到ElasTST等
varied-horizon invariant methods。

## 3. Sharing-demand heterogeneity audit

### 3.1 Five-dataset ranking

五个dataset都能找到由全部五个sharing extents稳定占据至少两个regions的
validation origin；因此前三个lexicographic fields及10/10 qualified crossing
pairs形成tie。后续winner margin与headroom把ETTm2排在第一：

| Dataset | Mean winner margin | Oracle headroom | Best fixed scope |
| --- | ---: | ---: | ---: |
| ETTm2 | **10.266%** | **8.112%** | 720 |
| ETTm1 | 3.605% | 5.090% | 32 |
| Weather | 1.511% | 1.777% | 1 |
| ETTh2 | 1.417% | 1.751% | 1 |
| ETTh1 | 0.568% | 0.423% | 32 |

### 3.2 Selected ETTm2 evidence

ETTm2 origin=4177的12个60-step regions由以下sharing extents获胜：

```text
region:  1   2   3   4  5  6   7  8   9   10   11   12
winner: 128 128 128  8  8  1  32  1  32  720  720  720
```

Winner counts为：

```text
s1=2, s8=2, s32=2, s128=3, s720=3
```

全部10个scope pairs均出现超过冻结0.5% margin的bidirectional ordering
crossing。相对sample-best fixed scope s720，各region descriptive winner的MSE
改善约为：

```text
14.6%, 30.7%, 18.0%, 16.3%, 6.5%, 10.8%,
3.1%, 4.3%, 6.1%, 0%, 0%, 0%
```

12-region平均的descriptive region-oracle headroom为8.112%。

[Strong Evidence] 在该sample上，一个固定cross-step sharing extent不能同时实现
各future regions的最低matched risk；最佳extent覆盖从s1到s720的完整诊断范围，
不是继续集中于s128。

[Boundary] region winners与8.112% headroom使用同一validation labels进行
descriptive计算，因此只证明illustrative finite-capacity heterogeneity，不证明
一个learned allocation可以out-of-sample兑现oracle headroom。正式method
effectiveness仍由ISCF-BSCA matched test evidence承担。

## 4. Final figure design

### Figure 1: Prefix disagreement

1. panel a只展示48-step history与真正共同的96-step future prefix；
2. panel b显示H96/H192/H336相对H720的raw-value difference及mean absolute
   difference；
3. panel c显示全validation origins/channels的six-pair NCHPD heatmap。

这避免用完整720-step轨迹稀释重叠区域，同时把selected example和dataset-level
average evidence明确分开。

### Figure 2: Sharing-demand heterogeneity

1. panel a展示五个sharing extents相对sample-best fixed s720的12-region风险面；
2. 每个region winner用空心方框标出，不再用高噪声折线连接；
3. panel b以winner-colored bars展示每个region相对fixed s720实际实现的risk
   reduction；零高度regions仍以绿色marker标出s720 winner。

这使“winner identity发生变化”和“变化具有多大风险收益”在同一图中可直接读取。

### Proposed manuscript captions

**Figure 1 | Independently optimized horizon-specific forecasts can disagree
on the same future steps.** **a**, Predictions from four DLinear models trained
separately for horizons 96, 192, 336, and 720 on the same ETTh2 history. The
panel shows the final 48 observed steps and the first 96 future steps shared by
all four requested horizons. **b**, Prediction differences relative to the
720-step model on this common prefix. The displayed validation origin-channel
pair maximizes mean absolute disagreement aggregated over all six horizon
pairs among 15,127 candidates. **c**, Normalized cross-horizon prefix
disagreement (NCHPD) averaged over all ETTh2 validation origins and variables.
The selected example is illustrative and is not a prevalence estimate.

**Figure 2 | Preferred cross-step sharing extent varies across future
regions.** **a**, Region-wise MSE of five capacity-matched neutral decoders on
one ETTm2 validation example, expressed relative to the best fixed decoder
($s=720$). Each column aggregates 60 future steps and all variables; outlined
squares mark the lowest-risk sharing extent. **b**, MSE reduction of each
region winner relative to the fixed $s=720$ decoder, with colors denoting the
winning extent. All five extents win two or three regions, and the descriptive
region-wise minimum yields 8.1% lower average MSE than the best fixed extent.
The example is selected on validation labels and does not represent
out-of-sample routing performance.

## 5. QA

| Check | Result |
| --- | --- |
| Nature source preflight | 13 PASS / 1 WARN / 0 FAIL |
| backend | Python/matplotlib only |
| editable SVG/PDF text | pass；SVG包含61/75个text nodes |
| PNG | 300 dpi export |
| TIFF | 600 dpi；4388×2570与4388×2083 |
| PDF | one page each；约185.8 mm wide before manuscript scaling |
| visual inspection | labels、legends、winner markers、colorbars与footer可读 |

唯一WARN为static validator无法从`DOUBLE_COLUMN_WIDTH=183/25.4`变量解析final
width。实际PDF约185.8 mm，插入manuscript时统一缩放至183 mm；该1.5%缩放后最小
文字仍高于5 pt。

## 6. Decision and failure attribution

Decision=`two_intro_figures_pass_illustrative_gate_etth2_prefix_ettm2_sharing`。

- `hypothesis_false`：否；两项illustrative existence均得到清晰支持；
- `intervention_point_wrong`：未见；
- `readout_or_head_design_wrong`：未见；
- `optimization_or_numeric_pathology`：未见；45/45 artifacts完整且alignment pass；
- `capacity_control_explains`：不适用于prefix；sharing family参数与主要path匹配，
  但本轮不把sample oracle包装为method gain。

停止dataset/sample search。下一步把两图与准确caption整合进Introduction和
Problem Formulation；不新增method、loss或router。
