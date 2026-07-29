# ISCF-BSCA Introduction Round 1修订记录与问题存在性证据计划

## 1. 文档状态

| Field | Content |
| --- | --- |
| `date` | `2026-07-29` |
| `source_draft` | `docs/paper-drafts/iscf-bsca-introduction-initial-draft.md` v0.1 |
| `revised_draft` | 同文件 v0.2-round1 |
| `revision_role` | 根据作者对blind review的逐项回复形成初步正文，不视为最终冻结 |
| `current_gate` | Introduction positioning与方法概述可继续讨论；problem-existence result pending |
| `new_training_authorized` | false |
| `formal_test_authorized` | false |

## 2. 作者建议的处理决定

### 2.1 接受并已进入正文

1. 多数long-term forecasting benchmark仍采用horizon-specific训练；近期只有有限
   工作和foundation models开始支持varied/flexible horizons。
2. CHPC降为本文定义的varied-horizon forecasting system应具备的基本性质，不作为
   独立算法创新。
3. ElasTST与foundation models只在P1作为少量varied-horizon先例轻量定位，不在
   Introduction展开比较表；主判断是该方向相对horizon-specific研究仍缺少充分、
   系统的发展。
4. Introduction保持本文native decoder主线，不扩展无关结构路线。
5. 现有架构研究的主流注意力概括为history encoding或input-side temporal
   representation；本文的差异集中到output-side cross-step latent-state sharing。
6. ISCF与BSCA在Introduction只保留核心直觉；parameterization、gradient推导与
   mechanism controls分别留给Method、Problem Formulation和Experiments。
7. 不在当前稿中虚构具体性能数字；贡献段只说明需要评估的优势维度。
8. 全面压缩P5术语：删除`latent modes`、`scope-conditioned slices`、
   `weighted contraction`与`broad learning access`等首次阅读负担较大的说法。

### 2.2 Varied-horizon literature的冻结表述

正文按以下强度定位已有工作：

> 大多数long-term forecasting研究与benchmark仍采用horizon-specific设计；
> ElasTST以及少量foundation models已经探索varied/flexible horizons，但相较于
> 大量horizon-specific研究，该方向仍缺少充分、系统的发展。本文进一步对
> varied-horizon forecasting进行明确的task definition、problem analysis与
> targeted output-side decoder design。

Introduction不扩展成prior-art comparison，也不展开与本文native decoder主线
无关的结构路线。更细的结构边界留到Related Work。

## 3. v0.2正文的主要变化

### Paragraph 1

- 加入horizon-specific仍占主流、recent varied-horizon work有限且该任务仍缺少
  充分系统发展的双重定位；
- 轻量提及ElasTST、TimesFM与Time-MoE；
- 明确本文将task definition、problem analysis与targeted decoder design形成一条
  完整研究链，核心research gap落到`output-side representation sharing within
  the future domain`。

### Paragraph 2

- CHPC的前提显式固定为same forecast origin与identical history；
- cost claim收紧为服务多个horizons的total system burden，不声称单次unified
  training天然更省算力。

### Paragraph 3

- CHPC明确写成varied-horizon forecasting的`basic property`；
- 删除future-step-specific synthesis实现细节；
- `arbitrary horizons`改为`every/any supported horizon`；
- 不使用max-$T$-then-crop宏观叙事。

### Paragraph 4

- 先指出history encoder与input-side modeling长期占据主要研究注意力；
- 加入broad/fine sharing的finite-capacity bias--variance桥梁；
- 显式声明问题不改变pointwise-MSE Bayes target；
- 保留一个可搜索的`PENDING MOTIVATION RESULT`占位，不虚构未完成实验结果。

### Paragraph 5

- ISCF只保留`multiple sharing scopes -> step-specific synthesis ->
  target-conditioned soft allocation`；
- BSCA只保留`allocation-mediated gradient access -> train-only
  co-adaptation`；
- 删除Introduction层不必要的tensor、slice和contraction术语；
- 将`mitigates`收紧为`designed to ... reduce`。

### Paragraph 6

- 三项贡献仍按`problem/formulation -> architecture -> training/evidence`组织；
- CHPC只属于Contribution 1中的basic requirement；
- 暂不填具体数字，不预先声称SOTA或全面超越；
- 优势维度收紧为unified deployment、accuracy、CHPC、output-side adaptation与
  transferability。

## 4. 必须补齐的problem-existence evidence

### 4.1 研究问题

在固定history、pointwise MSE且不输入requested horizon时，有限容量decoder是否
存在稳定的：

> region-dependent sharing-scale preference

即不同future regions对cross-step latent-state sharing extent的matched empirical
risk optimum是否不同，以及这种差异相对best fixed sharing extent是否留下material
headroom。

### 4.2 Primary evidence carrier

建议采用两层证据，但只把第一层作为因果主证据：

1. **Primary：neutral matched decoder family**
   - 共享同一简单history encoder；
   - 共享future-step descriptors与step-specific synthesis接口；
   - 只改变history-conditioned latent state在多少连续future steps之间共享；
   - 参数量、训练目标、初始化类、checkpoint selector与optimization尽量匹配；
   - end-to-end共同训练，不使用frozen replacement作方向级判断。
2. **External-validity sensitivity：现有baseline的unified adaptation**
   - 选择一个linear/MLP baseline和一个patch/Transformer baseline；
   - 只用于验证现象不是neutral carrier特有；
   - 不让最终ISCF作为问题存在性的primary evidence。

### 4.3 核心统计量

对future region $\mathcal B_b$与sharing extent $s$：

$$
R_{b,s}
=
\frac{1}{|\mathcal B_b|C}
\sum_{\tau\in\mathcal B_b}
\sum_{c=1}^{C}
\left(
\hat y_{\tau,c}^{(s)}-y_{\tau,c}
\right)^2.
$$

需要报告：

1. `best_scale[b]`：
   $s_b^\star=\arg\min_s R_{b,s}$；
2. `crossing_margin[b,s_1,s_2]`：
   不同sharing-risk curves在region间发生方向反转的幅度；
3. `best_fixed_risk`：
   $\min_s\sum_bw_bR_{b,s}$；
4. `region_oracle_risk`：
   $\sum_bw_b\min_sR_{b,s}$；
5. `adaptive_headroom`：

   $$
   \frac{
   R_{\text{best fixed}}-R_{\text{region oracle}}
   }{
   R_{\text{best fixed}}
   };
   $$

6. 跨dataset、seed、variable group与future-region的sign stability。

### 4.4 Split角色

- validation：冻结scope set、region bins、matched budgets和可视化样式；
- official test：只在完整预注册matrix上报告最终问题存在性；
- 不允许使用test选择scope、region boundary或figure中的有利cells；
- 如果只完成validation，Introduction继续使用hypothesis措辞，不能写
  `we demonstrate`。

### 4.5 Failure attribution

若没有稳定crossing或headroom，需要区分：

- `hypothesis_false`：匹配设计下不存在material region preference；
- `capacity_control_explains`：收益来自参数量或generic multi-head capacity；
- `readout_or_head_design_wrong`：probe没有真正只改变sharing extent；
- `optimization_or_numeric_pathology`：个别scope难以训练；
- `intervention_point_wrong`：sharing发生位置与声称的decoder问题不一致。

负结果只允许拒绝exact diagnostic，除非matched、稳定、end-to-end设计能够直接检验
problem hypothesis。

## 5. 面向Introduction的可视化设计

### 5.1 推荐主图：三联叙事图

建议将Introduction Figure 1设计为从问题到方法的横向三联图：

1. **Panel A — Fragmented horizon-specific forecasts**
   - 同一history下叠加$H=96,192,336,720$的overlapping forecasts；
   - 用局部放大框显示同一future step的预测不一致；
   - 辅以小型CHPC-disagreement heatmap。
2. **Panel B — Sharing-risk landscape**
   - 横轴为future step或future region；
   - 纵轴为sharing extent；
   - 颜色为相对best-fixed decoder的normalized MSE；
   - 叠加一条`best-scale ridge`，直观看出最优sharing extent随future region变化；
   - 用阴影或bootstrap interval标记跨seed稳定性。
3. **Panel C — ISCF-BSCA response**
   - 用不同宽度的region-shared latent states展示multiple sharing scopes；
   - target-conditioned allocation用简化heatmap表示；
   - 输出一条CHPC unified forecast；
   - BSCA以train-only箭头进入，不画成额外inference module。

该图能够同时服务P2、P4与P5，是比单独method architecture图更有叙事力的
Introduction figure。

### 5.2 辅助图

1. `Risk-crossover curves`：不同sharing extents随future step变化的normalized
   MSE曲线；
2. `Dataset × future-region best-scale map`：展示外部有效性；
3. `Best-fixed vs region-oracle headroom bars`：显示问题是否有material规模；
4. `Sample/variable small multiples`：避免平均结果掩盖heterogeneity。

### 5.3 关于多边形/radar图

多边形图可作为graphical abstract或综合能力概览，候选轴包括accuracy、CHPC、
model count、storage、latency与transferability。但它不宜替代main result或
problem-existence evidence：

- 各轴量纲与归一化方式不同；
- 面积受轴顺序影响；
- 视觉面积容易夸大很小的metric差异。

如果使用，应同时保留原始table或bar chart，并在结果冻结前预先定义normalization。
Introduction的primary evidence仍建议使用`sharing-risk landscape`，它与核心问题
一一对应，叙事性更强。

## 6. 当前决定与下一步

Decision=`intro_v02_round1_positioning_pass_problem_evidence_pending`。

下一步按顺序讨论：

1. 确认P3把CHPC定义为basic property的表述；
2. 冻结P4 problem-existence diagnostic与Introduction Figure 1；
3. 再讨论P5中ISCF/BSCA的创新性表达是否足够简洁；
4. main experiments完成后决定P6是否只写qualitative advantage，或加入一条
   compact quantitative headline。

本轮不修改model、loss或training code，不启动remote training或formal test。
