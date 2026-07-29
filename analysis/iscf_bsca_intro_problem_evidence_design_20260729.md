# ISCF-BSCA Introduction 两组问题证据实验设计

## 1. 文档状态

| Field | Content |
| --- | --- |
| `date` | `2026-07-29` |
| `document_role` | Introduction P2 与 P4 的问题存在性证据设计 |
| `internal_protocol_id` | `SC-UVHF-INTRO-EVIDENCE-v1` |
| `current_step` | Step 6 design freeze；implementation/prelaunch pending |
| `paper_candidate` | frozen `ISCF-BSCA-v1`；本设计不修改最终方法 |
| `experiment_1` | horizon-specific baseline prefix disagreement |
| `experiment_2` | future-region sharing-demand heterogeneity |
| `new_training_authorized` | initial 9-run exploratory visualization pilot only |
| `formal_test_authorized` | false |
| `decision_status` | `formal_design_deferred_visualization_pilot_step7b_pass` |

本设计将两项证据拆成两条独立链：

1. **P2 证据**回答：标准horizon-specific多模型系统是否会对同一个future
   time step给出不同预测？
2. **P4 证据**回答：在matched finite-capacity decoder中，适合future steps
   共享的latent-state范围是否随future region稳定变化？

两项实验不合并成“问题—方法”三联图，也不使用最终ISCF-BSCA的内部结果反向
证明研究动机。

## 2. 共同边界

### 2.1 可以建立的结论

- Experiment 1可建立：
  `independently trained horizon-specific models do not constitute a
  cross-horizon prefix-consistent forecasting system`。
- Experiment 2可建立：
  `a single fixed cross-step sharing extent can incur region-dependent
  finite-capacity risk, leaving out-of-sample headroom for region-varying
  sharing`。

### 2.2 不允许越界的结论

- Experiment 1不证明horizon-specific models的accuracy更差，也不证明所有
  varied-horizon methods都缺少CHPC。ElasTST明确研究了inference-horizon
  invariance，因此本文的对象是prevailing horizon-specific protocol，而不是
  “所有已有模型”。
- Experiment 2不证明ISCF的allocation一定能学到最优scope，不证明BSCA有效，
  也不证明canonical ISCF scope set或contiguous partition必然最优。
- validation只用于checkpoint selection、实现健康检查与figure pipeline调试；
  正式问题存在性判断必须在冻结的official-test matrix上完成。
- 本项目的official test已用于历史candidate决策，因此两项新证据均必须标记为
  `test_informed`，不得宣称使用了untouched holdout。正式audit仍需记录
  `test_access_date`、user authorization、candidate/config version、checkpoint
  hashes、matrix completeness与全部negative cells。
- D18 specialists、A6/ISCF historical arms和frozen replacement只可做pipeline
  smoke或sensitivity，不能作为Introduction的primary problem evidence。

## 3. Experiment 1：Horizon-Specific Prefix Disagreement

### 3.1 研究命题

对固定forecast origin、相同history $\mathbf X$ 和两个horizons
$H_i<H_j$，分别训练的horizon-specific models
$f_{\theta_{H_i}}$与$f_{\theta_{H_j}}$没有结构约束保证：

$$
f_{\theta_{H_i}}(\mathbf X)
=
\operatorname{Prefix}_{H_i}
\left(
f_{\theta_{H_j}}(\mathbf X)
\right).
$$

实验要检验的不是“理论上没有保证”这一显然事实，而是这种disagreement在标准
benchmark checkpoints上是否具有可见且稳定的经验规模。

### 3.2 Baseline 与复用策略

正式baseline families冻结为：

1. `DLinear`：lightweight linear family；
2. `PatchTST`：patch-based Transformer family；
3. `iTransformer`：variate-token Transformer family。

正式结果必须来自各方法的native upstream implementation或经过source-faithful
审计的adapter。本仓库现有`baselines/dlinear`只是local adaptation/sanity
floor，不能直接标成official reproduction；PatchTST与iTransformer当前也尚未
在仓库中形成native paper-facing runs。

不为这张motivation figure单独重复训练一个完整baseline矩阵。正式执行时，把
per-origin predictions导出接入主实验的horizon-specific checkpoints：

$$
3\ \text{families}
\times
5\ \text{datasets}
\times
4\ \text{horizons}
\times
3\ \text{seeds}
=180
\ \text{main-baseline checkpoints}.
$$

这180个checkpoints属于后续主结果本来就需要的矩阵；本实验的新增成本只应是
prediction export与analysis。若需要先验证evaluator，可使用D18/A6 artifacts
做`pipeline_smoke_only`，但D18缺少完整H720 specialist且训练角色不同，不进入
论文图表。

### 3.3 数据与配对单位

- datasets：`ETTh1, ETTh2, ETTm1, ETTm2, Weather`；
- horizons：$\mathcal H=\{96,192,336,720\}$；
- seeds：$\{2021,2022,2023\}$；
- input length、split、scaler、channel order和forecast origins在同一model
  family内部严格一致；
- 每个horizon checkpoint按其native horizon-specific validation protocol选择；
- 所有horizon pairs共享完全相同的test origins；
- prediction先恢复到统一物理尺度；若paper main table在standardized space
  评估，则同时保留standardized predictions供contract核验。

建议保存：

```text
pred[m, d, H, seed, origin, tau, channel]
target[d, origin, tau, channel]
origin_timestamp[d, origin]
train_scale[d, channel]
```

### 3.4 Primary statistic

对$H_i<H_j$，先定义raw L1 disagreement：

$$
\operatorname{CHPD}_{1}
(m,d,H_i,H_j)
=
\mathbb E_{o,c}
\left[
\frac{1}{H_i}
\sum_{\tau=1}^{H_i}
\left|
\hat y^{(H_i)}_{o,\tau,c}
-
\hat y^{(H_j)}_{o,\tau,c}
\right|
\right].
$$

跨变量与dataset比较使用train-split scale归一化：

$$
\operatorname{NCHPD}_{1}
=
\mathbb E_{o,c}
\left[
\frac{
\frac{1}{H_i}\sum_{\tau=1}^{H_i}
\left|
\hat y^{(H_i)}_{o,\tau,c}
-
\hat y^{(H_j)}_{o,\tau,c}
\right|
}{
\sigma^{\mathrm{train}}_{d,c}+\epsilon
}
\right].
$$

使用train-split $\sigma$而不是test target variance，避免test labels参与
normalization定义。若输入输出已按同一train scaler标准化，则应验证两种计算
数值等价。

每个model family共有六个horizon pairs：

$$
(96,192),(96,336),(96,720),
(192,336),(192,720),(336,720).
$$

### 3.5 Secondary statistics

1. squared disagreement：

   $$
   \operatorname{CHPD}_{2}
   =
   \mathbb E
   \left[
   \left(
   \hat y^{(H_i)}-\hat y^{(H_j)}
   \right)^2
   \right];
   $$

2. per-origin disagreement distribution，而不只报告均值；
3. relative disagreement amplitude，仅作materiality描述：

   $$
   \operatorname{RDA}
   =
   \sqrt{
   \frac{\operatorname{CHPD}_{2}}
   {\frac12(
   \operatorname{MSE}_{H_i}^{\mathrm{overlap}}
   +
   \operatorname{MSE}_{H_j}^{\mathrm{overlap}})
   +\epsilon}
   };
   $$

4. NCHPD随$H_j-H_i$变化的curve；
5. model count、stored parameters与total training cost另入efficiency table，
   不混入disagreement统计量。

### 3.6 Evaluator controls

1. `self-replay-zero`：同一checkpoint/prediction artifact与自身比较，
   $\operatorname{CHPD}=0$；
2. `unified-replay-zero`：同一horizon无关checkpoint按两个supported horizons
   读取overlap，$\operatorname{CHPD}=0$；
3. `origin-alignment`：打乱或错位一个origin必须被hash/timestamp guard拒绝；
4. `scale-roundtrip`：standardized与inverse-transformed计算在归一化后匹配；
5. `prefix-shape`：每个pair只能比较$\tau\leq H_i$，禁止padding参与统计。

### 3.7 Introduction Figure 1

Figure 1放在P2与P3之间，建议只保留两个主panel：

**Panel A — Same-history forecast overlay**

- pre-register为`Weather / PatchTST / seed2021`；
- 在该cell内按六个pair的mean NCHPD为每个test origin打分；
- 选择最接近median的origin，而不是最大disagreement样本；
- 画observed history、ground truth以及四条horizon-specific forecasts；
- 对$1{:}96$ overlap做局部放大，标出同一future time step的预测差异。

该选择规则只依赖predictions，不依赖test error，避免用ground truth挑选“漂亮”
案例。

**Panel B — Horizon-pair disagreement heatmaps**

- 每个baseline family一个$4\times4$ upper-triangular small multiple；
- cell颜色为五datasets、三seeds的macro
  $\operatorname{NCHPD}_{1}$；
- cell内可写mean，完整dataset/seed intervals放Problem Formulation或Appendix；
- 对角线固定为0并以浅灰表示，不把缺失下三角误画成0。

建议caption核心句：

> Independently optimized horizon-specific models can assign different
> values to the same future time step. The heatmaps quantify this disagreement
> over all overlapping horizon pairs, while the trajectory panel shows a
> pre-registered median-disagreement example.

### 3.8 决策规则与claim边界

本实验不需要用任意$p$值证明“非零”，因为独立floating-point models几乎必然
不完全相同。需要判断的是能否作为material visual motivation：

- `material_prefix_disagreement_supported`：
  至少$2/3$ families同时满足macro
  $\operatorname{NCHPD}_1\geq0.01$与
  $\operatorname{RDA}\geq0.10$，并且这些families各自在至少$4/5$
  datasets上达到相同阈值。前者表示平均prediction disagreement达到train
  scale的1%，后者表示disagreement amplitude达到平均forecast-error
  amplitude的10%；
- `formal_property_only_material_motivation_weak`：
  CHPD非零但相对train scale和forecast error极小，P2只保留“无保证”的formal
  argument，不把Figure 1作为核心motivation；
- `unresolved`：
  origin、scaler、checkpoint protocol或artifact provenance不匹配。

无论结果大小，都不能写成“horizon-specific forecasting is inaccurate”或
“all existing models violate CHPC”。

## 4. Experiment 2：Future-Region Sharing-Demand Heterogeneity

### 4.1 研究命题

固定history、pointwise MSE且requested horizon不携带额外信息时，同一个future
time step的Bayes conditional mean不因requested horizon改变。本实验因此不输入
$H$，也不设计horizon embedding/router。它检验的是finite-capacity问题：

> 当decoder只允许一种固定的cross-step latent-state sharing extent时，不同
> future regions的matched empirical-risk optimum是否稳定不同？

这是P4所称`future-region sharing-demand heterogeneity`的可检验表现。

### 4.2 Primary carrier：neutral single-scale decoder family

内部diagnostic ID为`SC-UVHF-FRSD-D1`。每次training只包含一个sharing
extent，不含multiple scopes、allocation、fusion、arm loss或BSCA。

#### Tensor path

给定：

$$
\mathbf X\in\mathbb R^{B\times L\times C},
$$

使用channel-wise raw-history MLP encoder：

$$
\mathbf R
=
E_\psi(\mathbf X)
\in
\mathbb R^{B\times C\times D}.
$$

为每个future time step设置共享的step descriptor：

$$
\boldsymbol\phi_\tau\in\mathbb R^{D_\tau},
\qquad
\tau=1,\ldots,T.
$$

先对每个future time step计算相同parameterization的candidate state：

$$
\mathbf U_{b,c,\tau}
=
G_\omega
\left(
\left[
\mathbf R_{b,c},
\boldsymbol\phi_\tau
\right]
\right)
\in
\mathbb R^{D_z}.
$$

对sharing extent $s$，把future domain按固定连续blocks
$\mathcal G_s=\{\mathcal G_{s,g}\}$分组，并计算：

$$
\mathbf Z^{(s)}_{b,c,g}
=
\operatorname{LayerNorm}
\left(
\frac{1}{|\mathcal G_{s,g}|}
\sum_{\tau\in\mathcal G_{s,g}}
\mathbf U_{b,c,\tau}
\right).
$$

每个future time step仍由自己的synthesis vector生成：

$$
\hat y^{(s)}_{b,c,\tau}
=
\left\langle
\mathbf a_\tau,
\mathbf Z^{(s)}_{b,c,g_s(\tau)}
\right\rangle
+b_\tau.
$$

因此：

- $s=1$时，每个future step使用自己的latent state；
- $s=T$时，全部future steps复用同一个latent state；
- 中间$s$控制连续future steps之间共享state的范围；
- $\mathbf a_\tau$始终step-specific，不能把broad sharing误解为所有steps输出
  同一个数；
- $E_\psi,G_\omega,\phi_\tau,\mathbf a_\tau,b_\tau$的shape和parameter count在
  所有$s$下完全相同；
- 所有$s$都先计算完整$\mathbf U$，只有parameter-free pooling topology不同，
  避免$s=1$比$s=T$多执行$G_\omega$造成大幅compute confound。

v1冻结：

```text
L=96, T=720, D=64, D_tau=32, D_z=64
E: channel-wise MLP 96 -> 128 -> 64
G: MLP 96 -> 128 -> 64
sharing extents S_diag={1, 8, 32, 128, 720}
```

$S_{\mathrm{diag}}$采用独立于最终ISCF中间scopes的粗略log-spaced grid，避免用
最终method配置循环定义问题。最后一个不足$s$的block自然保留，不padding进loss。
这些数值在Step7A只做parameter/shape/compute contract确认，不得根据
validation/test winner调整。若contract无法成立，必须形成显式v1.1设计修订，
不能静默改grid或width。

### 4.3 Training protocol

- datasets：五dataset标准suite；
- seeds：$\{2021,2022,2023\}$；
- primary matrix：

  $$
  5\ \text{scales}
  \times
  5\ \text{datasets}
  \times
  3\ \text{seeds}
  =
  75\ \text{end-to-end runs};
  $$

- primary objective使用uniform full-domain pointwise MSE：

  $$
  \mathcal L_{\mathrm{uniform}}
  =
  \frac{1}{TC}
  \sum_{\tau=1}^{T}\sum_c
  \left(
  \hat y_{\tau,c}-y_{\tau,c}
  \right)^2;
  $$

- 使用uniform full-domain loss是为了避免四个prefix losses造成early steps重复
  加权，从而把supervision exposure误当成data/model sharing demand；
- requested horizon不进入model或loss；
- checkpoint selector仍按项目默认的validation
  `mean MSE over {96,192,336,720}`，所有scales完全相同；
- optimization、epoch budget、early stop、normalization、batch order class和
  initialization class全部匹配；
- validation只能选checkpoint并检查pathology，不选择scale set、future bins或
  claim方向；
- official test只允许在75/75 checkpoints、hash、config与matrix全部冻结后一次
  性执行。

`mean-four-prefix MSE`可在primary conclusion冻结后作为objective sensitivity，
但不是首轮必要matrix，避免把motivation experiment膨胀成loss search。

### 4.4 Future regions

Primary statistical regions不沿用requested horizons，也不沿用ISCF scopes：

$$
\mathcal B_b
=
\{60(b-1)+1,\ldots,60b\},
\qquad
b=1,\ldots,12.
$$

即12个等长60-step regions。这样可避免把“region preference”预设成
$96/192/336/720$边界。

Secondary reporting可使用benchmark-aligned disjoint regions：

$$
[1,96],\ [97,192],\ [193,336],\ [337,720],
$$

只用于衔接paper horizons，不参与scale选择或primary gate。逐step risk curve
只作可视化，并用固定的31-step centered moving average平滑；统计始终基于未平滑
60-step regions。

### 4.5 Region-wise risk

对dataset $d$、seed $r$、region $\mathcal B_b$和sharing extent $s$：

$$
R_{d,r,b,s}
=
\mathbb E_o
\left[
\frac{1}{|\mathcal B_b|C}
\sum_{\tau\in\mathcal B_b}
\sum_c
\left(
\hat y^{(s)}_{o,\tau,c}
-
y_{o,\tau,c}
\right)^2
\right].
$$

为便于画图，定义region内normalized regret：

$$
\operatorname{Regret}_{d,r,b,s}
=
\frac{
R_{d,r,b,s}
-
\min_{s'}R_{d,r,b,s'}
}{
\min_{s'}R_{d,r,b,s'}+\epsilon
}.
$$

`Regret`只用于展示哪个scale在region内更接近最优；正式risk仍报告原始MSE/MAE。
Introduction主图不使用同一test split的$\min_s$作为reference，而使用下节
validation-selected fixed scale：

$$
\operatorname{RelativeRisk}^{\mathrm{test}}_{d,r,b,s}
=
\frac{
R^{\mathrm{test}}_{d,r,b,s}
-
R^{\mathrm{test}}_{d,r,b,s_{\mathrm{fixed}}^{\mathrm{val}}}
}{
R^{\mathrm{test}}_{d,r,b,s_{\mathrm{fixed}}^{\mathrm{val}}}
+\epsilon
}.
$$

`Regret`只进入Appendix或internal audit。

### 4.6 Validation-selected region schedule

不能用test labels先选每个region的winner再在同一test上报告oracle gain。主统计采用
validation-selected schedule：

1. 在三个seeds的validation predictions上汇总风险；
2. 选择一个全域fixed scale：

   $$
   s_{\mathrm{fixed}}^{\mathrm{val}}
   =
   \arg\min_s
   \sum_b w_bR^{\mathrm{val}}_{b,s};
   $$

3. 对每个region选择：

   $$
   s_b^{\mathrm{val}}
   =
   \arg\min_s
   R^{\mathrm{val}}_{b,s};
   $$

4. 将同一个validation-frozen region schedule应用于三个test seeds；
5. 比较：

   $$
   \operatorname{CFH}_{d}
   =
   \frac{
   R^{\mathrm{test}}_{d,\mathrm{fixed}}
   -
   R^{\mathrm{test}}_{d,\mathrm{region\ schedule}}
   }{
   R^{\mathrm{test}}_{d,\mathrm{fixed}}
   }.
   $$

这里的`CFH`是cross-fitted headroom。region schedule只是由五个独立models拼接的
diagnostic upper bound，不是可部署method，也不证明history-conditioned
allocation可识别。test-oracle best scale可放Appendix，但不得作为primary
support。

### 4.7 Crossover 与稳定性统计

除CFH外，报告：

1. `best_scale_val[b]`与test risk winner的一致率；
2. 一个dataset内validation-selected schedule使用的distinct scales数量；
3. pairwise crossover：

   $$
   \Delta_{b}(s_i,s_j)
   =
   R_{b,s_i}-R_{b,s_j},
   $$

   若不同regions中$\Delta_b$稳定变号，则存在risk crossover；
4. dataset、seed、channel group与region的winner stability；
5. macro CFH、dataset CFH、seed CFH与positive counts；
6. 以forecast-origin为paired unit的risk differences。由于相邻origins的targets
   高度重叠，interval使用moving-block bootstrap，block length固定为720
   origins；若某dataset有效blocks不足，则只报告interval为descriptive并降低
   证据等级。

### 4.8 Controls 与failure attribution

#### 必须通过的matched controls

1. exact parameter-count equality across $s$；
2. same encoder/readout/objective/optimizer/checkpoint selector；
3. same initialization class和paired seed；
4. no non-finite、>100% degradation、boundary-checkpoint pile-up；
5. $s=1$与$s=720$ endpoint models均具有非零gradient和有效forecast；
6. step-specific synthesis row在所有$s$下都实际参与输出；
7. scale label shuffle只能改变analysis ordering，不能改变artifact。

#### Conditional specificity control

如果正文要进一步声称“**contiguous temporal regions**具有特殊意义”，则在
primary support后追加parameter-matched random grouping：

- 只随机化中间scales的group membership；
- group-size multiset保持一致；
- partition seed在训练前冻结；
- ordered-vs-random是contiguity specificity，不是basic
  sharing-demand existence gate。

该control未完成前，允许的claim是region-dependent sharing-extent preference，
不能claim canonical temporal contiguity本身已被证明。

#### Failure attribution

- `hypothesis_false`：无pathology时单一scale稳定支配，或validation-selected
  region schedule在test无正headroom；
- `capacity_control_explains`：参数、compute或generic width差异解释风险变化；
- `readout_or_head_design_wrong`：pooling后数值尺度、step synthesis或endpoint
  function退化；
- `optimization_or_numeric_pathology`：部分scales不收敛、checkpoint堆在预算边界
  或出现异常退化；
- `intervention_point_wrong`：改变的不是claimed latent-state sharing extent；
- `unresolved`：上述设计/pathology问题尚未排除。

frozen replacement或A6 sensitivity的负结果不得将方向标为
`hypothesis_false`。

### 4.9 Introduction Figure 2

Figure 2紧跟P4，建议三panel但只展示问题证据，不提前画ISCF：

**Panel A — Sharing-risk landscape**

- x轴：12个future regions；
- y轴：$s\in\{1,8,32,128,720\}$；
- color：相对validation-selected fixed scale的test `RelativeRisk`；
- 叠加validation-selected best-scale ridge；
- 用边框区分ridge在test上是否仍为region winner。

**Panel B — Risk crossover curves**

- 选择预注册的三个scales：$s=1,32,720$；
- x轴为future step，y轴为relative MSE to
  $s_{\mathrm{fixed}}^{\mathrm{val}}$；
- 31-step smooth只用于显示；
- 阴影为three-seed range或block-bootstrap interval；
- dataset small multiples固定为五个，不只展示正向dataset。

**Panel C — Out-of-sample headroom**

- 五dataset bars加macro diamond；
- 值为validation-selected region schedule相对validation-selected fixed scale
  的test CFH；
- 同时画0线与interval；
- negative datasets必须完整保留。

建议caption核心句：

> Capacity-matched single-scale decoders exhibit region-dependent risk
> crossovers. A region schedule selected only on validation is then evaluated
> on the test split, separating out-of-sample sharing-demand evidence from a
> same-split oracle.

不建议用radar/polygon chart承担这一问题证据，因为面积和轴归一化不能直接显示
risk crossover或out-of-sample headroom。

### 4.10 决策规则

正式test前冻结三类结果：

#### `future_region_sharing_demand_supported`

同时满足：

1. validation-selected schedule在至少$3/5$ datasets使用不少于两个distinct
   scales；
2. macro test CFH $>0$，且至少$3/5$ datasets为正；
3. 至少$2/3$ seeds的macro CFH为正；
4. 至少$3/5$ datasets存在跨seed稳定的pairwise risk crossover；
5. 所有matched/numeric controls通过。

若进一步达到macro CFH $\geq0.5\%$、$4/5$ datasets为正且hierarchical/block
interval下界大于0，可标记为
`strong_intro_level_support`。这个strong gate用于决定是否把结果作为
Introduction headline，不用于事后调scale或region。

#### `future_region_sharing_demand_not_supported`

在matched/numeric controls全部通过时，best fixed scale稳定支配，或macro
CFH $\leq0$且无稳定risk crossover。此时P4必须保留为有限容量直觉或删除
“we demonstrate”表述，不能用最终ISCF结果救回问题存在性。

#### `future_region_sharing_demand_unresolved`

capacity、readout、optimization、numeric或intervention-point问题使exact
diagnostic不能回答命题。只允许回Step 5/6修复diagnostic，不能直接否定或确认
论文方法。

## 5. 执行顺序与授权边界

### Step 7A：local implementation and invariant gate

1. 实现prediction exporter与CHPD evaluator；
2. 实现neutral single-scale decoder；
3. 检查shape、parameter、compute、gradient、endpoint与loss contracts；
4. 只用synthetic/local smoke，不访问formal test。

### Step 7B：prelaunch freeze

1. 冻结native baseline source commits与180-checkpoint main-baseline manifest；
2. 冻结FRSD 75-run manifest、dataset profiles、seeds、selector与hash；
3. 冻结Figure 1/2 layout、sample rule、statistics与decision gates；
4. 写明test exposure和negative-result reporting。

### Step 8--10：只在新授权后执行

- Experiment 1优先复用main baseline trainings，不为motivation figure重复烧算力；
- Experiment 2完成75/75 validation-selected checkpoints后再申请一次formal test；
- test不得用于修改scope grid、region bins、figure sample或objective；
- 若remote run预计耗时，只需启动并记录，不持续短间隔值守。

原formal design本身不构成implementation、remote training或formal-test授权；
下方Section 8记录用户随后单独授权的缩减pilot。

## 6. 与论文段落的对应

| Manuscript location | Evidence | Allowed sentence |
| --- | --- | --- |
| P2后、P3前 | Figure 1：overlay + NCHPD heatmaps | horizon-specific systems can produce materially different overlapping forecasts |
| P3 | CHPC definition | CHPC是varied-horizon system的basic property |
| P4后 | Figure 2：risk landscape + crossover + CFH | matched fixed-sharing decoders exhibit region-dependent finite-capacity risk |
| P5 | ISCF-BSCA response | 只在前述证据成立后引出multiple sharing scopes与allocation |

若Experiment 1只有formal nonzero而缺少material magnitude，P2仍成立但Figure 1
降为Problem Formulation辅助图。若Experiment 2不支持，P4和P5之间不能继续使用
“our evidence shows”作为逻辑桥。

## 7. Primary-source audit

Search date=`2026-07-29`。来源以official proceedings、paper pages与official
repositories为主：

1. ElasTST，NeurIPS 2024 official proceedings：
   https://proceedings.neurips.cc/paper_files/paper/2024/hash/d7aa002885ccbe68cf6880da583761b2-Abstract-Conference.html
   。其inference-horizon invariance要求本文避免“所有已有模型均不一致”的绝对
   claim。
2. Ben Taieb and Hyndman，ICML 2014/PMLR：
   https://proceedings.mlr.press/v32/taieb14.html
   。它明确区分recursive与direct horizon-specific multi-step strategies，并讨论
   bias--variance trade-off；本文的sharing-demand是output-side finite-capacity
   诊断，不能包装成首个multi-step bias--variance讨论。
3. N-HiTS，AAAI 2023 official proceedings：
   https://ojs.aaai.org/index.php/AAAI/article/view/25854
   。其hierarchical interpolation和multi-rate synthesis已覆盖generic
   multi-scale output modeling，因此本文不能以“多尺度future synthesis”本身
   作为novelty。
4. PatchTST official repository：
   https://github.com/PatchTST/PatchTST 。
5. iTransformer official repository：
   https://github.com/thuml/iTransformer 。
6. DLinear paper：
   https://arxiv.org/abs/2205.13504 ；正式implementation阶段需以paper链接的
   source repository为准，并记录commit。

当前Zotero coverage未在本轮核验，因此上述来源记为external
primary-source audit；正式Related Work整理时仍需补library-presence记录。

## 8. Visualization-first pilot amendment（2026-07-29）

用户要求先降低run数量，并允许为Figure 1选择差异较明显但非极端的案例。正式
180-checkpoint/75-run designs不删除，但延后到需要generalization evidence时再
执行。

当前授权pilot：

1. DLinear × Weather × H96/H192/H336/H720 × seed2021，共4 runs；
2. neutral single-scale × Weather ×
   $\{1,8,32,128,720\}$ × seed2021，共5 runs；
3. 只使用validation，不访问test；
4. prefix origin与channel均选择aggregate disagreement的85% quantile nearest
   item，不取maximum/top-1%；
5. Weather得到清晰visualization后立即停止；fallback
   `ETTm1 -> ETTh2`只记录、不授权；
6. same-validation region oracle只作descriptive panel，不能写成out-of-sample
   CFH。

该pilot的角色是`exploratory_visualization_only`。它可以提供Introduction中的
illustrative existence case，但不能替代正式problem-existence、prevalence、
cross-seed stability或out-of-sample headroom evidence。详细实现与prelaunch见
`analysis/iscf_bsca_intro_evidence_visualization_pilot_20260729/step7a_implementation_and_prelaunch.md`。
