# SIFF-v2 Reactivation 与 TSAF Step 4-6 Narrative/Design Audit

## 1. Decision

| Field | Content |
| --- | --- |
| `current_step` | SIFF paperization reset；Step 4-6 narrative/design gate complete |
| `problem` | unified full-domain decoder如何让不同future coordinates使用不同history coupling scales，同时避免把requested horizon或不可识别的sample-wise arm competence当作信息？ |
| `existence_evidence` | SIFF-v2相对A6_FULL `+1.6436%`；ordered超过constant/permuted/Q1-wide；internal 7/7；但低于A6_MEASURE `-0.2366%`且learned fusion被cross-fit static convex超过`+2.2112%` |
| `idea` | 保留SIFF scale-indexed arm generator，以target coordinate × ordered log-scale allocation field替代失败的history-conditioned generic router |
| `theory_check` | 同一future coordinate的allocation不依赖requested H；arms仍依赖fixed past；target-scale field是对无法稳定识别的sample-wise routing自由度的结构化收缩 |
| `design` | immutable SIFF-v2 parent + provisional `SC1-SIFF-v3-TSAF-v1`；不增加第二loss，不恢复CCSF/D17-D21 |
| `narrative_gate` | conditional pass：完整贡献链可辩护，但effectiveness尚未测试 |
| `effectiveness_gate` | pending；当前没有新candidate result |
| `artifacts` | 本报告；`configs/stage_c_siff_v3_tsaf_step6.json` |
| `decision` | `reactivate_siff_paperization_freeze_tsaf_step6_local_implementation_next` |

[Decision] 用户于`2026-07-21`明确选择以SIFF-v2作为当前论文项目的最短落地路径。该决定恢复的是
**SIFF paperization program**，不是把历史Step9 failure改写为pass，也不是授权对冻结v2做seed、width、rank、readout
或objective盲调参。

`SC1-SIFF-v2-EQ-ATTR-v1`继续作为immutable parent和当前best performance-near anchor。任何结构变化均创建新的
`test_informed` candidate。SC-MNB降为supporting prior/control inventory；在新candidate的claim和matrix冻结前，
不运行65-run baseline reproduction。

## 2. 已经成立与尚未成立的证据

### 2.1 可以进入论文叙事的部分

1. SIFF-v2在完整50-run/200-cell audit中超过A6_FULL `+1.6436%`、PCSD_EQUAL `+0.5906%`；
2. ordered field超过constant `+0.9393%`、permuted `+0.3959%`和Q1-wide `+1.1619%`；
3. arms未collapse，policy、oracle、diversity、component intervention等internal gates为7/7；
4. SIFF在同一个$T=720$ generation graph内产生全部future coordinates，requested H只作用于最终crop；
5. scale variation、scale ordering与multi-scope partition不是无效装饰。

### 2.2 不能改写的失败

1. SIFF-v2低于A6_MEASURE `-0.2366%`；
2. ordered相对parameter-matched independent scopes仅`+0.2580%`，低于冻结`0.3%` gate且存在validation→test reversal；
3. v1 policy best-arm match仅29.24%，policy-skill alignment为0.0277；
4. row-cross-fit static convex相对learned adaptive fusion为`+2.2112%`；
5. CCSF contrast architecture、RELCAL、region teacher、mixture-risk与temperature/sharpness routes均已关闭；
6. D17-D21 post-hoc context/interaction routes关闭，不得作为本轮rescue。

[Strong Evidence] 当前最可行动的失败归因不是“SIFF field无效”，而是history-conditioned policy拥有一项没有得到
稳定证据支持的sample-wise competence自由度。继续向router提供更多features或auxiliary labels既缺乏证据，也与
generic MoE prior高度重叠。

## 3. Paper problem reformulation

对fixed past $X=x$和future coordinate $t$，pointwise MSE的Bayes predictor为

$$
f_t^*(x)=\mathbb E[Y_t\mid X=x].
$$

请求长度$H$不为同一$t$增加信息，因此不能用requested-H router制造伪条件差异。但$t$改变时，目标函数本身改变；
近端、周期中段与远端坐标可能需要不同history coupling extent。问题不是“模型是否知道用户请求到多远”，而是：

> 一个统一decoder能否把future coordinate与history coupling scale组织成共享、连续且低容量的生成几何，同时保留
> history-conditioned arm forecasts，而不要求模型逐sample猜测噪声较大的best scope？

该问题把三个维度分开：

- `history dependence`：保留在每个scope arm $F_s(X,t)$中；
- `target dependence`：由future coordinate $t$决定scale allocation；
- `request invariance`：同一$t$的计算不依赖requested $H$。

## 4. Core method: SIFF + Target-Scale Allocation Field

### 4.1 保留的SIFF generator

SIFF在ordered log-scale coordinate $z_s$上，用共享components生成scope-conditioned modes：

$$
M_s(h)=\sum_q \phi_q(z_s)M_q(h),
\qquad
F_s(X,t)=\mathcal G_t(M_s(h(X)),s).
$$

这里$F_s$仍是history-conditioned full-domain forecast arm。五个scopes不是任意experts，而是同一decoder field在
不同output-coupling extents上的结构化切片。

### 4.2 新candidate：TSAF

provisional `SC1-SIFF-v3-TSAF-v1`定义共享target-scale scorer：

$$
e_{t,s}=v^\top\operatorname{GELU}
\left(W_t\psi(t)+W_s\eta(z_s)+b\right),
\qquad
\pi_{t,s}=\operatorname{softmax}_s(e_{t,s}),
$$

其中：

- $\psi(t)$复用full-domain DCT future-coordinate field；
- $\eta(z_s)=[z_s,z_s^2]$显式编码ordered log scale；
- scorer参数在所有scales、targets、samples和channels间共享；
- $\pi_{t,s}$广播到`[B,C,T,S]`，不读取history hidden、requested H或future labels；
- forecast为
  $$
  \hat Y_t(X)=\sum_s\pi_{t,s}F_s(X,t).
  $$

TSAF删除的是没有稳定证据支持的sample-wise routing freedom，不删除forecast对history的依赖。它也不是post-hoc
static ensemble：allocation与SIFF arms从同一初始化类end-to-end joint training。

### 4.3 Training role

保留SIFF-v2的single `equal_skill` contract：harmonic dense-prefix fused loss训练最终forecast，uniform per-arm skill项
避免scope collapse。它是method optimization contract，不被包装为独立Contribution 2；不增加competence teacher、
KL router supervision、temperature loss或第二decoder。

## 5. Theory and code-design checks

### 5.1 Request invariance

对任何$H_1,H_2\ge t$，$F_s(X,t)$与$\pi_{t,s}$都只由完整$T$ graph中的$X,t,s$决定，因此

$$
\hat Y_t^{(H_1)}(X)=\hat Y_t^{(H_2)}(X).
$$

该性质是设计结果而非论文唯一贡献；constraint reset后它不是所有候选的先验硬约束。

### 5.2 Containment and restriction

- scorer输出全零时，TSAF包含equal fusion；
- 去掉target variation时，包含learned global scale allocation；
- SIFF arm generator与frozen v2完全相同；
- TSAF不包含任意history-conditioned routing，因此是有意的finite-capacity restriction，而非更大function class。

[Hypothesis] 当sample-wise competence不可由fixed past稳定识别，而coordinate-wise scale preference更稳定时，该
restriction应改善validation/test transfer并降低policy variance。

[Self-critique] 一般Bayes predictor并不要求scale mixture weights与history无关。若真实数据存在可识别的
sample-conditioned scale choice，TSAF可能欠拟合；因此它必须与frozen direct-policy parent和parameter-matched
categorical target-only control比较，不能仅凭叙事升格。

## 6. Latest primary-source boundary

检索日期：`2026-07-21`。来源优先使用conference proceedings、OpenReview、arXiv primary pages与official code。
Zotero coverage不作为novelty完整性证据。

| Prior work | 已覆盖内容 | SIFF/TSAF必须收紧的边界 |
| --- | --- | --- |
| [ElasTST, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d7aa002885ccbe68cf6880da583761b2-Abstract-Conference.html) | one-model varied-horizon、multi-scale patches、horizon reweighting、invariance | 不claim首次varied-H或首次multi-scale；强调decoder target-scale coupling field |
| [CATS, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cf66f995883298c4db2f0dcba28fb211-Abstract-Conference.html) | future horizon parameters作为queries读取past | 不claim首次future-coordinate query；TSAF只分配structured output scopes，不执行query retrieval |
| [Pathformer, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2be6705de7412adf107900add727a795-Abstract-Conference.html) | input-side multi-scale Transformer与sample-adaptive pathways | 不claim generic adaptive multi-scale routing；TSAF是decoder-side、target-conditioned、sample-shared allocation |
| [MoLE, AISTATS 2024](https://proceedings.mlr.press/v238/ni24a.html) | forecast experts与input-dependent router | 不claim generic forecasting MoE；SIFF arms共享continuous field而非独立pattern experts |
| [BasisFormer, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e150e6d0a1e5214740c39c6e4503ba7a-Abstract-Conference.html) | learnable history/future bases与attention coefficients | 不claim首次learned basis；贡献边界是scale-indexed output-coupling field |
| [Implicit Forecaster, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e82ef0c89df6a6eff8734ea7e27c42f-Abstract-Conference.html) | decoder-side global wave synthesis | 不claim首次future-stage decoder；SIFF建模的是coupling-scale family而非wave components |
| [TimePerceiver, NeurIPS 2025](https://arxiv.org/abs/2512.22550) | target-timestamp queries与generalized input/target positions | 不claim首次target-position decoding；本任务仍是fixed-past extrapolation |
| [SRSNet, NeurIPS 2025](https://arxiv.org/abs/2510.14510) | selective patching与dynamic reassembly | 不claim首次selective history representation；SIFF不选择或重排input patches |
| [Moirai-MoE, ICML 2025](https://proceedings.mlr.press/v267/liu25an.html) | foundation-model token-level sparse specialization | 与from-scratch decoder scale allocation分表，不泛称unified MoE |

[Decision] primitive-level overlap很强，但尚未发现primary source覆盖完整链：
`fixed-past request invariance -> target-specific coupling-scale demand -> shared scale-indexed decoder field ->
target-scale allocation without sample router -> one full-domain forecast`。novelty为`provisional contribution-level`，不是
component-first claim。

## 7. Paper narrative and contributions

建议working title：

> **Scale-Indexed Forecast Fields for Unified Multi-Horizon Time-Series Forecasting**

单一paper-core contribution为SIFF/TSAF unified decoder。论文不预设第二个loss或router contribution。贡献链可写为：

1. **Problem insight**：区分requested-horizon information与future-coordinate-specific scale demand，说明前者在
   fixed-past MSE中不提供额外Bayes information，而后者仍要求不同finite computation；
2. **Method**：提出shared Scale-Indexed Forecast Field，并以Target-Scale Allocation Field在future coordinate与
   ordered history coupling scale之间建立低容量连续映射；
3. **Evidence**：通过constant、permuted、Q1-wide、categorical target-only、independent-scope、A6_MEASURE与
   direct-policy controls，分别审计scale variation、ordering、capacity、coordinate field、sharing与performance。

第三项是evaluation/attribution contribution，不应包装为第二个method contribution。

## 8. Frozen control and evaluation design

### 8.1 Candidate identity

- parent：immutable `SC1-SIFF-v2-EQ-ATTR-v1`；
- new candidate：`SC1-SIFF-v3-TSAF-v1`；
- inherited carrier/profile：与parent相同；
- training：from-scratch joint encoder-decoder；
- checkpoint：validation `{96,192,336,720}` mean MSE；
- test-informed：true；任何正式test matrix必须完整报告negative cells。

### 8.2 Required controls

1. `A6_FULL`、`A6_MEASURE`：carrier与strong objective controls；
2. frozen `SIFF-v2 direct policy`：parent performance/control；
3. `PCSD_EQUAL`：无scale-coordinate generator；
4. `SIFF categorical target-only`：去history但不显式共享scale geometry；
5. `TSAF permuted-scale`：保留参数量、破坏scale semantics；
6. `TSAF no-target/global`：检验future-coordinate variation；
7. `SIFF independent target-only`：检验shared ordered field是否只等价于independent scopes。

已有references可在hash、profile、checkpoint rule一致时复用；所有new method arms必须end-to-end训练，禁止frozen
replacement用于direction rejection。

### 8.3 Effectiveness and attribution gates

正式Phase A前冻结：

- TSAF相对A6_MEASURE与SIFF-v2 parent：MSE macro至少`+0.3%`，datasets至少`3/5`、horizons至少`3/4`、
  cells至少`11/20`，MAE macro非负；
- TSAF相对categorical target-only、permuted-scale与no-target controls分别满足macro正、datasets至少`3/5`；
- shared ordered field相对independent target-only至少macro正，若不足`0.3%`只能claim compact structured bias，
  不能claim strict superiority；
- internal health必须包含finite、arm diversity、nonconstant target-scale surface、scale-order sensitivity、
  allocation entropy与component contribution；
- validation只能做implementation/checkpoint选择，不得pass或reject机制；official test才执行formal gate。

## 9. Failure attribution and rollback

- candidate与categorical target-only均低于parent：`history_free_allocation_hypothesis_false`，回Step2/4；
- categorical target-only提升而TSAF不提升：`scale_field_allocation_design_wrong`，只关闭TSAF exact v1；
- TSAF超过parent但不超过A6_MEASURE：`performance_partial_pass`，可作为SIFF ablation但不自动成为paper core；
- performance通过但permuted/no-target解释：`capacity_or_coordinate_control_explains`；
- numeric/gradient异常：`optimization_or_numeric_pathology`，不得方向拒绝；
- 不得回到CCSF、region teacher、temperature、rank/seed/readout sweep作rescue。

## 10. Authorization and next action

当前授权：

`SIFF paperization=true / Step4-6=true / local model implementation=true (Step7A 26/26) /
remote training=false / official test=false / confirmation=false / SC-MNB execution=false`。

Step7A现已完成：TSAF及permuted/no-target modes进入production path，26/26 tensor、parameter、
request-invariance、semantic、gradient与constructor cases通过。下一步为Step7B prelaunch；只有其通过且文档另行
授权后，才可启动remote或official test。
