# SC1-SIFF-v2-CCSF Step 5 Theory Feasibility

## 1. 11-step record

| Field | Content |
| --- | --- |
| `current_step` | Step 5 theory feasibility complete |
| `problem` | SIFF arms有skill/diversity/headroom，但history-only policy无法识别每个future target上的relative scope competence |
| `existence_evidence` | v1 policy skill alignment 0.0277；static convex比learned fusion好2.2112%；target-free contrast cross-fit diagnostic 5/5 gates pass |
| `idea` | 用arm forecasts之间的target-free contrast修正v1 policy logits，并用confidence-weighted relative competence作同步弱监督 |
| `theory_check` | projectivity、v1 function-class inclusion、target-free inference、contrast identifiability通过；fusion gain与training stability仍为conditional |
| `design` | 本Step只冻结tensor/loss/proof/control边界；不实现、不训练、不授权remote |
| `narrative_gate` | conditional pass；generic output-aware gating与error-supervised routing已有prior art，claim必须限定到projective output-coupling scopes |
| `effectiveness_gate` | not started；Step6必须冻结A6_MEASURE、v1、loss-only、architecture-only、capacity、shuffle与independent controls |
| `artifacts` | `diagnostics/*.csv/json`、`configs/stage_c_siff_ccsf_step5_theory.json` |
| `decision` | `conditional_theory_pass_to_step6`；显式A6 anchor branch退出method；training=false |

## 2. Step 4假设的关键修正

Step4把候选写成`arm-contrast-aware policy + synchronous competence calibration`。代码审计后必须收紧：

1. `PCC.py`已经实现同一次forward内由detached arm error构造capability、再以KL监督policy；
2. `pointwise_prior_composed`已经是equal-skill + pointwise route supervision；
3. 历史PCSD上pointwise prior相对equal几乎持平，SIFF上的PCC transport相对SIFF_EQUAL为`-0.2663%`；
4. 因此“同步calibration loss”不是未测试的新想法，更不能单独作为CCSF创新。

[Strong Evidence] 真正尚未测试、且由当前失败直接指出的变量是：policy是否能读取arms实际产生的forecast contrast。
calibration只能作为该information path的co-designed training signal，并必须设置loss-only control。

## 3. Target-free contrast identifiability diagnostic

### 3.1 计划与构造

诊断protocol在读取结果前冻结于`configs/stage_c_siff_ccsf_step5_theory.json`。每个dataset使用SIFF_EQUAL保存的：

- `probe_arms [256,5,720]`；
- `probe_targets [256,720]`；
- 对齐的`arm_row_bin_mse [256,8,5]`与`policy_row_bin_usage [256,8,5]`。

对每个row × future bin，从target-free arm predictions构造：

- 五arm consensus；
- 每个arm减去consensus后的mean/std/RMS/end-minus-start slope；
- 三个固定bin coordinate。

一个固定`C=1` multinomial logistic model在128 rows拟合best-arm label，另128 rows评估，然后交换fold。它不是
候选训练，也没有把offline classifier带入模型；作用只是检验“arm contrast中是否存在可被低容量readout提取的
relative competence线索”。

### 3.2 结果

| Feature | best-arm accuracy | allocation gain over uniform |
| --- | ---: | ---: |
| existing learned policy | 28.83% | -0.1502% |
| coordinate only | 34.72% | +3.6392% |
| consensus + coordinate | 40.07% | +4.4365% |
| shuffled contrast + coordinate | 33.98% | +3.7620% |
| **contrast + coordinate** | **44.14%** | **+5.4034%** |
| full consensus + contrast | 45.57% | +5.7955% |

预冻结五项gate全部通过：

1. contrast相对coordinate expected arm MSE `+1.8348%`，门槛`0.5%`；
2. 10/10 folds为正，门槛7；
3. contrast相对shuffled `+1.7085%`；
4. 10/10 folds为正；
5. best-arm accuracy相对existing policy `+15.31` percentage points，门槛`+2`。

[Strong Evidence] 低维、target-free arm contrast包含稳定的relative competence信息；当前history-only policy没有利用
这部分信息。该结果支持进入architecture theory，而不是证明CCSF model已经有效。

[Boundary] 该probe来自test artifacts，且只保存256 rows；classifier使用bin summaries而production model尚未定义。
因此不能用`+5.40%`作为模型收益，也不能从本结果选择dataset-specific descriptor或temperature。

## 4. Provisional CCSF tensor contract

### 4.1 Parent path

保持v1 Encoder与SIFF scale field不变：

$$
H\in\mathbb R^{B\times C\times R}
\longrightarrow
A\in\mathbb R^{B\times C\times S\times T},
\qquad S=5,\;T=720.
$$

`A_{bcst}`是scope $s$对target $t$的forecast。CCSF不增加horizon ID，不改变arms，也不先冻结后替换组件。

### 4.2 Target-free contrast field

将arms转为`[B,C,T,S]`后定义：

$$
\mu_{bct}=\frac1S\sum_s A_{bcts},\qquad
\delta_{bcts}=A_{bcts}-\mu_{bct},
$$

$$
v_{bct}=\sqrt{\frac1S\sum_s\delta_{bcts}^2+\epsilon},\qquad
z_{bcts}=\frac{\delta_{bcts}}{v_{bct}}.
$$

对每个scope使用其已有group $G_s(t)$，构造shared、dimensionless descriptor：

$$
c_{bcts}=\left[
z_{bcts},\ |z_{bcts}|,\
\operatorname{mean}_{u\in G_s(t)}z_{bcsu},\
\operatorname{rms}_{u\in G_s(t)}z_{bcsu},\
z_{bcs,\max G_s(t)}-z_{bcs,\min G_s(t)},\
\log(1+r_{bct})
\right],
$$

其中$r_{bct}=v_{bct}/(S^{-1}\sum_s|A_{bcts}|+\epsilon)$。这些量只来自model forecasts，不读取target，
不依赖requested horizon，也不使用benchmark horizon bins。Step3的bin summary仅是offline existence proxy；
production descriptor必须使用scope-native groups。

### 4.3 Contrast-conditioned logit correction

保留v1 base logits $\ell^0(H,e_t)\in\mathbb R^S$，再用跨scope共享的scorer：

$$
\Delta\ell_{bcts}
=g_\theta\!\left(p(H_{bc}),e_t,\xi_s,c_{bcts}\right),\qquad
\alpha_{bcts}=\operatorname{softmax}_s(\ell^0_{bcts}+\Delta\ell_{bcts}),
$$

其中$e_t$是原future coordinate，$\xi_s$是normalized log-scale coordinate。最终输出仍为：

$$
\widehat Y_{bct}=\sum_s\alpha_{bcts}A_{bcts}.
$$

`g_θ`的final layer零初始化。令其参数为零时，CCSF严格退化到v1 policy，因此在function class上包含v1；这不是
trained checkpoint warm-start，也不能声称保留了learned capacity。Step6必须加入参数匹配的
`generic logit correction without contrast` control，排除额外MLP capacity解释。

## 5. Projectivity与continuity

对任意requested $H\le T$，CCSF先在固定完整domain上计算$A_T$、$c_T$、$\alpha_T$和$F_T$，最后只执行：

$$
F_H=\mathcal R_HF_T.
$$

因此对$H_1\le H_2$：

$$
\mathcal R_{H_1}F_{H_2}
=\mathcal R_{H_1}\mathcal R_{H_2}F_T
=\mathcal R_{H_1}F_T
=F_{H_1}.
$$

[Fact] 只要implementation不在crop前读取requested $H$，prefix projectivity严格成立。scope group可能查看$t$之后的
arm forecasts，但这些forecast同样来自fixed past与固定T domain；它不会造成不同requested horizon输出不一致。

scale coordinate使用normalized $\log(s/T)$，scorer跨scope共享；因此没有horizon-specific model。当前只在五个
离散scales训练，不能claim对未见scale连续泛化。

## 6. Synchronous calibration：保留什么、修正什么

### 6.1 为什么旧PCC teacher可能不适合equal-skill SIFF

旧PCC capability先对arm errors做center，再除以cross-arm standard deviation。该操作会把“只有极小差异”的arms也
标准化成单位幅度teacher。已有full row-bin diagnostic显示：

- PCC standardized teacher normalized entropy为0.8032；
- 其teacher confidence与relative error dispersion的dataset-macro correlation为`-0.4047`；
- 即teacher强度没有随真实relative skill margin增强，反而可能在微小/噪声差异处给出过强route signal。

这与equal-skill把arm loss CV压到3.45%的环境存在结构性张力。

### 6.2 Confidence-weighted relative competence（provisional）

训练时使用raw arm error $e_{bcts}=|A_{bcts}-Y_{bct}|$，定义scale-invariant relative regret：

$$
\rho_{bcts}=\frac{e_{bcts}-\bar e_{bct}}{\bar e_{bct}+\epsilon},
\qquad
q_{bcts}=\operatorname{softmax}_s\left(-\operatorname{sg}(\rho_{bcts})/\tau\right),
$$

$$
m_{bct}=\operatorname{sg}\left(1-\frac{\mathcal H(q_{bct})}{\log S}\right),
$$

$$
L_{\mathrm{cal}}
=\mathbb E_{b,c}\sum_t\omega_t m_{bct}
\operatorname{KL}(q_{bct}\Vert\alpha_{bct}).
$$

若arms误差完全相同，则$q$为uniform、$m=0$，不强迫router学习任意winner；整体error乘常数不改变$\rho$。target
只用于训练loss并经stop-gradient进入teacher，inference只读取$H/A/e_t/\xi_s$，因此没有label leakage。

总目标暂定：

$$
L=L_{\mathrm{fused}}+L_{\mathrm{equal\text{-}skill}}
+\lambda_{\mathrm{cal}}L_{\mathrm{cal}}.
$$

`L_fused`仍是主预测目标；calibration不是性能保证。对fixed arms，expected KL的最优policy是
$\alpha(Z)=\mathbb E[q\mid Z]$，所以Step3的cross-fit result正是在检验feature $Z=(H,e_t,\xi_s,c)$是否可能预测$q$。

### 6.3 不能声称的结论

1. per-arm error最小不等于convex fused error最小，预测误差可能相互抵消；
2. 因此更高best-arm accuracy不必然带来更低fused MSE；
3. calibration只作为弱regularizer，最终仍由fused loss决定；
4. `τ`与`λ_cal`不得从本次test-derived temperature grid选择。Step6只能冻结shared validation-only选择规则，
   禁止per-dataset tuning。

teacher geometry grid只证明relative-regret family可从near-uniform到selective连续调节：例如$\tau=0.1$时entropy
0.8012，confidence与relative dispersion correlation为`+0.6410`；它不是推荐超参数。该grid使用artifact中保存的
row-bin MSE，而provisional training formula使用与fused objective一致的L1，因此数值只作teacher-geometry proxy。

## 7. A6_MEASURE anchor审计

Step4曾考虑把A6_MEASURE作为显式anchor arm。Step5决定**不把它纳入CCSF method**：

1. 新增一条完整A6 branch会引入明显capacity/ensemble confound；
2. 若性能恢复只能由anchor解释，SIFF/CCSF mechanism仍未成立；
3. anchor概念与近期anchored/uncertainty-gated MoE相邻，削弱贡献边界；
4. 现有SIFF `map_a6_parameters_`只证明存在A6 function witness，不证明from-scratch learned capacity被保留；
5. CCSF通过`Δlogit=0`包含v1，而不是通过额外forecast branch兜底。

因此`A6_MEASURE`继续作为mandatory external effectiveness baseline；若未来只为debug加入anchor，必须标记
`diagnostic_only`并设置parameter-matched generic ensemble control。

## 8. Source-informed novelty boundary

检索日期2026-07-18，external primary-source first；Zotero presence未检查，不以Zotero覆盖判断novelty。

| Work | Primary source | 已覆盖 | CCSF必须收紧的边界 |
| --- | --- | --- | --- |
| Pathformer, ICLR 2024 | https://openreview.net/forum?id=lJkOCMP2aW | multi-scale temporal modeling + input-adaptive pathways | 不能claim一般multi-scale adaptive routing |
| Spatial MoE, NeurIPS 2022 | https://proceedings.neurips.cc/paper_files/paper/2022/file/4c5e2bcbf21bdf40d75fddad0bd43dc9-Paper-Conference.pdf | dense regression中的self-supervised routing/error signal | calibration loss本身不是新贡献 |
| MoGU, arXiv v2 2026 + official code | https://arxiv.org/abs/2510.07459；https://github.com/yolish/moe_unc_tsf | time-series expert-specific uncertainty作为native gating signal | 不能泛称output-aware/confidence-aware gating；CCSF使用deterministic inter-scope contrast而非probabilistic variance |
| AdaMixT, arXiv 2025 | https://arxiv.org/abs/2509.18107 | adaptive weighted multi-scale expert forecasting | 必须强调experts是同一projective decoder的output-coupling scopes，不是多套patch encoders/models |
| FAME, arXiv 2026 | https://arxiv.org/abs/2606.08896 | offline expert-suitability targets | CCSF必须保持same-forward synchronous、无offline teacher |

provisional contribution chain为：

> fixed-past unified multi-horizon generation
> → one projective decoder需要多个output-coupling scopes
> → scope competence不能只由history-only router识别
> → deterministic arm-contrast field提供target-wise、scope-native evidence
> → shared contrast-conditioned logit correction与confidence-weighted synchronous calibration
> → full-domain computation后prefix crop。

这是complete-chain novelty的conditional pass，不是component-level first claim。

## 9. Theory risks与failure attribution

| Risk | 当前判断 | Step6/7 falsifier |
| --- | --- | --- |
| contrast只反映forecast level，不反映skill | offline 5/5 gate反驳exact low-capacity版本 | shuffled/zero contrast与generic correction打平full CCSF |
| loss-only即可解释 | 历史PCC未支持，仍需matched rerun | old policy + new calibration等于full CCSF |
| extra policy capacity解释 | untested | generic no-contrast correction匹配full CCSF |
| calibration破坏ensemble cancellation | real risk | architecture-only优于full calibration |
| arms为迎合teacher而collapse | stop-gradient/equal-skill降低但不消除 | oracle/diversity/component-use明显下降 |
| output feedback形成数值不稳定 | theory可微，empirical untested | >100% degradation、NaN、logit/gradient explosion |
| independent field仍解释收益 | v1边界未解决 | ordered CCSF不超过independent CCSF |

失败归因规则：

- architecture-only不优于generic/shuffled：`intervention_point_wrong`或contrast descriptor false；
- architecture-only正、full calibration负：`objective_design_wrong`，不否定CCSF architecture；
- full正但independent control打平：ordered SIFF specificity仍失败；
- numerical pathology：只能否定exact implementation，不能方向级拒绝。

## 10. Step 5 decision与Step 6输入

Decision=`conditional_theory_pass_to_step6`。

通过项：

1. target-free contrast identifiability 5/5；
2. exact prefix projectivity proof；
3. v1 function-class inclusion witness；
4. no offline teacher / no inference target；
5. generic prior-art boundary已收紧。

未通过或未测试项：

1. end-to-end fused performance；
2. contrast branch optimization stability；
3. calibration是否优于architecture-only；
4. ordered field是否超过independent field；
5. A6_MEASURE effectiveness。

Step6必须设计至少以下factorial controls：

- v1 EQUAL；
- old policy + new relative calibration（loss-only）；
- CCSF + EQUAL（architecture-only）；
- CCSF + confidence-weighted calibration（full）；
- CCSF shuffled/zero contrast；
- parameter-matched generic logit correction；
- independent-field CCSF；
- A6_MEASURE external baseline。

在Step6 narrative/control gate完成前，`SC1-SIFF-v2-CCSF-v0-theory`不得实现、不得remote、不得访问新结果。
