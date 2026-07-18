# SC1-SIFF-v2-CCSF Step 6 Narrative And Control Gate

## 1. 结论先行

`SC1-SIFF-v2-CCSF-v1-preimplementation`通过Step 6 narrative/control gate，decision为
`step6_pass_step7a_local_only`。这只授权Step 7A本地实现与construction tests，不授权validation
temperature pilot、remote training、formal test或confirmation seeds。

本次把Step 5的理论候选收紧为两项必须联合、也必须可拆开归因的候选贡献：

1. **CCSF architecture**：让projective coupling scopes产生的forecast contrast进入fusion policy，解决
   history-only policy无法识别target-wise relative scope competence的问题；
2. **Relative competence calibration**：用confidence-weighted relative regret在同一次forward内弱监督上述
   contrast-conditioned policy；它不是独立的“首个error-supervised routing”claim。

只有完整候选同时超过`A6_MEASURE`、冻结parent、architecture/loss/capacity/permutation/teacher/independent
controls，且内部路径健康，才允许把两项写成联合paper contribution。任一层失败都按预冻结decision map降级或回滚，
不能用oracle headroom挽救negative effectiveness。

## 2. 11-step record

| Field | Content |
| --- | --- |
| `current_step` | Step 6 method、control、narrative与experiment contract complete |
| `problem` | SIFF scopes有skill/diversity/headroom，但history-only policy没有直接观察不同scope对同一target的分歧，未把conditional headroom转为超过A6_MEASURE/independent的fused forecast |
| `existence_evidence` | v1 policy match 29.24%、skill alignment 0.0277；two-fold contrast diagnostic相对coordinate/shuffle分别+1.8348%/+1.7085%，5/5 gates |
| `idea` | v1 logits上增加scope-shared、target-free contrast correction；以confidence-weighted relative competence作co-designed弱监督 |
| `theory_check` | full-domain-then-crop projectivity、v1 inclusion、target-free inference与contrast identifiability通过；end-to-end gain仍未测试 |
| `design` | 10 arms、2×2 architecture/objective core、teacher/capacity/semantic/field controls、validation-only shared temperature protocol与four-layer gates |
| `narrative_gate` | conditional pass；claim限定为projective output-coupling scopes中的contrast-conditioned fusion与co-designed calibration |
| `effectiveness_gate` | not started；正式阶段必须official test五dataset × 四horizon完整矩阵 |
| `artifacts` | frozen config、static matrix/parameter/claim audits、本报告 |
| `decision` | `step6_pass_step7a_local_only`；local implementation=true，remote/test=false |

## 3. Method contract

### 3.1 Tensor path

保持冻结parent的Encoder、SIFF ordered scale field与五个coupling scopes：

$$
H\in\mathbb R^{B\times C\times R}
\longrightarrow
A\in\mathbb R^{B\times C\times S\times T},
\qquad S=5,\ T=720.
$$

对`A`在scope轴计算consensus、centered normalized contrast与relative disagreement，再按各scope已有的
native group构造六维descriptor：pointwise signed/absolute contrast、group mean/RMS/endpoint difference及
log relative disagreement。descriptor不读取label、requested horizon或benchmark horizon bin。

保留v1 base logits $\ell^0$，以一个跨scope共享的两层scorer产生correction：

$$
\Delta\ell_{bcts}=g_\theta(p(H_{bc}),e_t,\xi_s,c_{bcts}),\qquad
\alpha_{bcts}=\operatorname{softmax}_s(\ell^0_{bcts}+\Delta\ell_{bcts}),
$$

$$
\widehat Y_{bct}=\sum_s\alpha_{bcts}A_{bcts}.
$$

scorer input为`32 history + 4 target-coordinate + 1 scale-coordinate + 6 contrast = 43`，hidden dimension为64，
参数量为2,881。final layer零初始化，且base logits始终保留；令correction为零时严格包含v1 function class。
这不是warm-start或learned-capacity preservation，正式训练仍需from-scratch end-to-end joint optimization。

### 3.2 Projectivity contract

所有arms与policy先生成固定$T=720$完整forecast，requested $H$只执行prefix crop：

$$
F_H=\mathcal R_HF_T.
$$

因此$H_1\le H_2$时$\mathcal R_{H_1}F_{H_2}=F_{H_1}$。Step 7A必须在多种prefix下验证最大gap不超过
`1e-7`，并审计crop前没有读取requested $H$。

### 3.3 Objective contract

主预测与equal-skill项不变。新增teacher只使用本次forward的detached arm absolute error：

$$
\rho_s=\frac{e_s-\bar e}{\bar e+\epsilon},\qquad
q=\operatorname{softmax}(-\operatorname{sg}(\rho)/\tau),
$$

$$
m=\operatorname{sg}\left(1-\frac{\mathcal H(q)}{\log S}\right),\qquad
L_{cal}=\mathbb E\sum_t\omega_t m_t\operatorname{KL}(q_t\Vert\alpha_t).
$$

冻结`calibration_weight=0.1`。relative regret令整体error尺度变化不改变teacher；entropy confidence令near-tie
cells接近零监督。它仍可能破坏ensemble cancellation，因此必须通过architecture-only、loss-only与standardized
teacher controls归因，不能因best-arm accuracy提高就宣称fused MSE会提高。

## 4. Factorial attribution matrix

### 4.1 核心2×2

| Architecture | EQUAL objective | Relative calibration objective |
| --- | --- | --- |
| parent SIFF v1 | `siff_v1_equal` | `siff_v1_relcal` |
| contrast-conditioned CCSF | `ccsf_equal` | `ccsf_relcal` |

该2×2分别给出architecture main effect、objective main effect与interaction：

$$
I=G(\text{ccsf\_relcal},\text{ccsf\_equal})
-G(\text{siff\_v1\_relcal},\text{siff\_v1\_equal}).
$$

joint claim要求$I\ge0.2\%$，从而证明relative calibration并非在任何policy上都同样有效，而是与contrast
information path形成co-design。

### 4.2 必要controls

| Arm | 作用 | 排除的替代解释 |
| --- | --- | --- |
| `a6_measure` | external effectiveness baseline | CCSF只修复了比A6差的复杂parent |
| `ccsf_stdcal` | old standardized teacher geometry | 收益只来自任意error teacher |
| `ccsf_no_contrast_equal/relcal` | 相同2,881参数、contrast固定零 | 多一个MLP或policy capacity即可解释 |
| `ccsf_permuted_contrast_relcal` | fixed arm-axis permutation | 只要输入任意output statistic就有效 |
| `ccsf_independent_relcal` | dataset-wise parameter-matched independent field | ordered scale field不是贡献所必需 |

独立field沿用已审计matched ranks：ETTh1/ETTh2/ETTm1/ETTm2/Weather分别
`109/116/116/106/116`。加入同一correction scorer后，相对ordered CCSF总参数gap均小于`0.5%`。显式
`A6_MEASURE anchor`不进入method，因为它会把architecture机制与额外ensemble/capacity混在一起。

## 5. Hyperparameter、checkpoint与evaluation protocol

### 5.1 Shared temperature pilot

`tau`只能从validation选择，grid冻结为`{0.05, 0.1, 0.25}`。每个temperature在五个datasets各跑一次，共15 runs；
selection score为五dataset、H96/H192/H336/H720 validation MSE的macro mean。五个datasets共用一个temperature，
并列时选择更大的temperature。不得使用test labels，也不得按dataset分别选择。

该pilot尚未授权；Step 7A只需完成local objective/gradient smoke与CLI contract。pilot完成后将产生新的formal
candidate version与checkpoint hash，再进入Step 7B prelaunch审核。

### 5.2 Formal Phase A与confirmation

Phase A冻结为10 arms × 5 datasets × seed2021，共50 runs、200个standard-horizon test cells；每个run从头
joint train 20 epochs，patience 5，batch size 32，learning rate `1e-4`。checkpoint只由validation四horizon
mean MSE选择，test labels不得选择epoch。

seeds2022/2023 confirmation为100 runs、400 cells，只有Phase A完成四层Step 9审核后才能授权。本Step不授权
任何remote或formal test访问。

## 6. Frozen gates

每项hard comparison要求同时满足：

- MSE macro gain不少于`+0.3%`；
- 至少3/5 datasets为正；
- 至少3/4 horizons为正；
- 至少11/20 dataset-horizon cells为正。

`full_over_a6_measure`与`full_over_v1`还要求MAE macro不劣。joint paper claim要求以下10项全部通过：full分别超过
A6/v1、architecture分别超过v1/capacity、calibration超过architecture-only、full超过loss-only、relative teacher
超过standardized teacher、true contrast超过zero/permuted、ordered超过independent。

内部健康层同时检查finite、projectivity、oracle headroom、arm diversity、policy entropy、best-arm accuracy gain、
skill alignment、allocation gain及correction activity。内部健康只能解释机制是否按理论运行，不能覆盖negative
official-test gate。

## 7. Claim-control contract

| Conditional paper claim | 必须通过的controls |
| --- | --- |
| paper-facing performance | full over A6_MEASURE；full over v1 |
| contrast-conditioned architecture | architecture over v1/capacity；true over zero/permuted |
| relative competence calibration | calibration on CCSF；relative over standardized teacher |
| architecture-objective co-design | full over loss-only；positive factorial interaction |
| ordered scale-field necessity | ordered over parameter-matched independent field |

若全部成立，允许的两项贡献表述为：

1. `contrast-conditioned projective scope fusion for unified multi-horizon generation`；
2. `confidence-weighted relative competence calibration co-designed for output-coupling scopes`。

禁止claim“首个output-aware MoE”“首个error-supervised routing”“best-arm accuracy保证fused improvement”，也不能把
A6 anchor写进CCSF。由于候选是在多次official-test反馈后设计，必须明确标记`test_informed`。

## 8. Failure attribution与rollback

| Evidence pattern | Decision |
| --- | --- |
| 全部hard + internal通过 | Step9后才可申请confirmation seeds |
| architecture通过、objective失败 | 保留CCSF architecture；Contribution 2回Step4 |
| architecture失败、objective通过 | 关闭exact CCSF；training-only结果不足以支撑当前双贡献 |
| performance正、matched attribution失败 | 仅保留`performance_partial_pass`，回失败control层 |
| ordered不超过independent | 回Step4审计scale-field necessity，不否定contrast fusion |
| NaN、爆炸或>100%异常退化 | `optimization_or_numeric_pathology`；只退回Step7A，不方向级否决 |

## 9. Static audit与Step 6 decision

`scripts/check_stage_c_siff_ccsf_step6.py`对冻结config生成三类审计：

- matrix contract：9/9 checks通过；
- parameter contract：5/5 datasets通过；
- claim-control contract：5/5 claims有对应controls。

汇总gate为5/5：matrix、parameter、claim-control、conditional narrative与Step7A-local-only授权全部通过。

最终decision：`step6_pass_step7a_local_only`。下一步只实现production tensor path、objective、10 arms adapters、
prefix/gradient/parameter/control tests与remote refusal gate；在Step7A完整通过前，不进入validation pilot或Step7B。
