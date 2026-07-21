# ISCF-v1-CPSI Step6 Control Design and Test-Decision Policy

## 1. Long-stage record

| Field | Record |
| --- | --- |
| `current_step` | ISCF Step6 concrete design complete；Step7A production-local implementation next |
| `problem` | 如何公平地区分CPSI interaction、generic nonlinear capacity、linear sharing、common-only和post-synthesis placement，同时避免validation轻微负向过早关闭机制？ |
| `existence_evidence` | Step4 D1.1确认common/private pre-synthesis response；Step5证明fixed linear mixing可重参数化，CPSI product提供非线性function path |
| `idea` | 保持CPSI为candidate；四个matched controls既是intermediate diagnostics，也是formal test attribution arms；validation不承担方向级淘汰 |
| `theory_check` | SELF/LINEAR/COMMON exact `3Lr`；POST-SYNTH直接作用于`[B,C,S,T]`且总模型参数gap最大0.041%；所有arms zero-init exact parent morph |
| `design` | 5 new arms × 5 datasets × seed2021=`25` new trainings；ISCF-v0/A6_FULL各5个historical references；全部有效arms一次性进入official test |
| `narrative_gate` | `step6_design_pass_with_test_first_effectiveness_and_diagnostic_controls` |
| `effectiveness_gate` | pending official-test MSE/MAE；validation不能pass/reject |
| `artifacts` | 本报告与`configs/stage_c_iscf_v1_cpsi_step6.json` |
| `decision` | `step6_pass_step7a_local_authorized`；implementation local only；remote/test execution pending Step7A/7B |

## 2. User-directed governance update

用户于`2026-07-21`明确要求：`CPSI-SELF/CPSI-LINEAR/CPSI-COMMON/POST-SYNTH`作为中间诊断，不因轻微负向
提前放弃机制，应走到official-test MSE/MAE后再判断effectiveness。

本报告采纳该原则，但保持Four-Layer Mechanism Evaluation边界：

1. controls不再是validation access gates；
2. 只要run protocol有效，五个new arms必须全部完成训练并一次性进入冻结official-test audit；
3. test MSE/MAE决定performance viability；
4. controls决定claim attribution，而不是决定是否允许看test；
5. positive CPSI但controls未分离时，保留performance candidate，状态只能是`performance_partial_pass`；
6. mild negative进入`inconclusive`而不是direction rejection；只有预注册的material negative或hard pathology才触发明确回滚。

这不是放宽reporting：所有negative cells、datasets、MSE与MAE仍必须完整报告，test不能用于调rank、挑dataset或改arm。

## 3. Shared tensor and parameter contract

令ISCF independent modes为

$$
X\in\mathbb{R}^{B\times C\times S\times L},\qquad L=DK,quad S=5,quad D=4,
$$

并冻结pre-synthesis bottleneck rank

$$
r=32.
$$

所有pre-synthesis interaction arms均使用三组shared、无bias matrices：两组$L\to r$和一组$r\to L$，新增参数

$$
N_{\mathrm{pre}}=3Lr.
$$

所有new arms均在base ISCF parameters初始化完成后再创建interaction parameters，以保证同seed下encoder、independent
`mode_weight/mode_bias`、scope synthesis与policy的initial state一致。$W_o=0$；input projections使用同一fan-in
initialization class和paired seed tensors。

## 4. Candidate and exact pre-synthesis controls

### 4.1 `CPSI`

$$
\mu=\operatorname{mean}_s X_s,\qquad \delta_s=X_s-\mu,
$$

$$
m_s=W_o\left[
\operatorname{GELU}(W_c\mu)\odot
\operatorname{GELU}(W_p\delta_s)
\right],
\qquad X'_s=X_s+m_s.
$$

Role：working method candidate；同时需要common与private path。

### 4.2 `CPSI-SELF`

$$
m_s=W_o\left[
\operatorname{GELU}(W_aX_s)\odot
\operatorname{GELU}(W_bX_s)
\right].
$$

Role：exact-parameter nonlinear capacity/depth control。它保留product与两组input projections，但不读取其他scope或set
mean。若SELF接近或超过CPSI，只削弱relation attribution，不自动否定CPSI的forecast performance。

### 4.3 `CPSI-LINEAR`

$$
m_s=W_o\left(W_c\mu+W_p\delta_s\right).
$$

Role：exact-parameter linear sharing/optimization control。三组matrices与CPSI完全同shape，但移除GELU与Hadamard
product。该path对$h$仍是linear/affine reparameterization；若其表现相当，说明nonlinear product necessity未建立。

### 4.4 `CPSI-COMMON`

$$
m_s=W_o\left[
\operatorname{GELU}(W_a\mu)\odot
\operatorname{GELU}(W_b\mu)
\right],
$$

并将相同$m$ broadcast到全部scopes。

Role：exact-parameter common-only nonlinear control。它保留两投影、product和set aggregation，只移除scope-specific
$\delta_s$。所以它比“单branch common MLP + dormant padding”更诚实，不依赖nonfunctional parameter counting。

### 4.5 Exact shared invariants

四个pre-synthesis arms均满足：

- input/output `[B,C,S,D,K]`；
- added params exact `3Lr`；
- `W_o=0`时对任意input exact equal ISCF-v0；
- 不读取requested H、scale order、future label或oracle；
- shared matrices使scope/metadata joint permutation保持equivariance；
- objective、policy、scope partitions、ranks与checkpoint selector完全相同。

## 5. `POST-SYNTH` placement diagnostic

令原始scope forecasts为

$$
A\in\mathbb{R}^{B\times C\times S\times T},\qquad T=720.
$$

在`_scope_forecast`之后、existing direct policy fusion之前计算

$$
\bar A=\operatorname{mean}_s A_s,\qquad d_s=A_s-\bar A,
$$

$$
q_s=U_o\left[
\operatorname{GELU}(U_c\bar A)\odot
\operatorname{GELU}(U_pd_s)
\right],\qquad A'_s=A_s+q_s.
$$

$U_c,U_p\in\mathbb{R}^{r_{post}\times T}$、$U_o\in\mathbb{R}^{T\times r_{post}}$，无bias，$U_o=0$。
它直接在forecast space工作，不使用DCT、pseudoinverse、frozen projection或representation replacement，因此不存在任意
projection与co-adaptation confound。

冻结derived rank rule：

$$
r_{post}=\operatorname{round}\left(\frac{Lr}{T}\right).
$$

这是由pre-synthesis parameter budget确定的解析规则，不按dataset performance选择。

| Dataset | $L$ | CPSI params | $r_{post}$ | POST params | Module gap | Total-model gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 436 | 41,856 | 19 | 41,040 | -1.9495% | -0.02243% |
| ETTh2 | 464 | 44,544 | 21 | 45,360 | +1.8319% | +0.03959% |
| ETTm1 | 464 | 44,544 | 21 | 45,360 | +1.8319% | +0.04013% |
| ETTm2 | 424 | 40,704 | 19 | 41,040 | +0.8255% | +0.00491% |
| Weather | 464 | 44,544 | 21 | 45,360 | +1.8319% | +0.03959% |

[Strong Evidence] added-module gap小于1.95%，总模型gap小于0.041%。POST-SYNTH不是exact parameter equality，故只可
判断placement evidence，不可用小差异作方向级拒绝。若POST明显更好，应删除“pre-synthesis necessary”claim，而不是把
CPSI performance自动判失败。

## 6. Initialization and local implementation gate

### 6.1 Paired initialization

对同一dataset/seed：

1. 五个new arms的base ISCF state hashes必须一致；
2. CPSI/SELF/LINEAR/COMMON的两组input matrices分别从paired tensors初始化；
3. 四臂$W_o$全零且shape一致；
4. POST使用同一initialization distribution与独立记录的derived rank；
5. zero-init forward必须与ISCF-v0在float32 tolerance内一致。

### 6.2 Step7A hard blockers

只有以下项目可以阻止进入prelaunch：

- tensor shape、scope permutation或CLI contract错误；
- parameter count违反exact/declared bound；
- initial morph超过`1e-6` max absolute error；
- output、loss或gradient出现NaN/Inf；
- $W_o$首个backward为zero/nonfinite，或一次optimizer step后input projection仍永久zero-gradient；
- interaction启用后forecast完全不响应nonzero synthetic perturbation；
- OOM或训练路径不能执行。

Step7A不比较validation MSE/MAE，不因随机小样本loss变化阻止candidate。

## 7. Validation, test and diagnostic severity

### 7.1 Validation role

validation只用于：

- mean MSE over `{96,192,336,720}`选择checkpoint；
- early stopping和ordinary implementation health；
- 记录product RMS、message/base ratio、common/private branch RMS、arm diversity与gradient norms；
- 发现numeric pathology。

所有protocol-valid arms无论validation相对排序如何都进入formal test。validation labels不得选择arm、rank、dataset或test
subset。

### 7.2 Severity classes

| Class | Definition | Consequence |
| --- | --- | --- |
| `hard_invalid` | NaN/Inf/OOM、morphism/shape/parameter contract failure、permanent zero path、artifact incomplete | repair exact design before test；不能作mechanism conclusion |
| `diagnostic_warning` | finite但branch/message很小、control轻微领先、单dataset异常、validation负向 | 完整保留；不阻止test；不调参 |
| `test_inconclusive` | CPSI vs ISCF macro MSE gain位于`[-0.5%, +0.3%)`，或positive但support不足；且无material catastrophe | 不方向级拒绝；保留candidate，依据预注册confirmation/redesign policy决定下一步 |
| `material_test_negative` | macro MSE `<= -0.5%`且至少4/5 datasets为负，或任一dataset degradation `>=5%`且无明确numeric pathology | exact CPSI-v1 performance fail；按failure attribution回Step4/5，不否定ISCF carrier |
| `initial_test_supported` | macro MSE `>= +0.3%`、至少3/5 datasets及10/20 cells正向，且MAE不低于`-0.3%` | performance viability pass；仍需matched attribution与后续confirmation |

阈值在任何new training/test之前冻结。`[-0.5%,+0.3%)`是不确定区，不是“负向也算成功”。它仅防止单seed小波动被
误写成方向级失败。

### 7.3 Control interpretation bands

对CPSI相对SELF/LINEAR/COMMON/POST的macro test MSE gain：

- `>= +0.3%`且至少3/5 datasets正向：相应attribution得到initial support；
- `(-0.3%, +0.3%)`：unresolved/tied，不作正负claim；
- `<= -0.3%`且至少3/5 datasets为负：control解释相应mechanism；只降级claim，不自动覆盖CPSI vs ISCF effectiveness；
- 任何`<0.3%`差异不得使用“显著优于”“证明必要性”等措辞。

特别地：

- SELF解释收益 -> `performance_partial_pass_capacity_unresolved`；
- LINEAR解释收益 -> nonlinear product necessity unsupported；
- COMMON解释收益 -> private modulation necessity unsupported；
- POST解释收益 -> pre-synthesis placement necessity unsupported；
- 多个controls同时解释 -> CPSI可保留engineering candidate，但paper-core mechanism claim blocked。

## 8. Formal matrix

### 8.1 New training arms

| ID | Role | New training |
| --- | --- | --- |
| `iscf_v1_cpsi` | working candidate | 5 datasets × seed2021 |
| `iscf_v1_cpsi_self` | nonlinear capacity diagnostic | 5 datasets × seed2021 |
| `iscf_v1_cpsi_linear` | linear sharing diagnostic | 5 datasets × seed2021 |
| `iscf_v1_cpsi_common` | common-only diagnostic | 5 datasets × seed2021 |
| `iscf_v1_cpsi_post` | placement diagnostic | 5 datasets × seed2021 |

共`25`个new trainings。每个run训练full-$T=720$ unified model，evaluation horizons固定为
`{96,192,336,720}`，MSE/MAE全部报告。

### 8.2 Historical effective references

- `iscf_v0`：复用既有seed2021 five-dataset checkpoints；prelaunch必须验证base initialization/contract兼容；
- `a6_full`：复用既有seed2021 five-dataset formal reference，只承担performance baseline，不承担CPSI mechanism attribution；
- SIFF-v2保持portfolio reference，不是本次primary gate，可在summary中列出但不改变matrix decision。

有效scorecard为`7 arms × 5 datasets × 4 horizons = 140` test MSE cells与140 MAE cells，并同步产生validation
scorecard。若historical artifact/hash不满足当前contract，必须在test前把相应reference改为from-scratch new run，不能静默
混用。

### 8.3 Test access order

1. Step7A local production checks；
2. Step7B runner/analyzer/prelaunch与historical hash audit；
3. commit/push；remote pull；`nvidia-smi`；两个resource smokes；
4. 完成25/25 training，不访问test；
5. 确认25/25 selected checkpoints与artifacts完整；
6. 一次性执行35 effective runs的official-test audit；
7. 冻结analyzer输出四层decision，不按test重跑rank/profile/arm。

用户本轮授权的是“走到test MSE/MAE再判断机制是否有效”的研究路径。实际remote/test execution仍须在Step7A/7B
machine checks、commit-pinned code和GPU preflight成立后执行；confirmation seeds、post-test tuning与第二机制不在授权内。

## 9. Four-layer decision

1. `paper_facing_effectiveness`：CPSI vs ISCF-v0和A6_FULL的完整test MSE/MAE；
2. `matched_mechanism_attribution`：CPSI vs SELF/LINEAR/COMMON/POST；
3. `internal_mechanism_health`：common/private states、product、message、base ratio、gradients、arms与policy；
4. `failure_attribution`：按capacity、linearity、common-only、placement、numeric或exact hypothesis映射。

可能的组合状态：

- effectiveness supported + attribution supported -> `initial_core_candidate_pass_confirmation_pending`；
- effectiveness supported + attribution unresolved/failed -> `performance_partial_pass_claim_blocked`；
- effectiveness inconclusive -> `test_inconclusive_keep_candidate_no_claim`；
- material negative且protocol valid -> `cpsi_v1_exact_performance_fail_return_step4_5`；
- pathology -> `diagnostic_invalid_for_direction_rejection_repair_design`。

## 10. Narrative/design decision

Step6现在解决了Step5的两个开放问题：

1. COMMON使用双common projections与product，做到exact functional parameter matching；
2. POST直接在full forecasts上做同构interaction，避免任意projection，仅保留小于0.041%的total-model parameter gap。

因此design gate通过：

```text
narrative_gate = step6_design_pass_with_test_first_effectiveness_and_diagnostic_controls
decision = step6_pass_step7a_local_authorized
```

授权边界：

```text
active_candidate = ISCF-v1-CPSI
active_method = none_until_step7a_implementation_pass
local_implementation = true
remote_training = false_until_step7b_prelaunch_and_commit
formal_test_execution = false_until_25_of_25_training_complete
confirmation_seeds = false
router_or_second_loss = false
```
