# Post-D19 Compact Statistic Viability Audit

## 1. Audit identity

| Field | Frozen value |
| --- | --- |
| `candidate_id` | `SC-D20-CST` |
| `role` | `diagnostic_only` |
| `test_informed` | `true`：问题来自D19 official-test结果 |
| `current_step` | Contribution 1 Step2/4完成；D20 Step2/3 proposed；下一步Step6 diagnostic freeze |
| `paper_method` | `false` |
| `implementation_authorized` | `false` |
| `remote_authorized` | `false` |
| `rollback_point` | Contribution 1 Step2/4 |

本报告回答D19关闭后能否直接把`compact spectral decoder`提升为新Contribution 1。结论是不能；但D19留下了
一个更小、可证伪的问题，值得用一次严格的transfer/specificity diagnostic回答。

## 2. What we plan to test

固定问题为：

> 在相同720-point normalized history、相同A6 Encoder与相同learned full-trajectory basis下，一个低维
> history-spectrum summary能否稳定改善A6 coefficient operator，并且其收益超过同维random orthogonal
> history projection？

该问题包含两个不可互换的条件：

1. `transfer`：D19的skip收益能否从IF readout迁移到strong A6 coefficient operator；
2. `specificity`：若有收益，它是否来自frequency semantics，而不是新增一条generic history path或新增参数。

只有两者同时成立，才说明“被压缩状态遗漏的、具有频率结构的history information”是一个可继续研究的问题。

## 3. Why this matters

D19给出四条已确认事实：

1. `IF_MEASURE`相对`A6_MEASURE`为`-3.6117%` MSE，不能作为新decoder候选；
2. `IF_MEASURE`相对parameter-matched direct nonlinear control为`-0.8075%`，polar/frequency synthesis没有
   matched-control支持；
3. `IF_MEASURE`相对`IF_NOSKIP_MEASURE`为`+1.6191%`、16/20 cells，direct history spectrum在IF路径内有用；
4. D19 internal health全通过，但heads为A6参数量的7.94--10.29倍且12/15 new arms在epoch 1选中，表明exact
   readout存在scale/optimization mismatch，不能方向级否定所有structured decoder。

因此，直接缩小IF只是在失败架构上做engineering rescue；直接把spectrum接到A6上又只能得到generic feature
augmentation。D20的价值是先判断这个较窄的existence question是否成立，而不是验证某个论文方法。

## 4. Source-informed boundary

外部primary-source audit见`Papers/post-d19-compact-statistic-decoder-audit.md`。其边界为：

- FITS已覆盖low-pass spectrum与complex frequency interpolation；
- FBM覆盖显式time-frequency basis mapping；
- Implicit Forecaster覆盖amplitude/phase heads与input-spectrum skip；
- PhaseFormer覆盖compact phase representation与routing；
- BasisFormer、FlowState、N-HiTS与TimePerceiver分别覆盖basis-conditioned forecasting、dynamic-horizon
  functional basis、multirate interpolation与target-coordinate decoding；
- ICML 2024的linear-model analysis进一步说明：当transform可逆且后接unconstrained linear head时，Fourier
  coordinates本身不产生新的function class。

所以本项目不能claim first spectral decoder、first phase decoder、first compact decoder或first dynamic-horizon
basis。prior-art overlap没有自动否决完整contribution chain，但已否决“把频谱接入预测头”作为独立创新。

## 5. Theory check

### 5.1 Algebraic no-go boundary

令normalized history为$x\in\mathbb{R}^{720}$，若$Q\in\mathbb{R}^{720\times720}$可逆，且

$$
s=Q^\top x,\qquad \hat y=W_s s+b,
$$

则

$$
\hat y=W_sQ^\top x+b
$$

仍是普通affine history-to-future map。因而full-spectrum linear head不能建立独立机制。

D20必须使用compact $q\ll720$ statistic，并与相同$q$的random orthogonal projection比较。即使compact spectrum
获胜，也只通过problem gate；后续paper method仍需说明为何该subspace与multi-horizon full-trajectory coefficient
generation构成native contract。

### 5.2 Projectivity contract

所有arms保持A6的shared full-$T$ synthesis：

$$
\hat Y_T=\Phi_T a(x),\qquad \hat Y_H=P_H\hat Y_T.
$$

history statistic只允许进入full-trajectory coefficient computation，不能读取requested horizon，也不能为不同
horizon创建独立head。这样D20仍检验fixed-past unified multi-horizon carrier，而不是horizon-specific models。

### 5.3 Fairness contract

D20是matched end-to-end diagnostic：

- 所有arms从相同initialization class训练；
- A6 Encoder不冻结、不复用与旧decoder共同训练的representation；
- dataset profile、objective、optimizer、seed、checkpoint selector与test matrix完全相同；
- spectrum与random projection的summary dimension、coefficient-head shape和新增参数完全一致；
- random projection固定且不训练，避免control自行学习成另一个有利subspace。

## 6. Proposed diagnostic design

本节冻结family和比较逻辑；精确$q$、normalization、FFT bins、initialization与machine-readable matrix必须在
Step6 code-theory audit中冻结，当前不得实现。

### 6.1 Arms

| Arm | History path | Purpose |
| --- | --- | --- |
| `A6_MEASURE_RETRAIN` | 原A6 compressed state | same-run end-to-end anchor |
| `A6_CST_SPEC` | compact real/imag history-spectrum summary | transfer candidate |
| `A6_CST_RANDOM` | same-dimensional fixed random orthogonal projection | generic history-access + parameter control |

三臂共享A6 Encoder、learned basis size、full-$T$ generation、measure objective与prefix crop。`SPEC/RANDOM`仅在
coefficient operator输入的低维summary不同；不得改变decoder depth、activation或训练时长。

`A6_CST_ZERO`暂不列为必需arm：zero summary会让新增输入权重失活，不能形成与SPEC/RANDOM同等的active
function class。若Step6发现bias/normalization带来额外差异，再将其加入numeric control，但不得用它替代random
projection。

### 6.2 Frozen evaluation surface

- datasets：ETTh1、ETTh2、ETTm1、ETTm2、Weather；
- primary horizons：96、192、336、720；
- primary metrics：official-test MSE与MAE；
- checkpoint：validation四horizon MSE平均；
- first-stage seed：2021；
- full matrix：5 datasets × 3 arms = 15 runs，60 official-test cells；
- test role：`test_informed formal mechanism diagnostic`，完整报告所有cells，不允许dataset/horizon tuning。

### 6.3 Primary gates

先冻结以下对称gate；Step6只可因统计定义错误或实现不可比而修改，不能因预期难以通过而放宽：

| Gate | Required comparison | Pass condition |
| --- | --- | --- |
| `transfer_mse` | SPEC vs A6 | macro MSE gain >=0.3% |
| `transfer_cells` | SPEC vs A6 | >=11/20 MSE cells |
| `transfer_coverage` | SPEC vs A6 | >=3/5 datasets and >=3/4 horizons positive |
| `transfer_mae` | SPEC vs A6 | macro MAE gain >=0% |
| `specificity_mse` | SPEC vs RANDOM | macro MSE gain >=0.3% |
| `specificity_cells` | SPEC vs RANDOM | >=11/20 MSE cells |
| `specificity_coverage` | SPEC vs RANDOM | >=3/5 datasets and >=3/4 horizons positive |
| `specificity_mae` | SPEC vs RANDOM | macro MAE gain >=0% |

其中gain定义继续沿用项目规则：

$$
\operatorname{gain}(A,B)=\frac{\operatorname{metric}(B)-\operatorname{metric}(A)}
{\operatorname{metric}(B)}\times100\%,
$$

正值表示第一个arm更好。

### 6.4 Internal mechanism health

Step9必须同时输出以下diagnostics，不能只看aggregate test MSE：

1. `summary_gradient`：SPEC/RANDOM summary-input weights的梯度finite且非零；
2. `summary_usage`：selected checkpoint对应的summary weight norm与contribution norm非退化；
3. `prediction_deformation`：SPEC/RANDOM相对A6输出不是数值同一；
4. `frequency_specificity_by_dataset/horizon`：SPEC-RANDOM gain是否只由单一dataset或H720驱动；
5. `checkpoint_health`：best epoch分布、train/validation trajectory、NaN/Inf与budget ceiling；
6. `projectivity`：同一full output的prefix crop误差满足implementation tolerance；
7. `parameter_match`：SPEC与RANDOM参数完全一致，A6差值只作attribution说明、不作candidate选择。

## 7. How results support or falsify the plan

| Observation | Interpretation | Decision |
| --- | --- | --- |
| SPEC不超过A6 | D19 skip收益没有迁移到strong A6 coefficient operator | 关闭direct compact-spectrum route；Contribution 1回Step2 |
| SPEC超过A6但不超过RANDOM | generic direct history access/capacity解释收益 | frequency claim失败；回Step2/4审计非频谱history path，不能升method |
| SPEC同时超过A6与RANDOM，但internal path退化或numeric异常 | exact diagnostic protocol无效或readout design有问题 | `diagnostic_invalid_for_direction_rejection`；Step6/7 redesign一次 |
| SPEC同时通过transfer、specificity与internal health | compact frequency-specific information problem成立 | 只回Step4设计native non-residual coefficient operator；concat head仍不是paper method |

## 8. Self-critique

[Strong counterargument] random orthogonal projection只是一个generic subspace control，并不能穷尽time-domain、
learned or multiresolution alternatives。SPEC胜过RANDOM仍不足以证明“频率是唯一正确语义”。因此positive D20只能
建立frequency-specific evidence relative to the frozen control，不得宣称frequency optimality。

[Risk] D20可能再次把A6为global basis decoder的co-adaptation当成frequency path失败。这里通过三臂共同
end-to-end训练、同一个intervention point与same-dimensional controls降低该风险；但若SPEC/RANDOM同时严重失败，
结论只能是当前coefficient-input intervention不适合，不能否定history information本身。

## 9. Narrative gate

`SC-D20-CST`本身为`diagnostic_only`，因此不要求方法级narrative pass；但它禁止结果后升级为paper core。

当前三个candidate-family结论：

- smaller implicit frequency decoder：`rejected_by_narrative_gate`；
- history-phase-continued future atoms：`rejected_by_narrative_gate_for_method`，仅可作control；
- statistic-conditioned A6 operator：`problem_unverified`，只允许D20 diagnostic。

## 10. Effectiveness gate

`not_started`。D19的skip-vs-no-skip正值不能替代D20 transfer/specificity gate，也不能作为新candidate performance。

## 11. Artifacts

- D19 four-layer result：
  `analysis/stage_c_post_ccsf_step24_reset_20260719/d19_step9/d19_step9_deep_audit.md`
- external primary-source boundary：
  `Papers/post-d19-compact-statistic-decoder-audit.md`
- this Step2/4 audit：
  `analysis/stage_c_post_ccsf_step24_reset_20260719/post_d19_step24_compact_statistic_viability_audit.md`

## 12. Decision

`step2_4_complete_d20_diagnostic_step6_next`。

1. compact spectral generator不能直接成为Contribution 1；
2. smaller IF和history-phase atom routes不进入implementation；
3. `SC-D20-CST`仅作为transfer/specificity diagnostic进入Step6 exact design；
4. 当前implementation、remote、official-test access与paper-method promotion均为false；
5. D20通过后回Step4，而不是直接进入method Step7；D20失败则按第7节区分direction、intervention与generic
   capacity explanation。
