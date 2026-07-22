# ISCF-FRSC Step 4–6 Design and Remote Gate

## 1. 11-step record

| Field | Record |
| --- | --- |
| `current_step` | exact SPS-v0 Step9 closed；FRSC-v0 Step4–6 complete；Step7A implementation pending |
| `problem` | hard scope projection诱导了role change，但删除shared full-rank forecast capacity；identity arms仍缺少可持续的scope-specific gradient conditioning |
| `existence_evidence` | SPS scope>global `+0.9041%`但scope<identity `-2.3123%`；FRSC frozen canonical `+0.7997%` MSE、5/5 datasets、4/4 horizons；random `-8.9750%` at alpha .55；global optimum `+0.8677%` |
| `idea` | 用full-rank SPD operator保留全部forecast directions，同时按native scope衰减out-of-scope component |
| `theory_check` | eigenvalues为`1`与`1-alpha>0`；no requested-H/new information/loss/router；gradient由同一operator conditioning |
| `design` | candidate scope-a055；same-alpha global-a055；best-tuned global-a045；random-a055；historical identity reference；five datasets、seed2021 |
| `narrative_gate` | `conditional_pass_as_full_rank_future_output_scope_conditioning`；generic spectral filtering claim prohibited |
| `effectiveness_gate` | frozen diagnostic不计method effectiveness；E2E validation必须超过identity、random和best-tuned global |
| `artifacts` | SPS Step9 audit、BSC-D0、FRSC-D1/D1.1、latest primary-source audit |
| `decision` | candidate=`SC-ISCF-FRSC-v0` narrative-ready；active method仍为none；implementation/remote training/test均未授权 |
| `rollback` | local fault -> Step5/6；candidate<=identity -> Step4；candidate<=global -> generic conditioning only；random fail -> binding attribution fail |

## 2. Evidence that changes the design

SPS 20/20 from-scratch validation runs完整且无numeric pathology。scope-canonical相对identity MSE/MAE为
`-2.3123%/-1.0937%`，但相对hard global projection为`+0.9041%/+0.8461%`。candidate
`removed_to_raw_rms=0.8769`，非point scopes只保留约`0.33–0.49` RMS，因此failure attribution是
`readout_or_head_design_wrong / hard capacity restriction too strong`，不是ISCF hypothesis false。

BSC-D0把identity policy barycenter保留为full-rank common forecast，仅投影arm deviations；canonical在20/20 MSE cells负向，
macro MSE/MAE=`-0.2709%/-0.1729%`。global control近似identity，说明BSC只是破坏co-adapted cancellation，没有即时
scope lead。该结果只关闭exact frozen BSC readout。

FRSC-D1/D1.1使用同一identity probes。canonical risk curve在`alpha=.55`达到MSE/MAE
`+0.7997%/+0.4219%`，18/20 MSE cells、5/5 datasets、4/4 horizons正向；同一alpha random为
`-8.9750%`，表明temporal binding对该operator非常敏感。global在`alpha=.45`达到`+0.8677%`，比canonical
envelope高`0.0680` percentage point，因此frozen evidence是`performance_partial_positive_attribution_unresolved`，不能把
generic low-pass conditioning包装成scope contribution。

## 3. Full-Rank Scope Conditioning

令SPS local projector为$P_s=P_s^\top=P_s^2$，FRSC对raw scope arm $a_s$使用

$$
Q_s(\alpha)a_s
=\left[P_s+(1-\alpha)(I-P_s)\right]a_s
=a_s-\alpha(I-P_s)a_s,
\qquad 0<\alpha<1.
$$

$Q_s$在$\operatorname{im}(P_s)$上的eigenvalue为1，在orthogonal complement上为$1-\alpha$。candidate
`alpha=.55`时minimum eigenvalue为`.45`、condition number为`2.2222`，故不存在rank deletion或zero-gradient direction。
对loss $L$，

$$
\frac{\partial L}{\partial a_s}=Q_s\frac{\partial L}{\partial \tilde a_s}.
$$

因此五个independent maps接收由future-output coupling extent定义的不同、但均非零的gradient conditioning。scope1保持
identity；中间scopes使用不同local groups；scope720使用global low-rank subspace的soft complement attenuation。所有arms、
policy与conditioning必须from-scratch end-to-end joint training。

该operator不扩大information set，也不改变fixed-past pointwise-MSE Bayes boundary。由于$Q_s$ invertible，FRSC不声称扩大
function class；合法机制是finite-capacity optimization/regularization bias。它不是post-hoc residual adapter，production path是
raw arm生成与policy fusion之间的native full-rank synthesis operator。

## 4. Primary-source boundary audit

检索日期为`2026-07-22`。query覆盖output spectral filtering、spectral bias、multi-scale future predictors、expert
specialization与full-rank spectral control；来源优先official proceedings、OpenReview或arXiv。

- [NHITS, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25854)已覆盖multi-rate sampling、hierarchical
  interpolation与不同frequency/scale components；FRSC不得claim首次multiresolution forecast synthesis。
- [TimeMixer, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a7ac8a21e5a27e7ab31a5f42a0117bdb-Abstract-Conference.html)
  已覆盖multiscale decomposition与multiple future predictors；parallel predictors/composition不新。
- [CFPT, ICML 2025](https://proceedings.mlr.press/v267/kou25b.html)已覆盖cross-frequency interaction；frequency-aware
  operator本身不能作为novelty。
- [xCPD, ICLR 2026](https://openreview.net/forum?id=uIPAuyno4Z)以graph spectral decomposition路由channel-patch
  dependencies；spectral routing/oversmoothing motivation已有直接prior，但作用轴不是future-output coupling groups。
- [SALT, ICLR 2026 submission](https://openreview.net/forum?id=d3zrIHukon)把spectral bias与gradient interference作为问题，
  并用component extraction + separable training处理；该submission进一步要求本项目不得泛称首次spectral-gradient specialization。
- [MoHETS, arXiv 2026](https://arxiv.org/abs/2601.21866)覆盖shared continuity expert与routed Fourier local experts；
  generic shared/private expert或frequency specialists claim均不可用。

因此只保留完整贡献边界：

```text
fixed-past full-domain forecast
-> independent future-output coupling-scope arms
-> scope-indexed full-rank SPD synthesis/gradient conditioning
-> target-wise composition
-> identity / same-alpha global / best-tuned global / random-binding attribution
```

该边界是`conditional contribution-level novelty`，不是component-first novelty。若E2E candidate不超过best-tuned global，完整
chain断裂，FRSC只能降为generic regularization evidence。

## 5. Frozen Step7B design

Step7A通过后才可生成runner。新训练矩阵为4 arms × 5 datasets × seed2021=`20` runs：

| Arm | Operator | Alpha | Role |
| --- | --- | ---: | --- |
| `frsc_scope_a055` | canonical scope | .55 | candidate；D1.1 global validation optimum |
| `frsc_global_a055` | global | .55 | exact same-alpha generic control |
| `frsc_global_a045` | global | .45 | strongest validation-tuned generic control |
| `frsc_random_a055` | random scope binding | .55 | exact structure control |

`sps_identity_canonical`五个existing checkpoints作为frozen reference；它们使用相同seed/profile/objective和四standard-horizon
validation selector。Step7A必须证明candidate/control base-parameter initialization hashes与references一致，alpha不创建trainable
parameters，且`alpha=0`与identity exact equal。

所有new runs继续以validation H96/192/336/720 mean MSE选checkpoint；official test不创建loader。candidate相对identity
的primary validation gate为macro MSE至少`+0.3%`、MAE正、MSE 12/20 cells、3/5 datasets、3/4 horizons正，且无dataset
退化超过3%。机制归因还必须同时满足：

1. candidate相对`frsc_global_a045` macro MSE至少`+0.1%`且3/5 datasets正；
2. candidate相对`frsc_global_a055` macro MSE为正；
3. candidate相对random macro MSE至少`+0.3%`且4/5 datasets正；
4. five arms gradients finite/nonzero、minimum operator eigenvalue合同成立、policy/arm diversity不collapse；
5. 20/20 matrix和所有negative cells完整报告。

validation只能决定是否申请一次新candidate的formal test，不能建立paper-facing effectiveness。若candidate只超过identity/random但不
超过global，decision=`performance_partial_pass_generic_conditioning_explains`；不追加loss/router或per-dataset alpha rescue。

## 6. Authorization boundary

- model implementation: `false`，等待本Step4–6记录同步后的Step7A；
- remote training: `false`，需要Step7A/7B全部local gates通过并获得用户独立授权；
- formal test: `false`；
- confirmation seeds / modern baselines: `false`；
- requested-H embedding、new router、auxiliary specialization loss: `false`。

## 7. Step7A actual result

production/local contracts全部通过：

- alpha=0 path与ISCF parent max absolute gap=`0`，full/prefix gap=`0`；
- parent、identity、candidate、same-alpha global、best-global与random的parameter hash完全一致；
- candidate minimum operator eigenvalue=`0.45`，没有zero-gradient direction；
- candidate相对parent/global-a055/global-a045/random max gaps分别为
  `1.19329/0.83122/0.61751/0.91589`，controls非恒等；
- five scope mode-map gradient norms=`[0.16609,0.07302,0.06870,0.05083,0.04085]`，全部finite/nonzero；
- production model输出`[1,720,2]`与prefix`[1,96,2]`，prefix gap=`0`；
- production CLI固定readout、scope projection、alpha .55、rank109、four validation horizons与val-only split。

Step7A decision=`iscf_frsc_step7a_contract_pass`。该结果不含training/validation/test evidence。下一步只授权local Step7B
prelaunch implementation；remote training与formal test仍false。

Decision=`FRSC_v0_step7a_pass_prelaunch_next_remote_false`。
