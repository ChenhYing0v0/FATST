# Post-D20 Contribution 1 Step 2/4 Reassessment

## 1. Research record

| Field | Content |
| --- | --- |
| current_step | D20 closed at Step10；Contribution 1 rollback Step2/4；D20-D1 posthoc diagnostic next |
| problem | raw history structure的predictive utility是否随future coordinate与distribution改变，而uniform additive injection无法校准 |
| existence_evidence | SPEC-vs-RANDOM weak positive且随distance衰减；SPEC-vs-A6 validation→test reversal |
| idea | 先诊断contribution direction/scale，再评估future-distance predictive-support operator |
| narrative_gate | no new method yet；generic spectral robustness与scalar gate repair fail |
| effectiveness_gate | D20 failed；new candidate not started |
| rollback_point | Step2 if support problem unsupported；Step4 only after stable problem diagnostic |

## 2. External source refresh

- 检索日期：`2026-07-20`；
- query scope：unified multi-horizon decoder、frequency persistence、spectral shift、robust forecasting、dynamic horizon；
- source type：external arXiv、OpenReview、PMLR primary pages；
- Zotero：未以收录情况判断novelty或freshness。

Primary-source boundary：

| Work | Relevant coverage | Boundary for this project |
| --- | --- | --- |
| [Frequency Matters, 2025](https://arxiv.org/abs/2511.05619) | spectral mismatch导致跨domain generalization degradation | 不能把“frequency shift影响forecast”作为新问题 |
| [DropoutTS, ICML 2026](https://openreview.net/forum?id=7sksHLUvhH) | spectral sparsity估计instance noise并自适应dropout | sample-adaptive spectral robustness已有直接强prior |
| [Fremen, withdrawn ICLR 2026](https://openreview.net/forum?id=4IZzgIyD91) | 测量frequency-wise stationarity并downweight unstable components | frequency stability weighting有直接邻近链；因withdrawn只作medium-confidence压力 |
| [Adaptive Energy Amplification, ICLR 2026 submission](https://openreview.net/forum?id=O5uoS9ICec) | 避免按energy盲目放大frequency、强调generalization | frequency energy calibration不是空白primitive |
| [Implicit Forecaster, NeurIPS 2025](https://openreview.net/forum?id=gqoeQPhQcE) | frequency/amplitude/phase global synthesis与history-spectrum skip | wave decoding与spectral skip已有覆盖 |
| [FlowState](https://openreview.net/forum?id=R50AT6nAsM) | functional basis decoder与dynamic horizons | flexible-horizon basis generation已有覆盖 |
| [TimePerceiver](https://arxiv.org/abs/2512.22550) | target-coordinate queries与decoder-training co-design | generic coordinate-conditioned decoder不是独立novelty |

[Strong Evidence] D20的generalization现象与最新spectral-shift文献一致，但该文献也阻止我们把generic robust
frequency filtering包装成Contribution 1。创新必须落在完整的fixed-past multi-horizon generation contract上。

## 3. Candidate-family screening

### R1 — D20 scalar gate / normalization repair

`diagnostic_only / rejected_by_narrative_gate_for_method`。

对summary做RMS normalization、加一个learned scalar gate或降低LR可能改善scale，但只是在失败additive bypass上做
engineering rescue。D20-D1可以用oracle判断这类headroom是否存在，却不能因此直接启动v2 training。

### R2 — Generic frequency-stability robust forecaster

`rejected_by_overlap_and_scope`。

Frequency Matters、DropoutTS、Fremen与AEA已覆盖spectral shift、frequency stationarity、spectral noise和adaptive
energy。该方向也没有自然解释同一个full trajectory如何服务不同prefix，不能作为当前multi-horizon主线。

### R3 — Future-distance predictive-support operator

`provisional_problem_family / not method-ready`。

[Hypothesis] 同一history pattern对future coordinate$\tau$的predictive support并非常数。D20中SPEC-vs-RANDOM
从H1–48的`+0.985%`衰减到H513–720的`-0.011%`，提示结构信息的有效范围可能随future distance衰减；uniform
coefficient injection把这种局部有效性扩散到完整trajectory，造成shortcut和shift sensitivity。

潜在完整chain为：

`fixed past -> historical modes/features -> coordinate-wise predictive-support envelope -> one full-T trajectory -> prefix crop`。

它与requested-horizon conditioning不同：模型不读取deployment horizon$H$，只在一次full-domain generation中使用
固定future coordinate$\tau$。它与generic spectral robustness也不同：目标不是清洗input，而是约束每类history
evidence可以影响哪些future coordinates。

但是D20只测试一个low-frequency subspace，尚未证明mode-wise support可由past识别、可跨split泛化，或能超过A6。
因此R3不能直接进入Step4 method implementation。

## 4. Provisional two-contribution chain

以下只作为研究蓝图，不是当前paper claim：

1. `Contribution 1 — Support-Calibrated Trajectory Operator`：让history evidence通过future-coordinate support
   envelope进入full-trajectory generation，而不是uniform additive coefficient bypass；
2. `Contribution 2 — Retrospective Support Calibration`：在训练窗口内部构造多个past-only pseudo cut points，端到端
   监督“某类history evidence能稳定预测多远”，避免offline teacher和requested-H semantics。

两者能够形成统一叙事：Contribution 1提供需要校准的multi-horizon operator，Contribution 2利用仅限train history的
retrospective tasks学习其support。反方意见是multi-cut auxiliary training与frequency stability已有邻近工作；在正式
Step4前必须进一步检索并用problem diagnostic证明support envelope不是多余模块。

## 5. Immediate next diagnostic

先执行`SC-D20-D1-CONTRIB`，不训练新模型：

- 从SPEC/RANDOM checkpoint artifacts恢复within-model base与summary contribution；
- 分future bins计算actual contribution gain、oracle optimal scale与residual alignment；
- 若actual path有益但完整model仍差于A6，优先归因co-adaptation/redundancy；
- 若$0<\alpha^*<1$但$\alpha=1$有害，保留scale-miscalibration evidence；
- 若$\alpha^*\le0$占主导，则当前statistic direction本身不成立。

D1只决定下一次Step2/4问题诊断的优先级，不授权D20-v2、R3 implementation、remote training或paper claim。

## 5.1 D1 returned result

D1的90行oracle audit全部finite、重构误差为0。SPEC contribution相对其co-adapted base平均`+26.8928%`，39/40
future-bin cells有益，median optimal alpha为`1.2649`；RANDOM也为`+9.0422%`、35/40、alpha`1.4115`。因此
scalar shrinkage不是答案。新增path在joint model内部很重要但完整model仍差于A6，说明prediction responsibility被
重新分配，而不是证明新增独立information。

D1也没有证明future-distance support：SPEC contribution几乎所有bins均有益。R3继续`problem_unverified`，下一步
必须回Step2/3寻找跨机制、跨split且past-identifiable的existence evidence，不能进入method Step4。

## 6. Decision

`d1_complete_scalar_fix_rejected_coadaptation_explains_return_step2_3`。

不做confirmation、不做D20小修补。下一步只允许设计一个train-only、split-stable的predictive-support existence
diagnostic；在其problem gate通过前，Support-Calibrated Operator与Retrospective Support Calibration均不实现。
Contribution 2继续停在Step2，直到Contribution 1出现真实training mismatch。
