# ISCF-BSCA-v1 Step4–6：Balanced Scope Co-Adaptation

## 当前结论

`current_step=Step 6 complete; Step 7A authorized`。本候选以冻结的 ISCF-v0 架构为基底，只改变训练目标，不引入 requested-H、new router、新参数或 inference-time 分支。`narrative_gate=conditional_pass_to_one_test_informed_candidate`。

## problem

[Fact] ISCF 的五个 independent temporal-scope arms 通过 direct policy 融合。policy 不只决定 inference mixture；在 joint training 中，fused loss 对第 $s$ 个 arm 的梯度还乘以 $p_s$。因此 policy 同时承担 prediction allocation 与 gradient allocation。

[Strong Evidence] PSA-D1 已排除 run drift：contemporary EQUAL 与 historical EQUAL 的五组 checkpoint、metrics、fused output、arms 与 policy 全部 exact equal；ARMERR 与 SHUFFLED 却都相对 EQUAL 获得约 $0.66\%$ validation MSE 改善，且 D0 post-hoc smoothing 无收益。现有收益最稳定的解释是 train-time route regularization 改变了 co-adaptation，而不是 target-aware credit 本身。

## existence_evidence

检索日期：2026-07-22。检索范围：MoE load balancing、router gradient、expert specialization、time-series expert routing；来源只采用 arXiv、OpenReview、PMLR、NeurIPS proceedings 等 primary sources。

| Primary source | 与本设计的关系 | 采用/拒绝 |
|---|---|---|
| [BASE Layers, ICML 2021](https://proceedings.mlr.press/v139/lewis21a.html) | 证明 balanced assignment 是成熟 primitive | 不声称 balancing component novelty |
| [Dense Backpropagation Improves Training for Sparse MoE, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f69707de866eb0805683d3521756b73f-Abstract-Conference.html) | 支持 router/gradient path 会影响 expert training stability | 采用 gradient-allocation motivation |
| [Advancing Expert Specialization for Better MoE, NeurIPS 2025 Oral](https://openreview.net/forum?id=iydmH9boLb) | 指出 uniform load balancing 可能造成 expert overlap | 作为主要反证与 internal-health 风险 |
| [Similarity Preserving Routers, 2025](https://arxiv.org/abs/2506.14038) | 指出常规 balancing 可能产生不一致 routing/冗余 experts | 不外推 generic balancing 必然有效 |
| [Fast Training MoE for Time Series Forecasting via Expert Loss Integration, 2026](https://arxiv.org/abs/2605.10330) | 说明 forecasting 中 expert loss 会改变训练 dynamics | 仅作为相邻工作，不采用其具体 objective |
| [AME-TS, 2026](https://arxiv.org/abs/2605.25166) | 说明 TS expert specialization 可由 structured routing prior 稳定 | 区分其 MoE router prior 与本工作 dense scope-output coupling |

覆盖缺口：2026 sources 较新，部分仅有 preprint；因此 novelty 只在完整 contribution chain 上陈述，不声称 uniform KL 首创。

## idea

ISCF-BSCA-v1 在 EQUAL objective 上增加 train-only broad scope anchor：

$$
L_{\mathrm{base}}=L_{\mathrm{fused}}+L_{\mathrm{equal\ skill}},
$$

$$
q_s=1/5,\qquad
L_{\mathrm{BSCA}}=\mathbb E_{b,c,t}\left[\omega_t\,
\frac{\mathrm{KL}(q\|p_{bct})}{\log 5}\right],
$$

$$
L=L_{\mathrm{base}}+\lambda(u)L_{\mathrm{BSCA}}.
$$

$\lambda(u)$ 在前 25% optimizer progress 线性从 0 升至 0.1，之后保持 0.1；与已产生稳定 carrier clue 的 ARMERR/SHUFFLED route-weight schedule 完全一致。

## theory_check

Tensor contract：`policy p:[B,C,T,5]`；`q=full_like(p,0.2)`；`measure omega:[T]`；route KL 最终为 scalar。该 loss 只直接对 policy logits 反传，不直接依赖 `target:[B,T,C]` 或 `arm_forecasts:[B,C,T,5]`。但它通过改变 $p_s$，间接改变 fused loss 分配给各 scope arms 与 shared encoder 的梯度，因此测试的是 balanced co-adaptation，而不是 post-hoc calibration。

Bayes boundary：BSCA 不输入 requested horizon，也不声称改变 fixed-past pointwise-MSE 的 Bayes conditional mean；它只改变 finite-capacity joint optimization。

Self-critique：[Hypothesis] broad anchoring 可能抑制有用 specialization，或者只产生 generic regularization。故 official-test gain 与 internal policy/arm health 必须同时报告；entropy/usage 本身不能替代 MSE/MAE。

## design

- Architecture：与 ISCF-v0 exact same `siff-independent-scope-control`。
- Training：five datasets，seed2021，from scratch，H720 full objective。
- Checkpoint：validation mean MSE over H96/H192/H336/H720；test 不参与选择。
- Matched control：historical ISCF-EQUAL，same architecture/objective except BSCA anchor；其已有 official-test artifacts 复用，不重训。
- Formal test：只有 5/5 training artifacts 完整后，运行一次 BSCA five-dataset test；报告 5 datasets × 4 horizons × MSE/MAE 全部 cells，验证 checkpoint SHA256 before/after 不变。
- 不做 lambda search、per-dataset tuning、partial-cell selection 或 confirmation seeds。

## narrative_gate

`conditional_pass`。问题真实且有 ISCF-specific causal evidence；tensor/gradient path 清晰；成本低且 inference unchanged。贡献边界为：`dense future-output temporal scopes -> policy-mediated fused-gradient allocation -> train-only balanced co-adaptation -> unified-horizon effectiveness`。generic uniform KL/load balancing 不是 component novelty。

## effectiveness_gate

相对 ISCF-EQUAL official test 同时满足：macro MSE gain $\ge 0.3\%$；macro MAE gain $>0$；至少 3/5 datasets 与 3/4 horizons 的 mean MSE 改善；20/20 cells 完整；数值、checkpoint 与 scope diagnostics 健康。单 seed 通过只记为 `performance_partial_pass_pending_confirmation_seed`，不伪装成最终多 seed 结论。

## artifacts

- Frozen config：`configs/stage_c_iscf_bsca_v1.json`
- Objective：`baselines/timealign_official/layers/PCC.py`
- Step7A checker：`scripts/check_stage_c_iscf_bsca_step7a.py`
- Remote runner：`scripts/remote/run_stage_c_iscf_bsca_v1.sh`

## decision

Decision=`step4_6_conditional_pass_step7a_and_frozen_single_test_authorized`。若 formal test 无 pathology 但为负，failure=`hypothesis_false_for_exact_BSCA_v1`，rollback Step4；若出现 numeric/gradient pathology，仅回 Step7 修复 exact implementation，不据此否定 fixed ISCF architecture。
