# SC2 Projective Coupling Credit Step2-4 Source And Idea Audit

## Status

| Field | Value |
| --- | --- |
| `current_step` | SC2 Step2/3 problem confirmed；Step4 candidate proposed；Step5 conditional pass |
| `candidate` | `SC2-PCC-v0`（Projective Coupling Credit） |
| `role` | fully end-to-end training strategy candidate；not implemented |
| `problem` | fused loss下structured coupling arms缺少skill-preserving、target-specific credit |
| `narrative_gate` | conditional pass；完整claim仍须Step6冻结task-specific controls |
| `effectiveness_gate` | not started |
| `remote/test/confirmation` | false |

## Problem Evidence

PCSD-CF Step7B提供了比generic“router可能collapse”更具体的证据：

1. 25/25 DIRECT-run scope arms相对相同scope独立E2E fixed training退化，median `89.95%`；
2. learned policy不是one-hot collapse，但future-bin usage variation仅L1 `0.0051-0.0440`；
3. DIRECT相对A6为0/5、macro -1.5833%，但相对dense capacity control为5/5、+2.3492%；
4. same-run row/bin oracle仅3/5为正，说明“存在credit headroom”和“所有dataset均可学target router”必须分开。

[Strong Evidence] 问题是joint fused training没有维持structured arm capability，而非简单parameter shortage。
[Hypothesis] 若同一次forward可给arms与router分配forecast-risk-aware credit，PCSD representation signal可能转化为
effectiveness；具体PCC尚未得到实验支持。

## External Search Record

- `search_date`: 2026-07-16
- `scope`: MoE under-training/collapse、load balancing、soft routing、forecasting MoE、forecast ensembling、
  multi-horizon loss shaping
- `source_policy`: 仅primary paper/proceedings/arXiv；Zotero未作为发现源
- `Zotero status`: 下列条目均由external search发现；是否已存在Zotero未复核

### Key source boundaries

1. [Switch Transformers, JMLR 2022](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf)使用batch-level
   usage/confidence auxiliary loss逼近uniform expert load。它解决计算利用率，不识别哪个scope对哪个future target
   的forecast risk更低；uniformity不能直接作为PCC。
2. [Expert Choice Routing, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html)
   明确指出poor routing会使experts under-trained，并通过expert选择tokens固定bucket size。其离散capacity routing
   与PCSD dense future-output fusion不同，且不提供forecast-loss capability credit。
3. [Soft MoE, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/79fea214543ba263952ac3f4e5452b14-Paper-Conference.pdf)
   证明fully differentiable dispatch/combine可避免hard-routing问题；但PCSD当前已经是soft convex fusion，25/25
   arm starvation说明“保持可微”本身不充分。
4. [Auxiliary-Loss-Free Load Balancing, 2024](https://arxiv.org/abs/2408.15664)用historical load更新expert bias，
   避免auxiliary interference gradients。它仍只平衡usage而非capability，而且bias在gradient objective外更新，
   不满足本项目one-stage end-to-end credit目标。
5. [Mixture-of-Linear-Experts, AISTATS 2024](https://proceedings.mlr.press/v238/ni24a.html)已在LTSF中端到端
   mixture多个forecast heads，并让channel-wise router组合outputs。故“forecast heads + router + joint loss”不是
   novelty；PCC必须claim output-coupling scopes与projective future risk credit的完整链。
6. [MoHETS, arXiv 2026](https://arxiv.org/abs/2601.21866)在forecasting MoE中使用prediction loss加assignment
   balance loss，并用heterogeneous temporal/frequency experts。它进一步阻止“多尺度experts + balance loss”claim。
7. [FAME, arXiv 2026](https://arxiv.org/abs/2606.08896)从validation performance挖掘expert-suitability targets再训练
   router，说明offline suitability supervision已有直接邻近工作；这也是本项目放弃CCRL式two-stage labels的原因。
8. [Expert-Router Coupling, arXiv 2025](https://arxiv.org/abs/2512.23447)用proxy activations约束router embedding与
   expert capability的一致性。PCC不能泛称“首次couple router and experts”，只能claimforecast-risk、future-region
   与projective output-coupling的task-specific coupling。
9. [Loss Shaping Constraints, ICML 2024](https://proceedings.mlr.press/v235/hounie24a.html)已经指出平均window loss会
   造成forecast-step error不均，并以primal-dual constraints塑造逐step loss。因此simple horizon reweighting或
   “关注不同future positions”不具备充分新颖性。
10. [Forecast Stacking Guarantees, ICML 2023](https://proceedings.mlr.press/v202/hasson23a.html)允许ensemble weights
    随item、forecast timestamp和quantile改变，但依赖cross-validated stacked generalization。PCC需明确区别于
    black-box post-hoc stacking：arms是同一decoder field内的structured views，credit在single forward中产生。

[Novelty assessment] 在本次检索范围内，没有primary work覆盖
`fixed-past projective full-domain generation -> shared output-coupling field -> same-forward forecast-risk credit ->
skill-preserving arm/router co-training`完整链。该判断为medium confidence；primitive-level overlap很强，不能claim
generic MoE credit、load balancing、loss shaping或forecast stacking首创。

## Rejected Step4 Shortcuts

| Candidate | Decision | Reason |
| --- | --- | --- |
| uniform load-balance loss | reject as paper core | usage不等于capability；可能强迫无用scope均匀化 |
| entropy/diversity regularization | control only | current entropy已不低，仍有25/25 under-training |
| loss-free router bias | reject for current problem | 调整load而非forecast regret；训练过程含objective外state update |
| independently pretrain/freeze arms | diagnostic only | 重回two-stage/frozen replacement不公平与训练不一致 |
| validation-mined suitability labels | reject | FAME/stacking/旧CCRL overlap，工程重且非one-stage |
| equal per-arm full loss alone | mandatory control | 可测试skill floor，但缺少target-specific credit与完整贡献边界 |

## Step4 Candidate: Projective Coupling Credit

令同一PCSD field产生$S=5$个scope forecasts$F_s(X,t)$，router产生$\pi_s(X,t)$，fused forecast为

$$
F(X,t)=\sum_{s=1}^{S}\pi_s(X,t)F_s(X,t).
$$

PCC只在training使用ground-truth计算same-forward counterfactual arm error$e_s(t)$；inference仍只输入history与
natural target coordinate。定义scale-normalized regret$\tilde e_s(t)$后，entropy-regularized capability target为

$$
q_s(t)=\operatorname{softmax}_s\left(-\operatorname{sg}[\tilde e_s(t)]/\tau\right),
$$

其中$\operatorname{sg}$为stop-gradient。为防止poor-at-start arms继续starve，加入skill floor

$$
q_s^{\epsilon}(t)=(1-\epsilon)q_s(t)+\epsilon/S.
$$

候选objective为

$$
\mathcal L_{\mathrm{PCC}}
=\mathcal L_{\mathrm{fuse}}
+\lambda_{\mathrm{skill}}\sum_t\omega_t\sum_s q_s^\epsilon(t)\ell(F_s(t),Y_t)
+\lambda_{\mathrm{route}}\sum_t\omega_t\operatorname{KL}(\operatorname{sg}[q(t)]\|\pi(t)).
$$

这里$\omega_t$不是benchmark horizon ID，而是full future domain上的projective measure。若目标是dense prefix MSE
AUC，可用exact identity

$$
\frac1T\sum_{H=1}^{T}\frac1H\sum_{t=1}^{H}e_t
=\sum_{t=1}^{T}\omega_t e_t,
\qquad
\omega_t=\frac1T\sum_{H=t}^{T}\frac1H.
$$

该identity本身不作为创新；它只规定same-forward capability credit如何对unified prefix family积分。PCC也不要求
requested $H$进入forward，full-T forecast仍只在最后prefix crop，因此保持projectivity。

## Why This Is More Elegant Than CCRL

- one forward同时得到arms、fused forecast、credit target与所有loss；
- 不训练fold × scale teachers，不减少有完整label的samples，不保存offline risk labels；
- arms、router、Encoder保持from-scratch E2E，不使用frozen replacement；
- capability target由当前structured arms的真实forecast error产生，而不是generic uniform usage；
- inference graph与PCSD-CF不变，训练辅助量全部删除。

## Self-Critique And Falsification

1. $q$是moving target，可能放大早期noise；$\epsilon$只给gradient lower floor，不能保证shared field的gradient
   conflicts可解；
2. per-target error可能过噪，局部平滑会引入额外kernel choice；不得用benchmark horizons作bins；
3. skill loss可能使arms趋同，削弱scope specialization；必须比较equal full-arm supervision与PCC；
4. dense-prefix measure可能单独解释short-horizon gain；必须有A6/PCSD measure-only controls；
5. ground-truth-generated$q$在training合法，但若router无法从history预测它，oracle headroom不会转化为inference gain；
6. fixed arms只有2/5超过A6，PCC即使修复credit也可能受shared-field readout quality上限阻断。

## Step5 Theory Feasibility Gate

下一步只做local theory/synthetic gate，不启动remote：

1. 推导plain fused loss的$\partial\mathcal L/\partial F_s=\pi_s\partial\mathcal L/\partial F$与PCC skill-floor
   gradient lower-bound，检查shared parameters下哪些结论只是output-level proxy；
2. 验证$\omega_t$ identity、normalization、arbitrary-prefix projectivity与no requested-H access；
3. synthetic crossed test必须同时出现history-dependent和target-dependent best scope，PCC需让五arms保持skill且router
   recovery超过plain/equal-skill/load-balance controls；
4. 计算额外activation/FLOP；PCC不得引入second forward、teacher checkpoints或$O(N^2)$sample storage；
5. Step5若不能区分generic skill floor与task-specific capability credit，则PCC降为training control，SC2返回Step4。

## Returned Step5 Gate

2026-07-16 local checker的15/15 cases通过：plain/PCC arm与router梯度解析式最大误差不超过`5.20e-18`；
dense-prefix identity误差`4.44e-16`；full-domain crop projectivity gap为`0`；crossed synthetic router的
capability KL为`1.50e-11`、argmax accuracy为`1.0`。完整解释见
`analysis/stage_c_sc2_pcc_step5_theory_20260716/step5_theory_feasibility.md`。

[Decision] Step5=`conditional_pass_step6_design_only`。该结果只证明algebra、projective measure与toy
recoverability，不证明真实arm skill、capability predictability或forecast gain。下一步只授权Step6 control matrix、
optimization schedule与rollback gate设计；implementation、remote、test、confirmation仍false。
