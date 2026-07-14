# SC1-PLGO Step 6 Tensor, Narrative And Control Gate

## Decision Summary

| Field | Decision |
| --- | --- |
| `candidate` | `SC1-PLGO-PAF` (Projective Atom Functional) |
| `current_step` | Step 6 complete |
| `tensor_gate` | pass：atomwise subset/prefix invariance max gap `4.547e-13` |
| `rank_gate` | pass：output effective rank不超过A6的256，不是full affine |
| `horizon_gate` | pass：descriptor与generator均不读取requested $H$ |
| `external_novelty_gate` | fail：generic form被DeepONet、NOMAD、BasisFormer与query decoders覆盖 |
| `internal_mechanism_gate` | fail/not established：B11 controls与B14 negative evidence不能被忽略 |
| `exact_A6_containment` | false；free atom table control才与A6等价 |
| `method_implementation` | `false` |
| `Step6_decision` | `narrative_not_ready_rollback_step2_3_d7` |
| `next` | `SC1-D7-RGNB-descriptor-sufficiency` diagnostic only |
| `rollback` | Step 2/3；D7失败则关闭descriptor-generator路线，RGNB只保留为component |

## 1. Reader Path

本报告依次回答：

1. Step 5留下的atom-conditioned generator能否形成明确tensor path？
2. 该path能否保持同一future function在不同prefix上的exact consistency？
3. function class是否被full-affine capacity解释？
4. 外部工作是否已经覆盖generic mechanism？
5. 内部历史实验是否支持把RGNB descriptors放进primary generator？
6. 当前是否可以进入Step 7，若不能应回滚到哪里？

## 2. Source And Search Record

- search date: 2026-07-14；
- search scope: branch-trunk operator、nonlinear manifold decoder、hypernetwork operator、basis coefficient
  attention、future timestamp queries、functional basis decoder、wavelet coefficient forecasting；
- source policy: external primary sources优先；Zotero未用于completeness判断；
- code status: BasisFormer official `model.py`已核对实际coefficient/future-basis tensor path；
- full-text boundary: DeepONet、NOMAD、BasisFormer正文已读；其余以official proceedings/arXiv/model card的
  method description为边界。

| Work | Verified mechanism | Occupied claim | PLGO remaining boundary |
| --- | --- | --- | --- |
| [DeepONet](https://arxiv.org/abs/1910.03193) | branch encodes input function；trunk encodes output locations；inner product生成function | shared input branch + descriptor/location trunk不是新机制 | descriptor改为RGNB atom仍属于branch-trunk family |
| [NOMAD](https://arxiv.org/abs/2206.03551) | nonlinear decoder $f(\beta,y)$表示target function manifold | latent+query nonlinear decoder已占据 | 不能以nonlinear atom decoder作为novelty |
| [HyperDeepONet](https://arxiv.org/abs/2312.15949) | hypernetwork把input function信息注入target function | input-conditioned decoder weights已占据 | HyperNetwork不是可辩护边界 |
| [BasisFormer](https://arxiv.org/abs/2310.20496) / [code](https://github.com/nzl5116190/Basisformer/blob/main/model.py) | basis与series bidirectional cross-attention生成`score [B,k,C,N]`，再加权future bases | basis-aware coefficient attention已直接占据 | 不能claim首次basis-conditioned coefficients |
| [TimePerceiver](https://arxiv.org/abs/2512.22550) | target timestamp queries cross-attend encoded input | target-query retrieval已占据 | PLGO禁止改名为atom-query retrieval |
| [CATS](https://arxiv.org/abs/2405.16877) | future horizon-dependent parameters作为cross-attention queries | future query与parameter sharing已占据 | query subset本身不是novelty |
| [FlowState](https://research.ibm.com/publications/flowstate-sampling-rate-invariant-time-series-foundation-model-with-dynamic-forecasting-horizons) | encoder输出coefficients，functional basis decoder生成continuous horizon | coefficient-space dynamic-horizon decoder已占据 | 剩余只能是local/global support-projective组合 |
| [WaveToken](https://arxiv.org/abs/2412.05244) | autoregressively forecast quantized wavelet coefficients | localized coefficient forecasting已占据 | RGNB coefficient generation不是首次 |
| [Implicit Forecaster](https://papers.nips.cc/paper_files/paper/2025/hash/0e82ef0c89df6a6eff8734ea7e27c42f-Abstract-Conference.html) | 预测frequency/amplitude/phase形成future waves | implicit wave decoder已占据 | global wave generation不能作为claim |

[Decision] generic `atom descriptor + shared branch/trunk/cross-attention`没有独立novelty。若PLGO继续，贡献边界
必须来自**RGNB support algebra + atomwise separability + empirically identified geometry constraint**的组合，
而不是generator primitive。

## 3. Internal Evidence Audit

### 3.1 Evidence supporting continued diagnosis

- [Strong Evidence] D6在disjoint validation window确认local b144与global DCT存在short/long crossing；
- [Strong Evidence] D3/D4确认basis main effect与contiguous locality，但exact balanced midpoint不特异；
- [Fact] Step 5证明RGNB可stable co-synthesize global DCT root与interval-local details。

这些证据支持研究output basis/support geometry，不自动支持history retrieval或descriptor generator。

### 3.2 Historical evidence that blocks direct implementation

1. `B11-BCF`已经把learned basis descriptors用于continuous coefficient field。相对A6仅`-0.1019%`，而
   `no_basis`几乎完全复制该收益；B11相对no-basis只有`-0.0012%`、`2/12` wins，constant-slot整体更好。
   failure attribution为`capacity_or_head_effect_suspected`。
2. `B14-FURD`的model-independent retrieval-demand diagnostic只有`1/6` settings通过，跨dataset gate为
   `0/3` datasets；对“不同future units需要不同history patch retrieval”形成strong negative evidence。
3. D2 formal5已否定exact depth grouping；不得把atom depth改名为新scale experts。

[Decision] Step 6禁止atom-to-memory retrieval/cross-attention。一个合法的successor只能读取与A6一致的shared
flattened memory/latent，并在primary temporal operator中检验RGNB geometry；否则是在没有新problem evidence的
情况下复活B14。

## 4. Narrowed Tensor Contract

唯一尚可诊断的narrow form记为`PLGO-PAF`。令

- A6 history memory $M\in\mathbb R^{B\times C\times P\times D}$；
- flattened shared history $h=\operatorname{vec}(M)\in\mathbb R^R$；
- shared branch latent $z=Ah+a\in\mathbb R^K$，$K=256$；
- RGNB synthesis $Q_T=[q_1,\ldots,q_T]\in\mathbb R^{T\times T}$；
- fixed atom descriptor $d_j\in\mathbb R^8$，包含global/detail type、support endpoints、support length、
  depth与within-node canonical order，不包含$H$；
- shared trunk $\psi_j=T_\theta(d_j)\in\mathbb R^K$。

每个coefficient独立生成：

$$
\alpha_j=\langle\psi_j,z\rangle.
$$

对requested prefix $H$，只对active set
$\mathcal A_H=\{j:\operatorname{supp}(q_j)\cap[0,H)\ne\varnothing\}$计算：

$$
\widehat y_H=Q_{[0,H),\mathcal A_H}\alpha_{\mathcal A_H}+b_{[0,H)}.
$$

禁止：

- atom-to-atom attention或对active atoms做softmax normalization；
- atom-specific patch retrieval；
- requested $H$、benchmark horizon ID或target-set embedding；
- A6 output + local residual；
- free atom table与geometry trunk并存后声称geometry mechanism。

## 5. Projectivity And Function-Class Audit

`scripts/check_stage_c_plgo_step6_design.py`对$T=16,96,720,721$执行33个prefix cases：

| Check | Result |
| --- | --- |
| active-only coefficient vs full coefficient subset | max `5.684e-14` |
| active-only synthesis vs full-output prefix | max `7.105e-14` |
| simultaneous atom ordering permutation | max `4.547e-13` |
| tolerance | `1e-10` |

[Fact] 因$\alpha_j=F_\theta(h,d_j)$不读取其他atoms，active subset不会改变任何保留coefficient；这比“最后
crop full output”更强，但只是一条architecture invariant，不是独立novelty。

把全部trunk rows堆叠为$\Psi_\theta(D)\in\mathbb R^{T\times K}$，则effective temporal operator为

$$
B_\theta=Q_T\Psi_\theta(D),\qquad
\widehat y=B_\theta(Ah+a)+b.
$$

因此linear PAF的output rank不超过256，不是full affine。若$\Psi$改为free table，则它与A6/M0完全等价；
若$\Psi$由descriptor trunk生成，则不再exact包含所有A6 temporal tables。

## 6. Parameter And Attribution Boundary

两种非调参、只用于theory audit的width：

- compact width=`256`，保持trunk feature width与A6 rank一致；
- near-A6-budget width=`694`，由A6 temporal table parameter budget代数求得，不按dataset选择。

| Profile group | Compact/A6 params | Near-budget/A6 params |
| --- | ---: | ---: |
| Weather / ETTm1 / ETTh2 ($R=768$) | `0.6957` | `0.9996` |
| ETTh1 ($R=1536$) | `0.7991` | `0.9997` |
| ETTm2 ($R=3072$) | `0.8804` | `0.9998` |

[Risk] compact PAF可能重复PMFO的capacity restriction；near-budget PAF又有足够capacity记忆有限的720个
descriptors，使true geometry与random/ID descriptors难以区分。params差异不用于超参数选择，但这里必须用于
mechanism attribution。

形成Step 6 design trilemma：

1. exact包含任意A6 temporal table需要free/per-atom freedom；
2. 强制geometry sharing需要限制per-atom freedom；
3. 仅通过扩大descriptor MLP匹配参数会削弱geometry identifiability。

三者不能被一句“parameter matched”同时解决。

## 7. Candidate And Control Decisions

| Arm | Role | Decision |
| --- | --- | --- |
| `A6` | frozen carrier | baseline control |
| `PLGO-M0-FREE` | free RGNB atom table | exact morphism control；无novelty |
| `PLGO-PAF-GEO` | true RGNB descriptors | hypothesis only；D7前不得实现method |
| `PLGO-PAF-PERM` | permuted true descriptors | mandatory geometry control |
| `PLGO-PAF-RANDOM` | fixed random descriptors | mandatory descriptor-capacity control |
| `PLGO-ATOM-ATTN` | atom-specific patch retrieval | rejected；source overlap + B14 negative evidence |

## 8. Narrative Gate And Failure Attribution

[Decision] `SC1-PLGO-PAF`没有通过Step 6 narrative gate：

- `hypothesis_false`：尚未证明。D6仍支持output support-scale interaction；
- `intervention_point_wrong`：atom-to-memory retrieval被B14阻断；已从candidate删除；
- `readout_or_head_design_wrong`：B11证明late descriptor field容易被no-basis/constant controls解释；
- `optimization_or_numeric_pathology`：本轮未出现；33个algebra cases稳定；
- `capacity_control_explains`：generic/free/matched descriptor generator存在实质风险，尚未被排除。

否定边界是generic PAF作为paper method，不是RGNB数学scaffold。Step 6回滚Step 2/3，先验证一个更窄问题：

> canonical RGNB geometry是否能在相同frozen-memory、rank与optimization protocol下，比permuted/random
> descriptors更有效地参数化coefficient-row structure？

## 9. SC1-D7 Authorization

只授权`SC1-D7-RGNB-descriptor-sufficiency`，角色为`diagnostic_only`。D7通过也只能返回Step 4/6，不能直接
升为method。D7必须：

1. 使用五个frozen natural profiles与validation-only artifacts，不读test；
2. 同时比较compact与near-budget widths，防止capacity reduction或memorization单独解释；
3. true geometry必须同时优于permuted与fixed-random descriptors；
4. free-M0作为accuracy upper control，不把parameter少写成贡献；
5. 若true geometry未通过cross-dataset与matched-width gates，关闭descriptor-generator路线；
6. 不实现cross-attention、retrieval、Encoder、MoE或MIPR。

## 10. 11-Step Record

| Field | Record |
| --- | --- |
| `current_step` | Step 6 complete；rollback Step 2/3 |
| `problem` | local-prefix/global-domain output support interaction已成立，但descriptor sufficiency未成立 |
| `existence_evidence` | D6 supports RGNB scaffold；B11/B14 oppose generic descriptor/retrieval successors |
| `idea` | atomwise PAF over RGNB coefficients |
| `theory_check` | projectivity/rank pass；exact A6 containment、novelty与mechanism attribution fail/pending |
| `design` | generic PAF不冻结为method；仅冻结D7 diagnostic arms |
| `narrative_gate` | fail/not ready |
| `effectiveness_gate` | not started |
| `artifacts` | 3 CSV/JSON artifacts、source matrix、protocol、code explanation、本报告 |
| `decision` | `narrative_not_ready_rollback_step2_3_d7`；training false |
