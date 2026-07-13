# StageC Step 4-6：PMFO/PIR External Prior-Art、Theory 与 Narrative Gate

## Decision Summary

| Field | Decision |
| --- | --- |
| `search_date` | 2026-07-13 |
| `current_step` | Step 4-6 completed；下一步为 Step 7 local implementation/invariant gate |
| `SC1-PMFO` | `narrative_ready`，具体候选为 `PMFO-RCT`（Refinement-Conservative Tree） |
| `SC2-PIR` | `narrative_ready`，formal objective 收紧为 `MIPR`（Measure-Induced Projective Risk） |
| `remote_training_authorized` | `false`；先实现本地 operator invariants 与 capacity controls |
| `encoder_decision` | 首轮严格保留 A6 `memory: [B,C,P,D]`；不同时修改 Encoder |
| `basis_decision` | A6 dense learned basis降为matched control；不作为新方法初始化或理论对象 |
| `rollback` | PMFO-RCT若被dense/no-transition control解释，回滚 Step 4；MIPR若只复制raw weighting，回滚 Step 2 |

[Decision] Step 4-6 的结论不是“wavelet decoder值得一试”，而是一个更窄、更可证伪的设计：未来输出由
mixed-radix interval tree逐层细化，detail严格位于父尺度的正交补中，因此细尺度不能改写已经建立的coarse
projection；requested horizon只裁定需要展开的prefix domain，不进入learned state。训练侧不再声称PIR比
raw weighting更“measure-aligned”，而是把deployment measure诱导的quadratic metric投影到同一组
refinement blocks上，显式移除cross-scale coupling。

## 1. Literature Search Protocol And Coverage

### 1.1 Search rule

本轮按更新后的研究规则执行：Zotero只作为user-curated seed/reference，不用于证明检索完整性、时效性或
novelty。默认从外部广泛检索，优先使用arXiv/OpenReview、会议论文集、官方project page与official code。

本轮主题/queries覆盖：

- varied/arbitrary-horizon forecasting、horizon invariance、horizon reweighting；
- functional/basis/implicit decoder、hierarchical interpolation；
- multiresolution analysis、lifting、multiwavelet neural operator；
- target timestamp query、decoder-training co-design；
- temporal hierarchy/coherence与wavelet forecasting（含2025-2026工作）。

Zotero live connector在本轮不可用；下表的`Zotero seed`只依据repo中已落地的`Papers/INDEX.md`，不能解释为
当前library的完整盘点。外部全文/official code是本次判断的主要证据。

### 1.2 Key source matrix

| Work | Source / implementation checked | Discovery | Novelty pressure on StageC |
| --- | --- | --- | --- |
| ElasTST, NeurIPS 2024 | [paper](https://arxiv.org/abs/2411.01842), [official code](https://github.com/microsoft/ProbTS/tree/elastst) | Zotero seed + external verification | structured mask已占据horizon invariance；harmonic reweighting已占据raw horizon-measure weighting |
| N-HiTS, AAAI 2023 | [paper](https://arxiv.org/abs/2201.12886), [official code](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/nhits.py) | external | hierarchical interpolation、multi-rate sampling与coarse-to-fine additive forecast已被占据 |
| BasisFormer, NeurIPS 2023 | [paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e150e6d0a1e5214740c39c6e4503ba7a-Abstract-Conference.html), [official code](https://github.com/nzl5116190/Basisformer) | external | learned/interpretable future basis、coefficient alignment与smoothness不能作为独立novelty |
| Multiwavelet Operator, NeurIPS 2021 | [paper](https://arxiv.org/abs/2109.13459), [official code](https://github.com/gaurav71531/mwt-operator) | external | nested spaces、fixed analysis/reconstruction filters与resolution-independent operator已有成熟先例 |
| FlowState, ICML 2026 | [paper](https://arxiv.org/abs/2508.05287), [official code](https://github.com/ibm-granite/granite-tsfm/tree/gift-flowstate/tsfm_public/models/flowstate) | external | continuous functional basis、dynamic target length/sampling scale已被占据 |
| TimePerceiver, NeurIPS 2025 | [paper](https://arxiv.org/abs/2512.22550), [official code](https://github.com/efficient-learning-lab/TimePerceiver) | Zotero seed + external verification | target timestamp queries与decoder/training co-design已被占据 |
| Implicit Forecaster, NeurIPS 2025 | [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e82ef0c89df6a6eff8734ea7e27c42f-Abstract-Conference.html), [official code](https://github.com/rakuyorain/Implicit-Forecaster) | external | amplitude/phase/frequency wave synthesis使“global constituent waves”不再新颖 |
| Shifting Time, ICML 2025 | [OpenReview](https://openreview.net/forum?id=emkdmORaj4) | external | continuous time-shift neural operator、super-resolution与irregular sampling压缩generic operator claim |
| MoHETS, 2026 under review | [paper](https://arxiv.org/abs/2601.21866) | Zotero seed + external verification | convolutional patch decoder也已声称single-model arbitrary-horizon generalization |
| AdaWaveNet, 2026 | [article](https://doi.org/10.1007/s44443-026-00537-5) | external | learnable lifting用于input-side adaptive decomposition，禁止把“learnable lifting”本身当贡献 |
| TimesWave, 2026 preprint | [paper page](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6992565) | external | adaptive lifting + inverse reconstruction进一步增加wavelet命名压力；venue/evidence confidence低于正式会议论文 |
| Temporal hierarchy forecasting, AISTATS 2023 | [paper](https://proceedings.mlr.press/v206/rangapuram23a.html) | external | aggregation coherence与本项目prefix projectivity不同，论文中必须主动区分 |

### 1.3 Source-derived implementation facts

- ElasTST official code的`get_weights('random')`直接构造随future position衰减的近似log/harmonic权重；其
  training config固定`train_pred_len_list=720`。因此SC2的raw step weighting必须是control，不是method。
- FlowState official FBD先把encoder state线性映射为basis coefficients，再按`prediction_length`与
  `scale_factor`采样Legendre/Fourier kernel并做matrix multiplication。仅把A6 basis换成continuous basis不能
  形成novelty。
- TimePerceiver official decoder由future positional query对history representation做cross-attention；本项目
  不采用target timestamp query，避免重新打开explicit target-query路线。
- Implicit Forecaster official code用history spectrum与encoder state预测amplitude/phase，再用`irfft`生成
  forecast。SC1不能被简化成DCT/Fourier coefficient prediction。
- Multiwavelet operator official code使用fixed filter buffers执行多层decompose/reconstruct；这支持我们的
  algebra feasibility，但不支持“首次使用nested operator”的claim。

## 2. Novelty Triage

### 2.1 Rejected paper-core formulations

以下表述在本轮narrative gate中明确拒绝：

1. “首个支持arbitrary horizon的basis decoder”；FlowState、ElasTST、MoHETS及foundation models已形成
   直接压力。
2. “用coarse-to-fine interpolation改善long horizon”；N-HiTS已直接覆盖。
3. “用learned wavelet/lifting捕获多尺度”；multiwavelet operator、AdaWaveNet及大量2025-2026 wavelet
   forecasting已覆盖。
4. “用future coordinate/query得到不同长度输出”；TimePerceiver/FlowState已覆盖，且违背当前H不进入
   learned semantic path的约束。
5. “按horizon distribution给step loss加权”；ElasTST已给出理论与官方实现。

### 2.2 Defensible remaining gap

[Strong Evidence] 上述工作没有在已核对的论文与官方实现中同时给出以下四点：

- history-to-future的primary decoder采用**refinement-conservative** tree synthesis；
- 每层detail被结构性限制在父尺度正交补中，fine refinement不能改变coarse projection；
- requested horizon只触发support-intersection pruning，不成为query、embedding或router feature；
- deployment measure在相同projectors上诱导block metric，并把raw risk与cross-scale-decoupled risk严格区分。

[Uncertainty] “没有同时出现”是截至2026-07-13本轮外部检索的结果，不是数学意义上的absence proof。尤其
2026 wavelet/lifting论文增长很快，投稿前必须重复freshness search，并对PMFO-RCT的四项组合做citation
chaining。

## 3. SC1 Design: PMFO-RCT

### 3.1 Horizon and resolution are different variables

- $H$：requested output domain $\{1,\ldots,H\}$；
- $\ell$：同一future function的refinement level。

[Decision] 所有$H$都使用到unit-resolution的同一tree；禁止令短H选择fine branch、长H选择coarse branch。
multi-horizon统一性来自同一projective family，不来自horizon-specific scale preference。

### 3.2 Frozen input contract

A6 Encoder保持不变：

$$
M\in\mathbb{R}^{B\times C\times P\times D},\qquad
z=\operatorname{flatten}_{P,D}(M)\in\mathbb{R}^{B\times C\times 768}.
$$

不同dataset仍使用已冻结的natural profile；`P*D=768`不变。首轮不引入multi-patch Encoder、future
placeholder、MoE、router或auxiliary reconstruction。

### 3.3 Mixed-radix future tree

canonical domain长度$T=720$，采用D1已审计的nested partitions：

$$
90\rightarrow30\rightarrow10\rightarrow5\rightarrow1,
\qquad r=(3,3,2,5).
$$

coarse scaling coefficients与四层detail group的维度为：

$$
(8,16,48,72,576),\qquad 8+16+48+72+576=720.
$$

对radix $r$，令$u_r=\mathbf 1_r/\sqrt r$，$Q_r\in\mathbb R^{r\times(r-1)}$满足
$Q_r^TQ_r=I$且$Q_r^Tu_r=0$。每个parent的children scaling coefficients为：

$$
a_{\ell,m}^{child}=a_{\ell-1,m}u_{r_\ell}+Q_{r_\ell}d_{\ell,m}.
$$

因此：

$$
u_{r_\ell}^Ta_{\ell,m}^{child}=a_{\ell-1,m},
\qquad
Q_{r_\ell}^Ta_{\ell,m}^{child}=d_{\ell,m}.
$$

这是`refinement-conservative`的核心：detail无论如何学习，都不能改变父尺度projection。

### 3.4 Learned state path

最小候选采用future tree state，而不是dense coefficient rotation：

1. `seed(z) -> u0: [B,C,8,d_s]`，并由`head0(u0) -> a0: [B,C,8]`；
2. level $\ell$对每个active parent独立应用shared `split_l`，得到
   `u_l: [B,C,N_l,r_l,d_s]`；
3. shared `detail_l`从parent/child state产生
   `d_l: [B,C,N_l,r_l-1]`；
4. fixed $u_r,Q_r$完成conservative synthesis；child states reshape后进入下一层；
5. 最后一层scaling coefficients即normalized forecast `[B,C,H]`，再走A6 RevIN denormalization。

所有normalization必须只沿feature dimension执行；禁止跨active node count的attention、BatchNorm或
sequence normalization，否则requested H会改变prefix内state，破坏exact projectivity。

### 3.5 Domain-only execution and proofs

H只执行两项deterministic操作：

- 每层取与$[1,H]$相交的前$\lceil H/b_\ell\rceil$个parents；
- 最后返回前H个leaves。

对boundary parent需要完整生成其$r_\ell-1$个detail，产生至多常数radix overhead。理论审计得到：

- orthogonality max error `1.33e-15`；
- refinement recovery max error `8.88e-16`；
- pruned-prefix与full-tree crop max error `4.44e-16`；
- basis-defined projectors与block projectors max error `2.22e-16`。

active coefficient count在H=`1/48/96/192/336/720`时为
`10/52/104/199/342/720`。该统计只证明out-of-prefix atoms可不求值，不等价于完整FLOPs。
任何显式输出H个值的方法都有$\Omega(HC)$写出下界，论文不得宣称sublinear total generation。

### 3.6 Why this is not just a rotated dense head

若直接预测全部720个orthogonal coefficients再inverse transform，在线性条件下只是dense output head的坐标
旋转，不能成为PMFO。paper-core候选必须同时包含：

- parent-to-child shared state transition；
- fixed conservative synthesis；
- support-local H-pruning；
- no-transition与dense matched controls。

若`no-transition`或`dense-MLP`在matched budget下解释收益，则`capacity_control_explains`，PMFO-RCT不能
作为paper core。

## 4. SC2 Design: MIPR

### 4.1 Exact deployment risk

令$e_t=\hat y_t-y_t$，deployment horizon distribution为$\mu(H)$：

$$
\mathcal R_{raw,\mu}(e)
=\mathbb E_{H\sim\mu}\left[\frac1H\sum_{t=1}^{H}e_t^2\right]
=e^TW_\mu e,
$$

其中：

$$
W_\mu=\operatorname{diag}(w_1,\ldots,w_T),\qquad
w_t=\sum_{H=t}^{T}\frac{\mu(H)}{H}.
$$

[Fact] 这就是random horizon sampling的期望，也是ElasTST-style reweighting的对象。它是exact
measure-aligned risk，不能被包装为我们的创新。

### 4.2 Measure-induced block metric

令$Q_0,\ldots,Q_L$为PMFO nested spaces的orthogonal increment projectors：

$$
Q_\ell^2=Q_\ell,\quad Q_\ell Q_k=0\ (\ell\ne k),\quad \sum_\ell Q_\ell=I.
$$

定义：

$$
\widetilde W_\mu=\sum_{\ell=0}^{L}Q_\ell W_\mu Q_\ell,
$$

$$
\mathcal R_{MIPR,\mu}(e)
=e^T\widetilde W_\mu e
=\sum_{\ell=0}^{L}(Q_\ell e)^TW_\mu(Q_\ell e).
$$

$\widetilde W_\mu$是$W_\mu$在由$\{Q_\ell\}$定义的block-diagonal operator subspace上的Frobenius
orthogonal projection。它保留within-scale measure weighting，删除
$Q_\ell W_\mu Q_k,\ell\ne k$的cross-scale coupling。

### 4.3 Exact boundary

$$
\mathcal R_{raw,\mu}-\mathcal R_{MIPR,\mu}
=\sum_{\ell\ne k}e^TQ_\ell W_\mu Q_ke.
$$

因此：

- 一般情况下两者不相等，且MIPR既不是upper bound也不是lower bound；
- 当$W_\mu\propto I$时cross blocks为零；本项目的`delta_720`正是该情况；
- MIPR是decoder-aligned structured surrogate，不得称为“比raw risk更measure-aligned”；
- L2下有exact quadratic/projector algebra；Huber/L1没有对应的Parseval/block-metric等价，首轮不实现
  `Huber-PIR`或`L1-PIR`。

### 4.4 Theory audit result

| Measure | Off-block energy fraction | Mean random-error risk gap | Interpretation |
| --- | ---: | ---: | --- |
| `delta_720` | `0.000000` | `0.000000` | exact equality control |
| `uniform_h` | `0.003456` | `0.003091` | non-zero but weak |
| `log_uniform_h` | `0.205154` | `0.107832` | strong scale coupling; primary mechanism setting |
| `benchmark_h` | `0.002480` | `0.002686` | weak，吻合D1 benchmark projected excess `0/3` |

[Decision] SC2的研究价值是**measure-induced scale decoupling**，不是horizon weighting。它对
log-uniform continuous deployment最有mechanistic headroom，对四个benchmark horizons的预期主效应很弱。
因此论文必须包含dense-horizon risk curves，不能只在`96/192/336/720`上宣称training贡献。

## 5. Narrative Gates

### 5.1 SC1-PMFO

| Criterion | Decision |
| --- | --- |
| clear problem | pass：D1显示residual nested structure且current basis无refinement/local support |
| novelty boundary | pass with medium confidence：不claim arbitrary horizon/wavelet/basis；claim future-side conservative refinement + domain pruning |
| tensor path | pass：`memory -> tree states -> details -> fixed synthesis -> prefix` |
| theory | pass：orthogonality/refinement/restriction全部numeric invariant通过 |
| controls | pass：dense/no-transition/no-conservation/parameter-FLOP controls已预注册 |
| status | `narrative_ready`；只授权Step 7 local implementation |

### 5.2 SC2-PIR/MIPR

| Criterion | Decision |
| --- | --- |
| clear problem | pass conditional：continuous deployment measure会诱导非零cross-scale coupling |
| novelty boundary | pass：区别于raw harmonic weighting与generic task balancing |
| gradient path | pass：$e\to Q_\ell e\to W_\mu$，projector与decoder共用 |
| theory | pass for L2 only；Huber/L1 rejected from first implementation |
| measure relevance | log-uniform strong；uniform weak；benchmark weak |
| status | `narrative_ready`，但implementation serialized after PMFO-RCT operator contract |

## 6. Mandatory Controls And Falsification

### 6.1 Architecture controls

1. frozen `A6-LBF-natural-baseline`；
2. `dense-MLP-matched`：相同nonlinearity与近似params，但直接预测future rows；
3. `tree-no-conservation`：相同tree states，children unconstrained，检验orthogonal complement identity；
4. `tree-no-transition`：各level直接从history state产生，检验recursive refinement是否被capacity解释；
5. fixed DCT/Haar coordinate control；
6. params与measured forward FLOPs均报告，差异不作为profile选择因素，但必须用于mechanism attribution。

PMFO effectiveness gate：至少ETTm1与ETTh2 seed2021上，PMFO相对A6与dense/no-transition controls具有稳定
dense-horizon MSE改善；若收益被任一matched control解释，则状态降为`failed_as_core_candidate`并回滚Step 4，
不能通过加入Encoder/MoE修补。

### 6.2 Objective controls

1. full H720 uniform-step MSE；
2. exact raw `uniform_h/log_uniform_h/benchmark_h` weighting；
3. stochastic horizon sampling与其expected raw weights的一致性control；
4. MIPR on aligned projectors；
5. matched-rank random/permuted projector partition，检验“decoder alignment”是否真实。

MIPR effectiveness gate：相对same-$\mu$ raw weighting，在log-uniform dense-horizon expected risk上形成跨dataset
主效应，且benchmark measure无超过`+0.5%`的稳定退化；若random projector同样有效，判定
`capacity_or_regularization_control_explains`。

## 7. Step 7-10 Experiment Sequence

1. **Step 7A local invariants**：实现PMFO-RCT module、shape tests、full-vs-prefix、refinement recovery、local
   perturbation support与parameter count；不训练。
2. **Step 7B architecture gate**：ETTm1+ETTh2、seed2021，统一natural profiles，比较A6、dense matched、
   no-transition、no-conservation、PMFO-RCT；先用full MSE，不加入MIPR。
3. **Step 9-10 SC1 decision**：若SC1通过，再冻结operator contract；若失败，按intervention/readout/
   optimization/capacity归因回滚。
4. **SC2 isolated gate**：在通过的PMFO上比较same-measure raw versus MIPR，并加入random-projector control。
5. **2x2 factorial**：A6/PMFO × raw/MIPR，确认decoder与objective独立主效应及interaction。
6. **full matrix**：3 datasets × 3 seeds × dense horizons；最后才加入第二backbone generality与official
   native baselines。

## 8. Self-Critique And Remaining Risks

- PMFO-RCT的function class仍可覆盖普通720维输出；贡献主要来自parameterization、conservation与domain
  execution，而非更大的expressivity。dense/no-transition controls必须严格。
- mixed-radix partition来自T=720与D1 block diagnostic，存在benchmark-length依赖。若以后claim arbitrary
  $T$，必须证明自动构树或boundary handling，而不是只支持720。
- tree transition可能重现B13中“transition不如no-transition”的历史风险；本次已把no-transition设为
  blocking control，而不是事后ablation。
- MIPR主动删除cross-scale terms，因此优化目标有bias；即使MSE改善，也不能把它写成exact deployment-risk
  minimization。
- 本轮Zotero live metadata/fulltext不可用，但按新规则不阻断外部调研；投稿前仍需把最终引用回填到Zotero并
  做一次freshness/citation-chain复查。

## 9. 11-Step Record

| Field | Record |
| --- | --- |
| `current_step` | Step 4-6 complete；Step 7 local implementation next |
| `problem` | dense global basis无conservative refinement/local execution；raw measure risk包含cross-scale coupling |
| `existence_evidence` | D1-v2 3 datasets × 3 seeds；本轮external primary-source audit与numeric algebra gate |
| `idea` | PMFO-RCT + measure-induced projective risk |
| `theory_check` | mixed-radix invariants pass；MIPR exact for L2 block metric、not exact raw risk |
| `design` | frozen A6 Encoder；future tree primary path；serialized SC1 then SC2 |
| `narrative_gate` | SC1 pass；SC2 pass with measure/L2 boundary |
| `effectiveness_gate` | pending；无method performance evidence |
| `artifacts` | `theory_gate_report.md/json`、tree/measure CSV、本文 |
| `decision` | 只授权PMFO-RCT本地实现与controls；remote training仍false |
