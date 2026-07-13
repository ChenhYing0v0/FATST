# StageC Step 4 Source-Informed Redesign Audit

## Decision Summary

| Field | Decision |
| --- | --- |
| `candidate_id` | `SC1-FPMO`（provisional）：Function-Preserving Multiresolution Operator Morphism |
| `current_step` | Step 4 completed；仅授权进入Step 5 theory feasibility |
| `PMFO-RCT-v1` | 保持关闭；不做tuning复活 |
| `source_decision` | generic tree、learned scale scheduling、nested basis、lifting、network morphism均不能单独构成novelty |
| `diagnostic_decision` | v1 function class不包含A6；fixed partition无跨dataset operator-boundary证据；root history profiles近乎同构 |
| `retained` | domain-only $H$、exact restriction、conservation/perfect reconstruction、ordered memory、A6 carrier |
| `removed` | fixed `90/30/10/5/1` factorization、shared recursive state、learned horizon/scale router、A6整体替换 |
| `method_implementation` | `false`；Step 5 theorem与Step 6 preregistration通过前不得编码或训练 |
| `rollback` | Step 5无法构造exact A6 embedding或native restriction -> Step 2 problem reformulation |

## 1. What We Planned To Test

Step 7B只否定PMFO-RCT v1的具体head，不能直接否定projectivity。按failure attribution，Step 4必须分别回答：

1. v1是否真正保留了A6的function class，而不只是保留相同的256维瓶颈？
2. `90/30/10/5/1`是否得到数据/已训练operator支持，而不是720的方便因子分解？
3. `history memory -> 8 root states -> shared transition`是否形成了node-specific interface？
4. 在2025-2026 prior art下，什么边界仍足以构成SCI-level decoder contribution？

本轮不训练forecast model，不读取test prediction，不调整dataset profile。

## 2. Source-Informed Audit

### 2.1 Fresh external evidence

| Source | Verified mechanism | Pressure on StageC |
| --- | --- | --- |
| [PRISM](https://arxiv.org/abs/2512.24898) + [official code](https://github.com/nerdslab/PRISM) | paper以learnable tree-based partition表述multiscale forecasting；commit `a6342da`代码实际执行fixed overlapping binary history split、band routing，并为各level/band配置dense `pred_len` head后求和 | “learnable multiresolution tree forecast”已经高度拥挤；但它是history-side tree和fixed-H dense heads，不提供future-prefix restriction或A6 function preservation |
| [LeapTS](https://arxiv.org/abs/2605.10292) | hierarchical controller动态选择prediction scale与advancement length，并用neural CDE演化state | learned horizon/scale scheduling已被直接占据，也与本项目“$H$只定义domain”的连续性原则冲突 |
| [Hierarchical nested-basis network](https://arxiv.org/abs/1808.02376) | 用$\mathcal H^2$/FMM nested bases近似nonlinear operator，参数随离散维度线性增长 | nested transfer/basis是可行性工具，不是novelty本身 |
| [Lifting scheme](https://doi.org/10.1137/S0036141095289051) | second-generation wavelets可适配interval/domain并支持fast in-place transform | perfect reconstruction与interval-adaptive wavelet已有经典基础；不能声称首次learnable lifting |
| [Net2Net](https://research.google/pubs/net2net-accelerating-learning-via-knowledge-transfer/) / [Network Morphism](https://proceedings.mlr.press/v48/wei16.html) | architecture变化时保持parent network function | function-preserving transformation是已有一般原则；本项目只能把它收紧到future operator restriction contract |
| [Asymmetric MMF](https://arxiv.org/abs/1910.05132) | global low-rank component与hierarchical residual结合优于单独factorization | “A6 global output + multiresolution residual patch”既叙事弱，也有直接matrix-factorization prior-art压力 |

[Fact] PRISM official code完整可读；LeapTS截至2026-07-13未找到author-linked official implementation，故只使用论文机制，不据此评价实现细节。

[Decision] 新候选的novelty不能是tree、wavelet、nested basis、adaptive scale或function preservation中的任何单项，而必须是它们在**future output domain**上的特定组合：

> 将已验证的dense forecast operator精确morph为可restriction、可perfect-reconstruction的multiresolution operator；requested $H$只裁剪active supports，并且morph后的function class显式包含A6。

### 2.2 Routes excluded before design

- `learned scale/horizon controller`：与LeapTS重叠且重新引入horizon-specific shortcut；拒绝。
- `PRISM-style history tree + dense future heads`：不解决future restriction；只可作external baseline。
- `A6 output + hierarchical residual`：与Asymmetric MMF的global+residual结构相近，也违反当前非residual paper-core边界；拒绝。
- `fixed DCT/Haar decoder`：可作exact-transform control，不是method。
- `PMFO v1 + hyperparameter sweep`：Step 7B无numeric pathology，且本轮发现function-class obstruction；拒绝。

## 3. Artifact Construction And Statistics

输入为Step 7B `best_val_h720_mse` checkpoints，remote路径与SHA256见
`checkpoint_manifest.csv`。临时同步目录不作为artifact；可按manifest重新获取。分析脚本为
`scripts/analyze_stage_c_step4_operator_geometry.py`。

### 3.1 A6 effective operator

A6 normalized readout为

$$
y = B(Ah+a)+b,\qquad B\in\mathbb R^{720\times256},\quad A\in\mathbb R^{256\times768}.
$$

定义effective operator $W=BA\in\mathbb R^{720\times768}$。`operator_effective_rank`是$W$奇异值平方归一化后的entropy rank；`energy_rank_95/99`是累计Frobenius energy达到95%/99%的最小rank；`stable_rank=\|W\|_F^2/\|W\|_2^2`。

### 3.2 Function-class dimension obstruction

rank-$r$、$m\times n$矩阵流形维度为$r(m+n-r)$；加$m$维output bias后，A6 rank-256 affine operator family的维度为

$$
256(720+768-256)+720=316{,}112.
$$

`pmfo_readout_parameters`统计checkpoint中全部`pmfo_readout.*` trainable tensors。若它小于上述维度，analytic PMFO parameterization不可能覆盖全部A6 affine operator family。这个gate讨论function-family containment，不把parameter count当作模型优劣指标。

### 3.3 Partition-boundary statistic

对$t=1,\dots,719$定义operator jump

$$
j_t=\|W_{t+1,:}-W_{t,:}\|_2.
$$

对block size $s$，`boundary_to_all_jump_ratio`为$t\in\{s,2s,\dots\}$上的平均$j_t$除以全部位置平均。ratio显著大于1才支持该periodic boundary对应operator regime change；约等于1表示边界只是任意切点。

### 3.4 Local-rank capture

把$W$按block size切成连续矩阵，`mean_energy_capture`为每个block前$r$个奇异值平方占比的平均。它判断local compression headroom，不判断固定边界正确，也不自动证明tree transition有效。

### 3.5 History-interface statistic

A6 coefficient weight reshape为`[256,P,D]`；PMFO seed weight reshape为`[8,32,P,D]`。对每个latent/node，将weight平方沿feature维求和得到history-patch energy probability。

- `patch_entropy`：patch probability entropy除以$\log P$；接近1表示global/distributed use；
- `node_patch_profile_cosine`：8个root nodes的patch-energy profiles两两cosine均值；
- `seed_node_weight_cosine`：完整signed seed weights两两cosine均值。

profile相似但signed weight不同，表示nodes使用相似的history位置范围、但学习了不同投影方向；不能误写成8个nodes完全相同。

## 4. Results

### 4.1 Function class

| Dataset | A6 params | PMFO params | PMFO/A6 | Required affine dim. | Gap | A6 eff. rank | A6 rank@99% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 381,904 | 212,010 | 0.5551 | 316,112 | -104,102 | 144.25 | 234 |
| ETTm1 | 381,904 | 212,010 | 0.5551 | 316,112 | -104,102 | 115.88 | 231 |
| Weather | 381,904 | 212,010 | 0.5551 | 316,112 | -104,102 | 30.48 | 213 |

[Strong Evidence] “A6与PMFO都有256维state，所以capacity preserved”是错误命题。PMFO v1的整个readout parameter family比覆盖rank-256 affine operators所需维度少104,102；A6实际operator也不是低到可忽略的rank，99% energy需要213-234维。

[Boundary] dimension obstruction证明“不包含全部A6 family”，不证明Step 7B三个已训练A6函数中的每一个都绝对不可被某组PMFO参数拟合。它足以否定将v1称为capacity-preserving replacement。

### 4.2 Fixed partition

| Dataset | block 90 ratio | block 30 ratio | block 10 ratio | block 5 ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 0.9997 | 1.0093 | 1.0099 | 1.0072 |
| ETTm1 | 0.9889 | 0.9905 | 1.0054 | 0.9989 |
| Weather | 1.0011 | 1.0053 | 1.0128 | 1.0087 |

[Strong Evidence] coarse `90/30` boundaries没有一致operator discontinuity：90仅1/3 datasets大于1，30虽2/3略大于1但最大仅1.0093。现有证据支持“future operator有local/multiscale structure”，不支持“这些factorization boundaries就是自然structure”。

local rank进一步说明不能给所有dataset统一一个激进小rank：block90 rank16 capture为ETTh2 `0.4595`、ETTm1 `0.5312`、Weather `0.8025`；block30 rank16为`0.7312/0.7880/0.9061`。Weather更local-low-rank，ETT保留更高维结构。

### 4.3 History-to-node interface

| Dataset | A6 patch entropy | PMFO node entropy | PMFO max patch share | Node profile cosine | Signed weight cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 0.9969 | 0.9952 | 0.1225 | 0.9936 | 0.2005 |
| ETTm1 | 0.9896 | 0.9864 | 0.0939 | 0.9662 | 0.1549 |
| Weather | 0.9847 | 0.9762 | 0.1662 | 0.9360 | 0.1947 |

[Strong Evidence] 8个root nodes都从几乎全局的history patches取能量，位置profile高度相似；它们主要靠不同signed projections区分，而不是history-region specialization。再结合Step 7B transition相对no-transition仅macro `+0.0486%`，当前`root state -> shared recursive transition`没有形成受支持的scale-native interface。

[Uncertainty] global history use本身不一定错误，尤其周期预测可能需要全局context。本统计否定的是“v1已经学到清晰node-local routing”这一解释，不否定所有global-to-local decoder。

## 5. Candidate Redesign: SC1-FPMO

### 5.1 Core idea

暂定名称`Function-Preserving Multiresolution Operator Morphism (FPMO)`。它不是“A6 prediction + residual correction”，而是把整个A6 future operator改写到一组perfect-reconstruction multiresolution coordinates中：

1. 从A6的$W,b$构造可逆analysis/synthesis pair $(\mathcal A,\mathcal S)$；
2. node coefficients直接由ordered memory与support geometry生成，不经过shared parent-to-child state transition；
3. 参数空间存在显式submanifold，使$\mathcal S\mathcal A(W h+b)=W h+b$，因此A6是新operator的exact member；
4. $H$只选择与$[1,H]$相交的supports并执行native partial synthesis，不进入coefficient network；
5. conservation升级为perfect reconstruction / coarse-projection preservation，不绑定720的特定因子分解。

`support geometry`只描述每个future atom的domain interval/scale；它不是requested horizon embedding。Step 5必须验证这种描述不会退化为position-specific lookup。

### 5.2 Why this is not yet narrative-ready

[Hypothesis] FPMO可能同时解决v1三项缺陷，但目前只有design boundary，没有完整方法。以下任一项失败都应停止：

- exact A6 embedding只能通过保留一条dense A6 branch实现；
- native prefix execution仍需先生成全部720 outputs/coefficients；
- direct node heads的参数/FLOPs只能靠过度压缩而重现v1 capacity loss；
- 所谓support geometry实质成为learned horizon/position ID；
- 从scratch训练失败，只有morphing trained A6 checkpoint才有效。

最后一点尤其重要：trained-checkpoint morph可作为capacity-preservation diagnostic和初始化control，但不能自动成为paper贡献或主训练协议。

## 6. Narrative Gate And Next Decision

### Step 4 narrative audit

| Criterion | Result |
| --- | --- |
| clear problem motivation | pass：v1有dimension、partition、interface三项直接证据 |
| novelty after latest sources | conditional pass：只有future-domain operator morphism组合可继续 |
| explainable tensor path | provisional：`memory -> multiresolution coefficients -> partial synthesis` |
| contribution boundary | pass：明确排除tree/wavelet/router/morphism单项claim |
| theory feasibility | pending Step 5：exact embedding与restriction theorem尚未完成 |

[Decision] Step 4完成，`SC1-FPMO`状态为`source_informed_candidate / theory_pending`；只进入Step 5，不进入Step 6 implementation design，更不授权Step 7。

### Step 5 required proofs and kill gates

1. **Embedding theorem**：给出$\theta_{A6}\mapsto\theta_{FPMO}$，对任意$h$满足$F_{FPMO}(h;720)=F_{A6}(h)$；不得依赖额外dense branch。
2. **Restriction theorem**：native执行满足$F(h;H)=R_HF(h;720)$，且不计算prefix外atoms。
3. **Function-space budget**：证明新增parameterization至少包含A6 family；parameter/FLOP control只用于公平比较，不参与是否保留capacity的逻辑。
4. **Interface contract**：`memory [B,C,P,D]`直接进入各scale coefficients；禁止shared recursive state成为唯一history path。
5. **Irregular-length contract**：tree由通用interval rule构造，对任意$T,H\le T$定义；不得再从720质因数选择radices。
6. **Scratch boundary**：morphed-checkpoint exact equality是diagnostic；paper method必须另有from-scratch effectiveness path。

Step 5若通过，Step 6才设计最小controls；若embedding或restriction失败，rollback Step 2，不以更多head、Encoder或MIPR补救。

## 7. Failure Attribution

本轮进一步确认PMFO-RCT v1的主因是`readout_or_head_design_wrong`，具体包括function-family restriction、unsupported fixed partition和weakly supported recursive interface。未出现`optimization_or_numeric_pathology`；decoder-specific hyperparameter仍是次要不确定性，但已不足以优先于结构重设计。

本轮没有证明projectivity hypothesis false。conservation在Step 7B有跨dataset正向证据，perfect-reconstruction/lifting与operator morphism也提供理论可行性；因此方向级状态是`theory_plausible_but_new_design_unproven`。

## 8. 11-Step Record

| Field | Record |
| --- | --- |
| `current_step` | Step 4 complete；Step 5 next |
| `problem` | v1不包含A6 family；fixed factorization和recursive interface缺证据 |
| `existence_evidence` | Step7B controls + 3-dataset checkpoint operator geometry |
| `idea` | future-domain function-preserving multiresolution operator morphism |
| `theory_check` | prior-art boundary完成；embedding/restriction proof pending |
| `design` | principles only；no module/tensor implementation frozen |
| `narrative_gate` | source-level conditional pass；full gate pending Step 5-6 |
| `effectiveness_gate` | not started |
| `artifacts` | source matrix、checkpoint manifest、4 diagnostic CSVs、summary JSON、本报告 |
| `decision` | `SC1-FPMO theory_pending`；new training unauthorized；rollback Step 2 if proof fails |
