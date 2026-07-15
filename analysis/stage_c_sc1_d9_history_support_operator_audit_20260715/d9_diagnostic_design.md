# SC1-D9 History-Support Operator Evidence Audit

## 1. What We Plan To Test

| Field | Pre-Registered Decision |
| --- | --- |
| `current_step` | Contribution 1 Step 4 existence diagnostic |
| `role` | `diagnostic_only`；不是method candidate或effectiveness gate |
| `problem` | A6 history memory是否包含与future local/global support尺度可识别耦合的learned operator structure？ |
| `existence_evidence` | D6 support×horizon crossing；D8 canonical RGNB geometry effect；尚无history-scale coupling evidence |
| `idea` | 先精确分解A6 learned operator；若通过，再做sample-dependent input-Jacobian确认 |
| `theory_check` | A6 decoder对`memory`严格线性，允许无近似地恢复`future time × history patch × feature`算子 |
| `design` | D9-A exact operator decomposition；D9-B仅在A通过后授权 |
| `narrative_gate` | not required；diagnostic已在launch前声明 |
| `effectiveness_gate` | not applicable；不训练、不用test、不比较forecast MSE/MAE |
| `decision` | A失败则回Step2/3；A通过只授权D9-B，不授权新decoder |

核心假设是：越local、越细的future detail subspace，会相对更多地读取history patch axis中的高频/细尺度
coordinates；global future subspace会相对更多地读取history低频/全局coordinates。这里不主张某个future atom
应检索某个history patch，也不使用future位置与history位置的一一对应。

## 2. Why The Exact Operator Is Available

A6 forward path为

$$
M\in\mathbb R^{B\times C\times P\times D}
\rightarrow h=\operatorname{vec}(M)\in\mathbb R^{B\times C\times PD}
\rightarrow z=hC^T+b_c
\rightarrow \widehat y=zB^T+b_t.
$$

其中`C = learned_basis_coeff.weight [256,PD]`，
`B = learned_temporal_basis [720,256]`。去掉不影响sensitivity的bias后，精确memory-to-future operator为

$$
W=BC\in\mathbb R^{720\times PD}.
$$

reshape得到$W[t,p,d]$。该乘积消除了A6 learned basis内部任意可逆换基造成的identifiability问题；分析的是
实际end-to-end decoder map，不是单独观察`B`或`C`。

## 3. Coordinate Construction

### 3.1 Future side

使用已通过algebra gate的`RGNB(T=720, global_rank=16)`正交synthesis matrix $S$。其16个global root atoms
与depth 0-5 local detail atoms形成七个support-scale groups。future coefficient sensitivity为

$$
G[a,p,d]=\sum_t S[t,a]W[t,p,d].
$$

这一步只是固定正交坐标变换，不把RGNB本身当作method。

### 3.2 History side

对$P\in\{12,24,48\}$个ordered history patches使用orthonormal DCT-II rows $U[k,p]$。history-scale
sensitivity为

$$
H[a,k,d]=\sum_p G[a,p,d]U[k,p],\qquad
E[a,k]=\sum_d H[a,k,d]^2.
$$

$k/(P-1)$是normalized history frequency。每个future atom的$E[a,:]$先归一化，防止高能量atom支配结论；
随后在七个future groups内等权平均。

## 4. Statistics And Meaning

1. `group_history_frequency_centroid`：每个future group的row-normalized history frequency期望；越高表示更依赖
   fine history variation。
2. `scale_rho`：七个future support-scale levels与上述centroid的Spearman correlation。正值支持
   local-future ↔ fine-history单调关系。
3. `fine_global_contrast`：depth-5 centroid减global-root centroid。它给出可解释的normalized-frequency effect
   size，避免只靠rank correlation。
4. `atom_label_permutation_p`：保持七个group size不变，随机打乱atom-to-group assignment 1024次，计算
   canonical `scale_rho`的one-sided empirical p-value。它检验结果是否只是各atom能量差异。
5. `random_history_basis_percentile`：用64个Haar-random orthogonal patch bases替代DCT，并对其columns使用同一
   normalized index，报告canonical `scale_rho`在random controls中的percentile。它检验ordered patch scale
   coordinate是否优于任意orthogonal coordinates。
6. `parseval_relative_gap`：coordinate transform前后operator Frobenius energy相对误差；只作implementation
   invariant，阈值$10^{-8}$（float64）。

不设置future-center ↔ history-position gate。即便位置统计显著，也不能据此恢复B14式atom-to-patch retrieval。

## 5. Artifact And Sampling Contract

- datasets：ETTh1、ETTh2、ETTm1、ETTm2、Weather；
- checkpoints：frozen natural A6 seeds 2021/2022/2023，共15个units；
- split：不读取train/validation/test samples；只读frozen checkpoint与effective config；
- forecast model：不更新；不替换任何component；
- random controls：固定seed `20260715`；
- primary aggregation：先在每个dataset内对三seed metric取mean，再做five-dataset gate；
- raw artifact：per-unit metrics、seven-group centroids、permutation/random-control summaries、config/environment。

## 6. Pre-Registered Gate

D9-A只有同时满足以下条件才`pass_to_d9b`：

1. five-dataset mean `scale_rho >= 0.25`；
2. 至少4/5 dataset的three-seed mean `scale_rho > 0`且`fine_global_contrast >= 0.05`；
3. 至少12/15 checkpoint units的`scale_rho > 0`；
4. 至少4/5 dataset的three-seed aggregate通过atom-label permutation one-sided $p\le0.05$；
5. 至少4/5 dataset的canonical DCT `scale_rho`高于random-history-basis controls的95th percentile；
6. 所有unit `parseval_relative_gap <= 10^{-8}`且checkpoint/profile contract完整。

阈值在读取结果前冻结。`pass_to_d9b`仍不授权method：下一步只能以同一尺度定义做sample-dependent
input-Jacobian/JVP confirmation。任一primary gate失败，decision为`operator_scale_hypothesis_not_supported`，
rollback到Step2/3；不得用调basis、改group或增加router挽救本诊断。

## 7. Failure Attribution Boundary

- 若Parseval或checkpoint contract失败：`diagnostic_invalid_for_direction_rejection`；先修实现/协议。
- 若exact decoder operator没有scale signal：支持`hypothesis_false_at_memory_operator_level`，不实现新decoder。
- 若D9-A通过但未来D9-B失败：说明decoder weights存在静态scale pattern，但Encoder/input local sensitivity不支持；
  归因`intervention_point_wrong_or_representation_semantics_unsupported`。
- D9不能判断新架构性能，也不能证明A6 Encoder已经是最终最佳Encoder。
