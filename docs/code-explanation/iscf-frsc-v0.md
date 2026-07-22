# ISCF-FRSC-v0 代码与理论说明

## Forward tensor flow

`FullRankScopeConditioningReadout`保留ISCF/SPS的既有生成路径：

1. `hidden [B,C,R]`经五个independent `mode_weight [S,D,R,K]`生成`components [B,C,S,D,K]`；
2. 每个scope的group states与shared synthesis生成`raw_arm [B,C,T]`；
3. SPS basis计算hard component `P_s raw_arm [B,C,T]`；
4. FRSC计算`conditioned_arm = raw_arm + alpha * (P_s raw_arm - raw_arm)`；
5. 五arms堆叠为`[B,C,S,T]`，原direct policy `weights [B,C,T,S]`逐target融合；
6. full-T forecast最后crop为`[B,H,C]`，requested H不进入arm或policy计算。

candidate使用`projection_mode=scope, alpha=.55`。same-alpha global、best-tuned global alpha=.45、random scope
alpha=.55与alpha=0 identity共享同一class和parameter initialization path；alpha是Python float，不是trainable parameter。

## Full-rank and gradient contract

对orthogonal projector $P_s$，实际operator为

$$
Q_s=P_s+(1-\alpha)(I-P_s).
$$

其eigenvalues只有`1`和`1-alpha`。candidate minimum eigenvalue为`.45`，所以不会像SPS一样删除forecast或gradient
directions。autograd给出$\partial L/\partial a_s=Q_s\,\partial L/\partial\tilde a_s$；不同scope的$P_s$不同，因此五个
independent maps收到不同structured conditioning，但全部directions仍可学习。

## Diagnostics and artifact meaning

继承的`projection_diagnostics`中，`projected_arms`现在表示FRSC conditioned arms，`removed_arms=raw-conditioned`表示被衰减
而非被完全删除的部分。`model_diagnostics.json`新增`frsc_conditioning_strength`与
`frsc_minimum_operator_eigenvalue`；`trained_invariants.json`新增`frsc_full_rank`。这些字段只证明实现合同，不能单独证明
specialization或effectiveness。

## Code-theory boundary

因为$Q_s$可逆，FRSC不改变单个arm的理论function range；预期收益只能来自finite-capacity optimization/regularization bias。
固定DCT与local groups是scope structure的proxy，不等于真实frequency label。论文机制必须由from-scratch E2E candidate相对identity、
random及两个global controls的完整validation/test证据支持；若best-tuned global解释收益，FRSC只能降为generic conditioning。
