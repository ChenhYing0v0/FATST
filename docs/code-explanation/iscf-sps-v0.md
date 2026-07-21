# ISCF-SPS-v0 代码与理论说明

## Forward tensor flow

`ScopeProjectedSynthesisReadout`继承ISCF-v0的independent scope field：

1. `hidden [B,C,R]`经`mode_weight [S,D,R,K]`得到`components [B,C,S,D,K]`；
2. 每个scope经既有`_scope_forecast`产生`raw_arm [B,C,T]`；
3. canonical/random `group_indices [G_s,s]`把arm gather为`grouped [B,C,G_s,s]`；
4. fixed orthonormal basis `basis_s [s,r_s]`先得到
   `coefficients [B,C,G_s,r_s]`，再reconstruct为`[B,C,G_s,s]`；
5. scatter恢复`projected_arm [B,C,T]`，五arms堆叠为`[B,C,S,T]`；
6. parent direct policy输出`weights [B,C,T,S]`，逐target融合并在最后crop为`[B,H,C]`。

`global` control不按groups处理，而把每个`raw_arm [B,C,T]`投影到同一个global DCT rank-$K$ subspace；`identity`
control保留每组full rank，数值上恢复parent ISCF。

## Projection rank

candidate使用

$$
r_s=\min(s,\max(1,\operatorname{round}(Ks/T))).
$$

该规则让除point scope外的每个arm在完整$T$域上保留约$K$个interpolation degrees，同时通过不同group count/local
rank组合形成不同resolution bias。DCT bases是buffers，不增加trainable parameters。

## Code-theory consistency

预期理论：scope projector同时限制forward forecast subspace和backward error subspace，从而给五个independent history maps
提供不同resolution的learning signal。代码通过orthonormal $C_s$和$P_s=C_sC_s^\top$实现；autograd自然产生
$P_s^\top$ gradient。

仍属于proxy的部分：DCT frequency不是数据语义feature的直接标签；不同projected outputs不等于有用specialization；
point scope的projected degrees高于其余arms；stored synthesis参数中存在projected-out directions。SPS必须通过E2E
validation/test、global smoothing control、random binding control及internal gradient/spectral diagnostics，才能支持paper claim。

局部合同用one-target error impulse进一步验证gradient support：五个projectors的nonzero support分别精确为
`[1,48,144,360,720]`。这证明scope width在代码中成为credit propagation width，但仍不证明这种credit分配会降低
forecast MSE。
