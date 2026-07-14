# SC1-D8-E2E 代码与理论一致性说明

## 目标与边界

本实现只服务`SC1-D8-E2E`：在相同natural dataset profile下，让A6与六个PAF arms均从头训练
Encoder和Decoder，消除frozen A6 Encoder天然适配A6 Decoder的比较偏差。requested horizon不进入任何
learned module；Step 7B只读取validation，不读取test。

## Forward Tensor Flow

对输入`x_enc [B,L,C]`，沿用A6 Encoder得到：

1. channel-independent patching与embedding：`memory [B,C,P,D]`；
2. 无损reshape：`hidden = flatten(memory) [B,C,R]`，其中`R=P*D`；
3. PAF branch：`z = A hidden + a [B,C,K]`，`K=256`；
4. fixed descriptor：`d_j [8]`，描述第$j$个RGNB atom的global/local类型、support、depth与组内位置；
5. shared trunk：`q_j = psi(d_j) [K]`；
6. coefficient：`alpha_j = <z,q_j> + b_j [B,C]`；
7. 对prefix $H$，只选择`atom_start < H`的atom，并与其固定RGNB basis row的前$H$步合成：

$$
\hat y_{1:H}=\sum_{j\in\mathcal A(H)}\alpha_j\phi_j[1:H].
$$

输出在`TimeAlign.Model`中转回`[B,H,C]`并沿用输入normalization的denorm。`H`只控制fixed atom subset和
basis crop，不进入branch、trunk、normalization或router。

## 为什么 Flatten 不等于丢失 Patch 信息

`hidden [B,C,R]`只是`memory [B,C,P,D]`的bijective reshape。branch weight可精确重写为
`A [K,P,D]`：

$$
z=Ah+a=\sum_{p=1}^{P}A_pm_p+a.
$$

因此每个patch仍有独立weight block。真正需要诊断的是共同`K=256` latent是否发生patch collapse，以及
separable history-atom map是否过强，而不是四维到三维这个shape变化本身。

## 七个 Arms

- `learned-basis-forecast-operator`：same-run A6 control；
- `plgo-paf-{geo,perm,random}-c256`：primary compact PAF及两个descriptor controls；
- `plgo-paf-{geo,perm,random}-m694`：width/optimization sensitivity，不允许事后替换primary。

`geo`使用canonical RGNB descriptors；`perm`固定打乱descriptor与basis atom的对应关系；`random`使用逐维
moment-matched fixed random descriptors。所有arms共享Encoder profile、optimizer、objective与seed policy，
且均为from scratch joint training。

## Patch Diagnostics

训练后在validation最多8个batches上计算，不参与checkpoint selection：

- `weight_norm_share[p]`：branch/coeff projection第$p$个patch block的Frobenius norm占比；
- `latent_contribution_share[p]`：$A_pm_p$ mean-squared energy占比；
- `patch_contribution_entropy`：上述占比的normalized entropy；
- `atom_patch_jacobian_norm[j,p]`：$\|q_j^TA_p\|_2$；
- `atom_patch_profile_diversity`：不同RGNB group归一化patch profile的pairwise mean absolute distance。

`flatten_block_sum_max_abs`同时检查直接linear与显式patch block sum是否一致。float32长向量累积的protocol
阈值为`1e-5`；RGNB float64参考正交性仍用`1e-10`。

## 产物与后验审计

每个run写出`effective_config.json`、`model_diagnostics.json`、`patch_diagnostics.json`、
`patch_diagnostics_by_patch.csv`；PAF额外写`atom_patch_jacobian_norm.npz`。训练后
`check_stage_c_sc1_d8_checkpoint_invariants.py`重新加载checkpoint，验证prefix projectivity、from-scratch
contract、无frozen parameters和patch artifact数值有效性。

## Code-Theory Consistency

[Fact] 代码实现了RGNB local/global fixed synthesis、horizon-agnostic atom-conditioned coefficient generation、
native prefix restriction与端到端Encoder/Decoder共同训练。

[Fact] `geo/perm/random`只改变fixed descriptor identity；`c256/m694`只改变trunk width，因此可分别审计
geometry attribution与width sensitivity。

[Limitation] 当前history-to-atom interaction经过shared latent并具有separable形式；它不是通用
atom-to-patch retrieval，也未证明最优。

[Falsification] 若GEO相对A6和matched descriptors稳定失败且patch usage与A6相当，则否定exact shared-latent
PAF方法，而不否定RGNB scaffold；若PAF相对A6发生patch collapse，则优先归因
`intervention_point_wrong/shared_history_interface_suspected`并回Step 4。
