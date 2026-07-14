# SC1-D8 End-to-End Co-Adaptation Screen Protocol

## Status

| Field | Value |
| --- | --- |
| `candidate_id` | `SC1-PLGO-PAF / SC1-D8-E2E` |
| `role` | paper-core method screening |
| `current_step` | Step 9-10 complete；exact PAF failed，rollback Step 4 |
| `narrative_gate` | conditional pass；RGNB + horizon-agnostic projectivity + atom-conditioned generation |
| `effectiveness_gate` | fail：GEO-c256 vs A6 -28.10%；geometry vs matched +14.33%、5/5 retained |
| `remote_training_authorized` | false；35/35 complete，不进入三seed；新候选需重过Step4-6 |
| `test_usage` | forbidden during screening |
| `rollback` | stable E2E fail -> Step4；pathology -> repair Step7 protocol only |

最终failure attribution为`exact_paf_failed_geometry_retained_rollback_step4`。5/5 GEO hit epoch cap，但最后5
epochs validation只改善0.02%-0.49%，不足以解释14%-46%的A6 gap；不做无边界longer-epoch sweep，也不把
结果升级为PLGO方向级否决。详见
`analysis/stage_c_sc1_d8_e2e_20260714/research_interpretation.md`。

## Primary Question

当Encoder与Decoder共同学习时，RGNB geometry-conditioned PAF是否同时：

1. 相对A6形成可用的完整forecast operator；
2. 相对PERM/RANDOM descriptor controls表现出geometry-specific benefit；
3. 在requested horizon不进入learned path的前提下保持prefix projectivity？

## Tensor Interface And Patch-Information Boundary

当前A6/PAF都先得到`memory M [B,C,P,D]`，再执行无损
`h = flatten(M) [B,C,R]`，$R=P D$。PAF branch可按patch blocks严格写成

$$
z=A h+a=\sum_{p=1}^{P}A_pm_p+a,
$$

所以flatten不等于pooling，也不会删除patch identity。潜在瓶颈是$R\rightarrow256$ shared latent及
$\alpha_j=\psi(d_j)^Tz$的separable atom-history map。A6同样使用$R\rightarrow256$ coefficient projection，
因此不能把256维压缩单独归因为PAF缺陷。

本轮不加入atom-to-patch attention/retrieval：B14的retrieval-demand为1/6 settings、0/3 datasets，且该机制有
source overlap，尚未重新通过Step4-6。若D8失败，patch diagnostics决定是否有依据重开该interface方向。

## Fairness Contract

- initialization：每个arm独立from scratch；相同initialization class与seed policy；
- trainable path：Encoder、normalization、PAF/A6 decoder全部trainable；
- forbidden：加载A6 Encoder checkpoint、只训练replacement head、用旧A6 test reference替代同批重跑；
- dataset profiles：固定five-dataset natural profiles，不按arm重新特调；
- objective：full-H720 pointwise L1；MIPR仍held；
- selection：best H720 validation MSE；screen不读test；
- evaluation：validation dense horizons H1..720，并报告预注册segments；
- optimization：所有arms共享optimizer class、base LR、epoch/patience与stopping rule。若出现pathology，只能启动
  model-family-wide、预注册的optimization repair，不能逐dataset精调后宣称method pass。

## Step 7A Arms

| Arm | Descriptor | Width | Purpose |
| --- | --- | ---: | --- |
| `a6_e2e` | free learned temporal basis | native | same-run effectiveness control |
| `geo_c256_e2e` | canonical RGNB | 256 | pre-registered primary PAF |
| `perm_c256_e2e` | fixed row permutation | 256 | matched geometry control |
| `random_c256_e2e` | moment-matched random | 256 | matched descriptor control |
| `geo_m694_e2e` | canonical RGNB | 694 | width/optimization sensitivity only |
| `perm_m694_e2e` | fixed row permutation | 694 | matched geometry control |
| `random_m694_e2e` | moment-matched random | 694 | matched descriptor control |

params只报告，不参与dataset profile选择或方法价值判断；两个width用于避免把exact implementation的失败误归因
为单一width restriction，不把parameter equality写成contribution。

## Step 7A Local Gates

1. 五profiles、七arms可实例化；
2. `memory [B,C,P,D] -> prediction [B,H,C]`在H1/48/96/192/336/720 shape正确；
3. full-vs-prefix max gap`<=1e-5`；该阈值用于 float32 长向量累积等价性，descriptor 的 float64 参考构造仍采用更严格检查；
4. `H`不进入descriptor、branch、trunk、normalization或router；
5. 所有PAF与Encoder parameters均获得finite nonzero gradients；
6. effective config显式记录trainable parameter groups，assert不存在frozen A6 checkpoint；
7. runner dry-run、analyzer synthetic smoke与code-theory consistency通过。
8. flatten branch与patch-block-sum rewrite max gap`<=1e-5`；每个patch block均有finite nonzero gradient。

Step7B trained-checkpoint后验使用`2e-5`的patch-block absolute tolerance：首个RANDOM-Weather checkpoint在
所有shape/prefix/finite条件通过时出现`1.1444e-5`，属于trained weight scale下float32求和顺序误差。
该修复只作用于代数等价性审计，不改变模型、训练、checkpoint selection或performance gate；prefix gate仍为
`1e-5`。

Runner支持`WORKER_OFFSET/WORKER_STRIDE`恢复原固定stride worker。若单个worker因后验审计退出，可在不重复
其他active jobs的前提下只续跑其原job子序列；已有完整metrics/invariants/patch artifacts的run会跳过。

## Mandatory Patch Diagnostics

以下统计不参与checkpoint selection，也不单独覆盖forecast gate：

- `patch_block_weight_norm_share[p]`：将branch/coeff matrix按`[P,D]`切块后，各$A_p/W_p$的Frobenius norm占比；
- `patch_latent_contribution_share[p]`：validation上$z_p=A_pm_p$的mean squared norm占比；
- `patch_contribution_entropy`：上述share的normalized entropy，诊断是否collapse到极少patch；
- `atom_patch_jacobian_norm[j,p]`：$\|\psi(d_j)^TA_p\|_2$；
- `atom_patch_profile_diversity`：不同RGNB support groups的normalized Jacobian profiles pairwise distance；
- A6使用$W_p$给出同定义的coefficient-patch control。

这些量分别来自trained parameter blocks、validation memory与解析Jacobian。若PAF forecast失败但patch usage与A6
相当，则不支持“flatten压缩丢失patch信息”；若PAF相对A6出现稳定patch collapse，failure attribution标记为
`intervention_point_wrong/shared_history_interface_suspected`，不得拒绝更广的PLGO family。

## Step 7B Matrix And Gates

初筛：ETTh1、ETTh2、ETTm1、ETTm2、Weather × 7 arms × seed2021 = 35 runs。

screening `partial_pass`要求：

1. pre-registered primary `geo_c256_e2e`相对same-run A6的dense-MSE macro improvement至少`+1.0%`；
2. 任一dataset相对A6不得恶化超过`0.5%`；
3. primary `geo_c256_e2e`相对compact PERM/RANDOM median至少`+0.5%`，至少4/5 datasets为正；
4. MAE macro非负，所有prefix/projectivity/trainability invariants通过；
5. 无divergence、>100% degradation、epoch-cap dominated或validation protocol mismatch。

不得在看到结果后选择width。matched-694只判断compact failure是否具有width/optimization sensitivity；若只有
matched通过，必须返回Step6重冻method contract，不能直接把它改成primary。compact通过后，对A6/GEO及
compact PERM/RANDOM在五datasets运行seeds2021/2022/2023。只有三seed结果可进入Step10 paper-core decision。

## Failure Attribution

- A6与PAF都stable，GEO输A6但赢matched descriptors：exact PAF effectiveness fail；geometry retained；Step4；
- GEO与matched descriptors相同：geometry end-to-end unsupported；收缩D7 claim；
- PAF出现slow convergence/divergence：`optimization_or_numeric_pathology`；修复Step7，不拒绝方向；
- PAF稳定失败且patch diagnostics相对A6 collapse：`intervention_point_wrong`；返回Step4审计patch-aware interface；
- PAF稳定失败但patch usage不collapse：关闭exact shared-latent PAF，geometry retained，返回Step4 redesign；
- only one width works：width-sensitive implementation；不得写universal method claim；
- single dataset positive：cross-dataset gate fail；不得升paper core。

## Why No Frozen Cross-Swap Next

Encoder-source × Decoder-source的$2\times2$ cross-swap可量化co-adaptation，但它必须先拥有分别端到端训练完成的
A6与PAF checkpoints，而且仍是secondary compatibility diagnostic。当前最缺的是PAF完整架构的公平结果，
因此不先消耗实验预算继续做freeze/replace。
