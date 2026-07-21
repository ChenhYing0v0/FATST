# SC-ISCF-SPS-v0 Step7B validation prelaunch

## 1. 要回答的问题

Step7B只回答：在ISCF-v0的independent maps、五个scope groups、direct policy和`equal_skill` training contract均不变时，
scope-native synthesis projection能否同时改善validation forecast与arm specialization。它不是formal test，也不以validation
结果建立paper-facing effectiveness claim。

## 2. Frozen matrix

矩阵为`4 arms × 5 datasets × seed2021 = 20`个from-scratch matched runs：

- `sps_scope_canonical`：候选，scope-local DCT projection与canonical groups；
- `sps_identity_canonical`：exact parent code-path control，projection为identity；
- `sps_global_canonical`：所有arms使用相同global DCT smoothing；
- `sps_scope_random`：scope projection不变，仅把中间scope binding改为random partition。

五个datasets使用冻结natural profiles，matched `pcsd_mode_rank`分别为ETTh1/ETTh2/ETTm1/ETTm2/Weather =
`109/116/116/106/116`。checkpoint只由validation H96/H192/H336/H720 mean MSE选择；loss、policy、optimizer、epochs、
seed与initialization class在四arms间一致。test split被runner硬禁用。

## 3. Statistics and roles

- validation effectiveness：每个dataset-horizon cell的gain为
  $100(1-\mathrm{MSE}_{candidate}/\mathrm{MSE}_{control})$，同时报告MSE/MAE、macro、dataset wins、horizon wins和全部
  negative cells；
- `arm_pairwise_normalized_rms`：projected arms两两RMS距离的均值除以target RMS，检查arm collapse；
- `oracle_headroom_percent`：逐row选最优arm的MSE相对learned fusion MSE的潜在改善，检查complementarity是否仍存在；
- `policy_normalized_entropy`：五scope direct-policy entropy除以$\log 5$；
- `projection_retained_rms_ratio`：normalized forecast space内每个scope的projected RMS/raw RMS；
- `scope_winner_count`：所有dataset × future-bin中成为minimum-MSE arm的不同scope数量。

primary gate要求scope candidate相对identity macro MSE至少`+0.3%`、至少3/5 dataset wins、3/4 horizon wins且macro
MAE不低于`-0.3%`。candidate还需MSE超过global control并通过finite、nontrivial projection、noncollapse、oracle、policy与
multi-winner health checks。random control只决定canonical binding attribution：若candidate不超过random，结果降为
`performance_partial_pass_scope_binding_unresolved`，不得据此拒绝ISCF architecture。

## 4. Tooling and local verification

runner从JSON生成20个jobs，actual launch前读取authorization；`remote_training_authorized=false`时exit 3。
`EVALUATION_SPLIT`只允许`val`，即使`DRY_RUN=1`也拒绝`test`。每个正常training完成后，checkpoint evaluator只在sequential
validation loader上生成`pcsd_validation_diagnostics.npz`与`trained_invariants.json`。

evaluator新增向后兼容的SPS payload：`probe_sps_raw_arms/projected_arms/removed_arms`与
`probe_direct_policy`。analyzer只读取validation metrics和这些NPZ，不包含formal-test branch；输出完整cell table、run audit、
specialization health、projection retention和decision JSON。

local gate为`19/19`通过，jobs为`20/20`唯一且完整：model shapes/finite、paired trainable initialization、三种projection
rank/degrees、runner syntax/dry-run、remote unauthorized rejection、test-split rejection、analyzer synthetic decision path、
GPU preflight hook与`rg/grep` log scanner fallback均通过。

首轮checker曾把projection中的$K$错误预期为固定`basis_rank=256`，而production code按Step7A公式使用dataset-matched
`pcsd_mode_rank`。这属于prelaunch config expectation fault；已修正为106/109/116三组显式contracts，没有修改model forward。

## 5. Failure attribution and decision

- validation negative且无pathology：只拒绝exact SPS-v0，rollback Step4；ISCF仍作为architecture prior；
- candidate positive但不超过global：`scope_specificity_unresolved`；
- candidate positive但不超过random：`scope_binding_unresolved`，random不得方向级拒绝；
- nonfinite、dead gradients或artifact mismatch：`optimization_or_numeric_pathology`/protocol fault，回Step7 repair；
- diversity/oracle collapse：优先归为`intervention_point_wrong`或`readout_or_head_design_wrong`，回Step4/5。

Decision=`step7b_prelaunch_pass_wait_remote_authorization`。当前没有training/validation结果；remote training、formal test、
confirmation seeds与modern baselines均未授权。

> 2026-07-22 authorization update：用户随后明确授权启动冻结20-run validation matrix。prelaunch evidence与gates不变；
> 当前remote-training状态以Step8 launch record和config为准，formal test仍未授权。
