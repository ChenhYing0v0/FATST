# ISCF-v1-CPSI Step7B Prelaunch Audit

## 1. 本步要回答什么

本步不回答CPSI是否有效，只回答冻结的Step6/7A candidate能否被诚实、完整地训练与审计。用户要求
`CPSI-SELF/CPSI-LINEAR/CPSI-COMMON/POST-SYNTH`只作中间诊断，不因validation轻微负向提前停止；因此
本protocol强制全部protocol-valid arms完成同一次official-test MSE/MAE audit。

Decision=`step7b_prelaunch_pass`。远程seed2021的25个new trainings与训练完成后的单次formal test已获用户授权；
confirmation seeds仍未授权。

## 2. 冻结矩阵

- new arms：CPSI、SELF、LINEAR、COMMON、POST-SYNTH；
- datasets：Weather、ETTm1、ETTh1、ETTh2、ETTm2；
- seed：2021；
- new training：$5\times5=25$；
- historical references：ISCF-v0与A6_FULL各5个，共10个；
- effective runs：35；
- official test：H96/H192/H336/H720的MSE与MAE，各140 cells。

validation仅用于`mean MSE(H96,H192,H336,H720)` checkpoint selector、early stopping与internal health。runner的
`FORMAL_TEST_ONLY=1`在25/25 training artifacts齐全前硬拒绝执行，test过程中验证checkpoint SHA256不变。

## 3. Historical reference审计

reference source固定为`stage_c_siff_equal_attribution_v2`。source config SHA256为
`f0600be7...38a`，run audit SHA256为`64d89c07...e366`；10/10 source rows均为`status=ok`、
`protocol_pass=True`且checkpoint hash与新config逐dataset登记值一致。新protocol不重训或选择historical
reference；远端formal analysis只核验冻结hash并读取既有test artifacts。

## 4. Internal health与统计量

test evaluator在不改变forecast path的前提下，对固定前256个series rows保存：

- `probe_cpsi_common_rms`：scope mean在interaction input上的RMS；
- `probe_cpsi_private_rms`：zero-sum scope deviation的RMS；
- `probe_cpsi_left_rms/right_rms`：两条projection branch的RMS；
- `probe_cpsi_latent_rms`：product或linear combination后的latent RMS；
- `probe_cpsi_message_rms`：回写scope modes/arms的interaction message RMS。

`model_diagnostics.json`另保存trained `cpsi_output_projection_norm`。health pass要求全部finite且message、latent和
output projection norm均大于$10^{-8}$。health只能识别dead path或numeric pathology，不能覆盖negative test
effectiveness。

## 5. Effectiveness、attribution与failure mapping

primary为CPSI相对ISCF-v0：macro MSE至少`+0.3%`、至少3/5 dataset wins、10/20 cell wins，且macro MAE
不低于`-0.3%`，才记为initial support。macro MSE落入`[-0.5%,+0.3%)`时为
`test_inconclusive_keep_candidate_no_claim`，不作方向级拒绝。

SELF/LINEAR/COMMON/POST与CPSI的差异使用`±0.3%` attribution band。它们可以阻断“common-private interaction
必要性”的claim，但不能阻断test access；若CPSI与controls共同优于ISCF，则归为`capacity_control_explains`或
更宽泛nonlinear/shared-capacity效果。只有material negative（macro不高于`-0.5%`且至少4/5 datasets为负，或任一
dataset退化至少5%）且无pathology，才关闭exact CPSI-v1并回Step4/5。

## 6. 本地验证

machine gate为18/18：

- profile/config/matrix/test-authorization contracts：4/4；
- user control governance：2/2；
- historical source/hash audit：3/3；
- five production constructors与paired parent hash：6/6；
- runner syntax、25-job dry-run、analyzer synthetic smoke：3/3。

五个ETTh1 constructors的interaction parameters分别为41856/41856/41856/41856/41040；parent initialization
hash只有一个unique value。所有probe tensors finite。`py_compile`、JSON parse、`bash -n`均通过。

## 7. 授权与下一步

Step7B通过后，按顺序执行：commit/push；远端`git pull`；`nvidia-smi` preflight；Weather-CPSI与
ETTm2-POST双resource smoke；25-run training。只有25/25 training完成且artifact audit通过，才执行一次25-run
formal test。不得根据validation排序删除arms、修改rank、调整dataset/horizon或提前访问test。

当前`confirmation_seeds_authorized=false`，router=false，second loss=false。若remote smoke出现OOM/NaN/CLI或
artifact fault，只归为`optimization_or_numeric_pathology`或Step7 design fault并修复，不据此拒绝机制方向。
