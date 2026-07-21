# ISCF-v1-CPSI Step8 Remote Launch Record

## Launch state

- date/time: `2026-07-21T17:09:43+08:00`；
- remote repo: `/home/yingch/projects/FATST`；
- commit: `5d2330e0f9f3ec6c57d580ed545ea4a1a4e63ea4`；
- output: `/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_v1_cpsi_v1`；
- config SHA256: `bb7bfe58ef9d11eb239c6d3f4f027581cd8574faba716052a402068ed80554f9`；
- profile SHA256: `80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a`；
- GPUs: 0/1/2，均为RTX 3090；preflight memory used均18 MiB，无compute process；
- supervisor PID: `2235133`。

remote repo有三处历史analysis CSV的既有tracked modifications，均与本阶段无重叠；本次pull为fast-forward且未修改、
提交或清理这些文件。

## Resource smoke

第一次smoke的两项训练均finite，但remote无`rg`使negated log command产生false-pass风险，因此该verdict作废并修复
runner。commit `5d2330e`加入`grep` fallback后重新执行：

- Weather / CPSI：2 train batches + 2 validation batches，finite，无OOM/Traceback/NaN/Inf；
- ETTm2 / POST-SYNTH：同上，finite，无OOM/Traceback/NaN/Inf。

修复后的Step7B machine gate为19/19。

## Formal training launch

matrix为25 new runs；初始状态`training=0/25, test=0/25`。GPU0/1/2分别从Weather CPSI、SELF、LINEAR开始。
launch record明确`formal_test_execution_mode=0`；本阶段只训练与validation checkpoint selection。

formal test只有在25/25 training artifacts完整后才允许由独立`FORMAL_TEST_ONLY=1`调用。不得依据validation结果删arm、
改rank或改dataset/horizon。confirmation seeds仍未授权。

## Completion

- training：25/25，`2026-07-21T18:03:54+08:00`完成；
- formal test：25/25，`2026-07-21T18:10:55+08:00`完成；
- formal-test launch commit：`c4e3669b81b6316019483fdd47df035b9a9c2d57`；
- checkpoint nonmutation / protocol / all-finite：25/25 pass；
- analysis decision：`cpsi_v1_exact_performance_fail_return_step4_5`。
