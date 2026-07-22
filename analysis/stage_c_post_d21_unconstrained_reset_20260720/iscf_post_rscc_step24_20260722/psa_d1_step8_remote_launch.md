# SC-ISCF-PSA-D1 Step8 Remote Launch

## 1. Decision

Decision=`psa_d1_five_run_validation_training_active_formal_test_disabled`。

Commit `f5275a469f39b7dae99ad23fef715b303534344c`已remote fast-forward；GPU、Weather resource smoke与
initialization contracts全部通过。five-run validation matrix已于`2026-07-22T16:00:40+08:00`启动。

## 2. Remote environment and preflight

| Field | Value |
| --- | --- |
| remote repo | `/home/yingch/projects/FATST` |
| commit | `f5275a469f39b7dae99ad23fef715b303534344c` |
| conda env | `moe` |
| GPUs | 0/1/2，NVIDIA GeForce RTX 3090 |
| preflight memory | each GPU 18 MiB used / 24107 MiB free |
| utilization | 0% on all three |
| compute processes | none |
| output root | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_psa_d1` |
| config hash | `a8318177d04af4b8dec8ea308a6d218c2edabf326bac06daa5cdb6dc1bd60c93` |
| profile hash | `80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a` |

Remote worktree在pull前已有three unrelated tracked analysis CSV modifications；它们与D1代码无重叠，未清理、覆盖或提交。

## 3. Weather resource smoke

| Contract | Result |
| --- | --- |
| objective | `equal_skill` |
| route weight | `0.0` |
| weighted route loss | `0.0` |
| scope gradients | `[0.23986,0.23642,0.20030,0.17342,0.10166]`；5/5 nonzero |
| smoke initialization hash | `8828e47b...9f368` |
| historical EQUAL hash | same |
| ARMERR hash | same |
| SHUFFLED hash | same |
| Traceback/OOM/NaN/Inf | none |

Smoke只执行one epoch × two train/eval batches，final evaluation disabled；不进入result matrix。

## 4. Launch record

| Field | Value |
| --- | --- |
| supervisor PID | `3975446` |
| start | `2026-07-22T16:00:40+08:00` |
| new runs | 5 |
| standard-horizon validation cells | 20 |
| effective references | historical EQUAL + ARMERR + SHUFFLED |
| effective matrix | 20 runs / 80 validation cells |
| official test | false |

Initial scheduling：

- GPU0：Weather，epoch 1 active；
- GPU1：ETTm1，epoch 1 active；
- GPU2：ETTh1，epoch 1 active；
- queued：ETTh2、ETTm2。

首次status=`validation=0/5`。依据同profile历史时长，预计约20--35分钟完成；实际受early stopping与Weather耗时影响。

## 5. Running boundary

- 5/5前不读取partial validation metrics作H2/H3选择；
- training期间不remote pull，不改config/gates/matrix；
- 5/5与20/20 new cells完整后才运行冻结analyzer；
- formal test、confirmation seeds与method promotion保持false；
- 若单run出现numeric/config/artifact failure，只作protocol repair，不产生research decision。
