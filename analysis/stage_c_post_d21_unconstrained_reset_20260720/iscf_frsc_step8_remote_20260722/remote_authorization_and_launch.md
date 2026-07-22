# SC-ISCF-FRSC-v0 Step8 Remote Authorization and Launch

## Authorization

| Field | Record |
| --- | --- |
| `authorization_date` | `2026-07-22` |
| `user_instruction` | `继续完成 FRSC Step7B，并推进至remote training` |
| `candidate_version` | `SC-ISCF-FRSC-v0-step7b` |
| `remote_training_authorized` | `true` |
| `formal_test_access_authorized` | `false` |
| `new_training_runs` | 20 |
| `effective_audit_runs` | 25，含5个frozen identity references |
| `checkpoint_rule` | best mean validation MSE over H96/H192/H336/H720 |
| `confirmation_seeds` | false |
| `modern_baselines` | false |

授权只覆盖四个FRSC arms × five datasets × seed2021的validation matrix。candidate、alpha、rank、partition seed、
objective、profiles、metrics和gates均已冻结；runner硬拒绝test split。

## Preflight and resource smoke

remote repo保留三份已知历史CSV修改，并从`e5edeb1`安全fast-forward到`9069e87`。GPU 0/1/2均为RTX 3090，
preflight时各18 MiB used、0% utilization且无compute process。remote checker再次`37/37`通过。

resource smoke覆盖Weather `frsc_scope_a055`与`frsc_random_a055`。两者均写出finite training artifacts；effective config为
`iscf-full-rank-scope-conditioning`、alpha `.55`，model diagnostics为minimum eigenvalue `.45`与
`frsc_full_rank=true`。runner日志扫描未发现Traceback、OOM、NaN或Inf。

## Launch record

| Field | Actual record |
| --- | --- |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `remote_commit` | `9069e87688faf8768aeab33a8a878b6636b395bf` |
| `config_sha256` | `32d508ab62e5013cbb48d2cd68ae1c99288bcd991acc21585cbbb6d35164aa24` |
| `profile_sha256` | `80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_frsc_v0_step7b` |
| `environment` | conda `moe` |
| `GPU` | 0,1,2；NVIDIA GeForce RTX 3090 |
| `launch_time` | `2026-07-22T10:41:20+08:00` |
| `supervisor_pid` | outer `3559157`；runner `3559159` |
| `initial_progress` | validation `0/20`；Weather scope/random/global-a055 at epoch 1, iter 200 |
| `initial_GPU` | about 1474–1493 MiB used；58–60% utilization |
| `estimated_finish` | approximately 1.5–2 hours，dependent on early stopping and validation diagnostics |
| `formal_test_access` | false |

Decision=`frsc_step8_training_active_formal_test_disabled`。运行期间冻结repo/config/gates，不做短间隔轮询。20/20完成后
先同步artifacts并执行validation effectiveness、matched attribution、internal health与failure attribution audit；不得自动访问
formal test。
