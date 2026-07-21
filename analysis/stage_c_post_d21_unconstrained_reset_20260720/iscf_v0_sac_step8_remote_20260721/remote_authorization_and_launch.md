# ISCF-v0 SAC Step 8 Remote Authorization and Launch

## Authorization

| Field | Record |
| --- | --- |
| `authorization_date` | `2026-07-21` |
| `user_instruction` | `继续推进SAC remote training` |
| `candidate_version` | `SC1-ISCF-v0-SAC-v1` |
| `remote_training_authorized` | `true` |
| `formal_test_access_authorized` | `false` |
| `new_training_runs` | 25 |
| `historical_reference_runs` | 35 |
| `effective_runs` | 60 |
| `checkpoint_rule` | best mean validation MSE over H96/H192/H336/H720 |
| `test_role` | formal mechanism attribution；仍等待独立授权 |

用户本次授权只覆盖frozen 25-run training matrix，不覆盖official test。runner在training阶段只写validation metrics；
`FORMAL_TEST_ONLY=1`仍会因config authorization为false而exit 3。

## Frozen launch sequence

1. local authorization/config/document commit并push；
2. remote repo `/home/yingch/projects/FATST`执行`git pull --ff-only`并核对commit；
3. `nvidia-smi`检查三张GPU memory与active processes；
4. 并行运行Weather-RANDOM seed2021与ETTm2-Q1 seed2023 two-batch resource smoke；
5. smoke artifacts finite、无Traceback/OOM/NaN/Inf后，启动25-run workload-aware matrix；
6. training期间不pull、不改config/gates；
7. 25/25完成后停止在validation artifacts，等待用户另行授权formal test。

## Launch record

本节将在remote preflight、resource smoke和正式launch后填入实际commit、GPU状态、命令、output root与initial progress。
