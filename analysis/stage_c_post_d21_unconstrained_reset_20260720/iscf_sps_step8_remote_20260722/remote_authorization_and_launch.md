# SC-ISCF-SPS-v0 Step8 Remote Authorization and Launch

## Authorization

| Field | Record |
| --- | --- |
| `authorization_date` | `2026-07-22` |
| `user_instruction` | `授权并启动20-run validation matrix` |
| `candidate_version` | `SC-ISCF-SPS-v0-step7b` |
| `remote_training_authorized` | `true` |
| `formal_test_access_authorized` | `false` |
| `training_runs` | 20 |
| `checkpoint_rule` | best mean validation MSE over H96/H192/H336/H720 |
| `confirmation_seeds` | false |
| `modern_baselines` | false |

本次授权只覆盖已冻结的scope/identity/global/random四arms × five datasets × seed2021 validation matrix。runner只允许
`EVALUATION_SPLIT=val`；不访问official test，不改变candidate、rank、partition seed、loss、policy、profiles或gates。

## Frozen launch sequence

1. 更新authorization、主线文档并commit/push；
2. remote repo `/home/yingch/projects/FATST`执行`git pull --ff-only`并核对commit；
3. 用`nvidia-smi`检查GPU 0/1/2 memory、utilization与active processes；
4. 执行scope-canonical Weather与identity-canonical Weather two-batch resource smoke；
5. smoke finite且无Traceback/OOM/NaN/Inf后，以GPU 0/1/2启动20-run workload-aware matrix；
6. training期间冻结repo/config/gates，不短间隔轮询；
7. 20/20 validation artifacts完成后先做Step9 validation audit，formal test仍需独立研究决策与授权。

## Launch record

待remote preflight和正式启动后填写commit、config hash、GPU状态、smoke结果、launch time与supervisor PID。
