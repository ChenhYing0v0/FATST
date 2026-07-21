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

| Field | Actual record |
| --- | --- |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `remote_commit` | `78cbcf47e1cb5f6d24a01ac5ad8fea8b0deebbb9` |
| `config_sha256` | `78d46c96ecb5f3f41129d6fe5dce274e127779545b8afca5930f16ceebccb49c` |
| `profile_sha256` | `80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_v0_sac_v1` |
| `environment` | conda `moe` |
| `GPU` | 0,1,2；NVIDIA GeForce RTX 3090 |
| `preflight` | each 18 MiB used、0% utilization、no compute process |
| `resource_smoke` | Weather-RANDOM seed2021 + ETTm2-Q1 seed2023；finite/no-OOM |
| `launch_time` | `2026-07-21T18:58:40+08:00` |
| `supervisor_pid` | `2383292` |
| `formal_test_execution_mode` | `0` |

remote repo原有三份与SAC无关的generated CSV修改，fast-forward时保留且与本次文件不冲突。`git pull --ff-only`
从`23bb8e5`更新到`78cbcf4`。

resource smoke前后三张GPU均空闲；two-batch smoke产出checkpoint、training log、effective config、initialization
contract与model diagnostics，runner log scanner未发现Traceback、OOM、NaN或Inf。

正式launch命令为training-only default mode：

```bash
GPU_IDS="0 1 2" bash scripts/remote/run_stage_c_iscf_v0_sac.sh
```

初始状态为training/test=`0/25,0/25`。首批三个jobs均为Weather：RANDOM seed2021、Q1 seed2022、RANDOM
seed2022；均进入epoch 1。初始GPU used memory约`1473/2746/1474 MiB`，utilization=`67/87/61%`。

Decision=`step8_training_active_formal_test_not_authorized`。训练期间不pull、不改config/gates，不进行短间隔轮询；
25/25完成后先审计validation artifacts并等待formal-test授权。

## Completion and validation handoff

25-run training于`2026-07-21T20:24:32+08:00`完整结束；status=`training 25/25, test 0/25`。25个checkpoint、
25份validation metrics与所需config/init/diagnostic artifacts齐全，log scanner无Traceback、OOM、NaN或Inf命中。
联合35个historical references的validation audit为60/60 runs、240/240 rows，internal health 15/15通过。

validation observation为ISCF over Q1-WIDE MSE/MAE `+1.0704%/+0.7538%`，canonical over RANDOM-PARTITION
`-0.1823%/-0.3075%`。后者是明确negative lead，但validation不得通过或拒绝机制。

Decision=`formal_test_ready_pending_user_authorization`；formal test仍未授权、test access仍为0。详见
`validation_artifact_audit_and_test_handoff.md`。
