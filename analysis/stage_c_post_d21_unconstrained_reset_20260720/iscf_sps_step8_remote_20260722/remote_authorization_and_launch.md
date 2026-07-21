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

| Field | Actual record |
| --- | --- |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `remote_commit` | `48afd1255370e1b12f5151a4d2184dc0b142b20a` |
| `config_sha256` | `5a84de593eb120b31ba9da3e68212f3407b8fa519a765c1fcb960a48d53c5255` |
| `profile_sha256` | `80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_sps_v0_step7b` |
| `environment` | conda `moe` |
| `GPU` | 0,1,2；NVIDIA GeForce RTX 3090 |
| `preflight` | each 18 MiB used、0% utilization、no compute process |
| `resource_smoke` | Weather scope-canonical + identity-canonical；finite/no-OOM |
| `launch_time` | `2026-07-22T00:17:31+08:00` |
| `supervisor_pid` | outer `2787168`；runner `2787170` |
| `formal_test_access` | false |

remote从`6bbc3fc` fast-forward到`48afd12`。原有三份历史generated CSV修改被保留，与本次pull不冲突。resource smoke
两个arms均生成checkpoint、training log、effective config、initialization contract和model diagnostics；日志扫描没有
Traceback、OOM、NaN或Inf。

正式launch使用`GPU_IDS="0 1 2" bash scripts/remote/run_stage_c_iscf_sps_step7b.sh`。初始状态为`validation=0/20`；
首批jobs为Weather的scope/identity/global arms，均进入epoch 1。初始GPU memory约`1485/1474/1486 MiB`，utilization约
`63/64/60%`。预计完整matrix约需1.5–2小时，实际取决于Weather early stopping和每run validation diagnostics。

Decision=`step8_training_active_formal_test_disabled`。运行期间不pull、不改config/gates；完成后同步validation artifacts并
执行Step9 audit，不自动访问test。
