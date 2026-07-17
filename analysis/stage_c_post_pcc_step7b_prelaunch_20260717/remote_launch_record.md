# SIFF/MCCA Step 7B Remote Launch Record

| Field | Value |
| --- | --- |
| `launch_time` | `2026-07-17T14:59:22+08:00` |
| `remote_host` | `529_Lab-3090` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `commit` | `7a9e5c784f36e7d2b109bff6f12db6a32a5be1c1` |
| `config_sha256` | `cf677dc28a7cc9b0dcf785f7e9b84c219283b8cc91f7743ebb0057cbc372a371` |
| `profile_sha256` | `80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a` |
| `GPU_IDS` | `0 1 2` |
| `preflight_free_memory` | GPU0/1/2 each `24110 MiB` |
| `resource_smoke` | `Weather × SIFF_MCCA` one-batch pass on GPU0 |
| `launcher_pid` | `2977094` |
| `jobs` | `55` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_post_pcc_step7b` |
| `launcher_log` | `${output_root}/launcher_seed2021.log` |
| `test_used` | `false` |

首批worker已确认启动：GPU0=`Weather/pcsd_mcca`，GPU1=`Weather/siff_equal`，GPU2=`Weather/siff_pcc`。
启动时GPU利用率分别为`88%/87%/82%`，显存占用`1983/1984/2046 MiB`，无OOM或资源异常。

按用户要求，启动确认后不继续轮询值守。待用户通知训练完成后，先执行55/55 artifact/protocol完整性检查，再进入
Step9 factorial、control与failure-attribution分析；不得依据partial artifacts提前作method judgment。
