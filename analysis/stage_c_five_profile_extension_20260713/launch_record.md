# StageC Five-Profile Extension Launch Record

| Field | Value |
| --- | --- |
| `server` | `529_Lab-3090` |
| `repo_commit` | `05a9c57c44a2516e66d8dc3d845da302c0571b41` |
| `config_hash` | `ed033d420e8e94ce46363fabb12a8a34ab0e5cd538b60700046e9d9083d3debc` |
| `environment` | conda `moe`；Python 3.12.13；torch 2.9.0+cu128；CUDA 12.8 |
| `gpu` | RTX 3090 ×3；launch preflight各15 MiB、0% utilization |
| `start` | 2026-07-13 23:58:08 +08:00 |
| `finish` | 2026-07-14 00:06:39 +08:00 |
| `output` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_five_profile_extension_{a,b,c}` |

正式命令：

```bash
GPU_IDS="0 1 2" bash scripts/remote/run_stage_c_five_profile_extension.sh
```

首次`4846d7c` launch因ETTh1 adapter registry缺失而在training前退出；修复commit `05a9c57`经过ETTh1
1-batch remote smoke后重启。正式矩阵按每GPU固定worker运行；Phase A/B/C分别为6/4/4 runs。
