# Phase5 StageB B13-FUCO-B2 Remote Launch Record

## Launch

| Field | Value |
| --- | --- |
| `current_step` | StageB Step 2/3 intervention-point repair |
| `role` | diagnostic-only；not end-to-end forecasting performance |
| `remote_host` | `529_Lab-3090` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `git_commit` | `013dd350ec45525ffd625a13e87ee239b143238e` |
| `conda_env` | `moe` |
| `torch` | `2.9.0+cu128` |
| `gpu` | GPU 0；NVIDIA GeForce RTX 3090 |
| `start` | `2026-07-10T11:44:33+08:00` |
| `finish` | `2026-07-10T11:46:12+08:00` |
| `remote_output_root` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b13_future_unit_hidden_probe` |
| `local_analysis_root` | `analysis/phase5_stage_b_b13_future_unit_hidden_composition_20260710/` |

启动前 `nvidia-smi`：

```text
0, NVIDIA GeForce RTX 3090, 24576, 15, 24110, 0
1, NVIDIA GeForce RTX 3090, 24576, 15, 24110, 0
2, NVIDIA GeForce RTX 3090, 24576, 15, 24110, 0
```

三张 GPU 均无 compute process；GPU 0 启动前 memory used `15 MiB`、free `24110 MiB`、utilization `0%`。

## Effective Probe Matrix

```text
memory_source=hidden
datasets=ETTh2 ETTm1 Weather
unit_sizes=180 240
seeds=2021 2022 2023
arms=parallel_no_transition prefix_causal_composed
state_dim=64
epochs=20
train/val/test rows=4096/1024/1024
batch_size=256
extract_batch_size=32
device=cuda
```

总计 `3 x 2 x 3 x 2 = 36` runs，形成 `18` 个 same-seed parameter-matched pairs。

## Checkpoint Provenance

Remote root：

```text
/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b13_probe_inputs/a6_clean
```

| Dataset | SHA256 |
| --- | --- |
| ETTh2 | `04b60bb074e17b80b9f52ecbbf0cedac36a718549c69d1b445d4db4079bb10e6` |
| ETTm1 | `79b4224c9c110076f824e3dbdafb256b10ec913bf8e54727cf21a7267e514754` |
| Weather | `bfb8b254c0437e708b515af5b49c52e75f503a3c39f8bc6303b9c46688a51b83` |

三份 remote checksum 与 local clean A6 checkpoints 完全一致。

## Command

```bash
GPU_ID=0 bash scripts/remote/run_phase5_stage_b_b13_hidden_memory_probe.sh
```

runner 记录于 `launch.log`，GPU inventory 记录于 `gpu_preflight.txt`。完成后 GPU 0 回到 `15 MiB` used、
`0%` utilization。

## Completion

远程 runner 正常输出：

- `b13_future_unit_probe_runs.csv`；
- `b13_future_unit_probe_comparisons.csv`；
- `b13_future_unit_probe_summary.csv`；
- `b13_future_unit_composition_report.md`。

结果通过 `scripts/sync_phase5_stage_b_b13_hidden_memory_probe_results.sh` 同步到本目录。
