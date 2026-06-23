# Phase2-C Region-Balanced Remote Launch Note

## Launch

- launch time: `2026-06-23T13:07:06+08:00`
- host: `529_Lab-3090`
- remote project: `/home/yingch/projects/FATST`
- git commit: `d67675fb12ecd24e08a103bdfbe086d1b78a16cd`
- conda env: `moe`
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase2_region_balanced_objective`
- run name: `PatchEncoderRegionBalanced`
- PID reported by launcher: `68178`

## GPU Check Before Launch

`nvidia-smi` summary before launch:

| GPU | Memory Used MiB | Memory Free MiB | Utilization |
| ---: | ---: | ---: | ---: |
| 0 | `2939` | `21186` | `0%` |
| 1 | `8696` | `15429` | `100%` |
| 2 | `8690` | `15435` | `45-100%` |

[Decision] GPU 0 was avoided because this project treats GPU 0 as higher
kill-risk. GPUs 1 and 2 already had running jobs but retained about 15GB free
memory each. The run was launched on GPUs `1 2` because the user allows
sharing occupied GPUs when memory safety is acceptable.

## Command

```bash
cd /home/yingch/projects/FATST
GPU_IDS="1 2" KEEP_HEAVY_ARTIFACTS=0 \
nohup bash scripts/remote/run_phase2_region_balanced_gate.sh \
  > /home/yingch/exp_outputs/r-2026-fatst/phase2_region_balanced_objective/_logs/phase2_region_balanced_outer.log 2>&1 &
```

## Initial Progress Check

At `2026-06-23T13:10:12+08:00`:

- `ETTm1`: running, `epoch=1/100`, ETA finish approximately
  `2026-06-23T16:02:40+08:00`;
- `Weather`: running, epoch not yet reported;
- `ETTh2`: queued;
- GPU memory after launch:
  - GPU 1: `10588 MiB` used, `13538 MiB` free;
  - GPU 2: `13290 MiB` used, `10836 MiB` free.

## Follow-Up

Use:

```bash
ssh 529_Lab-3090 'cd /home/yingch/projects/FATST && PID=68178 bash scripts/remote/check_phase2_region_balanced_progress.sh'
```

When complete, sync locally with:

```bash
bash scripts/sync_phase2_region_balanced_results.sh
```
