# Phase5 StageB B12-STBO Small Gate Launch Record

## Status

| Field | Value |
| --- | --- |
| `candidate_id` | `B12-STBO` |
| `current_step` | StageB Step 8 remote training |
| `status` | `remote_small_gate_running` |
| `launch_time` | `2026-07-08T18:13:42+08:00` |
| `remote_host` | `529_Lab-3090` / `star3090` |
| `local_commit` | `ae86f24 chore: add b12 stbo remote gate` |
| `remote_commit` | `ae86f24e1da4e3d91f7fae3df9f732c27c282e09` |
| `remote_repo` | `/tmp/yingch/projects/FATST_b12_stbo_ae86f24` |
| `output_root` | `/tmp/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b12_stbo_small_gate` |
| `launch_pid` | `3816978` |
| `launch_log` | `/tmp/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b12_stbo_small_gate/_launch_b12_stbo.log` |

## Quota Workaround

Remote preflight showed `/home` user quota was over the hard limit:

```text
/dev/sdb3    220G*   200G    220G    3天
```

`/home/yingch/projects/FATST` could not write `.git/index.lock`, so normal `git pull` in the persistent repo was
blocked by quota. To keep the launch reproducible while avoiding manual source copying, the experiment uses a fresh
Git clone under `/tmp` and writes outputs under `/tmp`.

Future sync command should set:

```bash
REMOTE_OUTPUT_ROOT=/tmp/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b12_stbo_small_gate \
ANALYSIS_ROOT=analysis/phase5_stage_b_b12_stbo_small_gate_20260708 \
bash scripts/sync_phase5_stage_b_b12_stbo_small_gate_results.sh
```

After launch, `/home/yingch/exp_outputs` was cleared by user request:

| Time | Path | Before | After | Quota After |
| --- | --- | ---: | ---: | ---: |
| `2026-07-08T18:15:44+08:00` | `/home/yingch/exp_outputs` | `112G` | `4.0K` | `109G / 200G` |

The active B12 run was not affected because it writes under `/tmp/yingch/exp_outputs/...`.

## GPU Preflight

At preflight, all three GPUs were idle:

| GPU | Model | Memory Used | Memory Free | Util |
| ---: | --- | ---: | ---: | ---: |
| 0 | NVIDIA GeForce RTX 3090 | `18 MiB` | `24107 MiB` | `0%` |
| 1 | NVIDIA GeForce RTX 3090 | `18 MiB` | `24107 MiB` | `0%` |
| 2 | NVIDIA GeForce RTX 3090 | `18 MiB` | `24107 MiB` | `0%` |

Initial process check after launch:

| GPU | PID | Process | Used Memory |
| --- | ---: | --- | ---: |
| 0 | `3816997` | `/home/yingch/.conda/envs/moe/bin/python` | `974 MiB` |
| 1 | `3816999` | `/home/yingch/.conda/envs/moe/bin/python` | `956 MiB` |
| 2 | `3817000` | `/home/yingch/.conda/envs/moe/bin/python` | `956 MiB` |

## Launch Matrix

| Field | Value |
| --- | --- |
| datasets | `Weather ETTm1 ETTh2` |
| arms | `a6_clean stbo_shared stbo_bank4 stbo_dct stbo_independent` |
| seed | `2021` |
| epochs / patience | `10 / 3` |
| checkpoint policy | `official-last` |
| `stbo_tile_len` | `48` |
| `stbo_rank` | `16` |
| `stbo_bank_count` | `4` |
| loss | `multi-prefix` |
| Python | `/home/yingch/.conda/envs/moe/bin/python` |

## Launch Command

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHON_BIN=/home/yingch/.conda/envs/moe/bin/python \
OUTPUT_ROOT=/tmp/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b12_stbo_small_gate \
GPU_IDS="0 1 2" \
DATASETS="Weather ETTm1 ETTh2" \
ARMS="a6_clean stbo_shared stbo_bank4 stbo_dct stbo_independent" \
nohup bash scripts/remote/run_phase5_stage_b_b12_stbo_small_gate.sh \
  > /tmp/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b12_stbo_small_gate/_launch_b12_stbo.log 2>&1 &
```

## Initial Log Check

The first three jobs started correctly:

```text
run_start=2026-07-08T18:13:42+08:00 arm=a6_clean dataset=Weather gpu=0
run_start=2026-07-08T18:13:42+08:00 arm=stbo_shared dataset=Weather gpu=1
run_start=2026-07-08T18:13:42+08:00 arm=stbo_bank4 dataset=Weather gpu=2
```

The three runs entered epoch 1 and printed training losses, so the launch was not an immediate startup failure.

## Gate Reminder

B12 cannot become a paper-core method merely by beating A6. The returned artifacts must show whether learned
`stbo_shared` / `stbo_bank4` beat `stbo_dct`, and whether any gain is not explained only by
`stbo_independent` capacity.
