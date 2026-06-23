# Phase2-B Error-Process Decoder Remote Launch Note

## Launch

- launch time: `2026-06-23T11:29:53+08:00`
- remote host: `529_Lab-3090`
- remote project dir: `/home/yingch/projects/FATST`
- remote git commit: `b760573`
- output root: `/home/yingch/exp_outputs/r-2026-fatst/phase2_error_process_decoder`
- run name: `PatchEncoderErrorProcessDecoder`
- shell PID reported by SSH launch: `3846033`
- conda environment: `moe`

## Command

```bash
GPU_IDS="1 2" KEEP_HEAVY_ARTIFACTS=0 \
  nohup bash scripts/remote/run_phase2_error_process_decoder_gate.sh \
  > /home/yingch/exp_outputs/r-2026-fatst/phase2_error_process_decoder/_logs/phase2_error_process_decoder_outer.log \
  2>&1 &
```

## GPU Selection

Pre-launch GPU snapshot after code sync:

| GPU | Used MiB | Free MiB | Util |
| ---: | ---: | ---: | ---: |
| 0 | 4255 | 19870 | 1 |
| 1 | 8696 | 15429 | 100 |
| 2 | 8690 | 15435 | 55 |

GPU 0 had more free memory but is avoided because this project treats it as
higher kill-risk unless explicitly accepted. GPUs 1 and 2 had enough free
memory for this model family, so the gate was launched on `GPU_IDS="1 2"` even
though both cards already had active jobs.

Post-launch snapshot at `2026-06-23T11:30:53+08:00`:

| GPU | Used MiB | Free MiB | Util |
| ---: | ---: | ---: | ---: |
| 0 | 4255 | 19870 | 2 |
| 1 | 10706 | 13420 | 99 |
| 2 | 13738 | 10388 | 100 |

[Inference] The launched jobs added roughly 2.0 GiB on GPU1 and 5.0 GiB on
GPU2 at startup. Both cards retained more than 10 GiB free memory, so the
launch is memory-safe under the user's instruction that existing-task GPUs may
be used when memory headroom is sufficient.

## Matrix

- datasets order: `ETTm1 Weather ETTh2`
- target horizons: `96,192,336,720`
- epochs: `100`
- seed: `2021`
- first wave:
  - `ETTm1` on GPU1;
  - `Weather` on GPU2.
- queued:
  - `ETTh2`, to run after a first-wave dataset completes.

## Progress Check

Run on the remote server:

```bash
cd /home/yingch/projects/FATST
PID=3846033 bash scripts/remote/check_phase2_error_process_decoder_progress.sh
```

Sync and analyze locally after completion:

```bash
cd /Users/river/PaperResearch/Project/R_2026_FATST
bash scripts/sync_phase2_error_process_decoder_results.sh
```
