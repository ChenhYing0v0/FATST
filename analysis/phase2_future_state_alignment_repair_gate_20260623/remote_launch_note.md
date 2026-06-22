# Phase2-R.1 Remote Launch Note

## Launch

- local commit: `2868fbf`
- remote synced commit: `2868fbf`
- remote host: `529_Lab-3090`
- remote project dir: `/home/yingch/projects/FATST`
- conda env: `moe`
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase2_future_state_alignment_repair`
- run name: `PatchEncoderFutureStateAlignmentConfWeighted`
- datasets order: `ETTm1 Weather ETTh2`
- horizons: `96,192,336,720`
- epochs: `100`
- selected GPU: `2`
- launch PID: `2518493`

Command:

```bash
GPU_IDS="2" EPOCHS="100" KEEP_HEAVY_ARTIFACTS="0" \
  bash scripts/remote/run_phase2_future_state_alignment_repair_gate.sh
```

## GPU Check Before Launch

Remote `nvidia-smi` before launch:

| GPU | memory.used MiB | memory.free MiB | utilization % | decision |
| ---: | ---: | ---: | ---: | --- |
| 0 | 5225 | 18900 | 2 | external occupancy; avoid |
| 1 | 10744 | 13381 | 0 | not treated as free two-card capacity |
| 2 | 18 | 24107 | 0 | selected |

## First Runtime Check

The first post-launch check showed:

- active dataset: `ETTm1` (`1/3`);
- GPU2 memory used: `2112 MiB`;
- GPU2 utilization: `91%`;
- remote log had entered `run_start` for `ETTm1`.

## Monitoring Note

A later long-sleep progress check failed at the SSH connection layer with:

```text
ssh: connect to host 10.15.90.61 port 3022: Operation not permitted
```

This was a local SSH connectivity failure, not evidence of remote training
failure. The launched process had already detached with `PPID=1`, so result
collection should resume once SSH connectivity is available again.

## Follow-Up Monitoring Update

Later local checks still failed at the SSH connection layer with the same
message:

```text
ssh: connect to host 10.15.90.61 port 3022: Operation not permitted
```

Commit `da7a959` adds per-epoch `epoch_progress` logging and
`scripts/remote/check_phase2_future_state_alignment_repair_progress.sh` for
future remote runs. The already-launched process was started from commit
`2868fbf`, so it will not automatically emit the new per-epoch progress lines.
After SSH recovers, use the outer log, dataset-level logs, and artifact
existence to determine whether PID `2518493` completed.
