# B14 Prerequisite Contextual Patch Encoder Launch Record

## Launch

| Field | Value |
| --- | --- |
| date | `2026-07-10 13:01:39 +08:00` |
| remote | `529_Lab-3090` |
| repo | `/home/yingch/projects/FATST` |
| commit | `1d5bdf66b494ced30cba9e359620d61012e42448` |
| conda env | `moe` |
| output root | `/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b14_contextual_patch_encoder_gate` |
| GPUs | `0,1,2` |
| preflight | all three RTX 3090 GPUs used `15 MiB`, free `24110 MiB`, no compute processes |
| launcher pid | `2768865` |

## Matrix

```text
datasets = Weather ETTm1 ETTh2
arms = cpe_p16s8 cpe_p48s24
seeds = 2021
epochs = 20
target prefixes = 96,192,336,720
checkpoint policy = best-val
```

Total small-gate runs：`2 arms × 3 datasets × 1 seed = 6`。

## Command

```bash
OUT=/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b14_contextual_patch_encoder_gate
nohup env \
  OUTPUT_ROOT="$OUT" \
  GPU_IDS="0 1 2" \
  DATASETS="Weather ETTm1 ETTh2" \
  ARMS="cpe_p16s8 cpe_p48s24" \
  SEEDS="2021" \
  EPOCHS=20 \
  bash scripts/remote/run_phase5_stage_b_b14_contextual_patch_encoder_gate.sh \
  > "$OUT/launcher.log" 2>&1 < /dev/null &
```

## Initial Runtime State

- GPU 0：Weather `P16-S8`，约 `10.9 GiB`；
- GPU 1：Weather `P48-S24`，约 `2.9 GiB`；
- GPU 2：ETTm1 `P16-S8`，约 `4.0 GiB`；
- no OOM / traceback；queue filler 将后续启动 ETTm1 `P48-S24`、ETTh2 两个 arms。

## Decision Boundary

本记录只证明 launch/runtime validity，不是 effectiveness evidence。必须等 6-run metrics 返回并通过
`scripts/analyze_phase5_stage_b_b14_contextual_patch_encoder_gate.py` 后，才允许选择 confirmation arm。

## Completion

- completed：`2026-07-10 15:07:57 +08:00`；
- artifacts：`6/6` metrics complete；
- `cpe_p16s8`：overall `+4.135%`，`1/12` wins；
- `cpe_p48s24`：overall `+4.799%`，`0/12` wins；
- decision：no confirmation arm；rollback Step 5/6 to hierarchical patch-memory repair。
