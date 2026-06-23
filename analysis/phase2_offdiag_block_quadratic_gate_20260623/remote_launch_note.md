# Phase2-E1 Off-Diagonal Block Quadratic Remote Launch Note

更新时间：2026-06-23 21:18 +08:00

## Scope

[Decision] 已在 `529_Lab-3090` 启动 Phase2-E1 local objective gate。

- run name: `PatchEncoderOffdiagBlockQuadratic`
- step loss weighting: `offdiag_block_quadratic`
- datasets: `ETTm1`, `Weather`, `ETTh2`
- target horizons: `96,192,336,720`
- seed: `2021`
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase2_offdiag_block_quadratic_objective`

该轮检验：在 R.3 `prefix_risk` base loss 上加入 train-split off-diagonal block residual
penalty，是否能超过 R.3。

## Code State

- FATST commit: `b3ba31d`
- conda env: `moe`
- dataset root:
  `/home/yingch/dataset`

## GPU Check Before Launch

```text
0, NVIDIA GeForce RTX 3090, 24576, 9219, 14906, 29
1, NVIDIA GeForce RTX 3090, 24576, 10810, 13315, 0
2, NVIDIA GeForce RTX 3090, 24576, 10694, 13431, 0
```

[Decision] 选择 GPU `1` 和 `2`。两张卡已有约 10.7GB 占用，但仍有约 13GB 空闲；
GPU `0` 继续避开。

## Launch Command

```bash
mkdir -p /home/yingch/exp_outputs/r-2026-fatst/phase2_offdiag_block_quadratic_objective/_logs
cd /home/yingch/projects/FATST
GPU_IDS="1 2" \
OUTPUT_ROOT="/home/yingch/exp_outputs/r-2026-fatst/phase2_offdiag_block_quadratic_objective" \
nohup bash scripts/remote/run_phase2_offdiag_block_quadratic_gate.sh \
  > /home/yingch/exp_outputs/r-2026-fatst/phase2_offdiag_block_quadratic_objective/_logs/phase2_offdiag_block_quadratic_outer.log 2>&1 &
```

- wrapper PID: `3337971`
- active train PIDs observed: `3338222`, `3338223`

## Initial Progress

At `2026-06-23T21:18+08:00`:

- `ETTm1`: reached `epoch=3/100`;
- `Weather`: reached `epoch=1/100`;
- `ETTh2`: queued until one GPU frees;
- no immediate `Traceback`, `RuntimeError`, or CUDA OOM was observed.

## Follow-Up

Check logs:

```bash
ssh 529_Lab-3090 'tail -40 /home/yingch/exp_outputs/r-2026-fatst/phase2_offdiag_block_quadratic_objective/_logs/phase2_offdiag_block_quadratic_gate/PatchEncoderOffdiagBlockQuadratic_ETTm1_mixed_seed2021.log'
```

After completion, sync:

```bash
bash scripts/sync_phase2_offdiag_block_quadratic_results.sh
```
