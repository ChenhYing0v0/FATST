# Phase2-D QDF Upstream Controls Remote Launch Note

更新时间：2026-06-23 19:43 +08:00

## Scope

[Decision] 已在 `529_Lab-3090` 启动 QDF upstream native reproduction 的 control gate。

- QDF modes: `META_TYPES="diag off_diag"`
- datasets: `ETTm1 Weather ETTh2`
- horizons: `96 192 336 720`
- seed: `2023`
- upstream model: `TQNet`
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_upstream_gate`

该轮将补齐 `meta_type=all` 的同协议对照，用于判断 full learned covariance 是否优于
diagonal-only 或 off-diagonal-only route。

## Code State

- FATST commit: `4f5f372`
- QDF upstream commit: `eb0693a`
- QDF source location:
  `/home/yingch/projects/QDF`

## GPU Check Before Launch

```text
0, NVIDIA GeForce RTX 3090, 5277, 18848, 30
1, NVIDIA GeForce RTX 3090, 18, 24107, 0
2, NVIDIA GeForce RTX 3090, 18, 24107, 0
```

[Decision] 选择 GPU `1` 和 `2`，避开 GPU `0`。

## Launch Command

```bash
cd /home/yingch/projects/FATST
git pull --ff-only
GPU_IDS="1 2" \
META_TYPES="diag off_diag" \
DATASETS="ETTm1 Weather ETTh2" \
HORIZONS="96 192 336 720" \
RERUN=0 \
NUM_WORKERS=0 \
OUTPUT_ROOT="/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_upstream_gate" \
nohup bash scripts/remote/run_phase2_qdf_upstream_gate.sh \
  > /home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_upstream_gate/_logs/phase2_qdf_upstream_controls_outer.log 2>&1 &
```

- remote wrapper PID: `2417682`
- runner PID: `2417795`

## Initial Progress Check

At `2026-06-23T19:42:37+08:00`:

- active jobs: `diag/ETTm1/h96`, `diag/ETTm1/h192`
- completed runs already present from `meta_type=all`: `12`
- metric files already present from `meta_type=all`: `12`
- GPU memory:

```text
0, NVIDIA GeForce RTX 3090, 5277, 18848, 27
1, NVIDIA GeForce RTX 3090, 392, 23733, 4
2, NVIDIA GeForce RTX 3090, 396, 23729, 5
```

[Fact] `diag/ETTm1/h96` and `diag/ETTm1/h192` have entered Meta Training Phase with
`Num Workers: 0`.

## Follow-Up

Check progress:

```bash
ssh 529_Lab-3090 'cd /home/yingch/projects/FATST && OUTPUT_ROOT=/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_upstream_gate bash scripts/remote/check_phase2_qdf_upstream_progress.sh'
```

After completion, sync and re-run the analyzer:

```bash
bash scripts/sync_phase2_qdf_upstream_results.sh
```
