# Phase2-D QDF Upstream Gate Remote Launch Note

更新时间：2026-06-23 18:27 +08:00

## Scope

[Decision] 已在 `529_Lab-3090` 启动 QDF upstream native reproduction 的第一轮最小 gate。

- QDF mode: `META_TYPES=all`
- datasets: `ETTm1 Weather ETTh2`
- horizons: `96 192 336 720`
- seed: `2023`
- upstream model: `TQNet`
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_upstream_gate`

该轮只收集 full-QDF (`meta_type=all`) 指标。最终 gate 仍需要后续补 `diag` control；
只有 `all` 结果本身不能判定 QDF mechanism pass。

## Code State

- FATST commit: `9217cb8`
- QDF upstream commit: `eb0693a`
- QDF source location:
  `/home/yingch/projects/QDF`

[Fact] 远程直接从 GitHub clone QDF 时遇到 TLS 中断，因此使用本地已审计的
`/tmp/fatst-qdf-audit` 通过 `rsync` 同步到 3090。QDF source 没有 vendor 到 FATST repo。

## Environment

- server: `529_Lab-3090`
- conda env: `moe`
- torch: `2.9.0+cu128`
- dataset root:
  `/home/yingch/dataset`

为满足 QDF upstream 顶层 imports，已在 `moe` 环境补装：

- `setproctitle`
- `reformer-pytorch`
- `torch-dct`
- `tensorboard`
- `torch-geometric`
- `ipython`
- `patool`
- `robustica`
- `PyWavelets`
- `numba`
- `pot`

[Fact] `cupy` 只在当前路径中用于 `cp.random.seed`。runner 已使用
`${OUTPUT_ROOT}/_shims/cupy.py` 提供 `random.seed` shim，避免安装重型 optional dependency。

## GPU Check Before Launch

```text
0, NVIDIA GeForce RTX 3090, 24576, 2649, 21476, 0
1, NVIDIA GeForce RTX 3090, 24576, 18, 24107, 0
2, NVIDIA GeForce RTX 3090, 24576, 1334, 22791, 0
```

[Decision] 选择 GPU `1` 和 `2`，避开 GPU `0`。

## Launch Command

```bash
cd /home/yingch/projects/FATST
GPU_IDS="1 2" \
META_TYPES="all" \
DATASETS="ETTm1 Weather ETTh2" \
HORIZONS="96 192 336 720" \
OUTPUT_ROOT="/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_upstream_gate" \
nohup bash scripts/remote/run_phase2_qdf_upstream_gate.sh \
  > /home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_upstream_gate/_logs/phase2_qdf_upstream_outer.log 2>&1 &
```

- launcher PID: `2181452`

## Initial Progress Check

At `2026-06-23T18:27:04+08:00`:

- active jobs: `ETTm1/h96`, `ETTm1/h192`
- completed runs: `0`
- metric files: `0`
- GPU memory:

```text
0, NVIDIA GeForce RTX 3090, 2649, 21476, 1
1, NVIDIA GeForce RTX 3090, 392, 23733, 1
2, NVIDIA GeForce RTX 3090, 1712, 22414, 8
```

[Fact] `ETTm1/h96` 已进入 Meta Training Phase，并输出到：

`/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_upstream_gate/_logs/phase2_qdf_upstream_gate/QDF_all_ETTm1_h96_seed2023.log`

## Follow-Up

使用以下命令检查进度：

```bash
ssh 529_Lab-3090 'cd /home/yingch/projects/FATST && OUTPUT_ROOT=/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_upstream_gate bash scripts/remote/check_phase2_qdf_upstream_progress.sh'
```

远程完成后同步：

```bash
bash scripts/sync_phase2_qdf_upstream_results.sh
```
