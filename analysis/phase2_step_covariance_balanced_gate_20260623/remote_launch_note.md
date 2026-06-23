# Phase2-C.2 Step-Covariance Balanced Remote Launch

## Launch

- launch time: `2026-06-23T17:13:25+08:00`
- remote host: `529_Lab-3090`
- remote project: `/home/yingch/projects/FATST`
- git commit: `f0f5d41501c25063c28ff226707798546f13a4fc`
- conda env: `moe`
- PID reported by launcher: `939507`
- run name: `PatchEncoderStepCovarianceBalanced`
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase2_step_covariance_balanced_objective`
- outer log:
  `/home/yingch/exp_outputs/r-2026-fatst/phase2_step_covariance_balanced_objective/_logs/phase2_step_covariance_balanced_outer.log`

## Command

```bash
GPU_IDS="1 2" nohup bash scripts/remote/run_phase2_step_covariance_balanced_gate.sh \
  > /home/yingch/exp_outputs/r-2026-fatst/phase2_step_covariance_balanced_objective/_logs/phase2_step_covariance_balanced_outer.log 2>&1 &
```

## GPU Snapshot Before Launch

```text
0, NVIDIA GeForce RTX 3090, 24576, 3964, 20161, 7
1, NVIDIA GeForce RTX 3090, 24576, 18, 24107, 0
2, NVIDIA GeForce RTX 3090, 24576, 1334, 22791, 2
```

Selected GPUs: `1`, `2`.

Reason: GPU 1 and 2 had low memory occupancy; GPU 0 was avoided according to
project policy.

## Runner Configuration

```text
dataset_root=/home/yingch/dataset
output_root=/home/yingch/exp_outputs/r-2026-fatst/phase2_step_covariance_balanced_objective
gpu_ids=1 2
seed=2021
epochs=100
target_horizons=96,192,336,720
run_name=PatchEncoderStepCovarianceBalanced
step_loss_weighting=step_covariance_balanced
step_loss_alpha=0.5
step_covariance_beta=0.5
step_covariance_eta=0.5
step_covariance_eps=1e-6
```

## Initial Progress

```text
run_start=2026-06-23T17:13:25+08:00 model=PatchEncoderStepCovarianceBalanced dataset=ETTm1 gpu=1
run_start=2026-06-23T17:13:25+08:00 model=PatchEncoderStepCovarianceBalanced dataset=Weather gpu=2
```

GPU snapshot after launch:

```text
0, 3964, 20161, 5
1, 1910, 22215, 96
2, 5934, 18192, 97
```

