# Phase5 StageB B9-FSN-SCF Small Gate Launch Record

## Run Identity

| Field | Value |
| --- | --- |
| `candidate_id` | `B9-FSN-SCF` |
| `stage_step` | Step 8 remote small gate |
| `launch_time` | 2026-07-07T16:51:45+08:00 |
| `remote_host` | `529_Lab-3090` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `git_commit` | `540cbde83899867245952da6d03152ad5ad98816` |
| `conda_env` | `moe` |
| `dataset_root` | `/home/yingch/dataset` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b9_fsn_scf_small_gate` |
| `checkpoint_policy` | `official-last` |

## GPU Preflight

At launch, `nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu` returned:

| GPU | Name | Total MiB | Used MiB | Free MiB | Util % |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | NVIDIA GeForce RTX 3090 | 24576 | 18 | 24107 | 0 |
| 1 | NVIDIA GeForce RTX 3090 | 24576 | 18 | 24107 | 0 |
| 2 | NVIDIA GeForce RTX 3090 | 24576 | 18 | 24107 | 0 |

## Launch Command

Remote command:

```bash
cd /home/yingch/projects/FATST
nohup bash scripts/remote/run_phase5_stage_b_b9_fsn_scf_small_gate.sh \
  > /home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b9_fsn_scf_small_gate/_logs/launcher_YYYYMMDD_HHMMSS.log \
  2>&1 &
```

Returned PID:

```text
b9_small_gate_pid=1883897
```

## Matrix

Arms:

- `a6_clean`: `readout_mode=learned-basis-forecast-operator`
- `b9_fsn_scf`: `readout_mode=stage-native-coefficient-field`
- `b9_no_stage`: `readout_mode=stage-native-coefficient-field-no-stage`

Datasets and ordering:

```text
Weather ETTm1 ETTh2
```

The runner uses dataset-major ordering and GPU round-robin over `0 1 2`, so the first three slow Weather runs are
distributed across all three GPUs instead of stacking on one GPU.

## Initial Runtime Check

Initial launcher log showed:

```text
run_start arm=a6_clean dataset=Weather gpu=0
run_start arm=b9_fsn_scf dataset=Weather gpu=1
run_start arm=b9_no_stage dataset=Weather gpu=2
```

Initial active GPU processes used approximately `974-1016 MiB` per process.

## Expected Artifacts

Each run should export:

- `effective_config.json`
- `environment.json`
- `training_log.csv`
- `checkpoint.pt`
- `model_diagnostics.json`
- `metrics_by_target_horizon.csv`
- `metrics_by_segment.csv`
- `predictions_test.npz`

After completion, sync with:

```bash
bash scripts/sync_phase5_stage_b_b9_fsn_scf_small_gate_results.sh
```
