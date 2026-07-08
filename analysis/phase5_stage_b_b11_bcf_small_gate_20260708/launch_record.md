# Phase5 StageB B11-BCF Small Gate Launch Record

## Scope

| Field | Value |
| --- | --- |
| `candidate_id` | `B11-ESA/BCF` |
| `current_step` | StageB Step 8 remote small gate |
| `local_commit` | `16367b2 feat: add b11 basis-conditioned coefficient field` |
| `remote_host` | `529_Lab-3090` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `remote_output_root` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b11_bcf_small_gate` |
| `checkpoint_policy` | `official-last` |
| `seed` | `2021` |
| `driver_pid` | `3599363` |

## Matrix

Required arms:

- `a6_clean`: `learned-basis-forecast-operator`;
- `b11_bcf`: `basis-conditioned-coefficient-field`;
- `b11_no_basis`: `basis-conditioned-coefficient-field-no-basis`;
- `b11_constant_slot`: `basis-conditioned-coefficient-field-constant-slot`.

Datasets:

- `Weather`;
- `ETTm1`;
- `ETTh2`.

Horizons:

- `96`;
- `192`;
- `336`;
- `720`.

Optional control not launched in this required gate:

- `b11_shuffled_basis`.

## Scheduling

Runner:

- `scripts/remote/run_phase5_stage_b_b11_bcf_small_gate.sh`.

Scheduling policy:

- `GPU_IDS="0 1 2"`;
- dataset-major order: `Weather ETTm1 ETTh2`;
- arm order: `a6_clean b11_bcf b11_no_basis b11_constant_slot`.

This spreads slower datasets across GPUs first and avoids stacking all Weather jobs on GPU0.

## GPU Preflight

Before launch:

```text
0, NVIDIA GeForce RTX 3090, 24576, 18, 24107, 0
1, NVIDIA GeForce RTX 3090, 24576, 18, 24107, 0
2, NVIDIA GeForce RTX 3090, 24576, 18, 24107, 0
```

After launch:

```text
0, NVIDIA GeForce RTX 3090, 24576, 997, 23128, 65
1, NVIDIA GeForce RTX 3090, 24576, 1104, 23021, 34
2, NVIDIA GeForce RTX 3090, 24576, 1106, 23019, 34
```

## Initial Log Check

The driver log showed the first three Weather jobs active:

- `a6_clean` on Weather;
- `b11_bcf` on Weather;
- `b11_no_basis` on Weather.

All three entered training and were printing epoch-1 iteration logs.

## Next Action

Use a long progress interval before checking again. When all artifacts return, run:

```bash
bash scripts/sync_phase5_stage_b_b11_bcf_small_gate_results.sh
```

The resulting report should decide whether B11-BCF beats both required controls:

- `b11_bcf` vs `b11_no_basis`;
- `b11_bcf` vs `b11_constant_slot`.
