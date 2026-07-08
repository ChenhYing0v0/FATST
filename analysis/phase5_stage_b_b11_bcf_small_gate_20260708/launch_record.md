# Phase5 StageB B11-BCF Small Gate Launch Record

## Scope

| Field | Value |
| --- | --- |
| `candidate_id` | `B11-ESA/BCF` |
| `current_step` | StageB Step 8 remote small gate |
| `local_commit` | `16367b2 feat: add b11 basis-conditioned coefficient field` |
| `remote_host` | `529_Lab-3090` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `remote_output_root_first_attempt` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b11_bcf_small_gate` |
| `remote_output_root_retry` | `/tmp/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b11_bcf_small_gate` |
| `checkpoint_policy` | `official-last` |
| `seed` | `2021` |
| `driver_pid_first_attempt` | `3599363` |
| `driver_pid_tmp_retry` | `3611775` |

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

## Launch Failure And Retry

The first `/home` output-root launch failed with `OSError: [Errno 122] Disk quota exceeded`.

Read-only quota check:

```text
Disk quotas for user yingch:
/dev/sdb3 space 220G*, quota 200G, limit 220G
```

The failed B11 output itself was only about `294M`; the failure came from account-level `/home` quota saturation.
`/tmp/yingch/exp_outputs/r-2026-fatst` was writable and had about `75G` available, so a retry was attempted there.

The second retry still failed because `conda run` itself tried to create a temporary wrapper and hit the same quota
path before training. The runner was updated to support direct Python execution with:

```bash
PYTHON_BIN=/home/yingch/.conda/envs/moe/bin/python
OUTPUT_ROOT=/tmp/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b11_bcf_small_gate
```

The next valid launch should use both variables above.

## Next Action

Use a long progress interval before checking again. When all artifacts return, run:

```bash
REMOTE_OUTPUT_ROOT=/tmp/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b11_bcf_small_gate \
bash scripts/sync_phase5_stage_b_b11_bcf_small_gate_results.sh
```

The resulting report should decide whether B11-BCF beats both required controls:

- `b11_bcf` vs `b11_no_basis`;
- `b11_bcf` vs `b11_constant_slot`.
