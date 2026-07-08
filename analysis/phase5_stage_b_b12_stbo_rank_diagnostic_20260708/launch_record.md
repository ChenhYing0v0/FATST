# Phase5 StageB B12-STBO Rank Diagnostic Launch Record

## Purpose

This diagnostic tests whether the failed B12-STBO small gate was primarily caused by the low local basis rank
(`tile_len=48`, `stbo_rank=16`).

The previous B12 result had two important confounds:

1. STBO local basis rank was much smaller than A6's `basis_rank=256`;
2. STBO parameter count was lower than A6, especially on ETTm1.

This launch therefore increases the effective local rank while keeping the DCT and independent-tile controls.

## Why Not Directly Set `stbo_rank=256` at `tile_len=48`

Current STBO uses local basis tensors:

```text
local_basis: [tile_len, stbo_rank]
tile_coeff: [B,C,tile_count,stbo_rank]
```

With `tile_len=48`, the local output matrix has linear rank at most `48`. Setting `stbo_rank=256` would be
overcomplete and non-identifiable, not 256 independent local dimensions. The implementation therefore requires:

```text
stbo_rank <= stbo_tile_len
```

To test higher rank honestly, this diagnostic increases `tile_len` together with `stbo_rank`.

## Launch Matrix

| Tag | `stbo_tile_len` | `stbo_rank` | Role |
| --- | ---: | ---: | --- |
| `l48_r32` | `48` | `32` | local-rank expansion while preserving the original 15-tile structure |
| `l96_r64` | `96` | `64` | medium-rank tile operator |
| `l144_r128` | `144` | `128` | high-rank tile operator; 5 future tiles |
| `l360_r256_capacity_probe` | `360` | `256` | capacity probe close to A6 rank; only 2 tiles, so not a preferred stage-local method |

Arms for each tag:

- `stbo_shared`;
- `stbo_bank4`;
- `stbo_dct`;
- `stbo_independent`.

Datasets:

- `Weather`;
- `ETTm1`;
- `ETTh2`.

The experiment intentionally does not rerun `a6_clean`. Analysis should compare against the already validated clean A6
anchor in `analysis/phase5_stage_b_b12_stbo_small_gate_20260708/` and
`analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/`.

## Remote Launch

| Field | Value |
| --- | --- |
| `status` | `running` |
| `remote_host` | `529_Lab-3090` / `star3090` |
| `launch_time` | `2026-07-08T18:54:00+08:00` |
| `launch_pid` | `3868765` |
| `remote_commit` | `f0eb7041ac0c2429499c474832cb81a30b4fd7ba` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b12_stbo_rank_diagnostic` |
| `launch_log` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b12_stbo_rank_diagnostic/_launch_rank_diagnostic.log` |
| `python` | `/home/yingch/.conda/envs/moe/bin/python` |

## GPU Preflight

At launch, GPUs 0/1/2 were idle:

| GPU | Model | Memory Used | Memory Free | Util |
| ---: | --- | ---: | ---: | ---: |
| 0 | NVIDIA GeForce RTX 3090 | `18 MiB` | `24107 MiB` | `0%` |
| 1 | NVIDIA GeForce RTX 3090 | `18 MiB` | `24107 MiB` | `0%` |
| 2 | NVIDIA GeForce RTX 3090 | `18 MiB` | `24107 MiB` | `0%` |

Initial check confirmed `l48_r32` started on all three GPUs:

| GPU | Arm | Dataset |
| ---: | --- | --- |
| 0 | `stbo_shared` | `Weather` |
| 1 | `stbo_bank4` | `Weather` |
| 2 | `stbo_dct` | `Weather` |

## Analysis Requirements

This diagnostic can only answer whether increasing local rank repairs the tested STBO implementation. It must not
promote B12 to paper-core unless all of the following hold:

1. learned STBO approaches or beats clean A6;
2. learned STBO beats same-rank DCT;
3. `stbo_bank4` shows non-uniform tile-bank specialization;
4. gains are not explained only by independent-tile capacity.
