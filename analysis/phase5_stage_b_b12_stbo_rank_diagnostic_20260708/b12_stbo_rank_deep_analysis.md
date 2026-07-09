# Phase5 StageB B12-STBO Rank Diagnostic Deep Analysis

## Scope

This report analyzes the repaired B12-STBO rank/capacity diagnostic.

| Field | Value |
| --- | --- |
| `candidate_id` | `B12-STBO` |
| `current_step` | StageB Step 9-10 |
| `remote_output_root` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b12_stbo_rank_diagnostic` |
| `local_analysis_root` | `analysis/phase5_stage_b_b12_stbo_rank_diagnostic_20260708` |
| valid configs | `L48-R32`, `L120-R64`, `L144-R128`, `L360-R256_capacity_probe` |
| invalid config | `L96-R64`, excluded because `96` does not divide `720` |
| arms | `stbo_shared`, `stbo_bank4`, `stbo_dct`, `stbo_independent` |
| datasets | ETTh2, ETTm1, Weather |
| horizons | 96, 192, 336, 720 |

The diagnostic is complete after repair:

```text
4 configs * 4 arms * 3 datasets = 48 metrics files
```

A6 was not rerun. Comparisons use the validated clean A6 anchor from the B12 small gate, which was already shown to
match the clean A6 rerun exactly.

## Executive Decision

[Decision] `rank_capacity_repair_insufficient`.

The rank diagnostic partially supports the user's concern: the first `rank=16` STBO gate was capacity-limited. Raising
local rank and tile length substantially recovers performance, especially in the `L360-R256` capacity probe.

However, B12-STBO still does not pass as a paper-core method:

1. no learned shared/bank STBO beats A6 overall;
2. the best learned shared/bank arm is still `+0.33%` MSE vs A6;
3. the best near-A6 arm is `stbo_independent`, not shared/bank STBO;
4. `stbo_bank4` keeps near-uniform bank entropy across all configs;
5. the only 256-rank recovery config uses `tile_len=360`, leaving only two future tiles, so it weakens the native
   stage-local multi-horizon narrative.

Therefore: rank/capacity was a real confound, but fixing rank does not rescue the current B12 method.

## Overall Result vs A6

Mean MSE relative to A6:

| Config | Shared | Bank4 | DCT | Independent |
| --- | ---: | ---: | ---: | ---: |
| `L48-R32` | `+0.47%` | `+1.08%` | `+0.89%` | `+0.54%` |
| `L120-R64` | `+0.42%` | `+1.37%` | `+1.14%` | `+0.64%` |
| `L144-R128` | `+0.84%` | `+1.37%` | `+0.97%` | `+0.51%` |
| `L360-R256` | `+0.33%` | `+1.01%` | `+1.31%` | `+0.014%` |

Key reading:

- Increasing rank does help relative to the original `L48-R16` gate, but not enough for shared/bank STBO.
- `L360-R256:stbo_independent` nearly matches A6, with `+0.014%` mean MSE and `4/12` wins.
- `L360-R256:stbo_shared` is the best learned shared/bank candidate, but still `+0.33%` mean MSE vs A6.

## Dataset Pattern

### Best learned shared/bank arm

The best learned shared/bank arm is `L360-R256:stbo_shared`:

| Dataset | Mean MSE vs A6 | Wins |
| --- | ---: | ---: |
| ETTh2 | `+0.98%` | `0/4` |
| ETTm1 | `-0.094%` | `4/4` |
| Weather | `+0.097%` | `0/4` |
| ALL | `+0.33%` | `4/12` |

It is useful evidence: high-rank shared STBO can beat A6 on ETTm1. But it does not generalize to ETTh2 or Weather.

### Best capacity probe

`L360-R256:stbo_independent`:

| Dataset | Mean MSE vs A6 | Wins |
| --- | ---: | ---: |
| ETTh2 | `-0.059%` | `2/4` |
| ETTm1 | `-0.043%` | `2/4` |
| Weather | `+0.145%` | `0/4` |
| ALL | `+0.014%` | `4/12` |

This proves the low-rank STBO failure was not purely an architectural impossibility: a high-rank tiled operator can
almost recover A6. But the winning form is independent/capacity-heavy, not a clean shared/bank stage-local method.

## Per-Horizon Evidence for `L360-R256`

| Arm | ETTh2 | ETTm1 | Weather |
| --- | --- | --- | --- |
| `shared` | `+0.957/+1.042/+0.875/+1.048%` | `-0.137/-0.050/-0.023/-0.165%` | `+0.072/+0.086/+0.078/+0.152%` |
| `bank4` | `+2.192/+2.693/+2.315/+3.039%` | `+0.546/+0.549/+0.514/+0.327%` | `-0.155/-0.112/-0.023/+0.273%` |
| `independent` | `+0.034/-0.233/-0.570/+0.534%` | `+0.024/-0.005/+0.010/-0.201%` | `+0.118/+0.105/+0.096/+0.262%` |
| `DCT` | `+2.441/+1.977/+2.929/+4.573%` | `+0.998/+0.672/+0.721/-0.050%` | `+0.170/+0.157/+0.488/+0.685%` |

Values are H96/H192/H336/H720 relative MSE vs A6.

Interpretation:

- `shared` is an ETTm1-only positive result.
- `bank4` is Weather-short-horizon only.
- `independent` is the only broadly near-A6 operator, but it is not the intended shared/bank mechanism.
- DCT is weak at `L360-R256`, so beating DCT here is not sufficient mechanism evidence.

## DCT Control Reading

Learned shared STBO beats same-rank DCT in several configs:

| Config | Shared vs DCT | Wins |
| --- | ---: | ---: |
| `L48-R32` | `-0.41%` | `10/12` |
| `L120-R64` | `-0.70%` | `10/12` |
| `L144-R128` | `-0.12%` | `6/12` |
| `L360-R256` | `-0.96%` | `12/12` |

This is better than the first `rank=16` gate, where learned and DCT were nearly tied. But the mechanism gate still
does not pass because A6 remains stronger overall. A method cannot claim novelty only by beating a weak DCT control
while failing the accepted carrier.

## Bank Specialization

`stbo_bank4` remains inactive:

| Config | Mean Bank Entropy |
| --- | ---: |
| `L48-R32` | `0.999702` |
| `L120-R64` | `0.999908` |
| `L144-R128` | `0.999978` |
| `L360-R256` | `0.999988` |

This is almost maximum entropy. Tile-bank logits stay close to uniform mixture. Therefore the bank route does not
learn future-stage specialization even when rank increases.

## Capacity and Parameter Confound

The rank diagnostic confirms capacity matters, but not in a paper-favorable way.

Parameter deltas vs A6:

| Config/Arm | ETTh2 | ETTm1 | Weather |
| --- | ---: | ---: | ---: |
| `L360-R256 shared` | `+17.8%` | `-3.0%` | `+23.7%` |
| `L360-R256 independent` | `+23.3%` | `+7.4%` | `+25.2%` |
| `L360-R256 bank4` | `+34.2%` | `+28.3%` | `+28.1%` |

So when STBO nearly catches A6, it often uses equal-or-larger capacity. The only near-tie is therefore not a clean
parameter-efficient architectural gain.

## Paper-Story Assessment

B12's intended paper story was:

```text
replace full-720 step basis with native stage/tile-local basis operators
```

The rank diagnostic modifies the story:

- [Supported] Low rank was a real bottleneck in the first gate.
- [Supported] Local/tiled factorization can approximate A6 when rank and tile length are large enough.
- [Not supported] Shared/bank STBO is a stronger paper-core method than A6.
- [Not supported] Banked basis learns future-stage specialization.
- [Weak] `L360-R256` has only two tiles, so it is closer to a coarse segmented full-trajectory head than a native
  multi-horizon stage-local operator.

## Failure Attribution

- `hypothesis_false`: not fully proven. High-rank tiled operators can approach A6.
- `readout_or_head_design_wrong`: yes for current B12. The intended shared/bank readout does not provide robust gains.
- `optimization_or_numeric_pathology`: no obvious divergence; failure is structural/performance, not numeric blow-up.
- `capacity_control_explains`: partially. Rank/capacity explains much of the original failure, but not enough to create
  a valid method.
- `generic_basis_control_explains`: weaker than before for shared, because shared beats DCT; still insufficient because
  A6 gate fails.
- `direction_level_rejection`: no. This blocks current B12-STBO, not all architecture search.

## Decision

[Decision] `B12-STBO blocked_by_rank_diagnostic`.

Do not promote B12-STBO to paper-core. Do not launch a full matrix.

Recommended rollback:

```text
StageB -> Step 2/3 architecture search
```

If this line is revisited, it should not be another rank sweep. It would need a different operator design that preserves
A6's full-basis expressiveness while giving a cleaner native multi-horizon interface. The current shared/bank tiled
basis family is not sufficient.
