# Phase5 StageB B12-STBO Diagnostic Report

`current_step`: StageB Step 2/3 problem-existence and feasibility diagnostic.

## Scope

[Fact] This diagnostic uses clean A6 checkpoints and train split labels. It does not train or evaluate a new model.

[Boundary] A positive reconstruction result is not a method result. It only decides whether B12 may enter Step 4-6 method design.

## 11-step Record

| Field | Content |
| --- | --- |
| `current_step` | StageB Step 2/3 B12-STBO diagnostic |
| `problem` | Can A6's full-720 step basis be replaced by a stage/tile-local subspace basis operator? |
| `existence_evidence` | A6 basis tile factorization, train-label tile factorization, coeff projection into tile subspaces |
| `idea` | Use shared/bank local basis tiles instead of full-720 step basis; short horizons activate only needed tiles |
| `theory_check` | Positive only if shared/bank tiles approach independent-tile upper bound and beat local DCT controls |
| `design` | Offline diagnostic, tile_len=48, gate_rank=16 |
| `narrative_gate` | not evaluated until Step 4-6 |
| `effectiveness_gate` | not applicable before implementation |
| `artifacts` | this analysis directory |
| `decision` | `diagnostic_not_enough_for_b12` |

## A6 Basis Tile Factorization

| Dataset | Shared | Bank4 | Independent | DCT | Shared Gap | Bank4 Gap | Bank4-DCT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | `0.481` | `0.505` | `0.572` | `0.444` | `0.091` | `0.067` | `0.061` |
| ETTm1 | `0.436` | `0.466` | `0.534` | `0.411` | `0.098` | `0.068` | `0.054` |
| Weather | `0.400` | `0.438` | `0.521` | `0.357` | `0.121` | `0.083` | `0.081` |

## Train-Label Tile Factorization

| Dataset | Shared | Bank4 | Independent | DCT | Shared Gap | Bank4 Gap | Bank4-DCT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | `0.975` | `0.975` | `0.975` | `0.973` | `0.000` | `0.000` | `0.002` |
| ETTm1 | `0.986` | `0.986` | `0.986` | `0.986` | `0.000` | `0.000` | `0.000` |
| Weather | `0.927` | `0.936` | `0.952` | `0.925` | `0.025` | `0.016` | `0.011` |

## Coeff Projection Into Tile Subspaces

| Dataset | Adjacent Cos | Far Cos | Distance Spearman | Projection Entropy | Output Entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | `0.283` | `0.124` | `-0.678` | `0.952` | `0.960` |
| ETTm1 | `0.030` | `0.164` | `-0.012` | `0.955` | `0.953` |
| Weather | `0.035` | `0.047` | `-0.214` | `0.879` | `0.892` |

## Gate Evaluation

- Basis bank pass: `none`.
- Basis shared pass: `none`.
- Label bank pass: `none`.
- Label shared pass: `ETTh2, ETTm1, Weather`.
- Coeff projection pass: `ETTh2`.
- Independent-tile-only risk: `none`.
- Local DCT control risk: `ETTh2, ETTm1, Weather`.

## Interpretation

[Fact] `shared_local_energy` fits one local basis `U[L,r]` across all future tiles.

[Fact] `bank4_energy` fits four local basis banks and is the main B12-B feasibility proxy.

[Fact] `independent_tile_energy` is an upper bound and is not sufficient by itself, because it may indicate a segmented Direct head.

[Fact] `local_dct_energy` is the generic smoothness control. If it matches shared/bank basis, B12 is not distinct enough from fixed local spectral bases.

[Fact] Coeff projection compares how the same A6 `coeff` is used across tile row-spaces. Adjacent > far supports tile-subspace structure.

[Decision] `diagnostic_not_enough_for_b12`.

## Failure Attribution

- `hypothesis_false`: not proven. The B11 sliding-window evidence and B12 A6-basis bank-vs-DCT gaps still show some basis-side structure.
- `generic_basis_control_explains`: yes on the label side. Train-label tile structure is very strong, but local DCT nearly matches shared/bank local bases on all datasets.
- `independent_tile_only`: not the main failure. Independent tile basis is better, but shared/bank is not catastrophically far behind; the gap is just not small enough for a method gate.
- `coeff_path_not_supported`: yes for cross-dataset evidence. The adjacent-vs-far coeff projection pattern is clear only on ETTh2, not on ETTm1 or Weather.
- `direction_level_rejection`: no. This diagnostic blocks the current B12-STBO Step 4-6 transition; it does not reject all future basis-operator redesigns.

[Next] Do not implement B12-STBO as currently defined. Either redesign the basis-operator problem with stronger non-DCT and coeff-path evidence, or roll back StageB to Step 2/3 architecture search.

## Output Files

- `b12_stbo_basis_factorization.csv`
- `b12_stbo_label_factorization.csv`
- `b12_stbo_coeff_projection.csv`
- `b12_stbo_gate_summary.json`
- `b12_stbo_report.md`
