# Phase5 StageB B11-BCF Small Gate Report

## Scope

Required arms: `a6_clean`, `b11_bcf`, `b11_no_basis`, `b11_constant_slot`.
Optional arm: `b11_shuffled_basis`.
Datasets: ETTh2, ETTm1, Weather. Horizons: 96, 192, 336, 720.

## Summary

| comparison | dataset | settings | mse_wins | mean_relative_mse_pct | mean_relative_mae_pct |
| --- | --- | --- | --- | --- | --- |
| b11_bcf_vs_a6_clean | ETTh2 | 4 | 4 | -0.35% | -0.23% |
| b11_bcf_vs_a6_clean | ETTm1 | 4 | 1 | +0.0079% | -0.04% |
| b11_bcf_vs_a6_clean | Weather | 4 | 0 | +0.04% | -0.05% |
| b11_bcf_vs_a6_clean | ALL | 12 | 5 | -0.10% | -0.11% |
| b11_bcf_vs_b11_constant_slot | ETTh2 | 4 | 0 | +0.08% | +0.05% |
| b11_bcf_vs_b11_constant_slot | ETTm1 | 4 | 4 | -0.0036% | -0.0023% |
| b11_bcf_vs_b11_constant_slot | Weather | 4 | 3 | -0.0012% | -0.0011% |
| b11_bcf_vs_b11_constant_slot | ALL | 12 | 7 | +0.03% | +0.01% |
| b11_bcf_vs_b11_no_basis | ETTh2 | 4 | 2 | -0.02% | -0.0088% |
| b11_bcf_vs_b11_no_basis | ETTm1 | 4 | 0 | +0.0084% | +0.0047% |
| b11_bcf_vs_b11_no_basis | Weather | 4 | 0 | +0.0074% | +0.0098% |
| b11_bcf_vs_b11_no_basis | ALL | 12 | 2 | -0.0012% | +0.0019% |
| b11_constant_slot_vs_a6_clean | ETTh2 | 4 | 4 | -0.43% | -0.28% |
| b11_constant_slot_vs_a6_clean | ETTm1 | 4 | 1 | +0.01% | -0.04% |
| b11_constant_slot_vs_a6_clean | Weather | 4 | 0 | +0.04% | -0.05% |
| b11_constant_slot_vs_a6_clean | ALL | 12 | 5 | -0.13% | -0.12% |
| b11_no_basis_vs_a6_clean | ETTh2 | 4 | 4 | -0.33% | -0.22% |
| b11_no_basis_vs_a6_clean | ETTm1 | 4 | 1 | -0.0005% | -0.05% |
| b11_no_basis_vs_a6_clean | Weather | 4 | 0 | +0.03% | -0.06% |
| b11_no_basis_vs_a6_clean | ALL | 12 | 5 | -0.10% | -0.11% |

## Gate Reading

[Decision] `capacity_or_head_effect_suspected`: B11-BCF improves over A6 but does not beat required controls.

- B11 vs A6: mean MSE -0.10%, wins 5/12.
- B11 vs no-basis: mean MSE -0.0012%, wins 2/12.
- B11 vs constant-slot: mean MSE +0.03%, wins 7/12.

## Detailed Interpretation

[Fact] `b11_bcf` 的整体收益很小：相对 `a6_clean` mean MSE 为 `-0.1019%`，只有 `5/12`
MSE wins。收益主要来自 ETTh2：ETTh2 为 `4/4` wins、mean MSE `-0.3494%`；ETTm1 基本持平
（`+0.0079%`），Weather 小幅变差（`+0.0359%`）。

[Strong Evidence] `no_basis` control 几乎完全解释了 B11 的收益。`b11_no_basis` 相对 A6 的整体
mean MSE 为 `-0.1007%`，与 `b11_bcf` 的 `-0.1019%` 几乎相同；而 `b11_bcf` 相对
`b11_no_basis` 只有 `2/12` wins，mean MSE 仅 `-0.0012%`，MAE 反而 `+0.0019%`。这说明把
learned basis geometry 放进 descriptors 并没有形成稳定、可归因的机制优势。

[Strong Evidence] `constant_slot` control 进一步削弱 row-wise continuous field claim。`b11_constant_slot`
相对 A6 的整体 mean MSE 为 `-0.1281%`，比 `b11_bcf` 更好；`b11_bcf` 相对 `constant_slot`
整体 mean MSE 为 `+0.0263%`。虽然 `b11_bcf` 在 ETTm1/Weather 有 `7/8` wins，但幅度只有
`-0.0036%` 和 `-0.0012%`；ETTh2 上 `constant_slot` 明显更好，`b11_bcf` 为 `0/4` wins。

[Mechanism Diagnostic] `model_diagnostics` 显示 B11 gate 训练后仍很小：ETTh2/ETTm1/Weather 的
`basis_field_gate_sigmoid` 分别约为 `0.0157/0.0111/0.0121`。这与结果一致：B11 branch 主要作为一个
很弱的 initialized field perturbation 存在，尚未证明 basis-conditioned row-wise routing 被有效使用。

## Failure Attribution

This result blocks the tested B11-BCF implementation as a paper-core method.

- `hypothesis_false`: not proven. The earlier B11 diagnostic still shows continuous basis subspace geometry.
- `intervention_point_wrong`: possible. The current implementation injects descriptors only at a hidden-level
  coefficient field; it may be too weak to force basis geometry into history aggregation.
- `readout_or_head_design_wrong`: possible. The row-wise mixture can collapse to behavior explainable by
  no-basis / constant-slot controls.
- `optimization_or_numeric_pathology`: not observed. Runs completed stably; no divergence or >100% degradation.
- `capacity_control_explains`: yes. `no_basis` and especially `constant_slot` explain the observed A6 gain.

Decision: `blocked_by_required_controls`. B11 should roll back to Step 4 redesign if continued. It must not be
promoted to paper-core from these metrics.

## Failure Attribution Rule

This report may reject only the tested B11-BCF implementation unless the required controls show that the broader basis-conditioned direction is false.
If `no_basis` or `constant_slot` explains the gain, classify the result as capacity/head effect, not basis-conditioned mechanism evidence.
