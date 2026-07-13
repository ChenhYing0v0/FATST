# SC1-D2 Formal5 Research Interpretation

## 1. Decision

| Field | Value |
| --- | --- |
| `current_step` | Step 2 rollback active |
| `role` | `diagnostic_only` |
| `matrix` | 5 datasets × 3 frozen checkpoints × 11 arms = 165/165 fits |
| `invariant_gate` | pass |
| `decision` | `scale_alignment_not_supported_reformulate_step2` |
| `method_implementation_authorized` | `false` |

[Fact] true balanced-interval basis具有稳定正信号，但true depth grouping不具有跨dataset的一般性。因此D2否定
的是“按balanced depth分配独立nonlinear blocks就是关键结构”这一精确problem formulation，不是否定所有
future-side geometry、basis/operator redesign或multi-horizon training方向。

## 2. Formal Gate Results

| Comparison | Macro MSE gain | Datasets with ≥2/3 positive seeds | Gate |
| --- | ---: | ---: | --- |
| full affine vs rank256 | +0.6780% | 3/5 | pass |
| strongest dense nonlinear vs full affine | -6.4715% | 2/5 | fail |
| true scale vs strongest dense | +4.5202% | 4/5 | pass |
| true scale vs random-group median | +0.0947% | 2/5 | **fail** |
| true scale vs random-basis median | +3.0635% | 5/5 | pass |

true scale的MAE macro相对strongest dense为+2.9166%，排除了“只改善MSE、显著损害MAE”的解释；但formal
hard gate要求random-group与random-basis两项都通过，因此overall decision仍为fail。

## 3. Per-Dataset Reading

| Dataset | True vs dense | True vs random-group | True vs random-basis | Group controls beaten |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | +6.8719% | -0.5350% | +3.6791% | 0.33/3 |
| ETTh2 | +12.5011% | -1.0148% | +5.1077% | 0.00/3 |
| ETTm1 | -0.8040% | -0.1263% | +0.6057% | 1.33/3 |
| ETTm2 | +3.6213% | +1.6723% | +4.6970% | 3.00/3 |
| Weather | +0.4104% | +0.4775% | +1.2278% | 3.00/3 |

[Strong Evidence] basis comparison在5/5 datasets、15/15 checkpoint seeds为正，且每个dataset平均击败3/3
random-basis controls。相反，depth grouping在两个hourly ETT上稳定为负，只在ETTm2与Weather达到2/3 seeds
为正。它更像dataset-dependent inductive bias，而不是unified multi-horizon operator的共同原则。

## 4. Optimization And Validity Audit

所有metadata均满足：不加载test、不更新forecast model、official validation不参与probe early stopping、basis
orthogonality gap最大`1.91e-6`、Parseval relative gap最大`3.36e-7`。五dataset的所有主要arms均在120 epochs
之前取得best inner-holdout epoch，没有non-finite loss或>100%异常退化。

ETTh1/ETTh2的dense heads虽然fit与inner-holdout优于affine，却在official validation更差，说明存在temporal
generalization gap；它会放大true-vs-dense gain，所以该gain不能单独支持scale机制。关键的true-vs-random-group
比较使用相同basis、block size、optimizer和parameter count，不受上述dense-control gap解释，仍然失败。

## 5. Failure Attribution

- `hypothesis_false`：**supported for the exact hypothesis**。true depth grouping没有稳定超过same-basis random
  grouping；
- `intervention_point_wrong`：仍可能。D2只测试final frozen-memory head，不能否定jointly-trained decoder中的
  future geometry；
- `readout_or_head_design_wrong`：对generic dense temporal generalization成立，但无法挽救group-vs-random
  的matched comparison；
- `optimization_or_numeric_pathology`：未观察到；
- `capacity_control_explains`：rank/full/dense不能解释稳定basis advantage，但random grouping足以解释绝大多数
  grouped architecture收益。

[Decision] 方向级不可写成“future scale structure不存在”。允许拒绝的边界仅为
`balanced-depth independent nonlinear grouping at the final frozen-memory head`。rollback到Step 2，而不是继续
给FPMO-DS叠加nonlinearity、MIPR、MoE或Encoder改造。

## 6. What Remains Untested

当前true/random controls不是完整$2\times2$ factorial：已有`true basis × true group`、
`true basis × random group`、`random basis × true group`，缺少`random basis × matched random group`。因此
“basis geometry是独立main effect”仍是[Hypothesis]，不能把+3.0635%直接提升为paper contribution。

下一步建议定义`SC1-D3 crossed basis-group diagnostic`：只补每个structure seed的缺失cell，并计算basis main
effect、group main effect与interaction。它仍是Step 2/3 `diagnostic_only`，预计新增5 datasets × 3 checkpoints ×
3 missing arms = 45 fits。gate、paired estimator与optimization audit必须先在protocol中冻结；D3通过也只允许
进入Step 4做external source-informed redesign。

## 7. Self-Critique

[Against our conclusion] random orthogonal basis是很强但未必最有意义的control；它同时破坏local support、
frequency ordering与target smoothness，因此true basis获胜可能只是一般structured-coordinate regularization，
并不专属multi-horizon projectivity。即使D3得到basis main effect，仍需加入identity/DCT或其他structured basis
controls并完成prior-art audit，才能判断是否有SCI-level novelty。

## 8. Artifacts

- `d2_pairwise_metrics.csv`：15个dataset-seed paired comparisons；
- `d2_dataset_summary.csv`：逐dataset三seed mean/std/positive count；
- `d2_summary.json`：formal hard gates与decision；
- `d2_diagnostic_report.md`：自动生成的gate report；
- `raw/`：gitignored per-arm metrics、training history、metadata与remote logs。
