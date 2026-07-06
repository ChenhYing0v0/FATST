# Phase5 StageB B3 DSR Diagnostic Code Explanation

本文档解释 `scripts/analyze_phase5_stage_b_b3_dsr_diagnostic.py`。该脚本只做 post-hoc diagnostic，
不修改 model、loss、runner 或 remote training protocol。

## Functional Scope

`B3-DSR` 的目标是验证：

> train-only `seasonal_residual` 是否能解释去除 forecast-distance trend 后的 residual difficulty。

脚本输入：

| Input | Role |
| --- | --- |
| A6-LBF-r256 `predictions_test.npz` | 计算 held-out unit error；仅作为 diagnostic label |
| local train split labels | 计算 train-only proxies |
| B1 helper functions | 复用 dataset split、rank/correlation、linear residual 工具 |

脚本输出：

| Output | Meaning |
| --- | --- |
| `stage_b_b3_unit_residuals.csv` | per dataset/block/unit 的 raw MSE、distance-normalized labels 和 train-only proxies |
| `stage_b_b3_detrending_robustness.csv` | proxy 与不同 residual labels 的 Spearman/Pearson/top-quartile overlap |
| `stage_b_b3_blocksize_robustness.csv` | block size `24/48/96` 下的 seasonal signal summary |
| `stage_b_b3_bootstrap_stability.csv` | seasonal proxy 的 bootstrap sign stability |
| `stage_b_b3_report.md` | B3 gate decision |

## Computation Flow

1. `prediction_unit_mses()` 逐 dataset 读取 `predictions_test.npz`，对 block size `24/48/96`
   计算 unit-level `mse`、`mae` 和 `sample_error_std`。
2. `train_proxy_by_offset()` 从 train split 计算每个 future offset 的三个 train-only proxy：
   `label_novelty`、`local_variation`、`seasonal_residual`。
3. `train_proxy_units()` 将 offset-level proxy 聚合到不同 block size，并添加两个 control：
   `step_index` 和 deterministic `shuffled_seasonal`。
4. `residual_labels()` 为每个 dataset/block 生成三个 distance-normalized labels：
   `linear_step_residual`、`rank_step_residual`、`prefix_normalized_residual`。
5. `collect_diagnostic_rows()` 组合 prediction labels 和 train-only proxies，生成四类 CSV。
6. `gate_decision()` 根据 block-size robustness、step control、bootstrap sign stability 给出 B3 gate。

## Statistic Definitions

For dataset $d$ and unit $u=[s:e]$:

$$
E_{d,u}=\operatorname{mean}_{i,t,c}(\hat{Y}_{i,t,c}-Y_{i,t,c})^2,\quad t\in[s,e].
$$

`linear_step_residual`:

$$
R^{linear}_{d,u}=E_{d,u}-(a_d+b_d s_u).
$$

`rank_step_residual`:

$$
R^{rank}_{d,u}=\operatorname{rank}(E_{d,u})-(a_d+b_d\operatorname{rank}(s_u)).
$$

If `rank_step_residual` has no variance, correlations are written as `nan`; this means the rank structure was fully explained by monotonic step trend under that block size.

`prefix_normalized_residual`:

$$
R^{prefix}_{d,u}=\frac{E_{d,u}}{\operatorname{neighbor\_baseline}(E_{d,u})}-1.
$$

The primary proxy is:

$$
P^{seasonal}_{train}(d,u)
=
\operatorname{mean}_{train,t,c}
\left(Y_{t,c}-Y_{t-p_d,c}\right)^2,\quad t\in u.
$$

`p_d` follows the same train-side period settings used by B1:

| Dataset | Period |
| --- | ---: |
| ETTh2 | `24` |
| ETTm1 | `96` |
| Weather | `144` |

## Code-Theory Consistency Evaluation

[Intended Theory] B3 assumes raw unit MSE is distance-confounded, but residual difficulty after removing step trend may still contain train-estimable structural signal.

[Code Realization] The script evaluates this by comparing `seasonal_residual` against three residual labels and two controls across three block sizes.

[Proxy Boundary] `linear_step_residual` is the weakest distance control, while `rank_step_residual` and `prefix_normalized_residual` are stricter. A method candidate should not be justified by linear residual alone.

[Falsification Evidence] B3 is falsified or blocked if the seasonal signal disappears under stricter detrending, is weaker than step/shuffled controls, or has low bootstrap positive fraction.

## Current Result

The generated B3 report gives:

```text
partial_pass_needs_stronger_proxy_or_method_boundary
```

This means `seasonal_residual` contains some non-distance signal, especially under `linear_step_residual`, but it is not robust enough to justify implementing reliability-aware loss weighting.
