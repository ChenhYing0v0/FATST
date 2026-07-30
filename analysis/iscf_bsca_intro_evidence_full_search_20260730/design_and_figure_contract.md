# Introduction Evidence Full Visualization Search Design

## 1. Current cursor

| Field | Content |
| --- | --- |
| `date` | `2026-07-30` |
| `protocol` | `SC-UVHF-INTRO-EVIDENCE-FULL-SEARCH-v1` |
| `role` | validation-only illustrative figure search |
| `datasets` | ETTh1, ETTh2, ETTm1, ETTm2, Weather |
| `seed` | 2021 |
| `formal_test` | false |
| `architecture_effectiveness_gate` | false |
| `new_prefix_runs` | 16 |
| `new_sharing_runs` | 15 |
| `total_new_runs` | 31 |

现有完整artifacts将restart-safe复用。新增矩阵补齐：

- prefix：ETTh1/ETTh2/ETTm1/ETTm2 × 4 horizons；
- sharing：ETTh1/ETTh2/ETTm2 × 5 sharing extents。

## 2. Figure contract

### Figure 1: prefix disagreement

```text
Core conclusion:
  Independently optimized horizon-specific systems can assign visibly
  different values to the same future steps for the same history.
Figure archetype:
  quantitative grid
Target/output:
  high-impact journal; editable SVG/PDF plus high-resolution PNG/TIFF later
Backend:
  Python / matplotlib only
Panel map:
  a: full history and horizon-specific forecasts
  b: overlapping-prefix differences relative to a fixed long-horizon reference
  c: six-pair NCHPD summary
Hero evidence:
  selected origin-channel difference trajectories
Validation evidence:
  all-origin horizon-pair disagreement matrix
Reviewer risk:
  cherry-picking and single-step spikes
Integrity response:
  deterministic maximum over a disclosed validation search space; score uses
  mean absolute disagreement over overlap steps and all six pairs
```

### Figure 2: sharing-demand heterogeneity

```text
Core conclusion:
  Within one sample, the preferred output-side sharing extent changes
  materially across future regions.
Figure archetype:
  quantitative grid
Target/output:
  high-impact journal; editable SVG/PDF plus high-resolution PNG/TIFF later
Backend:
  Python / matplotlib only
Panel map:
  a: sample-specific scope x future-region risk landscape
  b: selected scopes' step-wise relative-risk curves
  c: region-wise risk contrasts and winner distribution
Hero evidence:
  sample-specific best-scope ridge with multiple materially supported scopes
Validation evidence:
  matched parameterization and full five-scope risk tensor
Reviewer risk:
  label-noise-driven one-region winners and post-hoc sample selection
Integrity response:
  each region aggregates 60 future steps and all channels; selection first
  maximizes scopes winning at least two regions, then entropy, qualified
  crossover count, winner margin, and descriptive headroom
```

## 3. Prefix maximum-selection statistic

For every horizon pair $H_i<H_j$, validation origin $o$, and channel $c$:

$$
d_{o,c}^{(i,j)}
=
\frac{1}{H_i}
\sum_{\tau=1}^{H_i}
\left|
\hat y_{o,\tau,c}^{(H_i)}
-
\hat y_{o,\tau,c}^{(H_j)}
\right|.
$$

The aggregate cell score is:

$$
D_{o,c}
=
\frac{1}{6}
\sum_{i<j}d_{o,c}^{(i,j)}.
$$

For each dataset, select:

$$
(o^\star,c^\star)=\arg\max_{o,c}D_{o,c}.
$$

This is intentionally the strongest validation example, not a representative
or prevalence estimate. The figure caption must disclose
`maximum aggregate validation disagreement`. A mean-over-time score prevents
selection by a single isolated future-step spike.

Across datasets, rank the per-dataset maxima by the same train-scale-normalized
$D_{o^\star,c^\star}$.

## 4. Sample-level sharing heterogeneity statistic

For scale $s$, validation sample $o$, and 60-step region $\mathcal B_b$:

$$
R_{o,b,s}
=
\frac{1}{|\mathcal B_b|C}
\sum_{\tau\in\mathcal B_b}
\sum_c
\left(\hat y^{(s)}_{o,\tau,c}-y_{o,\tau,c}\right)^2.
$$

For every sample, compute:

1. `distinct_winner_count`: number of scales winning at least one region；
2. `supported_winner_count`: number of scales winning at least two regions；
3. `winner_entropy`: normalized entropy of the 12-region winner histogram；
4. `qualified_crossing_pair_count`: pairwise bidirectional ordering changes
   exceeding the frozen 0.5% margin；
5. `mean_winner_margin`: mean normalized gap between the best and second-best
   scale in each region；
6. `sample_oracle_headroom`: best fixed-scope risk versus per-region oracle。

Per-dataset sample selection uses the following lexicographic order:

```text
supported_winner_count
-> distinct_winner_count
-> winner_entropy
-> qualified_crossing_pair_count
-> mean_winner_margin
-> sample_oracle_headroom
```

The first criterion prevents a scale that wins only one region through noise
from dominating the selection. Each region already averages 60 steps and all
channels. Across datasets, the per-dataset candidates are ranked by the same
tuple.

## 5. GPU/runtime redesign

The previous runtime imbalance is traced primarily to the Python loop in
`pooled_states`: $s=1$ invokes 720 small pooling/LayerNorm operations, whereas
$s=720$ invokes one. This is an implementation overhead, not a mechanism
difference.

The exact-function optimization will:

- reshape all complete blocks into one batched tensor；
- compute block means and LayerNorm in one vectorized call；
- process at most one incomplete tail block separately；
- broadcast and reshape back to `[B,C,T,D_z]`；
- verify output and gradient equivalence against a loop reference for all five
  scales。

The remote matrix runner will use a global three-GPU queue rather than pinning
all $s=1$ work to GPU 0. Longer datasets and $s=1$ jobs enter the queue first；
each freed GPU immediately receives the next missing job. Existing complete
artifacts are skipped.

## 6. Claim and stopping boundary

- all selection uses validation only；
- test is neither constructed nor accessed；
- selected examples are intentionally strongest and must be disclosed as such；
- figures establish illustrative existence, not prevalence；
- no result in this screen can reject or modify the fixed ISCF-BSCA
  architecture；
- after all five datasets are ranked, select one prefix candidate and one
  sharing candidate, then stop dataset/sample search。

Decision=`full_five_dataset_visualization_search_design_frozen`。

## 7. Long-stage record

| Field | Current Record |
| --- | --- |
| `current_step` | Step7A local implementation and invariant gate complete；Step8 remote launch next |
| `problem` | 85% prefix example不够清晰；aggregate sharing图的winner过度集中于s128；fixed GPU workers产生GPU 0长尾 |
| `existence_evidence` | Weather prefix及Weather/ETTm1 sharing artifacts可复用，但尚不足以冻结最终Introduction figures |
| `idea` | 搜索disclosed maximum prefix cell与sample-level supported multi-scope winner pattern；五dataset统一排序 |
| `theory_check` | prefix score对overlap steps和六pairs取均值；sharing region对60 steps及all channels聚合，先要求至少两个regions支持一个winner |
| `design` | reuse 14 + new 31 validation runs；vectorized exact pooling；global dynamic three-GPU queue |
| `narrative_gate` | pass for illustrative diagnostic；不使用ISCF/BSCA，不把selection包装为prevalence |
| `effectiveness_gate` | not applicable；本轮不是method effectiveness或formal problem gate |
| `artifacts` | config、maximum/sample analyzers、ranker、dynamic runner、local equivalence checker；remote artifacts pending |
| `decision` | `full_search_step7a_pass_remote_launch_next` |

## 8. First-launch implementation fault and repair

首次remote launch在commit `7813932`上通过GPU preflight与generic dry-run，但
ETTh1/ETTm2 jobs在argument parsing阶段退出。根因是两个trainers共享的
`baselines/dlinear/dataset.py` registry历史上只列出ETTh2、ETTm1与Weather；
原dry-run只检查`--help`和synthetic paths，没有枚举full-search dataset names。

Failure attribution：

- `hypothesis_false`：否；
- `intervention_point_wrong`：否；
- `readout_or_head_design_wrong`：否；
- `optimization_or_numeric_pathology`：否；
- exact cause=`prelaunch_dataset_registry_coverage_fault`。

driver已在发现首批失败后立即停止，三张GPU恢复18 MiB idle；已完成artifacts保留，
失败jobs没有进入训练或产生结果。修复为：

1. registry加入ETTh1与ETTm2，沿用对应hour/minute标准split；
2. local checker显式要求五个full-search datasets全部注册；
3. 重新执行dataset construction、full checker、remote dry-run与GPU preflight后
   才允许restart-safe relaunch。

该故障只影响exact runner readiness，不产生visualization result，也不能用于
problem、method或architecture判断。
