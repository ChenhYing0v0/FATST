# SC-D21-EVS Step 2/3 Problem and Diagnostic Design Audit

## Research record

| Field | Content |
| --- | --- |
| `current_step` | Contribution 1 Step2/3 problem gate |
| `problem` | internal forecast-route validity是否具有可由past识别的sample × future-region interaction |
| `existence_evidence` | D14-A dual-carrier three-seed crossing与oracle headroom；D20只作弱motivation |
| `idea` | validation拟合past-only relative-risk probe，official test只评估transfer |
| `theory_check` | exact projectivity；requested H禁用；future truth只生成diagnostic labels，不进入inference features |
| `design` | D14 canonical scope arms；global/region/sample/additive/interaction/permuted controls |
| `narrative_gate` | problem-level conditional pass；generic routing/multiscale/query overlap已排除 |
| `effectiveness_gate` | not applicable；本实验只决定问题是否值得进入Step4 |
| `rollback_point` | interaction fail则Contribution 1回Step2，不设计EVS operator |

## What is tested

For carrier $c$, dataset $d$, row $i$, future bin $b$, and independently
trained D14 canonical coupling arm $s\in\{1,48,144,360,720\}$, collect

$$
L_{i,b,s}=\frac1{|b|}\sum_{\tau\in b}
(\hat y_{i,s,\tau}-y_{i,\tau})^2.
$$

The bins remain `short=[1,144]`, `mid=[145,360]`, and
`long=[361,720]`, matching the frozen D14 diagnostic. The policy receives only
features computed from the observed 720-step past. It never receives requested
$H$, future labels, dataset ID, validation/test performance, or future calendar
features.

To avoid mistaking generic difficulty prediction for route validity, the
regression target is centered log risk:

$$
r_{i,b,s}=\log(L_{i,b,s}+\epsilon)
-\frac1{|\mathcal S|}\sum_j\log(L_{i,b,j}+\epsilon).
$$

The selected route is the arm with minimum predicted relative risk. Realized
test MSE is computed from the selected arm's actual test loss; no test label is
used during fitting or hyperparameter selection.

## Data and checkpoint boundary

- checkpoint source: frozen D14-A1 `seed2021` canonical arms;
- carriers: `neutral_raw` primary and `a6_natural` sensitivity;
- datasets: ETTh1, ETTh2, ETTm1, ETTm2, Weather;
- fit split: official validation only;
- evaluation split: official test only;
- row alignment: deterministic sequential loader and common evenly spaced probe
  indices, with all channels of the selected rows retained in the recorded
  order;
- checkpoint mutation: forbidden;
- new model training: none; only offline risk probes are fitted.

D14 checkpoints used an historical H720 validation selector. This does not
qualify them for a paper-facing accuracy comparison under the current
four-horizon rule. D21 uses them only as matched independent route generators.
Any future method comparison must retrain under the current selector.

## Inference-visible past features

For each probe row, the evaluator records a fixed descriptor assembled from:

1. raw window mean, standard deviation, last value, and normalized last value;
2. normalized mean-pooled history over 60 contiguous 12-step blocks;
3. the latest 48 normalized observations;
4. real and imaginary parts of the first 32 non-DC Fourier coefficients;
5. autocorrelations at lags `1,6,12,24,48,96,168,336`;
6. recent mean and standard deviation at windows `24,48,96,192`.

The descriptor is fixed before results. A linear ridge probe is primary. A
small `HistGradientBoostingRegressor` is a sensitivity readout so that a weak
linear head cannot create a direction-level false negative. Descriptor and
readout are diagnostic only and are not candidate components.

## Frozen controls

| ID | Information available | Question |
| --- | --- | --- |
| `G0_GLOBAL_FIXED` | validation mean only | 是否一个route全局支配 |
| `G1_REGION_FIXED` | future bin only | 是否只是静态future-region偏好 |
| `G2_HISTORY_GLOBAL` | past only；one arm for all bins | 是否只是sample-level adaptive fusion |
| `G3_ADDITIVE` | past score + bin intercept | 是否sample与region主效应已经足够 |
| `G4_EVS_INTERACTION` | separate past-dependent risk per bin | 是否存在真正的past × future-region interaction |
| `G5_PERMUTED_HISTORY` | row-permuted validation descriptors | 是否capacity/label prevalence复制收益 |
| `G6_ORACLE` | test future truth；upper bound only | 剩余可兑现headroom，不参与pass |

## Frozen gates

All gains are relative MSE reductions, computed on the same test probe rows and
reported per dataset, carrier, and readout.

### P1 — transferable past identifiability

For at least one preregistered readout class, `G4` must beat `G1` by macro
`>=0.3%` and on at least `3/5` datasets for the neutral carrier. The A6
sensitivity carrier must have positive macro gain and at least `3/5` positive
datasets.

### P2 — multi-horizon interaction specificity

`G4` must beat `G3_ADDITIVE` by macro `>=0.1%` and on at least `3/5` datasets
for the neutral carrier. It must also beat `G2_HISTORY_GLOBAL`; otherwise the
problem collapses to sample-only fusion.

### P3 — non-spurious history signal

`G4` must beat `G5_PERMUTED_HISTORY` by macro `>=0.2%` and on at least `3/5`
datasets for the neutral carrier. No policy may show non-finite output or
`>10%` degradation against `G1` on more than one dataset.

### P4 — readout interpretation

- ridge and tree both pass: `problem_gate_passed_robust`;
- tree passes while ridge fails: `problem_partial_pass_nonlinear_identifiability`;
- ridge passes while tree is weak but non-pathological:
  `problem_gate_passed_simple_signal`;
- neither passes: only the exact descriptor/readout probe fails. Direction-level
  rejection requires a later representation-level probe or independent
  negative evidence.

### P5 — confirmation boundary

Seed2021 can authorize Step4 source-informed method design but cannot establish
a paper claim. Before implementation of a paper-core candidate, the
identifiability result must be confirmed on seeds2022/2023 or reproduced with a
jointly trained representation-level diagnostic.

## What a pass would and would not mean

A pass means the multi-horizon problem is empirically supported: future-region
preference is not static, and the extra variation can be predicted from the
past across splits. It does not mean that a router, mixture, PCSD, SIFF, or any
specific decoder is the correct solution.

It would authorize Step4 with two linked design obligations:

1. a projective full-trajectory operator that represents internal
   past-by-coordinate route validity;
2. an end-to-end same-forward credit principle that learns it without offline
   teachers or future-label features at inference.

## Authorization

`diagnostic_implementation=true`; `remote_checkpoint_evaluation=true`;
`official_test_evaluation=true`; `new_forecasting_model_training=false`;
`paper_method_implementation=false`; `confirmation_seeds=false`.
