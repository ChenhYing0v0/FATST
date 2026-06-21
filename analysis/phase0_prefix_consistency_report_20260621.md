# Phase0 Prefix Consistency Report

## Run Metadata

- [Fact] Remote host: `529_Lab-3090`
- [Fact] Code commit: `6243991`
- [Fact] Source artifacts: `/home/yingch/projects/FATST/artifacts/runs/phase0`
- [Fact] Report output:
  `/home/yingch/exp_outputs/r-2026-fatst/phase0_prefix_consistency`
- [Fact] Model: `PatchEncoderFixedHead`
- [Fact] Seed: `2021`
- [Fact] Datasets: `ETTh2`, `ETTm1`, `Weather`
- [Fact] Horizons: `{96,192,336,720}`
- [Fact] Max horizon checkpoint: `h720`

Raw result table:
`analysis/phase0_prefix_consistency_raw_20260621.csv`

## Method

For each dataset and prefix horizon $H$, compare:

$$
\hat{Y}_{1:H}^{fixed(H)}
$$

against the aligned prefix of the max-horizon checkpoint:

$$
\hat{Y}_{1:H}^{max(720)}.
$$

The comparison uses the first `N720` test windows from the horizon-specific checkpoint, because the
`h720` dataset has the fewest valid sliding windows and is a prefix-aligned subset of shorter horizon
test windows.

The alignment check passed:

```text
max truth_alignment_mse = 0.0
```

## Main Results

| Dataset | Prefix H | Fixed MSE | h720 Prefix MSE | Relative MSE Change | Prediction MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | 0.259405 | 0.254338 | -1.95% | 0.019680 |
| ETTh2 | 192 | 0.303968 | 0.297754 | -2.04% | 0.016569 |
| ETTh2 | 336 | 0.326730 | 0.322379 | -1.33% | 0.024159 |
| ETTm1 | 96 | 0.278985 | 0.292111 | +4.70% | 0.038447 |
| ETTm1 | 192 | 0.325119 | 0.325232 | +0.03% | 0.044742 |
| ETTm1 | 336 | 0.358600 | 0.361814 | +0.90% | 0.009066 |
| Weather | 96 | 0.146651 | 0.153676 | +4.79% | 0.013327 |
| Weather | 192 | 0.192861 | 0.195442 | +1.34% | 0.013207 |
| Weather | 336 | 0.248152 | 0.247440 | -0.29% | 0.013520 |

Here `Prediction MSE` means:

$$
\frac{1}{BHC}\|\hat{Y}_{1:H}^{fixed(H)}-\hat{Y}_{1:H}^{max(720)}\|_2^2.
$$

## Findings

[Strong Evidence] Fixed heads do expose a quantifiable variable-horizon/prefix issue.
The strongest performance degradation is:

- `Weather / H=96`: h720 prefix is `+4.79%` MSE worse than the h96 fixed head.
- `ETTm1 / H=96`: h720 prefix is `+4.70%` MSE worse than the h96 fixed head.

[Strong Evidence] Even when error is similar, the predictions from different fixed heads are not
the same. The largest direct prediction mismatch is:

- `ETTm1 / H=192`: `fixed_vs_max_pred_mse = 0.044742`

This means separate fixed horizon heads can produce materially different outputs for the same input
window and same prefix target.

[Strong Evidence] The issue is not a data-window alignment artifact, because every row has
`truth_alignment_mse = 0.0`.

[Inference] The problem is moderate rather than catastrophic. On ETTh2, the h720 prefix is slightly
better than horizon-specific heads for short prefixes. On ETTm1 and Weather, however, short-horizon
prefix degradation appears exactly where variable-horizon inference would matter.

## Phase0 Conclusion

[Fact] Phase0 has now completed all planned gates:

1. baseline gate across `DLinear`, `PatchEncoderFixedHead`, `SegTSFTDenseFixedHead`;
2. Weather-720 OOM rerun;
3. targeted controls against paper-gap hypotheses;
4. selected-base seed variance;
5. fixed-head prefix consistency diagnostic.

[Strong Evidence] `PatchEncoderFixedHead` remains the selected internal base.

[Strong Evidence] There is a measurable fixed-head prefix inconsistency problem, sufficient to justify
Phase1 `Variable-Horizon Decoder Gate`.

[Inference] Phase1 should optimize two targets:

1. keep or improve horizon-specific MSE/MAE;
2. reduce prefix inconsistency between short-horizon requests and long-horizon prefix outputs.

The first Phase1 gate should use `PatchEncoderFixedHead` as encoder base and compare fixed direct
head against query/segment variable-horizon decoders on `ETTm1` and `Weather`, because those datasets
show the clearest short-prefix degradation.
