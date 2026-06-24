# Phase4 Existing-Residual Projection Audit

## Purpose

[Fact] This diagnostic uses existing R.3 `predictions_test.npz` artifacts and train-label component bases. It does not train a model.

[Question] Do existing R.3 residuals have meaningful component-space structure, and does that structure help explain known horizon/segment gaps?

## Summary

- specialist gap rows: `4/12`.
- gap mean residual top16 energy share: `0.789`.
- non-gap mean residual top16 energy share: `0.828`.
- gap mean top16-over-label ratio: `0.870`.
- non-gap mean top16-over-label ratio: `0.923`.
- gap minus non-gap top16-over-label ratio: `-0.054`.
- segment gap top16 reconstruction share: `0.716`.
- segment non-gap top16 reconstruction share: `0.714`.
- segment gap minus non-gap top16 reconstruction share: `0.002`.

## Decision

- component residual structure exists: `True`
- top component energy separates gaps: `True`
- segment reconstruction separates gaps: `False`
- top-only component loss supported: `False`
- proceed to component objective design: `True`

[Decision] Component-space supervision remains a viable Step 4-6 candidate, but the result does not support a top-only TransDF-style loss. The next objective should be hybrid or variance-balanced.

## Main Horizon Rows

| Dataset | Horizon | Gap | Rel MSE vs fixed | Residual top16 | Label top16 | Top16 / label |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | False | -0.86% | 0.854 | 0.960 | 0.889 |
| ETTh2 | 192 | False | -2.20% | 0.833 | 0.941 | 0.886 |
| ETTh2 | 336 | False | -0.31% | 0.786 | 0.916 | 0.858 |
| ETTh2 | 720 | True | +0.75% | 0.735 | 0.873 | 0.841 |
| ETTm1 | 96 | True | +2.83% | 0.901 | 0.976 | 0.923 |
| ETTm1 | 192 | False | -2.38% | 0.867 | 0.959 | 0.904 |
| ETTm1 | 336 | False | -0.22% | 0.808 | 0.938 | 0.861 |
| ETTm1 | 720 | True | +1.09% | 0.670 | 0.881 | 0.761 |
| Weather | 96 | True | +0.64% | 0.850 | 0.892 | 0.953 |
| Weather | 192 | False | -1.43% | 0.833 | 0.863 | 0.966 |
| Weather | 336 | False | -2.39% | 0.828 | 0.836 | 0.991 |
| Weather | 720 | False | -0.71% | 0.814 | 0.790 | 1.031 |

## H720 Segment Rows

| Dataset | Segment | Gap | Rel MSE vs fixed | Top16 recon share | Top64 recon share |
| --- | --- | --- | ---: | ---: | ---: |
| ETTh2 | 1-96 | False | -1.80% | 0.636 | 0.791 |
| ETTh2 | 97-192 | False | -0.89% | 0.712 | 0.834 |
| ETTh2 | 193-336 | True | +4.07% | 0.726 | 0.838 |
| ETTh2 | 337-720 | True | +0.47% | 0.754 | 0.868 |
| ETTm1 | 1-96 | False | -2.72% | 0.727 | 0.859 |
| ETTm1 | 97-192 | False | -1.76% | 0.709 | 0.885 |
| ETTm1 | 193-336 | False | -1.20% | 0.624 | 0.895 |
| ETTm1 | 337-720 | True | +3.03% | 0.669 | 0.911 |
| Weather | 1-96 | False | -4.04% | 0.621 | 0.789 |
| Weather | 97-192 | False | -1.91% | 0.744 | 0.866 |
| Weather | 193-336 | False | -1.16% | 0.810 | 0.900 |
| Weather | 337-720 | False | -0.05% | 0.844 | 0.919 |

## Interpretation

[Inference] Residual energy is strongly structured in component space, so component supervision is more coherent than arbitrary horizon pairs.

[Inference] Known gaps are not more concentrated in dominant top components. Gap rows have slightly lower top16-over-label ratio, and H720 segment gaps are nearly indistinguishable by top16 reconstruction share. Therefore, a top-only component loss is risky.

[Decision] The next design should test a hybrid component objective: keep time-domain MSE while adding variance-normalized or component-balanced supervision, so lower-variance detail components are not ignored.

[Risk] This diagnostic still does not prove training will improve. It only decides whether component-supervised training is a coherent next candidate.
