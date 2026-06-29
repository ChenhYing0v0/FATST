# Phase5 TimeAlign-HSS D0 Head Gate

## Decision Template

[Question] Is the TimeAlign unified decrease mainly caused by the fixed `pred_len=720` head receiving only a full-horizon prediction loss?

## Summary

| dataset | settings | multi_prefix_wins | mean_relative_mse_pct | mean_relative_mae_pct |
| --- | --- | --- | --- | --- |
| ETTh2 | 4 | 4 | -3.36 | -1.71 |
| ETTm2 | 4 | 4 | -1.57 | -0.98 |
| Weather | 4 | 4 | -1.17 | -1.06 |
| ALL | 12 | 12 | -2.03 | -1.25 |

## Per-Horizon Comparison

| dataset | horizon | full_mse | multi_prefix_mse | relative_mse_pct | multi_prefix_win |
| --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | 0.249109 | 0.242773 | -2.54 | True |
| ETTh2 | 192 | 0.295961 | 0.283057 | -4.36 | True |
| ETTh2 | 336 | 0.326745 | 0.312852 | -4.25 | True |
| ETTh2 | 720 | 0.403252 | 0.394071 | -2.28 | True |
| ETTm2 | 96 | 0.167614 | 0.162777 | -2.89 | True |
| ETTm2 | 192 | 0.219519 | 0.216118 | -1.55 | True |
| ETTm2 | 336 | 0.269307 | 0.266372 | -1.09 | True |
| ETTm2 | 720 | 0.343566 | 0.340916 | -0.77 | True |
| Weather | 96 | 0.143170 | 0.141425 | -1.22 | True |
| Weather | 192 | 0.184864 | 0.182559 | -1.25 | True |
| Weather | 336 | 0.234520 | 0.231734 | -1.19 | True |
| Weather | 720 | 0.306776 | 0.303611 | -1.03 | True |

## Training Selector Diagnostic

| dataset | loss_mode | best_epoch | best_val_mean_mse | last_epoch | last_val_mean_mse | last_gap_to_best_pct |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | full | 1 | 0.406879 | 10 | 0.491329 | 20.76 |
| ETTh2 | multi-prefix | 1 | 0.400350 | 10 | 0.454948 | 13.64 |
| ETTm2 | full | 3 | 0.184627 | 10 | 0.186479 | 1.00 |
| ETTm2 | multi-prefix | 3 | 0.180145 | 10 | 0.183163 | 1.68 |
| Weather | full | 5 | 0.489505 | 10 | 0.490709 | 0.25 |
| Weather | multi-prefix | 7 | 0.489430 | 10 | 0.489774 | 0.07 |

## Reading Guide

- [Pass] If `multi-prefix` materially reduces ETTm2/Weather gaps without losing ETTh2, the head/interface confounder is strong and HSS should not start from supervision reliability alone.
- [Partial] If `multi-prefix` helps only one degraded dataset, keep D1 but treat unified head design as a co-factor.
- [Fail] If `multi-prefix` does not help, proceed to D1 supervision reliability diagnostic.
