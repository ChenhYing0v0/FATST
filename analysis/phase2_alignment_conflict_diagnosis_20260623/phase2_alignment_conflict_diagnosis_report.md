# Phase2-A Alignment Conflict Diagnosis

## Purpose

[Fact] This report uses the completed Phase2-A artifacts to diagnose why uniform future-state alignment improved `ETTm1` but degraded every `ETTh2` horizon.

## Main Finding

[Strong Evidence] The failure pattern is more consistent with target-state geometry conflict than with pure reconstruction-scale imbalance.

- `ETTh2` is the only dataset with all-horizon degradation vs R.3.
- `ETTh2` also has the lowest teacher/student cosine and highest local alignment loss.
- `Weather` has extremely large reconstruction loss but only slight average degradation/improvement mix, so raw reconstruction scale alone does not explain the failure.

## Dataset Summary

| Dataset | Mean MSE vs R.3 | Wins | Mean cosine | Local loss | Relation loss | Recon loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | +5.25% | 0/4 | 0.6363 | 0.3637 | 0.0953 | 0.6275 |
| ETTm1 | -1.29% | 4/4 | 0.8300 | 0.1700 | 0.0712 | 0.1843 |
| Weather | -0.07% | 3/4 | 0.8220 | 0.1780 | 0.0563 | 637.1230 |

## Correlation Diagnostics

| Quantity pair | Pearson r | Interpretation |
| --- | ---: | --- |
| MSE delta vs teacher/student cosine | `-0.8866` | negative means lower cosine tends to coincide with worse MSE delta |
| MSE delta vs local alignment loss | `0.8866` | positive means stronger mismatch tends to coincide with worse MSE delta |
| MSE delta vs relation alignment loss | `0.1235` | relation mismatch is weaker evidence if this value is small |
| MSE delta vs reconstruction loss | `-0.2911` | near zero or negative weakens the pure scale-imbalance explanation |

## Per-Setting Rows

| Dataset | Horizon | MSE vs R.3 | Cosine | Local loss | Relation loss | Recon loss | Bucket |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ETTh2 | 96 | +7.64% | 0.6434 | 0.3566 | 0.0435 | 0.4392 | degrade/low_cosine |
| ETTh2 | 192 | +6.97% | 0.6452 | 0.3548 | 0.0749 | 0.4863 | degrade/low_cosine |
| ETTh2 | 336 | +2.73% | 0.6378 | 0.3622 | 0.1128 | 0.6108 | degrade/low_cosine |
| ETTh2 | 720 | +3.64% | 0.6188 | 0.3812 | 0.1499 | 0.9738 | degrade/low_cosine |
| ETTm1 | 96 | -1.21% | 0.8363 | 0.1637 | 0.0566 | 0.0959 | improve/high_cosine |
| ETTm1 | 192 | -1.10% | 0.8345 | 0.1655 | 0.0672 | 0.1303 | improve/high_cosine |
| ETTm1 | 336 | -1.37% | 0.8300 | 0.1700 | 0.0748 | 0.1957 | improve/high_cosine |
| ETTm1 | 720 | -1.47% | 0.8194 | 0.1806 | 0.0860 | 0.3154 | improve/high_cosine |
| Weather | 96 | +0.41% | 0.8235 | 0.1765 | 0.0273 | 378.8353 | degrade/high_cosine |
| Weather | 192 | -0.18% | 0.8320 | 0.1680 | 0.0437 | 478.3215 | improve/high_cosine |
| Weather | 336 | -0.24% | 0.8228 | 0.1772 | 0.0650 | 585.6189 | improve/high_cosine |
| Weather | 720 | -0.29% | 0.8098 | 0.1902 | 0.0893 | 1105.7161 | improve/high_cosine |

## Decision Impact

[Inference] If Phase2-R.1 confidence weighting fixes `ETTh2`, the paper story should focus on reliability-aware future-state calibration. If it fails, the correct rollback is not larger teacher capacity or MoE, but step 2-3: redefine the decoder problem around output-process / error-process modeling.

[Next] When Phase2-R.1 artifacts become available, compare its confidence statistics against this Phase2-A conflict map. The decisive question is whether down-weighting low-reliability teacher anchors reduces `ETTh2` degradation without erasing `ETTm1/Weather` gains.
