# Post-D20 Predictive-Support Prior-Art Audit

## Metadata

- Search date: `2026-07-20`
- Scope: spectral shift、frequency stability/energy、robust forecasting、dynamic-horizon decoder、future-coordinate generation
- Discovery: external primary-source search；Zotero仅为seed，未用于absence或freshness判断
- Source status: proceedings/OpenReview/arXiv official pages；withdrawn/submission work仅作overlap pressure

## D20 evidence that triggered the audit

D20的low-frequency SPEC相对same-run A6在validation为`+0.5755%` MSE，到official test反转为`-0.7614%`；
相对同维RANDOM从validation `+0.7288%`缩到test `+0.1412%`。Dense test specificity从H1–48的`+0.9852%`
逐步衰减，到H513–720为`-0.0114%`。因此问题不是frequency path是否active，而是其predictive utility是否随
distribution和future distance改变。

## Primary-source boundary

| Work | Mechanism coverage | Boundary |
| --- | --- | --- |
| [Frequency Matters](https://arxiv.org/abs/2511.05619) | spectral mismatch解释TSFM跨domain degradation | spectral shift本身不是新问题 |
| [DropoutTS, ICML 2026](https://openreview.net/forum?id=7sksHLUvhH) | spectral sparsity估计instance noise并控制adaptive dropout | sample-adaptive spectral robustness已有强prior |
| [Fremen](https://openreview.net/forum?id=4IZzgIyD91) | frequency-wise stationarity measurement与unstable-frequency downweighting | frequency stability weighting有直接邻近；withdrawn，置信度降低 |
| [Adaptive Energy Amplification](https://openreview.net/forum?id=O5uoS9ICec) | selective frequency energy enhancement与noise suppression | energy calibration不是空白primitive；submission status |
| [Implicit Forecaster](https://openreview.net/forum?id=gqoeQPhQcE) | frequency/amplitude/phase full-trajectory synthesis与spectrum skip | spectral generation/skip已有覆盖 |
| [FlowState](https://openreview.net/forum?id=R50AT6nAsM) | functional basis decoder与dynamic horizons | flexible-horizon basis不是独立novelty |
| [TimePerceiver](https://arxiv.org/abs/2512.22550) | target-coordinate query与decoder/training co-design | coordinate query本身不能claim |

## Claim boundary

下列路线不进入paper method：

- 对D20 summary做RMS normalization或scalar gate；
- generic spectral denoising、frequency stability weighting或adaptive energy；
- 把future coordinate query、functional basis或spectral skip重新命名。

可能保留的完整问题链是：

`fixed past -> historical evidence -> future-coordinate predictive support -> one full trajectory -> prefix crop`。

其区别不在primitive，而在multi-horizon contract：同一history evidence只能在其可预测的future-distance范围内影响
trajectory，并且不能读取requested horizon。该chain目前只是hypothesis；必须证明support可以从past估计、跨split
稳定、超过generic coordinate/capacity controls，才有Contribution-level novelty讨论价值。

## Provisional research opportunity

- Contribution 1 hypothesis：`Support-Calibrated Trajectory Operator`；
- Contribution 2 hypothesis：train-only multi-cut `Retrospective Support Calibration`；
- current status：`problem_unverified / method_not_authorized`。

最近工作的overlap会收紧claim，但不自动否决完整chain。相反，如果后续只实现frequency reweighting或普通
coordinate gate，就会被上述prior substantially覆盖。

## D20-D1 update

D1显示SPEC contribution相对其co-adapted base在39/40 bins有益、macro gain `+26.89%`，median oracle scale
`1.26`；RANDOM path也在35/40 bins有益。由于完整SPEC/RANDOM仍差于same-run A6，within-model importance主要
反映joint responsibility relocation，而非新增信息。该结果否定scalar shrink rescue，也没有建立future-distance
support。两项provisional contributions继续保持`problem_unverified / method_not_authorized`。
