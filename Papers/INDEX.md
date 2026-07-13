# Project Paper Notes

Zotero collection `FSA`（collection key: `UCWGBDUQ`）是seed/reference library，不是完整性或时效性的
source of truth。专题调研默认使用external primary-source search，并把重要结果追加到本索引。

## Zotero Seed 条目

| 方向 | Paper note | Zotero key |
| --- | --- | --- |
| future-aware / alignment | [TimeAlign](timealign-bridging-past-and-future.md) | `9JK37FWJ` |
| prototype / interpretability | [ProtoTS](protots-hierarchical-prototypes.md) | `SLZEMUSJ` |
| generalized forecasting / future queries | [TIMEPERCEIVER](timeperceiver-generalized-forecasting.md) | `34AMEC37` |
| MoE / distribution shift | [TFPS](tfps-pattern-specific-experts.md) | `PXCHMY4H` |
| varied horizon | [ElasTST](elastst-varied-horizon.md) | `MXLVX75Z` |
| objective / multi-step loss | [QDF / MetaDF](qdf-quadratic-direct-forecast.md) | `R8YQ4UWY` |
| objective / transformed label alignment | [TransDF](transdf-transformed-label-alignment.md) | `BK8HKCXT` |
| structure-aware MoE | [AME-TS](ame-ts-anchored-moe.md) | `IZMRGTIG` |
| static-dynamic decomposition | [TimeEmb](timeemb-static-dynamic-disentanglement.md) | `VFRBDA4N` |
| non-stationarity / dual-domain | [DTAF](dtaf-temporal-stabilization-frequency-differencing.md) | `LAKF59WZ` |
| step-specific representation | [SRP++](srp-step-specific-representation.md) | `YRL4AHYC` |
| segment-wise MoE | [Seg-MoE](seg-moe-segment-wise-routing.md) | `PY6VZSMM` |
| heterogeneous experts | [MoHETS](mohets-heterogeneous-experts.md) | `WSKUSM6X` |

## External Thematic Audits

| 方向 | Paper note | Search date |
| --- | --- | --- |
| varied-horizon decoder / training objective | [StageC external decoder/objective audit](stage-c-external-decoder-objective-audit.md) | 2026-07-13 |

## 初步聚类

- one model for multi-horizon: ElasTST, TIMEPERCEIVER, SRP++, QDF, TransDF.
- future-aware architecture: TimeAlign, TIMEPERCEIVER, ElasTST, SRP++, TimeEmb.
- MoE and conditional computation: AME-TS, Seg-MoE, MoHETS, TFPS, DTAF.
- non-stationarity and representation correction: TimeAlign, TimeEmb, DTAF, TFPS.

## 当前边界

- Zotero seed notes来自`FSA` collection；它们不代表完整或最新文献覆盖。
- external thematic audits使用arXiv、OpenReview、正式会议论文集、official project page与official code；
  投稿前仍需执行freshness search并将最终引用回填Zotero。
- 未读取或迁移旧 `R_2026_FSA` 仓库内容。
- 本批 notes 是第一版机制速读，不等价于完整复现审计。
