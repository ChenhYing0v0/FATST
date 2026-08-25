# HoriScope Section 7: Conclusion

## Draft status

| Field | Content |
| --- | --- |
| `document_role` | Clean manuscript-facing initial draft of Section 7 |
| `version` | `v0.2-author-fixed` |
| `date` | `2026-08-24` |
| `review_status` | `temporarily_frozen_usable` |
| `freeze_date` | `2026-08-24` |
| `freeze_scope` | Section 7 body, two-paragraph structure, evidence selection and claim boundary |
| `unfreeze_condition` | A concrete contradiction during full-manuscript assembly, followed by explicit author approval |
| `structure_source` | `docs/paper-drafts/iscf-bsca-conclusion-structure-design.md` |
| `upstream_dependency` | Introduction v0.9, Related Work v0.2, Section 3 v0.7, Section 4 v0.7, Section 5 v0.13 and Section 6 v0.4 remain temporarily frozen and unchanged |
| `section_format` | No subsections; two compact paragraphs |
| `evidence_scope` | CHPC construction and the completed experiments in Sections 5.2--5.7 |
| `experiment_change` | None; no new implementation, training or formal test |
| `claim_boundary` | No new results, citations, mechanisms, limitations or generality claims are introduced |

The status table above is editorial metadata and is not part of the manuscript body submitted for review.

## 7. Conclusion

UVHF requires one model to serve different request endpoints without changing predictions for their shared future steps. We formulated this requirement as cross-horizon prefix consistency (CHPC) and identified future-region sharing-demand heterogeneity as a complementary decoder-side challenge. To address both, HoriScope constructs a scope-indexed forecast field through region-wise generation at multiple output-side sharing extents and integrates its Scope-conditioned Forecasts through Target-Adaptive Allocation. BSCA supports the joint learning of the fused trajectory and all scope lines without changing the inference graph; the resulting horizon-agnostic trajectory provides the nested prefixes returned for different horizon requests.

Across seven multivariate benchmarks and four forecast horizons, HoriScope used one unified model per dataset, achieved stronger aggregate forecasting accuracy than the evaluated horizon-specific baselines and ranked first under the one-model-all-horizons comparison. The unified architecture also achieved a favorable balance between forecasting accuracy, checkpoint storage and inference memory. Controlled ablations supported the contribution of multi-scope generation, scope-specific projections, Target-Adaptive Allocation and BSCA. A selected internal diagnostic illustrated distinct scope signals and region-dependent allocation, while experiments with DLinear and PatchTST supported decoder compatibility across the two evaluated Encoder families. Together, these results support output-side sharing granularity as a practical design axis for unified forecasters that serve multiple horizons through one prefix-consistent trajectory.
