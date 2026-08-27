# HoriScope Section 7: Conclusion

## Draft status

| Field | Content |
| --- | --- |
| `document_role` | Clean manuscript-facing initial draft of Section 7 |
| `version` | `v0.3-author-refinement` |
| `date` | `2026-08-27` |
| `review_status` | `author-refined candidate pending renewed confirmation` |
| `freeze_date` | `Pending renewed author confirmation` |
| `freeze_scope` | Section 7 body, two-paragraph structure, evidence selection and claim boundary |
| `unfreeze_condition` | A concrete contradiction during full-manuscript assembly, followed by explicit author approval |
| `structure_source` | `docs/paper-drafts/iscf-bsca-conclusion-structure-design.md` |
| `upstream_dependency` | Introduction v0.9, Related Work v0.2, Section 3 v0.7, Section 4 v0.7, Section 5 v0.15 and Section 6 v0.5 remain temporarily frozen and unchanged |
| `section_format` | No subsections; two compact paragraphs |
| `evidence_scope` | CHPC construction and the completed experiments in Sections 5.2--5.7 |
| `experiment_change` | None; no new implementation, training or formal test |
| `claim_boundary` | No new results, citations, mechanisms, limitations or generality claims are introduced |

The status table above is editorial metadata and is not part of the manuscript body submitted for review.

## 7. Conclusion

This work reframes multi-horizon service as the construction of one coherent future trajectory, with request endpoints exposed through nested prefixes. Within this formulation, CHPC defines the required cross-horizon coherence, and output-side sharing granularity becomes an explicit dimension of forecast generation. HoriScope realizes this principle by constructing Scope-conditioned Forecasts across multiple sharing extents and integrating them through Target-Adaptive Allocation. BSCA supplies balanced supervision to the jointly optimized scope lines. The resulting architecture combines region-sensitive forecast generation with prefix-consistent inference in a single horizon-agnostic model.

Across seven multivariate benchmarks, HoriScope achieved state-of-the-art aggregate accuracy among the evaluated methods while consolidating every supported horizon into one model. Its advantage persisted under a common one-model-all-horizons workflow, and controlled ablations and two-backbone studies supported component utility and decoder portability. These findings connect prefix-consistent serving with strong forecasting accuracy and efficient model consolidation in one architecture. They also position output-side sharing granularity as a productive design axis for unified time-series forecasting.
