# HoriScope Section 6: Discussion and Limitations

## Draft status

| Field | Content |
| --- | --- |
| `document_role` | Clean manuscript-facing initial draft of Section 6 |
| `version` | `v0.5-author-refinement` |
| `date` | `2026-08-27` |
| `review_status` | `author-refined candidate pending renewed confirmation` |
| `freeze_date` | `Pending renewed author confirmation` |
| `freeze_scope` | Section 6 body, paragraph functions, interpretation and methodological limitations |
| `unfreeze_condition` | A concrete contradiction from later manuscript assembly, followed by explicit author approval |
| `upstream_dependency` | Introduction v0.9, Related Work v0.2, Section 3 v0.7, Section 4 v0.7 and Section 5 v0.15 remain temporarily frozen and unchanged |
| `evidence_scope` | CHPC construction, Sections 5.2 and 5.3 experiments, Figure 3 motivation, Figure 5 diagnostics, Core-Ablation, Figure 6 and Figure 7 |
| `experiment_change` | None; no new implementation, training or formal test |
| `claim_boundary` | Discussion interprets the completed evidence and states methodological limitations; it does not introduce new effectiveness or generality claims |
| `narrative_spine` | unified forecasting system → output-side multi-scope design → methodological limitations |

The status table above is editorial metadata and is not part of the manuscript body submitted for review.

## 6. Discussion and Limitations

**Unified varied-horizon forecasting.** UVHF serves different request endpoints from a single prediction trajectory. Horizon-specific predictors optimize each endpoint independently and do not enforce CHPC across their overlapping outputs. HoriScope maps a shared history representation and future-step coordinates to one trajectory, whose prefixes answer requests of different lengths. The experiments in Sections 5.2 and 5.3 show that this structural consistency coexists with strong forecasting accuracy, and Figure 6 demonstrates the deployment benefit of consolidating all horizon requests into one checkpoint.

**Output-side multi-scope generation.** Existing multi-scale encoders extract and combine historical dependencies at multiple input resolutions, where scale describes the granularity of history representation before decoding \citep{challu2023nhits,chen2024pathformer,wang2024timemixer}. HoriScope introduces a distinct output-side formulation in which candidate forecasts reuse history-conditioned latent states over different contiguous future extents. Each scope therefore specifies the sharing granularity of forecast generation, shifting multi-scale modeling from history representation to future construction. Evidence from the scope-wise ablations, the future-region analysis in Section 3 and the Figure 5 diagnostics indicates that different future regions can benefit from different sharing extents. Target-Adaptive Allocation assigns step-specific preferences across these candidate forecast signals. Multi-scale encoders and HoriScope address separate yet complementary stages of the forecasting pipeline. Multi-scale encoders organize historical evidence, while HoriScope converts the resulting shared representation into forecasts at multiple future-sharing granularities.

**Methodological limitations.** The current formulation relies on a finite set of pre-specified contiguous scopes and future regions. This assumption may be restrictive when useful sharing patterns change at irregular locations, motivating data-adaptive region partitioning. Interactions among region forecasts are mediated through the shared History State and Future Coordinate. A dedicated cross-region forecasting mechanism could model dependencies that evolve across the future domain more explicitly. Multiple scope branches also increase decoder-side computation compared with a single-scope head, although shared encoding and unified serving reduce duplicated computation and storage. Future work should examine adaptive partitioning, cross-region interactions and more efficient multi-scope generation.
