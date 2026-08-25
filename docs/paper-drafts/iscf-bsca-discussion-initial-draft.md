# HoriScope Section 6: Discussion and Limitations

## Draft status

| Field | Content |
| --- | --- |
| `document_role` | Clean manuscript-facing initial draft of Section 6 |
| `version` | `v0.4-author-refinement` |
| `date` | `2026-08-24` |
| `review_status` | `temporarily_frozen_usable` |
| `freeze_date` | `2026-08-24` |
| `freeze_scope` | Section 6 body, paragraph structure, interpretation and methodological limitations |
| `unfreeze_condition` | A concrete contradiction from Section 7 or later manuscript assembly, followed by explicit author approval |
| `upstream_dependency` | Introduction v0.9, Related Work v0.2, Section 3 v0.7, Section 4 v0.7 and Section 5 v0.13 remain temporarily frozen and unchanged |
| `evidence_scope` | CHPC construction, Sections 5.2 and 5.3 experiments, Figure 3 motivation, Figure 5 diagnostics, Core-Ablation, Figure 6 and Figure 7 |
| `experiment_change` | None; no new implementation, training or formal test |
| `claim_boundary` | Discussion interprets the completed evidence and states methodological limitations; it does not introduce new effectiveness or generality claims |
| `narrative_spine` | unified forecasting system → output-side multi-scope design → methodological limitations |

The status table above is editorial metadata and is not part of the manuscript body submitted for review.

## 6. Discussion and Limitations

UVHF requires one forecasting system to serve different request endpoints from a shared prediction trajectory. Horizon-specific predictors optimize each request endpoint separately and therefore do not impose CHPC. HoriScope instead maps a shared history representation and future-step coordinates to one prediction trajectory, so requests with different endpoints become nested prefixes of the same forecast. The experiments in Sections 5.2 and 5.3 show that this consistency constraint can coexist with competitive accuracy, while Figure 6 shows the deployment benefit of consolidating the horizon service into one unified checkpoint.

Unlike many existing multi-scale encoder studies, which extract and mix historical dependencies at multiple input resolutions, HoriScope starts from the output side and organizes how one shared representation generates the future. In these encoders, scale refers to the granularity of historical representation before decoding \citep{challu2023nhits,chen2024pathformer,wang2024timemixer}. HoriScope uses a different axis: it generates candidate futures whose latent states are reused over different contiguous extents. A scope therefore indexes output-side sharing granularity rather than input resolution or receptive field. This distinction shifts the multi-scale question from representation learning to forecast generation. The scope-wise ablations, the future-region analysis in Section 3 and the Figure 5 diagnostics together suggest that different future regions can benefit from different sharing extents, while Target-Adaptive Allocation assigns step-specific preferences among these candidate signals. Multi-scale encoders and HoriScope are therefore methodologically distinct yet complementary: multi-scale encoders organize how historical evidence is represented, whereas HoriScope organizes how a shared representation generates forecasts with multiple future-sharing granularities.

Several methodological limitations delimit the current formulation. First, HoriScope uses a finite set of pre-specified contiguous scopes and future regions. This design assumes that useful sharing patterns can be described by fixed intervals; processes whose dependencies change at irregular locations may require data-adaptive region partitioning. Second, each future region is generated from its own scope-conditioned state, while relationships between regions are represented through the shared history state and future coordinates rather than a dedicated mechanism that passes information from one predicted region to another. Finally, retaining multiple scope branches increases decoder-side computation relative to a single-scope head, even though shared encoders and unified serving reduce duplicated computation and storage. These limitations motivate future extensions with adaptive region partitioning and explicit cross-region forecasting interactions.
