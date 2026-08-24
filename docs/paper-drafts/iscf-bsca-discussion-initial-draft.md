# ISCF-BSCA Section 6: Discussion and Limitations

## Draft status

| Field | Content |
| --- | --- |
| `document_role` | Clean manuscript-facing initial draft of Section 6 |
| `version` | `v0.1-methodological-limitations` |
| `date` | `2026-08-24` |
| `review_status` | Initial author-review draft; Section 5 v0.13 and Figures 5--7 remain unchanged |
| `upstream_dependency` | Introduction v0.9, Related Work v0.2, Section 3 v0.7, Section 4 v0.7 and Section 5 v0.13 remain temporarily frozen and unchanged |
| `evidence_scope` | CHPC construction, Main-I, Main-II, Figure 3 motivation, Figure 5 diagnostics, Core-Ablation, Figure 6 and Figure 7 |
| `experiment_change` | None; no new implementation, training or formal test |
| `claim_boundary` | Discussion interprets the completed evidence and states methodological limitations; it does not introduce new effectiveness or generality claims |
| `narrative_spine` | unified forecasting system → output-side multi-scope design → methodological limitations |

The status table above is editorial metadata and is not part of the manuscript body submitted for review.

## 6. Discussion and Limitations

The experiments support viewing varied-horizon forecasting as a system-level problem rather than a choice of output length. Horizon-specific predictors optimize each request endpoint separately and therefore do not impose CHPC. ISCF instead maps a shared history representation and future-step coordinates to one prediction trajectory, so requests with different endpoints become nested prefixes of the same forecast. Main-I and Main-II show that this consistency constraint can coexist with competitive accuracy, while Figure 6 shows the deployment benefit of consolidating the horizon service into one unified checkpoint. CHPC itself is a structural contract, not an accuracy guarantee; the empirical advantage comes from the decoder and training design evaluated with it.

ISCF also distinguishes output-side multi-scope forecasting from the multi-scale encoder designs discussed in Section 2. Multi-scale encoders generally extract and mix dependencies from the history at multiple input resolutions, so their scales index the granularity of the representation before decoding \citep{challu2023nhits,chen2024pathformer,wang2024timemixer}. ISCF uses a different axis: it starts from one shared history representation and generates candidate futures whose latent states are reused over different contiguous extents. A scope therefore indexes output-side sharing granularity rather than input resolution or receptive field. This distinction moves multi-scale design from representation learning to forecast synthesis. The scope-wise ablations, the future-region analysis in Section 3 and the Figure 5 diagnostics together suggest that different future regions can benefit from different sharing extents, while Target-Adaptive Allocation assigns step-specific preferences among these candidate signals. ISCF is therefore complementary to multi-scale encoders: the former organizes how a shared representation is converted into a future trajectory, whereas the latter organizes how historical evidence is encoded.

Several methodological limitations delimit the current formulation. First, ISCF assumes that future sharing can be described by a finite set of contiguous scopes and regions; processes with irregular, non-contiguous or rapidly changing dependencies may require learned boundaries or non-contiguous grouping. Second, each region is synthesized from a scope-conditioned state and a step-wise generator, while cross-region interactions are mediated through shared history and future coordinates rather than an explicitly evolving dependency graph. Third, Scope Probabilities form a soft convex allocation of point forecasts, which expresses preference over sharing granularity but does not represent predictive uncertainty or multimodal futures. Finally, retaining multiple scope branches increases decoder-side computation relative to a single-scope head, even though shared encoders and unified serving reduce duplicated computation and storage. These limitations motivate future extensions with probabilistic outputs, adaptive region boundaries and explicit interactions among future regions.
