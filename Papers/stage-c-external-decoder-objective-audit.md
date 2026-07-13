# StageC External Decoder / Objective Prior-Art Audit

## Scope

- Search date: 2026-07-13
- Role: PMFO/PIR Step 4-6 thematic note
- Discovery policy: Zotero is a seed/reference library；本轮以external primary-source search为主
- Full-text/code status: ElasTST、N-HiTS、BasisFormer、Multiwavelet Operator、FlowState、TimePerceiver与
  Implicit Forecaster均检查论文页面或全文及official implementation；2026部分preprint只作为freshness
  pressure，不作为已确立SOTA事实

## Core Finding

[Strong Evidence] “arbitrary horizon”“continuous basis”“hierarchical interpolation”“learned wavelet”与
“harmonic horizon weighting”均已有直接prior art。StageC可辩护的窄边界是：future-side
refinement-conservative tree、domain-only pruning，以及同一projectors上由deployment measure诱导的
cross-scale-decoupled quadratic risk。

完整source matrix、tensor contract、proof、controls与decision见：

- `analysis/stage_c_step46_pmfo_pir_theory_gate_20260713/step46_design_and_prior_art.md`
- `analysis/stage_c_step46_pmfo_pir_theory_gate_20260713/theory_gate_report.md`

## Adopted Evidence

- [ElasTST](https://arxiv.org/abs/2411.01842)：horizon invariance与raw horizon reweighting作为mandatory
  controls；不采用placeholder/query architecture。
- [N-HiTS](https://arxiv.org/abs/2201.12886)：证明coarse-to-fine interpolation本身不新；PMFO必须提供
  exact conservative refinement。
- [FlowState](https://arxiv.org/abs/2508.05287)：functional basis + dynamic target length已被占据；PMFO不
  claim continuous basis novelty。
- [TimePerceiver](https://arxiv.org/abs/2512.22550)：target timestamp query与decoder-training co-design已被
  占据；PMFO保持H不进入learned state。
- [Implicit Forecaster](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e82ef0c89df6a6eff8734ea7e27c42f-Abstract-Conference.html)：global wave synthesis已被占据；DCT/Fourier只作control。
- [Multiwavelet Operator](https://arxiv.org/abs/2109.13459)：采用nested-space与perfect-reconstruction数学
  工具作为feasibility evidence；不声称首次使用multiwavelet/operator。

## Rejected Transfer

- 不复制upstream architecture或API；
- 不把future coordinate、horizon ID、benchmark horizon embedding加入learned path；
- 不把input-side wavelet decomposition改名为future operator；
- 不把raw harmonic weighting包装为SC2；
- 不把temporal aggregation hierarchy的coherence等同于prefix projectivity。

## Freshness Risk

[Risk] 2025-2026 arbitrary-horizon、wavelet/lifting与neural-operator工作增长快。本note不是absence proof。
投稿前必须重新执行external search与citation chaining，并把最终采用的external papers回填Zotero。
