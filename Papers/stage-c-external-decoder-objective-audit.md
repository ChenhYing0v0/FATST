# StageC External Decoder / Objective Prior-Art Audit

## Scope

- Search date: 2026-07-13
- Role: PMFO/PIR initial Step 4-6 + PMFO-v1 failure后的source-informed redesign note
- Discovery policy: Zotero is a seed/reference library；本轮以external primary-source search为主
- Full-text/code status: ElasTST、N-HiTS、BasisFormer、Multiwavelet Operator、FlowState、TimePerceiver与
  Implicit Forecaster均检查论文页面或全文及official implementation；2026部分preprint只作为freshness
  pressure，不作为已确立SOTA事实

## Core Finding

[Strong Evidence] “arbitrary horizon”“continuous basis”“hierarchical interpolation”“learned wavelet”与
“harmonic horizon weighting”均已有直接prior art。PMFO-RCT v1又被Step 7B effectiveness gate否定，故
fixed future tree本身也不再是active claim。SC1当前只保留更窄的source-informed边界：**future-domain
function-preserving operator morphism + domain-only restriction + perfect reconstruction**。SC2继续是同一
projectors上由deployment measure诱导的cross-scale-decoupled quadratic risk，但在新operator冻结前held。

完整source matrix、tensor contract、proof、controls与decision见：

- `analysis/stage_c_step46_pmfo_pir_theory_gate_20260713/step46_design_and_prior_art.md`
- `analysis/stage_c_step46_pmfo_pir_theory_gate_20260713/theory_gate_report.md`
- `analysis/stage_c_step4_source_informed_redesign_20260713/step4_source_informed_redesign_audit.md`

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

## Post-v1 Redesign Evidence

- [PRISM](https://arxiv.org/abs/2512.24898)与[official code](https://github.com/nerdslab/PRISM)：generic
  multiresolution forecasting tree、band routing与level fusion已被直接占据；代码审计显示它使用history-side
  overlapping split与各level fixed-`pred_len` dense heads，不提供future-prefix restriction。
- [LeapTS](https://arxiv.org/abs/2605.10292)：dynamic prediction scale / advancement scheduling已被占据；
  StageC不采用learned horizon/scale controller。
- [Hierarchical nested-basis network](https://arxiv.org/abs/1808.02376)：$\mathcal H^2$ nested transfer是
  operator feasibility evidence，不是novelty claim。
- [Lifting scheme](https://doi.org/10.1137/S0036141095289051)：perfect reconstruction、interval/domain
  adaptation与fast transform已有经典基础；lifting只作为构造工具。
- [Net2Net](https://research.google/pubs/net2net-accelerating-learning-via-knowledge-transfer/)与
  [Network Morphism](https://proceedings.mlr.press/v48/wei16.html)：generic function-preserving architecture
  transformation已有系统研究；StageC必须证明更具体的future restriction contract。
- [Asymmetric MMF](https://arxiv.org/abs/1910.05132)：global low-rank + hierarchical residual已有直接
  matrix-factorization先例；因此拒绝“A6 output + multiresolution residual patch”作为paper core。
- [Unbalanced Haar](https://doi.org/10.1198/016214507000000860)：arbitrary breakpoint/interval上的
  orthonormal Haar-like basis已有统计学基础；FPMO采用data-independent midpoint版本解决non-dyadic $T$，
  不采用其data-adaptive basis selection，也不claim interval wavelet novelty。

## Step 5 Theory Outcome

[Fact] arbitrary-length interval morphism、exact A6 embedding与native prefix restriction在9个$T$、53个
$(T,H)$ cases上通过，max algebraic gap为`5.329e-14`。

[Boundary] exact shared-latent morph只是A6的bijective coordinate transform，只能作control。独立scale maps
虽然严格包含A6，但T720下扩展到full-affine class；它必须与同function-class dense control比较，不能把
capacity gain写成multiresolution mechanism。完整proof/no-go audit见
`analysis/stage_c_step5_fpmo_theory_20260713/step5_theory_feasibility.md`。

## Rejected Transfer

- 不复制upstream architecture或API；
- 不把future coordinate、horizon ID、benchmark horizon embedding加入learned path；
- 不把input-side wavelet decomposition改名为future operator；
- 不把raw harmonic weighting包装为SC2；
- 不把temporal aggregation hierarchy的coherence等同于prefix projectivity。
- 不把trained-checkpoint morphing等同于from-scratch method effectiveness或paper novelty。

## Freshness Risk

[Risk] 2025-2026 arbitrary-horizon、wavelet/lifting与neural-operator工作增长快。本note不是absence proof。
投稿前必须重新执行external search与citation chaining，并把最终采用的external papers回填Zotero。
