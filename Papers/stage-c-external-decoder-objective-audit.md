# StageC External Decoder / Objective Prior-Art Audit

## Scope

- Search date: 2026-07-13
- Role: PMFO/PIR initial Step 4-6 + PMFO-v1 redesign + FPMO Step5-6 theory/narrative audit
- Discovery policy: Zotero is a seed/reference library；本轮以external primary-source search为主
- Full-text/code status: 上述forecasting/operator与matrix-factorization工作均检查论文页面或全文；对
  source-informed implementation相关的forecasting工作另审计official code。2026部分preprint只作为
  freshness pressure，不作为已确立SOTA事实

## Core Finding

[Strong Evidence] “arbitrary horizon”“continuous basis”“hierarchical interpolation”“learned wavelet”、
“harmonic horizon weighting”与“linear factorization带来implicit optimization bias”均已有直接prior art。
PMFO-RCT v1被Step 7B effectiveness gate否定；FPMO-DS又在Step 6被证明与full-affine DA拥有同一function
class，且该factorization不依赖真实scale coordinates。因此SC1目前没有paper-core candidate，先rollback
Step 2/3诊断scale-aligned nonlinearity。SC2仍是在同一projectors上由deployment measure诱导的
cross-scale-decoupled quadratic risk，但在新operator冻结前held。

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

## Step 6 Narrative Outcome

- [Implicit Regularization in Matrix Factorization](https://proceedings.neurips.cc/paper_files/paper/2017/hash/58191d2a914c6dae66371c9dcdc91b41-Abstract.html)
  与[Implicit Regularization in Deep Matrix Factorization](https://proceedings.neurips.cc/paper/2019/hash/c0c783b5fc0d7d808f1d14a6e9c8280d-Abstract.html)
  说明full-dimensional linear factorization即使不改变可表示矩阵集合，也可能改变gradient-based
  optimization的implicit bias。该mechanism category不是StageC的新贡献。
- T720下`FPMO-DS`每个scale width等于该组row count，故`D_l A_l`可表示任意block affine map；其class与
  `FPMO-DA`完全相同。把balanced interval basis换成random orthogonal basis、或随机分组rows，结论不变。
- 当前TimeAlign使用的optimizer/loss/joint training不满足直接宣称minimum nuclear norm等既有理论的
  条件；若DS实测改善，必须先视为optimization-parameterization hypothesis。
- per-scale nonlinear extension不是当前DS的implementation detail：它需要重新解决exact A6 containment、
  matched dense/random controls和N-HiTS/PRISM prior-art边界。

[Decision] `FPMO-DS rejected_by_narrative_gate`，不实现、不训练。下一步D2只诊断rank、generic
nonlinearity与true-scale alignment；若scale grouping不能稳定超过dense nonlinear和random controls，
Contribution 1 problem需再次重定义。完整报告见
`analysis/stage_c_step6_fpmo_narrative_control_20260713/step6_narrative_control_gate.md`。

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

## 2026-07-14 D6 Projective Local-Global Update

external primary-source refresh覆盖N-BEATS、N-HiTS、BasisFormer、FreTS、FBM、WaveToken、Implicit Forecaster
与FlowState。basis forecast generation、wavelet coefficients、multiscale interpolation、frequency global view与
dynamic target length均已有直接prior art。

D6的新证据边界不是“balanced basis首次用于forecast”，而是同一full-domain readout中local b144相对global
DCT在short horizons `+1.1964%`、long horizons `-1.2675%`，12/15 primary units crossing，并在disjoint
validation window复现。provisional `SC1-PLGO`只以“domain restriction自然协调local-prefix synthesis与
global-domain coherence”的组合进入Step 5 theory feasibility；balanced interval只保留为local support scaffold。

详细source matrix与rejected shortcuts见
`analysis/stage_c_sc1_step4_projective_local_global_audit_20260714/source_informed_audit.md`。
