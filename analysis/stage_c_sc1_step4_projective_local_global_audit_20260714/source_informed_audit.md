# SC1 Step 4 Source-Informed Audit: Projective Local-Global Synthesis

## Scope And Search Record

- search date: 2026-07-14；
- query scope: future basis generation、wavelet coefficient forecasting、frequency/global-view forecasting、
  multiresolution interpolation、arbitrary target length、local-global forecasting；
- source policy: external primary sources优先；Zotero未作为completeness证据；
- source status: 下表均由external search发现或重新验证，Zotero presence本轮未查询；
- full-text boundary: official proceedings/arXiv pages已验证；在提出implementation前仍需读candidate相关全文与official code。

## Prior-Art Matrix

| Work | Primary-source fact | Occupied claim | Remaining boundary |
| --- | --- | --- | --- |
| N-BEATS (2020) | neural basis expansion生成forecast | basis coefficients -> forecast不是新概念 | 不讨论unified prefix-domain support crossing |
| N-HiTS (AAAI 2023) | hierarchical interpolation、多rate sampling、不同frequency/scale顺序合成 | multiscale synthesis与hierarchical interpolation已占据 | fixed-horizon hierarchy不是本项目的domain restriction contract |
| BasisFormer (NeurIPS 2023) | learned bases、Coef module、future-view basis consolidation | learned future bases与basis selection已占据 | 未建立short-prefix locality vs long-domain coherence的projective gate |
| FreTS (NeurIPS 2023) | frequency global view与energy compaction改善MLP | “frequency更易学”机制已占据 | 主要是frequency-domain learning，不是future support restriction |
| FBM (NeurIPS 2024) | Fourier basis expansion混合time/frequency features | Fourier basis mapping已占据 | 不提供local-support/global-coherence crossed design |
| WaveToken (ICML 2025) | time-localized wavelet tokenization并forecast horizon coefficients | wavelet coefficients用于forecast generation已直接占据 | tokenizer/autoregression不同于single projective decoder |
| Implicit Forecaster (NeurIPS 2025) | frequency/amplitude/phase waves隐式形成future | global wave synthesis已占据 | 不以prefix support intersection组织local/global atoms |
| FlowState (ICML 2026) | functional basis decoder、dynamic horizon与sampling-rate equivariance | functional basis与varying target length已占据 | sampling-rate equivariance不等同support-scale × prefix crossing |

## Narrative Gate

[Decision] “balanced interval basis用于预测生成”可以作为组件创新，但不能单独成为Contribution 1：basis
forecasting、wavelet coefficients与multiscale synthesis均有直接prior art。D6提供的新问题证据是：在同一
frozen-memory readout中，b144相对global DCT的short/long effect稳定反转，且第二validation window复现。

provisional candidate命名为`SC1-PLGO`（Projective Local-Global Operator）。其可辩护贡献边界必须同时满足：

1. 一个shared future operator服务所有H，不读取离散/连续H作为learned semantic feature；
2. global smooth atoms提供long-domain coherence，interval-local atoms提供short-prefix synthesis；
3. H只限制输出domain并选择与prefix相交的supports；
4. local/global decomposition必须具有stable reconstruction与明确function class；
5. balanced interval construction只作为local scaffold，不claim midpoint/Haar novelty；
6. 实际head必须能选择性计算active coefficients，否则不能claim efficiency；
7. 必须与global-only、local-only、overcomplete-union、random-support、matched-capacity controls比较。

该组合通过Step 4 conditional narrative gate，但**尚未通过Step 5 theory feasibility**，method implementation仍为
false。

## Rejected Shortcuts

- 直接拼接global DCT与local block bases：构成overcomplete non-orthogonal dictionary，capacity与conditioning
  可解释收益，只能作control；
- 根据requested H选择expert/block size：重新引入horizon-specific semantic routing，违反主线；
- A6 output + local residual：属于已拒绝的residual patch路线；
- 把balanced midpoint tree改名为learned wavelet：D4 exact-specificity fail且prior art拥挤；
- 只报告short horizons收益：D6证明long horizons反向，paper必须正面解决crossing。

## Step 5 Theory Questions

下一步只做algebra/function-class feasibility，不训练：

1. 是否存在同时包含global smooth subspace与interval-local detail subspaces的stable analysis/synthesis pair？
2. arbitrary prefix restriction是否保持exact output consistency，而不需要H-conditioned coefficients？
3. 是否能在不退化为full-affine/overcomplete capacity expansion的前提下包含或function-preserve A6 control？
4. local/global atoms的coherence、frame bounds与coefficient identifiability如何控制？
5. active-support synthesis能否对应真实selective coefficient computation？

若1-4任一形成与FPMO相同的no-go boundary，rollback Step 4重新设计；不得用performance sweep绕过theory gate。
