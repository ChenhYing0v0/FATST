# Post-D19 Compact Statistic Decoder Prior-Art 审计

## 元数据

- 检索日期：`2026-07-19`
- 主题：compact forecast decoder、history-spectrum shortcut、sample-conditioned basis、phase-wise
  prediction、unified/dynamic horizon forecasting
- 来源规则：优先external primary-source search
- 来源：arXiv、OpenReview、NeurIPS/ICLR proceedings、official code repositories
- Zotero状态：不将其作为完整性证据，本轮未逐篇复核是否已收录
- 覆盖状态：下述关键边界均通过external search独立刷新
- 置信度：已发表论文的机制边界为high；精确的negative novelty判断为medium，因为2026 forecasting文献仍在快速变化

## 研究问题

D19表明：将720-point normalized-history spectrum加入IF decoder后，相对同一个frequency decoder的no-skip
control有所改善；但完整IF仍差于A6和parameter-matched direct nonlinear head。因此需要判断：**compact direct
history statistic**能否成为真正新颖的unified multi-horizon decoder的一部分。

## Primary-source 边界

| Work | Primary source | 已覆盖的相关机制 | 本项目边界 |
| --- | --- | --- | --- |
| FITS, ICLR 2024 | [OpenReview](https://openreview.net/forum?id=bWcnvZ3qMb), [arXiv](https://arxiv.org/abs/2307.03756) | low-pass history spectrum、complex linear frequency interpolation、约10k parameters | compact frequency interpolation并非新机制 |
| Linear forecasting model analysis, ICML 2024 | [arXiv](https://arxiv.org/abs/2403.14587) | FITS/DLinear等full linear transforms可与linear regression function-class等价 | Fourier reparameterization本身不能建立机制创新 |
| FBM, NeurIPS 2024 | [Proceedings](https://papers.neurips.cc/paper_files/paper/2024/hash/0fd4ce94d29be88a5a262a2c77a18f47-Abstract-Conference.html), [paper](https://papers.neurips.cc/paper_files/paper/2024/file/0fd4ce94d29be88a5a262a2c77a18f47-Paper-Conference.pdf) | Fourier basis expansion融合time-frequency features并处理starting-cycle/series-length ambiguity | explicit time-frequency basis mapping已有覆盖 |
| BasisFormer | [arXiv](https://arxiv.org/abs/2310.20496) | history/future basis views、coefficient similarity与basis consolidation | history-conditioned basis selection并非空白prior art |
| Implicit Forecaster, NeurIPS 2025 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e82ef0c89df6a6eff8734ea7e27c42f-Abstract-Conference.html), [OpenReview](https://openreview.net/forum?id=gqoeQPhQcE) | amplitude/phase heads、fixed frequency pool、input-spectrum skip、full-wave synthesis | wave decoding与spectral skip已有覆盖 |
| TimePerceiver, NeurIPS 2025 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c6c682ba9bd8839104f2a82901da4109-Abstract-Conference.html) | target timestamp queries与decoder-training co-design | generic target-coordinate decoding已有覆盖 |
| FlowState, 2025 | [OpenReview](https://openreview.net/forum?id=R50AT6nAsM) | functional basis decoder与dynamic forecasting horizons | functional basis加flexible horizon已有覆盖 |
| PhaseFormer, ICLR 2026 | [arXiv](https://arxiv.org/abs/2510.04134), [OpenReview](https://openreview.net/forum?id=Lk9SqMQzhX) | phase-wise prediction、compact phase embeddings、lightweight routing、约1k parameters | compact phase-based forecasting存在直接prior art |
| N-HiTS | [arXiv](https://arxiv.org/abs/2201.12886) | hierarchical interpolation与long-horizon multi-rate synthesis | compact multiscale interpolation已有覆盖 |

## 关键代数边界

设完整history transform为可逆线性映射$s=Qx$，则linear forecast head

$$
\hat y=W_s s+b=W_sQx+b
$$

与direct history-to-future map属于相同affine function class。ICML 2024对包含FITS在内的若干linear forecasting
architectures给出了更强的formal treatment。因此：

1. 使用全部real/imaginary Fourier bins不足以声称独立forecasting mechanism；
2. compact statistic必须施加有意义的subspace或interaction constraint；
3. frequency specificity必须超过同维random/alternative projection control；
4. 若generic added-history-feature control可以解释收益，就不能仅凭prediction gain归因于spectral semantics。

## Candidate family 决策

### R1 — Smaller implicit frequency decoder

`rejected_by_narrative_gate`.

缩小D19 hidden width可能改善generalization，但FITS、FBM、IF与PhaseFormer已覆盖compact frequency
interpolation、explicit Fourier basis features、implicit wave decoding与compact phase modeling。没有新的problem
contract时，smaller IF只是engineering rescue。

### R2 — History-phase-continued future atoms

`rejected_by_narrative_gate_for_method / retained_as_control_primitive`.

从observed history phase构造future sine/cosine atoms在数学上成立，但与classical spectral extrapolation过近，
同时受到FITS、FBM、BasisFormer、IF与PhaseFormer的强prior-art压力。它可以作为baseline或diagnostic primitive，
不能作为paper contribution。

### R3 — Statistic-conditioned A6 coefficient operator

`problem_unverified / diagnostic_only_next`.

可能形成差异的问题不是frequency features能否单独预测良好，而是strong A6 compressed state是否丢失了
forecast-relevant history information，以及能否在保留full-$T$ generation与prefix crop的同时，于**shared
full-trajectory coefficient operator**中紧凑恢复这些信息。

它还不是method candidate。把statistic直接concatenate到coefficient head属于generic mechanism，本身没有充分
narrative；必须先做transfer/specificity diagnostic。

## Source-informed control 要求

后续任何compact-statistic design必须包含：

- 不含direct history statistic的A6；
- 同维random orthogonal history projection；
- frequency/time-frequency statistic arm；
- head function class变化时加入generic capacity control；
- 先full-$T$ synthesis，之后只能prefix crop；
- prior-art/baseline边界纳入FITS、FBM、IF、PhaseFormer、BasisFormer与TimePerceiver；
- 不声称first spectral decoder、first phase model、first compact decoder或first dynamic-horizon basis。

## 当前结论

[Decision] external evidence阻止将`compact spectral generation`直接提升为Contribution 1。D19只提供了一条
path-specific existence clue——history spectrum帮助IF——但没有证明其可transfer到A6，也没有证明frequency
specificity。下一步只允许matched diagnostic-only transfer test；本审计不授权paper-core implementation或remote
training。
