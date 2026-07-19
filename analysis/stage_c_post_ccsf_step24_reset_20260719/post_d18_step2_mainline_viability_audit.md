# Post-D18 Step 2 主线可行性与下一诊断审计

## 1. 当前状态

| Field | Content |
| --- | --- |
| `current_step` | Contribution 1 Step 2；Contribution 2 Step 2 |
| `problem` | exact-projective unified generation成立，但当前还缺少能超过`A6_MEASURE`的decoder problem与method |
| `existence_evidence` | D18否定stable projectivity cost；A6_MEASURE相对A6_FULL在15/15 own-H cells正向 |
| `idea` | 暂不提出paper method；先用source-faithful structured decoder control判断trajectory-level decoding是否仍有headroom |
| `theory_check` | structured output generation可保持exact projectivity；但wave decoding与horizon weighting已有直接prior art |
| `design` | 下一候选仅为`SC-D19-IFC control_only`，需先完成source/code/theory gate |
| `narrative_gate` | method=false；D19 control narrative not required |
| `effectiveness_gate` | D18 fail；no active paper-core candidate |
| `artifacts` | D18 Step9 report；external primary sources；`Papers/implicit-forecaster-neurips2025.md` |
| `decision` | 关闭soft projectivity；Step 2保留trajectory-level decoder问题；不授权remote |

## 2. D18之后真正留下了什么

[Fact] 当前证据同时支持三件事：

1. 一个full-$T=720$ predictor再prefix crop，不会相对own-H specialists付出稳定accuracy cost；
2. unified prefix measure training相对flat H720 loss强且稳定；
3. 复杂SIFF/PCSD/PCC/CCSF路线尚未超过或不能归因地超过简单`A6_MEASURE`。

因此论文不能再把问题写成“strict projectivity限制了不同horizon的表达”。更合理的Step 2问题是：

> 在不把requested horizon作为semantic input、也不破坏shared prefix的条件下，能否把future trajectory作为
> 一个有内部结构的对象生成，而不是只依赖当前A6 global learned-basis projection？

这仍然是fixed-past unified multi-horizon decoder问题，因为模型生成同一条future function，任意horizon只读取
其prefix；但它不再依赖“不同horizon需要不同预测”这个已被D18削弱的前提。

## 3. 2026-07-19外部primary-source audit

本轮以外部搜索为主，未把Zotero覆盖当作完整性证据。

| Source | Primary evidence | 对本项目的边界 |
| --- | --- | --- |
| ElasTST, NeurIPS 2024 | https://proceedings.neurips.cc/paper_files/paper/2024/file/d7aa002885ccbe68cf6880da583761b2-Paper-Conference.pdf | horizon-invariant output与harmonic horizon reweighting已被直接覆盖；`A6_MEASURE`只能作control/protocol |
| Loss Shaping Constraints, ICML 2024 | https://openreview.net/forum?id=9CCoVyFuEp | per-step risk不均衡与约束优化已有直接工作；不能把“关注各future step”单独写成创新 |
| QDF, ICLR 2026 | https://openreview.net/forum?id=vpO8n9AqEG | label autocorrelation、heterogeneous step weights与adaptive quadratic objective已被直接覆盖 |
| Implicit Forecaster, NeurIPS 2025 | https://papers.nips.cc/paper_files/paper/2025/hash/0e82ef0c89df6a6eff8734ea7e27c42f-Abstract-Conference.html | 直接把forecasting phase建模为frequency/amplitude/phase wave composition，并提供Linear/MLP/Transformer controls |
| In Defense of the Unitary Scalarization, NeurIPS 2022 | https://arxiv.org/abs/2201.04122 | complex multi-task optimizer不自动优于简单scalarization；后续training contribution必须超过强静态measure control |

[Strong Evidence] 外部资料支持继续研究decoder的forecasting phase，但也把两条简单路线堵住：

- 直接实现harmonic measure weighting不够新；
- 直接把A6 head替换为Fourier amplitude/phase decoder也与Implicit Forecaster高度重叠。

## 4. 候选问题与淘汰边界

### 4.1 保留：trajectory-level structured generation

[Hypothesis] A6 learned basis与Implicit Forecaster分别代表两类全局trajectory decoder：

- A6：history state $\rightarrow$ sample coefficients $\rightarrow$ globally learned basis；
- IF：encoder state与input spectrum $\rightarrow$ amplitude/phase $\rightarrow$ fixed frequency pool。

如果source-informed IF control在相同unified measure protocol下稳定超过`A6_MEASURE`，说明A6的learned-basis
生成仍未充分利用future trajectory structure，decoder方向仍有真实headroom。

### 4.2 不保留：soft projectivity

D18已提供足够negative evidence。有限consistency penalty、horizon embedding、horizon-specific heads或
deformation operator均不进入下一步。

### 4.3 不保留：static measure weighting作为Contribution 2

其性能作用明确，但完整novelty chain不成立。ElasTST、Loss Shaping与QDF分别覆盖horizon reweighting、
stepwise risk shaping和future correlation-aware objective。

### 4.4 暂不重启：predictable frame / basis-only diagnostic

D4、D8、D12已分别审查locality、end-to-end structured basis和predictable frame。再次只做DCT/PCA/wavelet
residual fitting会重复旧证据，不能回答E2E decoder是否优于`A6_MEASURE`。

## 5. 下一诊断：SC-D19-IFC

`SC-D19-IFC`（Implicit-Forecaster Control）定位为`control_only`，不是paper contribution。

它要回答：

> 在same A6 natural Encoder class、same full-$T=720$ output contract、same measure objective与four-H
> checkpoint selection下，source-informed implicit wave decoder是否稳定超过A6 learned-basis decoder？

预期arms：

1. `A6_MEASURE`：mandatory reference；
2. `IF_MEASURE`：full-$T$ amplitude/phase frequency-pool decoder；
3. `IF_NOSKIP_MEASURE`：隔离input-spectrum skip；
4. `MLP_MATCHED_MEASURE`：排除generic nonlinearity/capacity。

公平性：

- 全部从相同initialization class端到端joint training；
- 不做frozen encoder replacement或cross-swap；
- dataset profile、rank、optimizer、measure objective与checkpoint selector一致；
- params仅报告，不作为选择偏好；
- official test只在完整预注册matrix后一次性评估。

### 进入实现前的hard gate

1. 完整读完IF paper与official implementation的amplitude、phase、frequency-pool、skip和initialization细节；
2. 证明local full-$T$ implementation对任意prefix保持crop equality；
3. 明确与A6 learned basis的function-class差异及MLP matched control；
4. 不把IF control或Fourier primitive表述为本项目创新；
5. 若实现需要改变Encoder contract或使用horizon-specific pool，回Step 2，不启动remote。

## 6. 当前决策

下一步正式进入`SC-D19-IFC Step 4/5 source-and-theory control audit`。在该audit通过前：

- model implementation=false；
- remote=false；
- official test=false；
- Contribution 1/2 method slots均保持open；
- SIFF-v2仍是immutable performance-near historical parent，不作为D19 warm start。

这一选择的价值不是马上得到新method，而是用一个最新、source-faithful且与A6正交的强decoder control，判断
“forecasting phase仍有无可开发headroom”。若D19 control也不能超过A6_MEASURE，fixed-past decoder主线需要
执行更高层的paper viability review，而不是继续内部组合basis/router/loss。
