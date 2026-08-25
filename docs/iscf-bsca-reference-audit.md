# ISCF-BSCA 参考文献与引用位置审计

## 1. 审计范围

| Field | Content |
| --- | --- |
| `search_date` | `2026-08-25` |
| `manuscript_bibliography` | `manuscript/ref.bib` |
| `reference_count` | `41` |
| `citation_key_status` | `41 cited / 41 defined / 0 missing / 0 uncited / 0 duplicated` |
| `seed_material` | `PDT_final.pdf`及其本地`ref.bib`只用于发现候选文献和交叉核对常用benchmark来源 |
| `primary_source_policy` | 优先conference/journal proceedings、OpenReview正式接收页、PMLR、AAAI、NeurIPS、Elsevier DOI和UCI原始数据页 |
| `version_policy` | 同时存在preprint与正式出版版本时，BibTeX采用正式conference/journal版本 |
| `zotero_status` | 本轮未逐条检查用户Zotero `FSA` subset；全部41条均记为`zotero_presence=not_checked`，不据此判断馆藏覆盖或文献完整性 |

本轮只在外部工作直接支撑论断、实验对象或数据来源的位置加入引用。CHPC、CHPD/NCHPD、Future Coordinate、ISCF、BSCA及本文实验结果属于本文定义、方法或证据，不通过相邻主题文献制造虚假的来源归属。

## 2. Section-wise citation design

| Section | Unique refs | Citation role |
| --- | ---: | --- |
| Section 1 | 13 | horizon-specific protocol、flexible-horizon/foundation forecasting以及history-side architecture背景 |
| Section 2 | 29 | classical multi-step strategies、UVHF近邻工作、forecast-generation decoder、multi-scale与expert allocation边界 |
| Section 3 | 1 | DLinear作为CHPD/NCHPD诊断载体的模型来源；原创定义和观测结果不外引 |
| Section 4 | 2 | patch-token Encoder接口的代表性prior；原创模块与公式不外引 |
| Section 5 | 15 | 13个实验baseline、AdamW/cosine schedule及transfer backbones的正式来源 |
| Section 6 | 3 | input-side multi-scale encoder与本文output-side scope的讨论边界 |
| Section 7 | 0 | Conclusion仅总结本文贡献和结果，不重复堆叠引用 |
| Appendix | 6 | ETT、ECL、Weather、Solar的数据来源及训练优化设置 |

## 3. Publication-version decisions

以下容易产生preprint/同名论文冲突的条目已单独核验：

| Key | Frozen publication record | Decision |
| --- | --- | --- |
| `zhang2024elastst` | NeurIPS 2024 | 使用正式NeurIPS版本，不使用arXiv条目 |
| `gao2024units` | NeurIPS 2024, pp. 140589--140631 | 使用正式题名`A Unified Multi-Task Time Series Model`、DOI与proceedings页码 |
| `das2024timesfm` | ICML 2024, PMLR 235 | 使用正式ICML版本 |
| `liu2024timer` | ICML 2024, PMLR 235 | 使用正式ICML版本 |
| `ansari2024chronos` | TMLR 2024 | 使用正式TMLR版本 |
| `woo2024moirai` | ICML 2024, PMLR 235 | 使用正式ICML版本 |
| `shi2025timemoe` | ICLR 2025 | 使用正式ICLR版本 |
| `li2025implicit` | NeurIPS 2025 | 使用正式NeurIPS版本 |
| `liu2025freqmoe` | AISTATS 2025, PMLR 258 | 使用正式AISTATS版本 |
| `liu2025moiraimoe` | ICML 2025, PMLR 267 | 使用正式ICML版本 |
| `hu2026timealign` | ICLR 2026 | 使用正式ICLR版本，不使用2025 arXiv条目 |
| `wang2026qdf` | ICLR 2026 | 使用正式ICLR版本，不使用匿名review稿 |
| `li2025tvnet` | ICLR 2025 dynamic-convolution/3D-variation paper | 排除同名但不同方法的Elsevier TVNet论文 |
| `yu2024leddam` | ICML 2024, PMLR 235 | 使用正式ICML版本 |
| `hu2025amd` | AAAI 2025, 39(16):17359--17367 | 使用AAAI正式出版信息 |
| `trindade2015electricity` | UCI dataset DOI `10.24432/C58C86` | 替换原先用TimesNet间接支撑ECL来源的单一引用 |

## 4. Coverage and claim boundary

- 41条参考文献中，28条发表于2023--2026年；近年文献占主体，同时保留multi-step strategy、optimizer和数据源所必需的经典条目。
- Section 5的所有对比模型均在首次成组介绍时给出正式来源，避免结果表中的baseline成为无出处名称。
- Section 3的Figure 2只引用DLinear模型论文；图中CHPD/NCHPD数值来自本文审计，不引用外部论文。
- Section 4只为可替换的patch-token Encoder接口提供代表性来源；Future Coordinate、Scope Matrix、Scope-region State、Scope Probability及三项BSCA目标均保持为本文方法定义。
- Appendix中ECL同时引用UCI原始数据与标准processed benchmark来源；Solar改用首次引入该公开benchmark的LSTNet工作。

## 5. Verification

最终机器检查应同时满足：

1. `manuscript/ref.bib`可被BibTeX解析；
2. 所有`\citep{...}` keys均在`ref.bib`定义；
3. `ref.bib`不存在未被稿件使用的占位条目；
4. `latexmk`可完成`pdflatex -> bibtex -> pdflatex`构建且无undefined citations；
5. PDT只作为seed，不把其中的arXiv记录覆盖为正式出版信息。
