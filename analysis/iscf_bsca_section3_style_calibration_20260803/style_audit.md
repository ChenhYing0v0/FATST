# ISCF-BSCA Section 3 顶会时序预测文风校准

## 审计状态

| Field | Content |
| --- | --- |
| `date` | `2026-08-03` |
| `scope` | Section 3 prose style only；不作novelty或literature-coverage结论 |
| `source_policy` | 仅使用官方conference proceedings / OpenReview页面 |
| `zotero_status` | 未检查；本轮为external primary-source style calibration，不据此判断coverage |
| `target_draft` | `docs/paper-drafts/iscf-bsca-problem-formulation-initial-draft.md` v0.4 |

## Primary-source reference set

| Paper | Venue | Official source | Style role |
| --- | --- | --- | --- |
| iTransformer: Inverted Transformers Are Effective for Time Series Forecasting | ICLR 2024 | [Proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/2ea18fdc667e0ef2ad82b2b4d65147ad-Paper-Conference.pdf) | task definition直接进入notation，随后给出structure overview |
| TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting | ICLR 2024 | [Proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/a7ac8a21e5a27e7ab31a5f42a0117bdb-Paper-Conference.pdf) | 从problem observation自然过渡到forecasting formulation与architecture |
| TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables | NeurIPS 2024 | [Proceedings page](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html) | 先用一段定义forecasting setting，再以`Problem Settings`直接形式化input/output |
| TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis | ICLR 2025 | [Proceedings page](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b187165e28fdfdc0ffb34d1bfff2b0c-Abstract-Conference.html) | 使用连续的problem-to-design叙事，不依赖rhetorical questions或meta roadmap |

## Calibrated style pattern

四篇论文虽具体措辞不同，但Methods / model section呈现出一致的领域写作习惯：

1. section opening直接承接前文problem observation，并在同一段说明本节要形式化或分析的对象；
2. task formulation通常由`Given ...`、`Consider ...`或直接的setting definition进入，不另设“本节将依次介绍”的roadmap段；
3. subsection topic sentence说明具体technical issue，随后立即给notation、operation或controlled comparison；
4. empirical motivation使用`As shown in Figure ...`连接observation与statistic，解释紧邻结果但不扩张claim；
5. prose以中等长度、连续因果句为主，较少使用Nature式短句、rhetorical questions、aphoristic system judgments或过度显式的claim-boundary口号。

## Application to Section 3 v0.4

- 将chapter opening压缩为一个承上启下段，删除原有两个rhetorical questions与`well-defined forecasting system`判断；
- 删除独立的section roadmap paragraph，让3.1直接进入same-history / shared-target setting；
- 将3.1--3.5统一为`setting -> notation -> contrast -> measurement -> observation -> implication`的时序预测论文语体；
- 将3.3标题改为中性的`Accuracy under naive unified forecasting`，仍明确现有证据不足以建立stable unified penalty；
- 保留CHPC、CHPD/NCHPD、$\operatorname{UP}_H$、$R_{o,b,s}$、CFH、Figures 2--3数值、matched controls与validation-only boundaries；
- Introduction v0.9、图像、method identity和实验授权均未修改。
