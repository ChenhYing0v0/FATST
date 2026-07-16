# PCC Training Prior-Art Audit

## Metadata

- `search_date`: 2026-07-16
- `scope`: expert-specific forecasting loss、loss-teacher routing、expert-router capability coupling、routing prior、
  multi-task gradient balancing
- `source_type`: arXiv full text、ICLR/NeurIPS/ICML/PMLR primary pages
- `discovery`: external web search；Zotero只作seed且本次未用于coverage判断
- `full_text_status`: Expert Loss Integration与主要conference papers可访问；AME-TS以official arXiv metadata为主
- `official_code_status`: 2026-07-16按标题/作者检索未定位到Expert Loss Integration、DiSCO、ERC或AME-TS的
  confirmed official repository；未导入第三方复现。Step7A只实现论文中明确的closest-prior loss controls，并以
  本地tensor contract与reference-value tests约束；若官方代码后续公开，paper submission前须补做implementation audit

## Claim Boundary

PCC不得claim以下primitive：

1. forecasting中的expert-specific auxiliary loss；
2. 从per-expert loss构造teacher distribution并以KL训练router；
3. uniform expert warm-up、load balance或generic expert specialization；
4. router–expert capability coupling的一般概念；
5. generic loss/gradient balancing或gradient surgery。

保留的candidate-level claim只允许是：

> 在fixed-past projective unified multi-horizon generation中，从全部nested-prefix risks构造coupling-scope
> capability，并通过harmonic incidence transport将其映射到不含requested-H的target-coordinate arm/router credit。

该claim只有在`PCC_TRANSPORT_FULL`超过`POINTWISE_PCC_V0`与`POINTWISE_PRIOR_COMPOSED`时才成立。

## Primary Sources

- [Fast Training of Mixture-of-Experts for Time Series Forecasting via Expert Loss Integration](https://arxiv.org/html/2605.10330)
- [Diverse and Sparse Mixture-of-Experts for Causal Subgraph-Based OOD Graph Learning](https://iclr.cc/virtual/2026/poster/10011548)
- [Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss](https://openreview.net/pdf?id=MpeyjgWbKt)
- [AME-TS](https://arxiv.org/abs/2605.25166)
- [GradNorm](https://proceedings.mlr.press/v80/chen18a.html)
- [Gradient Surgery for Multi-Task Learning](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
- [Conflict-Averse Gradient Descent](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html)
