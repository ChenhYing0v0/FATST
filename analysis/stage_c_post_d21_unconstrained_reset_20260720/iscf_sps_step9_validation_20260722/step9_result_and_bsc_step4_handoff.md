# ISCF-SPS Step9 Validation Result and BSC Step4 Handoff

## 1. What we tested

`SC-ISCF-SPS-v0`试图在ISCF五个raw arms进入direct policy fusion前施加scope-native hard projector，使scope extent同时
限制forward forecast和backward error。冻结matrix为4 arms × 5 datasets × seed2021，共20个from-scratch runs；本轮只使用
validation split，official test未授权且未访问。

## 2. Artifact and protocol audit

- 20/20 checkpoints、training logs、H96/192/336/720 validation metrics、effective configs、initialization contracts、
  model diagnostics、validation NPZ和trained invariants完整；
- 20/20 run audits为`ok`，80/80 standard-horizon metric cells完整；
- 20个checkpoint SHA256已保存；log failure scan为0；formal-test artifact count为0；
- candidate/global/identity/random全部finite，projection ranks/degrees与106/109/116 matched-rank contracts一致；
- training从`2026-07-22T00:17:31+08:00`运行至`01:07:02+08:00`，无OOM或numeric pathology。

## 3. Effectiveness and attribution

### 3.1 Candidate vs full-rank identity parent

scope-canonical相对identity MSE/MAE为`-2.3123%/-1.0937%`，只赢5/20与6/20 cells、1/5 datasets、0/4
horizons，primary validation gate明确失败。ETTh1四horizon MSE退化`3.47%–7.12%`，ETTh2退化`2.36%–8.66%`；
Weather近似tie，ETTm1/ETTm2大多小幅负向。

### 3.2 Local scope projection vs generic global smoothing

scope-canonical相对global projection MSE/MAE为`+0.9041%/+0.8461%`，16/20与19/20 cells、5/5 datasets、4/4
horizons均正。这说明在同样强的rank restriction下，group-local scope synthesis比对所有arms施加同一个global low-pass更有效。
该结果支持“local future-output geometry有条件价值”，但不能抵消candidate相对full-rank parent的material loss。

### 3.3 Canonical vs random binding

canonical相对random MSE/MAE为`-0.2701%/-0.3087%`，只赢2/5与0/5 datasets。按预注册规则，这只令
scope-binding attribution unresolved，不允许否定ISCF architecture。

## 4. Internal mechanism health

candidate all-finite，pairwise arm distance由identity的`0.1694`增至`0.1831`（约`+8.1%`），oracle headroom从
`5.0381%`增至`5.1450%`，五datasets合计有4个不同scope成为future-bin winner，normalized policy entropy为`0.6983`。
因此projection确实改变了roles且没有arm collapse。

问题在于restriction strength：candidate `removed_to_raw_rms=0.8769`；除scope1外，各scope retained RMS ratio大致只有
`0.33–0.49`。ETTh1/ETTh2/ETTm2 candidate均在epoch 1选中best checkpoint，而identity在ETTh1/ETTh2分别继续改善至
epoch 3/4。该组合更符合`readout_or_head_design_wrong / hard capacity restriction too strong`，不符合
`hypothesis_false`或`optimization_or_numeric_pathology`。

## 5. Latest primary-source boundary audit

检索日期为`2026-07-22`，queries覆盖`hierarchical interpolation forecasting`、`decomposable multiscale mixing`、
`shared/private heterogeneous forecasting experts`与`expert specialization`。来源均为conference proceedings或官方arXiv：

- [NHITS, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25854)：hierarchical interpolation、multi-rate
  sampling与sequential additive synthesis已覆盖“不同frequency/scale组件”；
- [TimeMixer, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a7ac8a21e5a27e7ab31a5f42a0117bdb-Abstract-Conference.html)：
  PDM/FMM已覆盖multiscale decomposition与multiple predictors的complementary composition；
- [MoHETS, arXiv 2026](https://arxiv.org/abs/2601.21866)：shared continuity expert + routed Fourier experts直接覆盖
  “shared carrier + heterogeneous local experts”的generic primitive；
- [Advancing Expert Specialization, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4598de7d243d528e38eb0c5d8155fb52-Abstract-Conference.html)：
  orthogonality/variance objectives已覆盖用额外loss促进specialization；本项目因此不把第二个diversity loss作为默认方案；
- [Learning to Specialize, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/03bb44eceb94537ee2e2f00a8dca60b1-Abstract-Conference.html)：
  joint gating-expert training对specialization的重要性进一步限制frozen diagnostic的结论边界。

Zotero semantic search返回了候选items但title metadata不可读；exact-title lookup因local Zotero connection refused失败，故上述
论文在用户curated `FSA` subset中的presence标记为`unresolved`，不能把Zotero coverage用于novelty结论。

## 6. Step4 next idea: Barycentric Scope Composition

hard SPS错误地把每个arm的common full-rank forecast一起投影掉。下一步不增加loss/router/requested-H，而把现有ISCF policy
barycenter定义为native common forecast：

$$
b(t)=\sum_s w_s(t)a_s(t),\qquad
\tilde a_s=b+P_s(a_s-b),\qquad
\tilde y(t)=\sum_s w_s(t)\tilde a_s(t).
$$

该结构只把arm-specific deviation放入scope-native subspace，full-rank barycenter不被丢弃。它不是post-hoc adapter：在候选
版本中arms、policy与affine composition必须end-to-end joint training。claim不能是generic shared/private expert或DCT，而只能是
`future-output policy barycenter -> affine scope deviations -> target-wise composition`这一完整task-specific chain。

在实现method前，先运行`SC-ISCF-BSC-D0` frozen validation function diagnostic：复用identity checkpoints前256条probe rows，
比较scope/global/random affine compositions。它只判断是否存在低成本function-level lead；由于representation与identity head
co-adapted，negative不能拒绝BSC/ISCF，positive也不能建立effectiveness，只能授权BSC Step4–6 E2E design。

## 7. 11-step record and decision

| Field | Record |
| --- | --- |
| `current_step` | SPS Step9 complete；BSC Step4 diagnostic proposed |
| `problem` | hard projector提高role diversity但删除shared full-rank forecast capacity |
| `existence_evidence` | scope>global +0.9041%；diversity/oracle active；removed RMS 0.8769；identity win material |
| `idea` | retain policy barycenter and project only arm-specific affine deviations |
| `theory_check` | no new information/H/loss/router；full-rank barycenter retained；prior boundary narrowed |
| `design` | frozen identity probes；scope/global/random projectors；validation only |
| `narrative_gate` | diagnostic-only pass；method gate pending D0 and full E2E controls |
| `effectiveness_gate` | exact SPS-v0 validation fail；BSC untested |
| `artifacts` | `remote_audit/`、`raw_lite/`、`remote_logs/`、`remote_records/` |
| `decision` | reject exact hard SPS-v0；retain ISCF architecture；run BSC-D0 |
| `rollback` | SPS -> Step4；D0 negative only rejects frozen readout；no direction rejection |

Decision=`exact_SPS_v0_validation_fail_readout_restriction_too_strong_BSC_D0_next`。formal test、SPS confirmation seeds与
modern baselines均不启动。
