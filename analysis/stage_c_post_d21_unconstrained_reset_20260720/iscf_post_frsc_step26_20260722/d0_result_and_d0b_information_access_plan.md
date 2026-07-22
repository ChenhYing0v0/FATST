# ISCF-SCC D0 Result and D0B Information-Access Plan

## 1. Long-stage record

| Field | Record |
| --- | --- |
| `current_step` | D0 Step9 complete；rollback to diagnostic design；D0B frozen |
| `problem` | coalition credit有强signal但fixed-label topology仅2/5 datasets跨seed稳定；该失败是否等价于credit不可学习 |
| `existence_evidence` | median oracle headroom=`17.9766%`；15/15 nondegenerate；15/15 shuffle-specific；policy-credit alignment弱 |
| `idea` | 用inference-available arms/policy/position拟合低容量ridge probe，held-out rows预测target-visible credit |
| `theory_check` | dynamic per-coordinate credit不要求independent seeds固定label同构；但必须存在target-free可预测分量并超过marginal shuffle |
| `design` | 15 frozen validation NPZ；60/40 blocked-row split；coalition/standalone labels；16 horizon-marginal shuffles |
| `narrative_gate` | unchanged=`conditional_pass_to_diagnostic_only` |
| `effectiveness_gate` | not applicable；无forecast model training/test |
| `artifacts` | D0 CSV/JSON/logs/invariants；D0B config/analyzer |
| `decision` | `d0_unresolved_d0b_information_access_authorized` |

## 2. D0 result

15/15 replay均满足`evaluation_split=val`、`uses_test_split=false`、256 probe rows、exact
`probe_direct_policy [256,720,5]` finite且simplex gap不超过`2.39e-7`。fusion reconstruction max gap为
`6.45e-7`。runner逐checkpoint执行SHA256前后相等检查并正常退出；hash未另存为独立before/after表，故只能确认
enforced pass，不能声称保存了双份hash provenance。

| Gate | Result | Decision meaning |
| --- | --- | --- |
| oracle headroom | median `17.9766%` | pass；存在显著target-visible coordination gap |
| nondegeneracy | 15/15；median positive scopes 2–3 | pass；不是single-winner退化 |
| standalone distinction | best-match median `.6178`；credit rho median `.9` | formal OR gate pass，但standalone解释力很强 |
| shuffled specificity | 15/15 | pass；scope binding不是纯marginal effect |
| fixed-label seed topology | stable datasets=2/5 | fail；ETTm1/ETTm2/Weather不稳定 |

coalition oracle相对standalone-error oracle的额外L1 gain中位数为`2.692` percentage points，但Weather仅约
`.074` points。该证据支持“coalition signal可能有增量”，不证明该增量可由fixed-past inference path获取。

因此D0 machine decision为
`coalition_credit_unresolved_requires_validation_diagnostic_redesign`。按预注册规则不得进入Step7，也不得把它写成
`hypothesis_false`：失败项测量的是different random seeds下固定scope label的global topology，而SCC target是每个
coordinate动态生成。

## 3. D0B question

D0B只回答一个必要问题：不读取future target时，现有ISCF forward中是否已有足够信息预测coalition credit，并在held-out
validation rows上转化为优于current fusion的重组收益。

feature全部在inference graph可用：五个coordinate-normalized arm deviations、五个direct policy weights、arm
dispersion、fused value、policy entropy，以及固定future-position basis。target只用于构造train label和held-out评价。

## 4. Split, probe and controls

每run按probe-row顺序做60/40 blocked split；不随机拆coordinate，减少同一source-row泄漏。固定ridge
`alpha=.01`，输出clip到nonnegative simplex。primary metrics为held-out credit Spearman/top-1、预测credit重组后的L1
gain，以及相对standalone-credit probe的gain。

16个control在每个future coordinate内shuffle整条credit vector的training row binding，保留horizon-specific marginal与
simplex joint distribution。只有coalition probe的gain、Spearman和top-1都超过shuffle p95才记为binding pass。

## 5. Frozen D0B gate

D0B返回Step5/6必须同时满足：

1. 至少12/15 runs通过三指标shuffle binding；
2. median predicted L1 gain至少`.1%`，且至少12/15为正；
3. 相对standalone-credit probe的median增益至少`.1` percentage point，且至少9/15为正；
4. 至少4/5 datasets的three seeds全部predicted gain为正。

若1–2失败，failure attribution=`credit_not_inference_accessible`，回Step2；若3失败，
`standalone_credit_sufficient`，回Step4；其余不完整或seed consistency失败保持`unresolved`。任何结论都只针对当前
arms/policy/feature intervention，不方向级否定ISCF architecture。

## 6. Authorization boundary

```text
active_method = none
D0 = unresolved
D0B_frozen_probe_authorized = true
forecast_model_training_authorized = false
method_implementation_authorized = false
formal_test_authorized = false
next_action = local_smoke_commit_push_and_remote_offline_d0b
```
