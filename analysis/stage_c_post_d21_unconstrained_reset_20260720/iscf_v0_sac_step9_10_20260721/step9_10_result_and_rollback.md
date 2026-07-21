# ISCF-v0 SAC Step 9/10 Result and Rollback

## 1. What was tested

Scope Attribution Confirmation（SAC）检验ISCF-v0的两项promotion-blocking attribution：

1. `independent_over_q1_wide`：independent scope-specific history maps是否超越near-matched wide shared map；
2. `canonical_over_random_partition`：contiguous/nested future-output partitions是否超越same-parameter、same-init的
   arbitrary grouping。

candidate、direct policy、equal-skill objective、five natural profiles、ranks、partition seed、four-horizon checkpoint
selector与gates均在test前冻结。25个new checkpoints与35个hashed historical references形成60 effective runs；primary
scorecard为5 datasets × 3 seeds × H96/H192/H336/H720，共240 standard-horizon rows，同时保留new runs的18,000
dense-horizon rows作为diagnostic。

## 2. Test access and protocol audit

| Field | Record |
| --- | --- |
| `test_access_date` | `2026-07-21` |
| `user_authorization` | “授权SAC formal test” |
| `candidate_version` | `SC1-ISCF-v0-SAC-v1` |
| `formal_test_commit` | `6bbc3fc962829cb7261197c9b67905000580ed16` |
| `config_sha256` | `f1be23adc831180cb2689d24d6411a46f261d31f52b7d3abd1908e376ae7b42a` |
| `test_start / finish` | `21:04:16 / 21:07:02 +08:00` |
| `new test runs` | `25/25` |
| `effective runs` | `60/60` |
| `standard test rows` | `240/240` |
| `new dense rows` | `18,000/18,000` |
| `test access count` | 1 |
| `checkpoint_retrained` | true：每个new control在Step8按frozen protocol从头训练；test阶段false |
| `checkpoint mutation` | false；25/25 pre/post SHA256一致 |
| `matrix_complete` | true |

首次launch因config遗漏evaluator-required diagnostic bins，在创建test loader前触发`KeyError`，test artifacts保持
0/25。exact repair只加入与FCC/CPSI一致的8-bin diagnostic contract及runner静态断言；真实checkpoint的validation-split
smoke通过后才重启。该事件归因为`exact_protocol_preflight_gap`，不属于model、numeric或result failure。

正式relaunch后25/25 `test_audit_invariants.json`均满足`pass=true`、`evaluation_split=test`、
`uses_test_split=true`、`test_access_authorized=true`；25个remote NPZ diagnostics完整。60/60 analyzer run audits为`ok`，
所有metrics finite。补同步FCC historical test CSV后，同一analyzer已完全从local lite artifacts复算并得到相同decision；
1.1GB remote NPZ不进入Git。

## 3. Statistics and frozen gates

每个dataset-seed-horizon cell的gain为

$$
g=100\left(1-\frac{m_{candidate}}{m_{reference}}\right).
$$

macro为60个cell gains的算术平均。Q1-WIDE要求MSE macro不低于`+0.5%`，RANDOM要求不低于`+0.3%`；
两者还必须同时满足MSE datasets `>=3/5`、horizons `>=3/4`、positive seed macros `>=2/3`与MAE macro
严格正向。两项primary comparisons必须全部pass才允许paperization与modern baselines。

## 4. Official-test results

| Comparison | MSE macro | MAE macro | MSE cells | Datasets | Horizons | Positive seeds | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ISCF-v0 over Q1-WIDE | `+0.8496%` | `+0.5996%` | `43/60` | `5/5` | `4/4` | `2/3` | **pass** |
| canonical over RANDOM | `-0.1990%` | `-0.4347%` | `24/60` | `1/5` | `0/4` | `1/3` | **fail** |
| ISCF-v0 over A6_FULL context | `+1.3584%` | `+0.9144%` | `49/60` | `5/5` | `4/4` | `3/3` | performance context positive |

### 4.1 Independent maps versus shared width

Q1 comparison的MSE dataset macros为Weather/ETTm1/ETTh1/ETTm2/ETTh2 =
`+0.1442/+1.0213/+0.1163/+1.7840/+1.1824%`；H96/H192/H336/H720 =
`+1.0150/+0.9634/+1.0338/+0.3863%`。seed2022为`-0.2352%`，但2021/2023为
`+0.8980/+1.8860%`，满足2/3 gate。

因此`capacity_or_shared_width_explains`不成立：在当前function family和training protocol内，independent maps具有可重复
收益。该结果不能单独paperize，因为generic independent predictors/branches已有充分prior-art pressure。

### 4.2 Canonical versus random grouping

RANDOM comparison只有ETTm2正向（`+1.7319%`）；Weather/ETTm1/ETTh1/ETTh2分别为
`-0.0764/-0.0185/-2.0183/-0.6138%`。H96/H192/H336/H720全部负：
`-0.3083/-0.0776/-0.0629/-0.3473%`。MAE四horizons也全部负，macro为`-0.4347%`。

这不是轻微threshold miss：macro、dataset、horizon、seed和MAE guards全部失败。validation中的
`-0.1823%/-0.3075%` MSE/MAE已给出同向negative lead，official test复现并强化该结论。

## 5. Four-layer mechanism evaluation

1. `paper_facing_effectiveness`：ISCF-v0相对A6_FULL为`+1.3584%/+0.9144%`，5/5 datasets与4/4 horizons
   正向；作为performance carrier通过。
2. `matched_mechanism_attribution`：independent maps超过Q1-WIDE，但canonical temporal grouping不超过exact RANDOM。
   SAC的all-primary rule失败。
3. `internal_mechanism_health`：15/15 dataset-seed pairs通过。canonical/random Encoder与PCSD initialization匹配，
   parameters完全相同，partition hashes不同；Q1 parameter gaps与预注册值一致。
4. `failure_attribution`：`temporal_scope_structure_not_supported`。不是numeric pathology、readout fault或shared-width
   capacity explanation；generic independent branches足以解释exact ISCF收益。

## 6. Narrative consequence and self-critique

SAC支持“独立映射有用”，但否定了“这些映射因contiguous/nested future-output coupling semantics而必要”。失去temporal
scope structure后，剩余机制接近generic independent multi-branch factorization；最新source audit已表明该claim空间拥挤。
因此不能把performance positive重新包装成paper-core attribution pass。

反方解释是RANDOM partition可能偶然提供更好的regularization，而不是temporal semantics完全无价值；但三seed、五dataset、
四horizon的matched result已足以否定当前论文所需的稳定necessity claim。进一步搜索partition seed、rank或另一个ordering会是
test-informed rescue，违反预注册rollback，也无法解决post-hoc narrative风险。

## 7. Decision and rollback

Decision=`temporal_scope_structure_not_supported_generic_independent_branches_explain`。

- ISCF-v0降为`strong carrier/control`，不promote为paper-core method；
- active method继续为none；
- modern-baseline execution不授权；
- 不做rank、seed、partition、loss、router、requested-H或second-contribution rescue；
- exact SAC/ISCF paperization route关闭，rollback到11-step loop的Step2/4 portfolio consolidation。

后续若继续当前论文，必须先重新决定paper contribution boundary；可以复用ISCF-v0作为高性能carrier，但任何新paper-core
mechanism都必须重新通过problem/narrative/design gate，不能把本次positive Q1 result单独升级为创新点。
