# SC-ISCF-PSA-D1 Contemporaneous EQUAL Control Design

## 1. Decision and role

Decision=`psa_d1_five_run_validation_training_active_formal_test_disabled`。

`SC-ISCF-PSA-D1`不是method candidate，也不测试新loss/router。它只补齐RSCC-v1 attribution matrix中缺失的
contemporaneous no-route EQUAL，使ARMERR/SHUFFLED公共gain可以在`training co-adaptation`与`run drift`之间归因。

用户已明确授权Step7A与five-run validation training；implementation已通过local contracts。remote launch仍须先通过
commit-pinned pull、GPU preflight与Weather resource smoke。三项现均通过，5 runs已启动；不访问official test，active
method仍none。

## 2. Problem and hypotheses

已有事实：

- historical EQUAL是RSCC matrix的reference；
- ARMERR与SHUFFLED是同轮new runs，均比historical EQUAL约`+0.656%` validation MSE；
- 两个new controls functionally near-equivalent；
- PSA-D0 post-hoc convex/marginal/temperature shrinkage均macro negative。

因此：

- `H2 training_coadaptation`：route loss只在joint training中改变arms/policy的共同动力学，frozen shrinkage无法复现；
- `H3 contemporaneous_retraining_drift`：即使没有route loss，同轮重训EQUAL也会取得类似提升。

D1不判断paper novelty，只判断哪一种解释更符合证据。

## 3. Frozen matrix

| Field | Frozen value |
| --- | --- |
| datasets | Weather, ETTm1, ETTh1, ETTh2, ETTm2 |
| seed | 2021 only |
| new arm | `iscf_equal_contemporaneous` |
| architecture | `siff-independent-scope-control`；identity/canonical；dataset-matched ranks |
| objective | `equal_skill`；route loss absent |
| training | from scratch；joint encoder/decoder；H720 training |
| selector | mean validation MSE over H96/H192/H336/H720 |
| optimizer | AdamW；lr `1e-4`；batch 32；max 20 epochs；patience 5 |
| output role | validation-only control |
| new runs/cells | 5 runs / 20 standard-horizon validation cells |
| official test | forbidden |

Matched ranks沿用RSCC matrix：ETTh1/ETTh2/ETTm1/ETTm2/Weather=`109/116/116/106/116`。

## 4. References and comparisons

三组read-only references：

1. historical `iscf_equal`：原FCC/SCC seed2021 EQUAL；
2. contemporaneous `iscf_equal_armerr`：RSCC Step7B；
3. contemporaneous `iscf_rscc_shuffled`：RSCC Step7B。

比较量：

- `new_equal_vs_historical_equal`；
- `armerr_vs_new_equal`；
- `shuffled_vs_new_equal`；
- new/historical EQUAL的per-cell metrics、training curves、function probes与policy entropy；
- initialization hash、effective config、checkpoint selector与artifact completeness。

不得重新运行或选择ARMERR/SHUFFLED，也不得增加seed、dataset或formal test。

## 5. Prelaunch contracts

在任何launch前必须满足：

1. current training/PCC source相对RSCC launch commit的`equal_skill` execution path无semantic diff；
2. new EQUAL与three existing arms逐dataset initialization contract一致；
3. `pcc_route_weight=0`，total loss逐值等于fused + equal-skill loss；
4. five scope gradients finite/nonzero；
5. config明确`remote_training_authorized=true`且`formal_test_access_authorized=false`；
6. local dry-run=5 jobs；Weather resource smoke finite且无OOM；
7. remote pull、GPU preflight与output root写入launch record。

第1--6项已通过local部分；第6项的Weather resource smoke与第7项remote records仍pending。smoke通过前不得launch full
matrix。

## 6. Frozen decision rules

先定义controls相对historical EQUAL的mean MSE gain：

$$
G_C=\tfrac12\left(G_{\mathrm{ARMERR},E_h}+G_{\mathrm{SHUFFLED},E_h}\right).
$$

定义new EQUAL recovery ratio：

$$
R=G_{E_n,E_h}/G_C.
$$

### `contemporaneous_run_drift_explains`

同时满足：

- new EQUAL vs historical EQUAL macro MSE gain `>= +0.3%`；
- `R >= 0.75`；
- 至少3/5 datasets与3/4 horizons为正；
- ARMERR、SHUFFLED相对new EQUAL均小于`+0.1%` macro MSE；
- protocol/init/numeric checks通过。

这说明大部分公共gain无需route loss即可复现；关闭ARMERR/SHUFFLED common-gain clue，回Step2重新找problem。

### `joint_training_route_regularization_supported_as_carrier_clue`

同时满足：

- new EQUAL vs historical EQUAL macro MSE绝对差 `<0.1%`；
- ARMERR与SHUFFLED相对new EQUAL均`>=+0.3%` macro MSE；
- 两个controls各至少3/5 datasets、3/4 horizons为正；
- new/historical EQUAL function与policy diagnostics没有material drift；
- protocol/init/numeric checks通过。

这只支持“某种no-binding route regularization在joint training中有效”，仍不说明scope semantics或paper novelty。随后必须回
Step4结合primary sources寻找ISCF-native且能由matched controls识别的机制；不能直接把ARMERR或SHUFFLED改名为method。

### `h2_h3_unresolved`

其他完整、健康但未满足上述任一集合的结果。不得按dataset/horizon调阈值，也不得补seed rescue；先做failure attribution，
再决定是否值得继续该clue。

## 7. Validation/test roles and failure attribution

- validation决定control attribution；不是paper-facing effectiveness；
- official test access=0；D1结果不授权formal test；
- numeric/OOM/config/init failure=`diagnostic_invalid_for_attribution`，只修复protocol后完整重跑；
- healthy negative不拒绝ISCF architecture；
- frozen references是合法的same-family historical/contemporaneous attribution inputs，但不能替代paper-core end-to-end
  comparison。

## 8. Authorization boundary

| Action | Authorized |
| --- | --- |
| D1 protocol/design documentation | true |
| config/runner implementation | true；Step7A complete |
| remote resource smoke | true；pending preflight |
| five-run validation training | true；conditional on smoke |
| official test | false |
| method promotion | false |

five runs当前active；5/5完整后运行冻结analyzer，不得读取partial结果修改gates。
