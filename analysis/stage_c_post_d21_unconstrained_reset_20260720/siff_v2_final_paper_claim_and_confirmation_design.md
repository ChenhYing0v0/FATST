# SIFF-v2 Final Paper-Claim Consolidation and Confirmation Design

## 1. Long-stage record

| Field | Record |
| --- | --- |
| `audit_date` | `2026-07-21` |
| `current_step` | SIFF-first Step 4-6 complete；A6_FULL-scope FCC Step7B prelaunch 25/25 pass；Step8 preflight pending |
| `candidate_version` | immutable `SC1-SIFF-v2-EQ-ATTR-v1`；evaluation tag `SC1-SIFF-v2-FCC-v1` |
| `problem` | one full-domain varied-horizon decoder如何把future-output coupling scope作为结构坐标，而不是把requested H、input scale或任意expert identity混为一谈？ |
| `existence_evidence` | SIFF超过A6_FULL、constant/permuted/Q1-wide与PCSD_EQUAL；internal 7/7；但未通过independent gate；A6_MEASURE历史negative保留但退出FCC |
| `idea` | 保留原样Q2 ordered Scale-Indexed Forecast Field；不使用TSAF、不增加loss/router、不改rank/readout |
| `theory_check` | Q2 field在ordered log-scale上共享history-conditioned operator components；requested H不进入forecast graph；direct policy只作既有fusion，不claim novelty |
| `design` | final claim confirmation不实现新method；SIFF/A6_FULL/independent × 5 datasets × 2 new seeds，复用seed2021形成three-seed evidence |
| `narrative_gate` | `conditional_pass_as_single_architecture_contribution` |
| `effectiveness_gate` | blocked pending FCC；A6_FULL承担method-package performance，independent承担ordered-field attribution |
| `artifacts` | SIFF Step9 four-layer audit、post-TSAF factorial audit、latest primary-source audit、FCC config/runner/analyzer/prelaunch report |
| `decision` | `fcc_a6_full_scope_authorized_prelaunch_pass_remote_not_started` |

## 2. Decision

[Decision] 原样SIFF-v2仍可形成一个**窄、单一、可证伪的architecture contribution**，但尚不能被写成已通过的
paper core。可辩护贡献不是“首次multi-scale/MoE/future query”，也不是TSAF，而是：

> **Scale-Indexed Forecast Field：把nested future-output coupling scope显式放在ordered log-scale coordinate上，
> 以共享history-conditioned operator components生成一个full-domain arm family，并在target coordinate上融合。**

这个完整链与input-side multi-scale patches、独立experts、future-query retrieval和requested-H conditioning不同；
因此在contribution-level具有provisional novelty。现有constant/permuted/Q1-wide controls支持scale variation、正确
order binding与multi-scale field的作用，PCSD_EQUAL comparison支持SIFF construction相对parent field的增量。

[Decision] 用户于`2026-07-21`指定FCC不以`A6_MEASURE`为对比，而改用`A6_FULL`。因此FCC的性能问题变为
“完整SIFF method package是否稳定超过A6_FULL”；由于两者同时改变architecture与training objective，该比较不能
单独归因于SIFF architecture。ordered-field attribution只由同objective、parameter-matched的
`SIFF_INDEPENDENT_EQUAL`承担。

[Strong Evidence] `A6_MEASURE`相对SIFF-v2的历史MSE `+0.2366%`不会从研究记录或论文limitations中删除，但它不再
进入FCC matrix、gate或machine decision。当前唯一仍直接阻塞ordered-field claim的是SIFF相对independent仅
`+0.2580%`、低于`+0.3%`且validation/test反转。

[Decision] 因此本节点通过的是**narrative/design gate**，不是effectiveness gate。下一步只允许原样SIFF-v2的
Final Claim Confirmation（FCC）；FCC前后均禁止修改method identity。若FCC失败，停止把SIFF-v2作为paper core，
不再用router、loss、rank、readout或seed selection rescue。

## 3. Exact method identity

### 3.1 Tensor path

Encoder输出flatten后的`hidden [B,C,R]`。SIFF-v2固定$Q=2$、coordinate dimension $D=4$、mode rank
$K=256$与五个coupling scopes $s\in\mathcal S$：

1. `mode_weight [Q,D,R,K]`和`mode_bias [Q,D,K]`把`hidden`变为
   `component_modes [B,C,Q,D,K]`；
2. `scale_basis [S,Q]`的第一列为1，第二列为centered unit-RMS ordered log-scale $z_s$；
3. `scale_modes [B,C,S,D,K]`由
   $$
   M_s(X)=M_0(X)+z_sM_1(X)
   $$
   生成；
4. 每个$M_s$进入对应nested scope pooling与shared synthesis，得到
   `arms [B,C,S,T]`；
5. 既有direct policy由history state与future coordinate产生`weights [B,C,T,S]`，凸融合得到
   `full [B,C,T]`；
6. requested horizon只在最后裁剪为`[B,H,C]`。

### 3.2 What is frozen

- readout：`siff-coupling-field`；
- policy：`direct`；
- objective：`equal_skill`；
- rank：256；
- coupling scales、natural profiles、optimizer、four-horizon selector与test protocol全部不变；
- 不加入TSAF、H embedding、CCSF、MCCA、PCC、CTD、new loss或第二router；
- 不复用warm start，不做frozen replacement effectiveness claim。

## 4. Paper problem and contribution boundary

### 4.1 Problem statement

varied-horizon forecasting通常把“scale”放在input tokenization、history retrieval、frequency experts或requested-H
conditioning中。SIFF处理的是另一个位置：**future output operator一次应耦合多宽的target region**。同一history
state需要支持point/local/block/global等nested output-coupling extents；把这些extents视为无关experts会失去它们的
order relation，把它们压成单一field又会失去scope variation。

### 4.2 Single contribution

论文只保留一项method contribution：

> 一个以ordered coupling scale为coordinate、由共享components生成nested full-domain forecast operators的
> Scale-Indexed Forecast Field，并在同一模型内产生所有preregistered horizons。

controls与dense-horizon analysis属于evidence，不包装为第二个method contribution。`equal_skill`只作为使五个arms
可共同训练的必要optimization contract，不宣称独立loss novelty。

### 4.3 Allowed claims

- SIFF明确建模future-output coupling scale，而非只建模input resolution；
- ordered scale variation相对constant、permuted与Q1-wide controls具有稳定增量；
- five arms保持diversity、oracle headroom与非零nonconstant component contribution；
- 一次full-domain computation可服务多个prefix horizons；这是method property，不宣称Bayes层面的H information。

### 4.4 Prohibited claims

- 首次multi-scale forecasting、首次one-model varied horizon、首次MoE/router、首次future-coordinate query；
- ordered field已被证明严格优于independent fields；
- direct history policy已被证明必要或能识别sample-wise best scope；
- TSAF/target-only allocation是贡献；
- SIFF-v2已超过A6_MEASURE或已通过paper-core gate；
- exact projectivity、full-$T$ crop或A6 interface是所有future methods的硬约束。

## 5. Latest primary-source boundary

检索日期为`2026-07-21`。本轮external discovery/verification确认：MoLE已覆盖forecast experts与input-dependent
router；TimeExpert已覆盖query-specific local/global expert selection；MoHETS与M²FMoE已覆盖heterogeneous、
frequency与multi-resolution expert fusion；TA-SparseMG已把multi-scale gating放入prediction head；Self-Gating
Attention已覆盖shared structure加input-dependent residual。故SIFF不能依靠这些primitive形成novelty。

尚未发现上述primary sources覆盖完整链：

```text
future-output coupling extent as an ordered coordinate
-> shared history-conditioned operator components over that coordinate
-> nested scope-specific full-domain synthesis
-> target-wise fusion in one varied-horizon forecast field
```

这是`conditional_pass`而非novelty定论：正式写作仍需逐篇full-text/code核对，且不能把没有搜到等同于不存在。

## 6. Evidence balance before FCC

| Layer | Positive evidence | Blocking evidence | Status |
| --- | --- | --- | --- |
| paper-facing effectiveness | vs A6_FULL `+1.6436%`；vs PCSD_EQUAL `+0.5906%` | vs A6_MEASURE `-0.2366%` | blocked |
| matched specificity | constant `+0.9393%`；permuted `+0.3959%`；Q1-wide `+1.1619%` | independent only `+0.2580%`，below gate | blocked |
| internal health | 7/7；oracle `+6.3937%`；pairwise NRMSE `0.1587`；component ratio `0.1475` | policy-skill alignment weak；dataset dependence | explanatory only |
| post-TSAF evidence | target-only main effect near zero；no numeric pathology | same-rank interaction negative；TSAF effectiveness fail | no successor support |

[Self-critique] 这一设计降低了FCC的performance comparator强度：seed2021中SIFF已经超过A6_FULL，而A6_MEASURE
更强。即使FCC通过A6_FULL，也只能支持完整method package相对source carrier的three-seed稳定性；ordered-field
必要性仍必须通过independent control，且历史A6_MEASURE negative必须在论文中诚实报告。

## 7. FCC frozen design

### 7.1 Role

`SC1-SIFF-v2-FCC-v1`不是新candidate architecture，而是immutable SIFF-v2的final claim confirmation。它只回答：

1. SIFF完整method package是否在three-seed evidence上超过用户指定的`A6_FULL`；
2. ordered field是否在three-seed evidence上超过matched `SIFF_INDEPENDENT_EQUAL`。

### 7.2 Matrix

| Arm | Role | New seeds | Datasets | New runs |
| --- | --- | --- | --- | ---: |
| `siff_equal` | immutable candidate | 2022, 2023 | ETTh1, ETTh2, ETTm1, ETTm2, Weather | 10 |
| `a6_full` | user-selected method-package baseline | 2022, 2023 | same five | 10 |
| `siff_independent_equal` | strongest function-class attribution control | 2022, 2023 | same five | 10 |

复用同protocol的seed2021 15 runs，形成45 effective runs。每run输出validation与official-test
H96/H192/H336/H720 MSE/MAE，共180 three-seed test cells。所有arms必须from-scratch joint training；每个
dataset/seed的Encoder initialization需paired，independent arm按现有parameter-matching rank，不因test结果改变。

### 7.3 Split and checkpoint roles

- validation：只以四个standard horizons mean MSE选checkpoint；
- official test：只在30/30 new training完整、run audit通过后执行一次完整matrix；
- test已是`test_informed`，不得称untouched；
- 不按dataset、horizon、seed选择模型或修改rank；
- seed2021既有结果必须完整保留，不能只报告new seeds。

### 7.4 Gates

对three-seed pooled dataset-horizon macro，延续原预注册margin，不因seed2021 near miss降低门槛：

1. `SIFF > A6_FULL`：MSE gain至少`+0.3%`，MAE严格为正，dataset wins至少3/5、horizon wins至少3/4；
2. `SIFF > INDEPENDENT_EQUAL`：同一组gates；
3. seed stability：两项comparison的MSE至少2/3 seed macro为正；
4. no reversal masking：报告每个seed/dataset/horizon cell，不以pooled macro隐藏单seed大负值；
5. internal health沿用all-finite、arm diversity、oracle、entropy与component-use，只解释，不替代1–3。

两项primary comparisons都通过，才得到`passed_core_candidate_pending_modern_baselines`。任一失败，decision=
`siff_v2_final_claim_not_confirmed_stop_paper_core_rescue`。

## 8. Rollback and authorization boundary

- narrative gate：conditional pass；
- code/model implementation：无变化，不需要Step7A；
- remote training：用户已授权，尚未启动；
- official test：用户已授权，但只允许在30/30 training完整后执行一次；
- local prelaunch：25/25 checks、30/30 jobs、15/15 historical references通过；
- 若FCC通过，下一步才是modern native baselines与完整formal ablations；
- 若FCC失败，回paper portfolio decision，而不是回SIFF rank/loss/router tuning。

最终decision：

```text
fcc_a6_full_scope_authorized_prelaunch_pass_remote_not_started
```
