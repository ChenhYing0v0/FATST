# SC-D17-PFC-v1 validation-to-test 结果与失败归因

## 1. 当前研究记录

| Field | Content |
| --- | --- |
| `current_step` | Step 2 problem diagnostic result |
| `problem` | frozen full-domain draft的ordered prefix context能否提供跨validation→test的增量correction？ |
| `existence_evidence` | v0 same-test cross-fit表面正向但protocol invalid；v1使用独立validation fit与test evaluation |
| `idea` | pointwise / causal ordered / row-shuffled / symmetric四类fixed ridge correction |
| `theory_check` | causal path严格prefix-invariant；requested horizon与prefix外target均不进入feature |
| `design` | SIFF_EQUAL、PCSD_EQUAL × 5 datasets；validation 256 rows fit；test 256 rows evaluate |
| `narrative_gate` | not applicable；diagnostic_only |
| `effectiveness_gate` | 6/7 static gates fail；只通过prefix invariance |
| `artifacts` | `d17_validation_to_test/`、`d17_validation_invariants/`、remote launch record |
| `decision` | exact frozen post-hoc correction protocol rejected；future-context direction unresolved；不进入Step 4 method design |

## 2. Protocol audit

remote validation inference在commit`746721b`上完成10/10：

- 2 carriers × 5 datasets；
- 10/10 `pass=true`、`protocol_pass=true`、`readout_contract_pass=true`；
- 每项validation artifact保存256 rows；
- 10个checkpoint hashes均唯一，SIFF五个hash与candidate freeze manifest一致；
- checkpoint前后hash不变；
- 未训练模型，未新增test访问，只复用既有authorized test probes；
- `prefix_invariance_max_abs_gap=0`。

因此本次不是v0的fold leakage，也没有numeric NaN/Inf或checkpoint mutation。

## 3. 主要结果

### 3.1 Macro

| Comparison | MSE gain |
| --- | ---: |
| pointwise vs parent | **-28.7314%** |
| causal vs parent | **-32.6392%** |
| causal vs pointwise | **-3.0356%** |
| causal vs row-shuffled | **-2.3616%** |
| symmetric vs pointwise | **-3.7090%** |

正值表示candidate更好。除projectivity外，所有预冻结gate均失败：

- carriers：0/2；
- datasets：3/5；
- carrier-dataset：6/10；
- horizons：1/4；
- causal vs pointwise positive cells：20/40；
- causal vs shuffled positive cells：23/40。

### 3.2 Dataset与horizon结构

| Dataset | pointwise vs parent | causal vs pointwise | causal vs shuffled | causal vs parent |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | -34.2693% | -0.8601% | -1.0890% | -35.4241% |
| ETTh2 | -149.0654% | +0.0764% | +3.6820% | -148.8751% |
| ETTm1 | +2.4418% | +1.3677% | +1.7703% | +3.7760% |
| ETTm2 | -21.7967% | +1.8653% | +1.7797% | -19.5248% |
| Weather | -18.3073% | -15.4772% | -15.4900% | -36.6179% |

| Horizon | pointwise vs parent | causal vs pointwise | causal vs shuffled | causal vs parent |
| --- | ---: | ---: | ---: | ---: |
| 96 | -13.7069% | +2.6503% | +2.4814% | -10.6934% |
| 192 | -11.0106% | -4.3321% | -3.6156% | -15.8197% |
| 336 | -28.3027% | -3.5662% | -2.7614% | -32.8782% |
| 720 | -54.2156% | -4.6692% | -3.6738% | -61.4162% |

Weather移除后，causal相对pointwise仅`+0.2451%`，仍低于`0.5%` gate；相对shuffled为`+1.0711%`。
这说明macro negative确受Weather放大，但并非删除一个dataset后就形成完整pass。

## 4. 结论该如何解释

### 4.1 已经否定的内容

[Strong Evidence] validation上学习的global pointwise calibration本身无法稳定迁移到test；加入causal或symmetric
future context没有修复，反而进一步恶化。因此以下exact implementation被否定：

`frozen draft -> validation-fitted global ridge correction -> test transfer`

它不能作为method initialization、teacher或post-processing方案。

### 4.2 不能否定的内容

本次包含ETTh2若干超过100%的退化，且parent在同一test上远好于所有corrections。按Diagnostic Failure
Attribution Rule，这属于`optimization_or_numeric_pathology`中的distribution-transfer pathology。又因为carrier
被冻结，按Frozen Component Replacement Fairness：

- 不能据此证明end-to-end jointly trained causal operator必然失败；
- 不能把future-coordinate interaction写成`hypothesis_false`；
- ETTm1、ETTm2与H96上的relative signal只可保留为局部线索，不能升为unified evidence。

正式failure attribution为：

`exact_protocol_failed / calibration_transfer_pathology / direction_unresolved`

## 5. Step 2 决策

D17没有通过预冻结problem gate，因此：

1. 不进入prefix-causal operator的Step 4 method design；
2. 不实现causal CNN、masked attention、triangular mixer或E2E refiner来“抢救”本结果；
3. 不把ETTm1、ETTm2或H96单独挑出来形成paper claim；
4. 不回到SIFF/CCSF router、teacher、scale或temperature sweeps；
5. Contribution 1与Contribution 2继续为空。

下一问题不应再问“怎样修这个ridge”，而应回到exact projectivity的代价：如果exact projectivity使requested
horizon在统计上完全冗余，那么是否存在足够大的horizon-specialization headroom，值得把contract从exact改为
controlled soft projectivity？这由新的`SC-D18-SPC` problem diagnostic先验证。

