# SC-ISCF-UPA-D2 Uniform Policy Anchor Diagnostic Design

## 1. Decision and role

Decision=`upa_d2_diagnostic_design_frozen_implementation_training_not_authorized`。

UPA-D2是D1后的唯一最小matched diagnostic。它不修改ISCF architecture，不引入requested-H，也不预设paper
Contribution 2；只用information-free uniform policy target检验ARMERR/SHUFFLED的公共train-time co-adaptation effect。

## 2. Mechanism question

现有两个positive controls的target semantics不同：ARMERR使用standalone pointwise error credit，SHUFFLED破坏coalition
scope binding；二者最终policy均near-uniform且performance近似。D1排除run drift，D0排除post-hoc smoothing。

剩余最小问题：

> 是否只需在joint training中把policy锚定到broad/uniform allocation，就能让five scope arms获得更健康的共同适配？

## 3. Exact objective

保留EQUAL原objective：

$$
\mathcal L_{\mathrm{base}}=\mathcal L_{\mathrm{fused}}+
\mathcal L_{\mathrm{equal\ skill}}.
$$

新增train-only uniform anchor：

$$
q_s=1/S,\qquad
\mathcal L_{\mathrm{UPA}}=\sum_s q_s\log\frac{q_s}{p_s},
$$

$$
\mathcal L=\mathcal L_{\mathrm{base}}+lambda(t)\mathcal L_{\mathrm{UPA}}.
$$

$\lambda(t)$必须与ARMERR/SHUFFLED完全相同：前25% progress线性ramp到`0.1`，之后保持。Route target不读取target、
arms、history、future coordinate或requested horizon；gradient只直接进入policy，但通过fused loss间接改变arms/shared
encoder的joint-training path。Inference graph与latency不变。

## 4. Frozen matrix

| Field | Value |
| --- | --- |
| new arm | `iscf_equal_uniform_anchor` |
| datasets | Weather, ETTm1, ETTh1, ETTh2, ETTm2 |
| seed | 2021 |
| training | H720；from scratch；same natural profiles/ranks |
| selector | mean validation MSE over H96/H192/H336/H720 |
| new runs/cells | 5 / 20 validation cells |
| read-only references | contemporaneous EQUAL, ARMERR, SHUFFLED |
| formal test | false |

## 5. Matched gates

### `uniform_anchor_sufficient`

全部满足：

- UPA vs EQUAL macro MSE `>=+0.3%`、MAE `>0`；
- 至少3/5 datasets、3/4 horizons与12/20 cells MSE positive；
- UPA相对ARMERR与SHUFFLED各自macro MSE `>=-0.1%`；
- final policy entropy median `>=0.95`；
- five-scope gradients、numeric、artifact、init、no-test checks全部通过。

Decision=`information_free_uniform_anchor_sufficient_for_coadaptation_clue`。这只确认problem mechanism，不通过paper
narrative gate。

### `target_variation_required_or_uniform_design_insufficient`

UPA vs EQUAL primary fail，但ARMERR/SHUFFLED historical controls仍通过。该结果说明constant uniform target不足；可能需要
sample/coordinate-varying但无需正确binding的training signal。Failure只能归因exact uniform design，不允许恢复SCC/RSCC
或直接推广为scope direction rejection。

### `unresolved`

Primary positive但不满足control noninferiority或stability。不得按dataset/horizon调weight、schedule或target。

## 6. Controls and failure attribution

- Architecture/objective control：EQUAL；
- no-binding positive controls：ARMERR、SHUFFLED；
- inference-only control：PSA-D0 frozen shrinkage negative；
- exact binding negative：RSCC；
- numeric failure只作protocol repair；
- control noninferiority failure=`capacity_or_generic_regularization_control_explains`；
- UPA positive不能建立scope-specific novelty。

## 7. Source and narrative boundary

本设计复用2026-07-22 Post-RSCC primary-source audit。GateTS与$\phi$-Balancing直接覆盖MoE balancing，forecast
combination/stacking literature覆盖weight shrinkage与complexity control，Dense Backpropagation覆盖training-time expert
signal。故UPA只作mechanism identification；不得以“首次uniform/load balancing”作为claim。

若UPA通过，下一步必须回Step4寻找ISCF-native necessity，例如policy对scope-arm gradient allocation的可测contract及其
forecasting-specific consequence；在此之前不实现paper method、不访问formal test。

## 8. Authorization boundary

| Action | Authorized |
| --- | --- |
| D2 design documentation | true |
| objective implementation | false |
| local Step7A | false |
| remote smoke/training | false |
| formal test | false |
| method promotion | false |
