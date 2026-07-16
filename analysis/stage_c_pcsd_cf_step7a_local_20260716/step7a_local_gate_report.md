# SC-D15-A PCSD-CF Step 7A Local Gate

## Reader path

本地gate测试`PCSD-CF-v1`的实现是否符合Step4-6冻结的数学与protocol contract。它不访问dataset、validation或
test，不训练模型，也不构成performance evidence。读取顺序为：shape/projectivity -> A6 containment ->
sharing topology -> arm/policy/gradient -> accounting -> decision。

## Gate result

- `overall_pass=true`
- `decision=step7a_local_pass_step7b_design_only_next`
- remote、effectiveness claim、Contribution 2与test仍为`false`。

| gate | pass |
| --- | --- |
| shape_prefix_checks | True |
| model_integration_checks | True |
| containment_checks | True |
| topology_checks | True |
| separation_checks | True |
| partition_checks | True |
| gradient_checks | True |
| accounting | True |
| protocol_contract_checks | True |

## Numerical evidence

### Arbitrary-A6 containment

| dtype | R | output_gap | arm_gap | pass |
| --- | --- | --- | --- | --- |
| float32 | 768 | 1.788e-06 | 1.907e-06 | True |
| float32 | 1536 | 2.503e-06 | 2.384e-06 | True |
| float32 | 3072 | 3.815e-06 | 1.669e-06 | True |
| float64 | 768 | 1.776e-15 | 4.885e-15 | True |
| float64 | 1536 | 2.665e-15 | 4.885e-15 | True |
| float64 | 3072 | 3.109e-15 | 5.329e-15 | True |

### Arm separation and equal-logit initialization

| partition | min_pair_nrmse | mean_pair_nrmse | uniform_gap | pass |
| --- | --- | --- | --- | --- |
| canonical | 0.131493 | 0.671231 | 0.000e+00 | True |
| random | 0.023079 | 0.624918 | 0.000e+00 | True |

### Static accounting

| dataset | R | field/A6 params | total/A6 params | PCSD/A6 FLOPs |
| --- | --- | --- | --- | --- |
| ETTh1 | 1536 | 3.3590 | 3.4487 | 10.5949 |
| ETTh2 | 768 | 3.0291 | 3.1006 | 13.9342 |
| ETTm1 | 768 | 3.0291 | 3.1006 | 13.9342 |
| ETTm2 | 3072 | 3.6184 | 3.7224 | 7.9742 |
| Weather | 768 | 3.0291 | 3.1006 | 13.9342 |

## What each artifact means

- `shape_prefix_checks.csv`：直接readout在5个dataset-aware state widths与13个dense/arbitrary horizons下的
  shape及full-domain prefix crop equality。
- `model_integration_checks.csv`：真实A6-natural encoder路径的`[B,C,P,D_e] -> [B,C,R] -> [B,C,5,720]
  -> [B,H,C]`接线检查。
- `containment_checks.csv`：将任意A6系数映射、basis和bias构造到PCSD constant mode，检查float32/float64
  output与所有scope arms的最大绝对误差。
- `topology_checks.csv`：`target_scope_descriptors=P_sQ[g_s(tau)]`正是group state对history modes的Jacobian；
  检查同组target共享Jacobian、跨组target不同、point/global端点分别有720/1个sharing class。
- `separation_checks.csv`：random trainable parameters下不同scope arms的pairwise normalized RMSE，以及
  direct policy final logits全零产生的equal initial weights。
- `partition_checks.csv`：canonical/random只改变fixed buffers，trainable parameter count与shape hash相同。
- `gradient_checks.csv`：两步local SGD；第一步field active，zero-logit policy output更新后第二步history/target
  policy path active，所有gradient tensors有限。
- `accounting.csv`：参数/DoF、multiply-add FLOP估算与activation估算。FLOP以单channel full-T forward计，
  GELU/softmax标量代价未计；activation是chunked operator的静态上界估算，不是GPU profiler实测。
- `protocol_contract_checks.csv`：requested H/future truth/test不进入learned path，frozen/warm-start/remote/SC2关闭。

## Failure attribution boundary

若本gate失败，只能定位为`implementation_or_theory_contract_mismatch`并回滚Step5/6；若通过，只证明实现忠于
PCSD-CF-v1设计且数值可微，不证明coupling-spectrum hypothesis、性能、paper-core effectiveness或SC2价值。
真正的effectiveness gate仍需单独授权后的Step7B validation-only matched end-to-end实验。
