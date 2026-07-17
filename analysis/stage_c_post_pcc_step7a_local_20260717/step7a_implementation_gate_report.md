# SIFF/MCCA Step 7A Implementation Gate

## 1. What We Tested

本轮把Step6冻结的两项设计落为production code，并只做local、synthetic、validation-protocol检查；未读取test，
未启动remote training。

- `SC1-SIFF-v1`：检查scale basis、tensor shape、full-domain prefix projectivity、PCSD Q1 containment、A6
  subspace、constant/permuted/independent controls，以及five-profile matched-rank误差。
- `SC2-MCCA-v1`：检查float64/float32 Sinkhorn marginals、与PCC完全相同的scope total mass、KL
  I-projection witness、scope floor和arm/policy gradients。

## 2. Tensor Path

```text
hidden [B,C,R]
 -> mode_weight [Q,D,R,K]
 -> component_modes [B,C,Q,D,K]
 -> scale_basis [S,Q]
 -> scale_modes [B,C,S,D,K]
 -> unchanged PCSD scope pooling and shared synthesis
 -> arms [B,C,S,T]
 -> policy [B,C,T,S]
 -> fused forecast [B,T,C]
```

MCCA不改该inference path。训练时将`batch × channel × target`展为assignment rows，以projective measure
$\omega_t/(BC)$作为row marginal，以same-mass capability budget作为column marginal，再执行64次log-domain
Sinkhorn；得到的credit stop-gradient后进入skill loss与route KL。

## 3. Results

| Group | Result | Key maximum/minimum |
| --- | ---: | --- |
| SIFF basis/shape/projectivity | pass | prefix gap `0` |
| Q1/constant/A6 containment | pass | Q1/A6 gap `0`；constant collapse `3.55e-15` |
| scale controls | pass | permutation value-set gap `0`；independent identity gap `0` |
| five-profile parameter matching | 10/10 | maximum relative gap `0.383856%` |
| MCCA float64 marginals | pass | `1.11e-16` |
| MCCA float32 marginals | pass | `4.47e-8`；credit simplex `1.67e-6` |
| same-mass PCC identity | 4/4 | maximum gap `2.78e-17` |
| KL I-projection witness | 3/3 | minimum advantage `0.002882` |
| gradient/floor | 4/4 | arm `0.09209`；policy `0.04093`；min scope mass `0.14329` |
| **Total** | **36/36** | `all_pass=true` |

原始逐case定义、数值和threshold见`cases.csv`；机器可读结论见`local_gate.json`。

## 4. Code-Theory Consistency

[Fact] SIFF确实在history-to-mode path引入scale coordinate，而requested horizon从未进入field；所有prefix仍由同一
$T=720$输出裁剪得到。

[Fact] MCCA与PCC在相同progress下具有相同column mass；因此后续`MCCA vs PCC`不是“总skill supervision更多”的
比较，而是competitive allocation与per-target uniform-floor allocation的比较。

[Strong Evidence] local algebra、numeric和gradient contracts均支持进入Step7B。

[Uncertainty] 本轮不证明SIFF会学习到有用的ordered-scale field，也不证明MCCA的competitive credit能改善
validation/test performance；这些属于Step9-10 effectiveness gate。

## 5. Failure Attribution Boundary

本轮没有numeric pathology。若Step7B失败，不能笼统归因为理论失败，必须分别检查：

1. SIFF是否被constant、permuted、Q1-wide或independent-scope controls解释；
2. MCCA是否被same-mass PCC、pointwise MCCA或uniform balanced OT解释；
3. arm diversity、policy entropy和scope mass是否正常；
4. joint candidate是否只是capacity或optimization差异。

Decision：`step7a_pass_step7b_prelaunch_authorized`。
