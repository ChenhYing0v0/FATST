# Post-PCC Step5 Theory Checker 代码说明

## Purpose

`scripts/check_stage_c_post_pcc_step5.py`不是production model，也不读取dataset。它用float64 synthetic cases检查
`SC1-SIFF`与`SC2-MCCA`是否具备进入Step6 source/design audit的最低数学条件。

## SIFF Checks

输入`hidden [B,C,R]`与scale-basis-conditioned weights `W [Q,D,R,K]`：

```text
hidden [B,C,R]
 -> components [B,C,Q,D,K]
 -> phi(scale) [S,Q]
 -> modes [B,C,S,D,K]
 -> synthesis [T,D,K]
 -> arms [B,C,S,T]
```

- 当`Q=1`且`phi_0=1`时，数值检查exact current-field containment；
- 先生成完整`T`再crop，检查prefix projectivity；
- 在constant-coordinate construction中，current field对不同scope不可区分，而`Q>1`通过scale coordinate产生非零
  contrast，用作non-absorbability witness。该witness只针对current Q=1 field，不证明超越所有generic wider heads。

## MCCA Checks

`projective_measure [T]`由dense prefix incidence产生；`capability [T,S]`与row/column marginals进入positive entropic
Sinkhorn：

```text
capability [T,S] + omega [T] + rho [S]
 -> allocation A [T,S]
 -> skill weights A
 -> router target A / omega[:,None]
```

检查row/column marginal误差、crossed capability下相对uniform的best-scope mass、dominant-arm case中的minimum column
coverage，以及skill/router gradients finite且非零。assignment在loss中stop-gradient；checker不宣称对Sinkhorn本身的
novelty。

## Boundary

通过只说明algebraic feasibility。production parameter budget、mini-batch marginal定义、generic BASE/SSR controls、
architecture non-absorbability与narrative claim仍属于Step6；本checker无权授权implementation、remote或test。
