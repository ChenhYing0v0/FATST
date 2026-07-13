# StageC PMFO-RCT Step 7A Local Gate

## Scope

本轮只验证实现与algebra，不训练、不读取test split，也不评估forecast effectiveness。
Step 7B固定验证集为`ETTm1`、`ETTh2`、`Weather`，但仍需单独启动。

## Gate Result

- decision: `step7a_pass`；
- shape/prefix cases: `90`；
- full-prefix max abs: `4.172e-07`；
- refinement recovery max abs: `2.384e-07`；
- conservation perturbation max abs: `2.682e-07`；
- locality outside-support max abs: `0.000e+00`；
- horizon path audit: `True`。

[Fact] 上述结果只证明代码满足Step 7A tensor/algebra contract；不能证明PMFO有效。

## Parameter And FLOP Audit

下表给出ETTm1 profile；decoder参数对三个profile相同，active参数随Encoder profile变化。
`state-dict params`包含TimeAlign兼容性保留但不进入当前forward的legacy `proj_x`，
因此mechanism attribution以`active params`和`active decoder`为准。
FLOPs是每条univariate series的linear-dominant estimate，包含Linear与显式synthesis/basis
乘加，不包含GELU、RevIN和tensor reshape，因此只用于matched-control审计。

| Variant | Active params | Active decoder | State-dict params | FLOPs H1 | FLOPs H720 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `learned-basis-forecast-operator` | 391408 | 381904 | 945088 | 833024 | 1201152 |
| `pmfo-rct` | 221514 | 212010 | 775194 | 862046 | 3058848 |
| `pmfo-rct-no-transition` | 563184 | 553680 | 1116864 | 1545310 | 1553568 |
| `pmfo-rct-no-conservation` | 222062 | 212558 | 775742 | 863079 | 3130688 |
| `dense-mlp-matched` | 224640 | 215136 | 778320 | 867840 | 867840 |

## Decision Boundary

Step 7A通过仅授权准备Step 7B。`dense-MLP-matched`或`no-transition`若在训练后解释收益，
仍触发`capacity_control_explains`并回滚Step 4；本地invariant通过不能覆盖该effectiveness gate。
