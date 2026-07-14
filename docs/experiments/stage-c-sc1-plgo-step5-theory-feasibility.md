# SC1-PLGO Step 5 Theory Feasibility Protocol

## Status

| Field | Value |
| --- | --- |
| `stage` | StageC-UVHF |
| `candidate` | SC1-PLGO |
| `step` | Step 5 theory feasibility |
| `experiment_type` | CPU float64 constructive/no-go audit；no training |
| `method_training_authorized` | false |
| `decision` | `partial_pass_step6_design_only` |
| `next_step` | Step 6 tensor/narrative/control design |

## What We Planned To Test

Step 4只授权检验以下问题：

1. stable global-local analysis/synthesis是否存在；
2. arbitrary prefix是否只需与domain相交的supports；
3. 是否能无dense bypass地包含A6；
4. square、overcomplete与independent-group variants的function class分别是什么；
5. support pruning是否等价于真实selective coefficient computation。

该protocol不允许dataset fitting、model implementation、remote training或以metric结果重新定义claim。

## Construction And Controls

- `RGNB`：DCT root subspace + balanced interval-local orthogonal complements；
- `PLGO-ONB-M0`：square RGNB下的A6 exact morph control；
- `PLGO-FRAME`：global DCT与square local basis直接拼接的overcomplete control；
- `PLGO-INDEPENDENT-GROUP`：按root/depth独立history maps的capacity/no-go control；
- `PLGO-ATOM-CONDITIONED-GENERATOR`：只记录为Step 6 open question，本轮不实现。

## Cases And Gates

构造cases：

```text
(1,1), (2,2), (3,3), (5,4), (7,4), (16,4), (96,8),
(720,1), (720,4), (720,8), (720,16), (721,16)
```

tuple为$(T,r_g)$。algebra tolerance为`1e-10`。必须同时通过：

- $Q^\top Q=I$；
- root projector等于global DCT projector；
- local details在interval外严格为零并与global prototypes正交；
- A6 morphism精确；
- selected prefix reconstruction精确；
- 所有$H$满足预注册active bound。

Function gate独立于algebra gate：若square construction只是bijective transform，则降为control；若union存在
kernel则降为overcomplete control；若independent groups达到full affine，则不得以local-global structure归因。

## Result And Decision

- 12个basis cases与101个selected prefixes通过，max gap `2.141e-13`；
- 3,731个all-$H$ active-bound cases通过；
- raw restricted DCT coordinates存在最高`3.110e17` condition number；stable Chebyshev local chart降到
  `1.784e3`；
- ONB-M0 exact A6 morph成立但没有function novelty；
- FRAME bounds为$[1,2]$但kernel dimension=$r_g$；
- T720 independent-group rank caps为`[16,16,32,64,128,256,208]`，sum=720且等价full affine；
- selective synthesis成立，generator-level efficiency未成立。

最终decision=`partial_pass_step6_design_only`。详细failure attribution与proof见
`analysis/stage_c_sc1_plgo_step5_theory_20260714/step5_theory_feasibility.md`。

## Rollback Rule

Step 6若不能提出非overcomplete、非full-affine-equivalent、非A6 residual且不读取$H$的coefficient mechanism，
则rollback Step 4 redesign。不得先进入Step 7训练，再根据性能包装claim。
