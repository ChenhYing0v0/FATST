# SC-D19-IFC Step 7A Local Implementation Gate

## 1. Decision

| Field | Content |
| --- | --- |
| `current_step` | D19 Step 7A complete；Step 7B prelaunch next |
| `candidate_version` | `SC-D19-IFC-control-v1.1` |
| `role` | `control_only` |
| `problem` | source-informed implicit trajectory control是否可在matched A6 contract中被公平、稳定地实现 |
| `existence_evidence` | D18 closure + IF paper/code audit |
| `idea` | IF、zero-spectrum no-skip与same-information parameter-matched direct三条production readout |
| `theory_check` | full-720 synthesis后prefix crop；same 720-point normalized history；paired E2E initialization |
| `design` | 15 new-run manifest；5 datasets × 3 new arms；A6 reference后续复用 |
| `narrative_gate` | not applicable；IF仍是prior-covered control |
| `effectiveness_gate` | not run |
| `artifacts` | manifest、109 local cases、gate summary、code explanation |
| `decision` | `step7a_pass_step7b_prelaunch_next` |

## 2. What was implemented

### `IF_MEASURE`

$$
h[B,C,R],X_n[B,720,C]
\rightarrow (A_x,\Phi_x)[B,C,361],
$$

三个独立two-layer MLP分别预测future amplitude、phase sine与phase cosine：

$$
\hat A=\left|\operatorname{LeakyReLU}_{0.5}(z_A)\right|,
\qquad
\hat\Phi=\operatorname{atan2}(\tanh z_s,\tanh z_c).
$$

然后：

$$
\hat Y_{full}
=\operatorname{irFFT}_{720}^{ortho}
\left(\hat A e^{i\hat\Phi}\right)
\in\mathbb R^{B\times C\times720}.
$$

### `IF_NOSKIP_MEASURE`

module、parameter shapes与initialization和IF完全相同，只把$A_x,\Phi_x$替换为zero。它隔离history-spectrum
skip，不通过减少参数或改变width制造差异。

### `DIRECT_NONLINEAR_MATCHED_MEASURE`

读取相同$[h,A_x,\Phi_x]$，经GELU MLP直接输出720 points；不使用polar spectrum或iFFT。它逐profile与IF参数
差小于0.01%，用于判断收益是否只是generic nonlinear capacity。

## 3. Gate construction

`scripts/check_stage_c_d19_if_control_step7a.py`构造：

1. v1.1 supersession、same-history、arm mapping、profile hash与authorization governance；
2. 5 datasets × 3 new arms = 15-row production CLI manifest；
3. 4 unique profile parameter contracts；
4. 3 readouts × 20 prefixes = 60 shape/projectivity cases；
5. paired A6/IF/no-skip/direct Encoder initialization hashes；
6. IF/no-skip identical decoder hash；
7. Encoder、amplitude、phase-sine、phase-cosine、direct gradient groups；
8. near-zero history numeric probe；
9. ALU、atan2 quadrant与orthonormal FFT source-reference cases；
10. three full-model wiring cases。

该gate不加载dataset、不训练模型、不读取validation/test。

## 4. Results

| Category | Passed / Total |
| --- | ---: |
| governance | 5 / 5 |
| CLI | 15 / 15 |
| parameter accounting | 12 / 12 |
| shape/projectivity | 60 / 60 |
| gradients | 10 / 10 |
| initialization | 2 / 2 |
| model wiring | 3 / 3 |
| numeric | 2 / 2 |
| prediction deformation | 2 / 2 |
| source invariants | 3 / 3 |
| **overall** | **114 / 114** |

关键数值：

- maximum full/crop prefix gap：`0.0`；
- IF vs no-skip paired-initialization prediction NRMSE：`1.0632`；
- IF vs matched direct initialization prediction NRMSE：`2.3237`；
- near-zero-history phase radius minimum：`0.0380`；
- orthonormal FFT roundtrip max gap：$1.33\times10^{-15}$；
- all required gradient groups finite and nonzero。

[Boundary] NRMSE只证明interventions没有在初始化时产生相同输出，不是accuracy或mechanism gain。

## 5. Code-theory consistency

| Intended theory | Code realization | Remaining proxy |
| --- | --- | --- |
| one full trajectory supports all horizons | every readout computes720 then crops | training/evaluation尚未运行 |
| IF/no-skip isolates spectrum skip | identical module/hash；features true vs zero | learned use需internal artifacts |
| wave structure vs generic capacity | same information + <0.01% params direct control | exact function classes仍不同，正是待测因素 |
| matched joint learning | same Encoder initialization class and production forward | Step7A只做local synthetic gradients |
| source-informed phase continuity | tanh sine/cosine + atan2 | full training仍需NaN/resource smoke |

可证伪条件：

1. Step7B real-batch smoke出现non-finite或phase gradient pathology；
2. remote artifacts缺失paired initialization或prefix invariant；
3. direct parameter gap超过0.1%；
4. formal IF未超过matched direct，则wave-specific headroom不成立；
5. IF与direct均不超过A6_MEASURE，则进入fixed-past decoder viability review。

## 6. Authorization

Step7A通过只允许进入Step7B prelaunch：

- production runner与completeness checker；
- remote GPU/resource smoke；
- 15 new-run matrix的launch authorization audit；
- official-test metadata freeze。

当前`remote=false`、`official_test=false`、`paper_method=false`。不得以114/114 local pass声称效果或贡献。
