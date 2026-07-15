# SC1-JAPO Step 7A Production Implementation Gate

## Decision Summary

| Field | Result |
| --- | --- |
| `candidate` | `SC1-JAPO` |
| `current_step` | Step 7A complete；Step 8 seed-2021 remote screen authorized |
| `decision` | `step7a_pass_remote_screen_authorized` |
| `shape_prefix_gate` | 210/210 pass |
| `gradient_gate` | 35/35 pass |
| `paired_encoder_initialization` | 5/5 datasets pass across seven arms |
| `paired_expert_bank_initialization` | 5/5 datasets pass across six JAPO arms |
| `maximum_prefix_gap` | `4.768e-7` |
| `maximum_patch_block_gap` | `5.722e-6` |
| `initial_gate_entropy_min` | `0.999944` |
| `initial_expert_usage` | `0.496363–0.503637` |
| `test_used` | `false` |
| `forecast_training_run` | `false` |
| `SC2` | held |

## 1. What We Tested

Step 6冻结了JAPO的数学与实验contract；Step 7A检验production code是否真的实现该contract，而不是只让
design-only tensor prototype通过：

1. 五个frozen natural profiles × 七arms × 六prefix，共210个shape/projectivity cases；
2. 五profiles × 七arms，共35个end-to-end gradient cases；
3. 相同dataset/seed下，七arms的Encoder initialization hash是否一致；
4. 六个JAPO arms的expert-bank initialization hash是否一致，同时两个experts是否彼此独立；
5. requested horizon是否只参与active atom selection，不进入任何learned module input；
6. `memory [B,C,P,D] -> hidden [B,C,PD]`与显式patch-block expert projection是否数值等价；
7. remote runner是否固定35 jobs、validation-only、full-H720 L1、best-val与seven-arm analyzer。

## 2. Production Tensor Flow

Encoder保持A6 carrier不变：

$$
M\in\mathbb R^{B\times C\times P\times D}
\rightarrow h\in\mathbb R^{B\times C\times R},\quad R=PD.
$$

两个独立experts执行：

$$
z_e=A_eh+a_e\in\mathbb R^{B\times C\times256},
$$

$$
r_{j,e}=V_{e,j:}z_e+c_{j,e}
\in\mathbb R^{B\times C\times T\times2}.
$$

`JAPO-JOINT-GEO` router执行：

$$
s=\operatorname{RMSNorm}(\tanh(W_h\operatorname{LN}(h))),
$$

$$
\phi_j=\operatorname{RMSNorm}(\tanh(W_dd_j)),
$$

$$
\pi_{j,:}=\operatorname{softmax}_e
\left(W_g\operatorname{RMSNorm}(s\odot\phi_j)/\sqrt{32}\right).
$$

最终只对`atom_start < H`的atoms计算：

$$
\alpha_j=\sum_e\pi_{j,e}r_{j,e},\qquad
\widehat y_H=Q_{[0,H),\mathcal A_H}\alpha_{\mathcal A_H}.
$$

因此$H$没有embedding、router argument或active-atom normalization；它只决定tensor subset与输出domain。

## 3. Seven-Arm Contract

| Arm | Production readout | Active router path |
| --- | --- | --- |
| A6 | `learned-basis-forecast-operator` | none |
| JOINT-GEO | `japo-joint-geo` | history × canonical atom |
| UNIFORM | `japo-uniform` | none；fixed 0.5/0.5 |
| HISTORY | `japo-history` | history only |
| ATOM | `japo-atom` | canonical atom only |
| JOINT-PERM | `japo-joint-perm` | history × permuted atom |
| JOINT-RANDOM | `japo-joint-random` | history × moment-matched random atom |

所有JAPO arms实例化相同参数结构。local hash audit与production
`initialization_contract.json`均确认：同一dataset/seed下expert bank完全paired，within-bank experts不相同。

## 4. Gate Results

`step7a_gate.json`的所有boolean gates为true：

- maximum full-prefix gap=`4.7683716e-7`，低于float32 protocol tolerance `1e-5`；
- maximum patch-block rewrite gap=`5.7220459e-6`，低于trained diagnostic tolerance `2e-5`；
- minimum normalized gate entropy=`0.9999443`；
- expert usage位于`0.4963628–0.5036372`；
- active parameters无missing、nonfinite或zero gradient；
- learned modules只接收tensor，requested $H$不进入learned module input；
- profile contract hash精确匹配
  `80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a`。

Runner dry-run进一步通过：35 jobs、七个synthetic checkpoint audits与analyzer synthetic fixture完整。

## 5. Artifact Definitions

### `shape_prefix_checks.csv`

- `full_prefix_max_abs`：同一model对native prefix与full-H720 output crop的最大绝对差；
- `pass`：shape精确匹配`[1,H,C]`且gap不超过`1e-5`。

### `gradient_checks.csv`

- `active_parameter_tensors`：由各arm真实forward path决定的Encoder、expert与router parameter tensors；
- `missing/nonfinite/zero_gradient_tensors`：对synthetic MSE反传后的异常数量；
- `flatten_block_sum_max_abs`：flatten linear map与逐patch block sum的最大绝对差。

### `initialization_contract.csv`

- `encoder_hash`：Encoder parameter bytes的SHA-256；
- `expert_bank_hash`：两个branch、atom basis与coefficient bias的SHA-256；
- `basis_hash/descriptor_hash`：RGNB synthesis与router descriptor buffers的SHA-256；
- `initial_gate_entropy/usage`：synthetic history上expert-only softmax的normalized entropy与mean usage。

Remote每个run还将保存`initialization_contract.json`、`patch_diagnostics.json`与
`trained_invariants.json`，analyzer会在读取metrics前先审计这些contracts。

## 6. Code-Theory Consistency

- intended theory：free full-rank experts恢复A6 operator freedom；joint history-atom routing解除fixed
  separability；expert-only softmax保持prefix projectivity；
- code realization：独立$E=2,K=256$ experts、$G=32$ factorized router、RGNB active-atom restriction与
  paired initialization artifacts；
- remaining proxy：local gates不能证明router会学习有效specialization，也不能证明约2x decoder capacity已被
  optimization完全公平地控制；
- falsification：JOINT不超过UNIFORM说明capacity control解释收益；不超过HISTORY说明atom interaction不必要；
  不超过ATOM/PERM/RANDOM说明joint或canonical geometry claim不成立。

## 7. Decision And Rollback

[Decision] Step 7A implementation gate通过，允许进入Step 8 seed-2021 validation-only remote screen。

[Boundary] 这不是effectiveness evidence，不授权test、SC2、hyperparameter tuning或architecture changes。

[Rollback] remote若出现NaN、prefix/hash/protocol失败，归为`optimization_or_numeric_pathology`或
`implementation_fault`并回Step 7A/6 repair；只有完整且稳定的same-bank结果才能触发Step 4 method attribution。
