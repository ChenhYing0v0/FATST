# SC-D19-IFC Step 6 Control Design

## 1. Decision

| Field | Content |
| --- | --- |
| `current_step` | D19 Step 6 complete；Step 7A local implementation authorized |
| `candidate_version` | `SC-D19-IFC-control-v1` |
| `role` | `control_only` |
| `problem` | implicit wave trajectory control是否超过A6 learned-basis与same-information matched nonlinear direct control |
| `existence_evidence` | D18 closure + IF source/code audit |
| `idea` | four-arm matched end-to-end comparison |
| `theory_check` | full-720 crop projectivity、source mechanism与function-class controls明确 |
| `design` | 5 datasets × 4 arms × seed2021；15 new runs + 5 reused A6 references |
| `narrative_gate` | not required；IF不得成为本项目claim |
| `effectiveness_gate` | frozen below；not run |
| `artifacts` | config、static checker、本文 |
| `decision` | `step6_pass_step7a_local_only` |

## 2. Frozen arms

| Arm | Input | Decoder | Purpose |
| --- | --- | --- | --- |
| `A6_MEASURE` | A6 hidden | rank-256 learned basis | mandatory reference |
| `IF_MEASURE` | A6 hidden + history amplitude/phase | 3 source-informed heads + polar iFFT | trajectory control |
| `IF_NOSKIP_MEASURE` | A6 hidden + zero spectrum tensors | identical IF parameters/init | skip attribution |
| `DIRECT_NONLINEAR_MATCHED_MEASURE` | A6 hidden + history amplitude/phase | parameter-matched direct MLP | nonlinear/capacity control |

`IF_NOSKIP`保留与IF完全相同的input shape与parameters，仅把sample-specific history spectrum替换为zero。
这会让对应columns无active gradient，但它正是被移除的信息路径；active-parameter差异不参与profile选择或结果
优劣判断。

## 3. Parameter contract

IF每个head的input为$R+49$，output为361，hidden width固定使用official default 2048。三个heads总参数：

$$
N_{IF}
=3\left[2048(R+49+1+361)+361\right].
$$

matched direct input为$R+98$，output为720：

$$
N_{direct}
=W_d(R+98+1+720)+720.
$$

冻结结果：

| Profile | $R$ | IF params | Direct width | Direct params | Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| P12-D64 | 768 | 7,244,859 | 4,565 | 7,245,375 | 0.0071% |
| P24-D32 | 768 | 7,244,859 | 4,565 | 7,245,375 | 0.0071% |
| P24-D64 | 1,536 | 11,963,451 | 5,080 | 11,964,120 | 0.0056% |
| P48-D64 | 3,072 | 21,400,635 | 5,500 | 21,401,220 | 0.0027% |

[Boundary] IF明显大于A6 rank-256 decoder；这是为什么`DIRECT_NONLINEAR_MATCHED`是hard control。参数量本身
不用于选择dataset profile，也不因其较大而拒绝IF。

## 4. Shared protocol

- datasets：ETTh1、ETTh2、ETTm1、ETTm2、Weather；
- seed：2021；
- all arms end-to-end from same initialization class；
- paired encoder initialization within dataset；
- full output：720；
- requested horizon不进入forward；
- training：harmonic-L1 `measure_only`；
- checkpoint：validation H96/H192/H336/H720 mean MSE；
- batch size 32、LR $10^{-4}$、最多20 epochs、patience 5；
- formal Phase A：15 new runs，复用5个A6_MEASURE references；
- official-test cells：20 artifact units × 4 horizons = 80；
- seeds2022/2023 held。

## 5. Hard gates

### Layer 1：IF相对A6

- MSE macro gain $\ge0.3\%$；
- positive cells $\ge11/20$；
- positive datasets $\ge3/5$；
- positive horizons $\ge3/4$；
- MAE macro gain $\ge0$。

### Layer 2：wave-specific attribution

`IF_MEASURE`相对`DIRECT_NONLINEAR_MATCHED_MEASURE`使用相同五项gate。若未通过，则generic nonlinear
capacity解释，不形成trajectory-structure evidence。

### Skip attribution

IF相对IF-no-skip MSE macro非负且至少3/5 datasets为正。失败不自动否定wave synthesis，但禁止声称history
spectrum skip有帮助。

### Layer 3：internal health

- all finite；
- full/crop max gap $\le10^{-6}$；
- amplitude非全零且finite；
- phase sine/cosine/atan2 gradients finite；
- IF与direct prediction NRMSE非零；
- paired encoder initialization；
- IF/no-skip decoder initialization完全相同；
- direct parameter gap $\le0.1\%$。

### Layer 4：decision map

| Outcome | Decision |
| --- | --- |
| IF > A6且IF > matched direct | trajectory-structure headroom supported；回Step2/4研究超越IF prior的新method |
| IF > A6但不胜matched direct | generic nonlinear/capacity explains；no wave claim |
| direct > A6、IF不胜A6 | generic direct headroom only |
| IF/direct均不胜A6 | close implicit control；启动fixed-past decoder paper viability review |
| numeric/protocol failure | 只修复exact control implementation，不拒绝方向 |

## 6. Authorization

Step 6只授权Step7A local implementation：

1. 新readout forward paths；
2. source invariant与shape/projectivity tests；
3. IF/no-skip initialization pairing；
4. parameter accounting；
5. CLI/config wiring；
6. zero/near-zero phase gradient tests。

remote、official test与paper method均为false。Step7A通过后还必须单独完成Step7B prelaunch。
