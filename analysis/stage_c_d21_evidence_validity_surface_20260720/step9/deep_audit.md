# SC-D21-EVS Step 9 Deep Audit

## Executive decision

`close_exact_evs_problem_split_stability_failed_return_step2`。

SC-D21-EVS没有通过冻结problem gate。D14的sample × region oracle headroom在official test仍然很大，但past-only
interaction相对更简单的additive sample+region解释没有稳定、实质的额外收益。不得降低阈值、补seed或进入EVS
method Step4。

这个结果关闭的是论文级命题：`past × future-region non-separable validity is split-stable and materially useful`。
它不证明所有representation-level interaction在数学上不存在；validation forward audit仍观察到局部可识别性，故该
现象只保留为diagnostic clue，不占paper contribution slot。

## 1. Artifact and protocol integrity

- remote exports：`100/100` NPZ、`100/100` invariant JSON；
- carriers：`neutral_raw`、`a6_natural`；
- datasets：ETTh1、ETTh2、ETTm1、ETTm2、Weather；
- arms：`c_s1/c_s48/c_s144/c_s360/c_s720`；
- split role：official validation只拟合risk probes；official test只评估transfer；
- probe rows：每个carrier/dataset/split固定4096；anchor descriptor为192维；
- checkpoint：只读；没有forecast model training或checkpoint mutation；
- numeric：所有export和policy MSE finite；没有traceback、runtime error或severe degradation。

因此本次failure不是missing artifacts、numeric pathology或未完成matrix造成的。

## 2. What each statistic means

For policy $p$, carrier $c$, and dataset $d$, `test_mse(p)` is the 720-point MSE obtained by selecting one D14 arm per
sample × future bin, weighted by bin lengths `144/216/360`. Relative gain against control $q$ is

$$
G(p,q)=\frac{\operatorname{MSE}(q)-\operatorname{MSE}(p)}{\operatorname{MSE}(q)}.
$$

`region_fixed` uses one validation-selected arm per future bin. `history_global` may change arm by sample but must use the
same arm over all bins. `additive_history_region` combines a sample-dependent arm score with a region intercept but forbids
past × region interaction. `evs_interaction` fits a separate past-dependent risk surface per region. `permuted_history`
breaks validation feature/label correspondence. `oracle` reads test future truth and is upper bound only.

To account for how much oracle opportunity is actually realized, define

$$
S_{\text{interaction}}
=\frac{\operatorname{MSE}(\text{additive})-
\operatorname{MSE}(\text{interaction})}
{\operatorname{MSE}(\text{region})-
\operatorname{MSE}(\text{oracle})}.
$$

This is not a gate metric. It measures how much of the available region-to-oracle gap is specifically supplied by the
non-separable interaction rather than additive main effects.

## 3. Frozen official-test gate

### Ridge primary readout

| Carrier | vs region | vs history-global | vs additive | vs permuted | Oracle vs region |
| --- | ---: | ---: | ---: | ---: | ---: |
| neutral | +0.1854% (2/5) | -0.0295% (2/5) | -0.1242% (2/5) | +0.2762% (4/5) | +7.6399% (5/5) |
| A6 | +0.6576% (3/5) | +0.3468% (3/5) | +0.2209% (4/5) | +0.6076% (5/5) | +10.4053% (5/5) |

Ridge在A6 carrier上有正向interaction clue，但neutral primary carrier没有通过region、history-global或additive
specificity。按冻结规则，不能由A6 sensitivity单独通过problem gate。

### HistGradientBoosting sensitivity readout

| Carrier | vs region | vs history-global | vs additive | vs permuted | Oracle vs region |
| --- | ---: | ---: | ---: | ---: | ---: |
| neutral | +0.6325% (3/5) | +0.0755% (3/5) | +0.0347% (4/5) | +0.6206% (4/5) | +7.6399% (5/5) |
| A6 | +1.1546% (3/5) | +0.1788% (3/5) | -0.0069% (2/5) | +0.9895% (5/5) | +10.4053% (5/5) |

Neutral HGB通过region、history-global与permuted controls，但相对additive只有`+0.0347%`，低于冻结`0.1%`
margin。A6上的additive specificity为负。结果后把阈值从0.1%降到0会构成posthoc gate relaxation，且会把一个
几乎可忽略的increment包装成核心问题，故禁止。

## 4. Oracle-headroom accounting

| Carrier | Readout | Additive main-effect share | Interaction-specific share | Total realized share |
| --- | --- | ---: | ---: | ---: |
| neutral | ridge | +4.295% | -1.435% | +2.860% |
| neutral | HGB | +7.990% | +0.568% | +8.558% |
| A6 | ridge | +4.460% | +2.497% | +6.957% |
| A6 | HGB | +11.734% | -0.027% | +11.707% |

最接近pass的neutral HGB只由interaction额外兑现oracle gap的`0.568%`；A6 HGB则没有interaction增量。
这说明D14的7.64%–10.41% oracle headroom主要仍未被past-only probe兑现，而已兑现部分又主要由sample/additive
main effects解释。Oracle很大不能挽救interaction specificity。

ETTm2是主要反例：neutral HGB interaction相对additive在test为`-0.6941%`，按oracle gap归一化为`-8.08%`；A6
HGB在ETTm1/ETTm2/Weather均不优于additive。这不是单一dataset小波动。

## 5. Posthoc validation-forward stability audit

为了区分`weak readout`与`validation→test transfer failure`，另做不改变gate的posthoc audit：每个validation probe
按sequential row order以前60%拟合，在后40%评估，feature、risk target、readout和controls不变。这个audit不读取
test，也不用于candidate selection。

| Carrier | Readout | Interaction vs additive | Positive datasets | Interaction vs region | Positive datasets |
| --- | --- | ---: | ---: | ---: | ---: |
| neutral | ridge | +0.0255% | 2/5 | -0.2333% | 2/5 |
| A6 | ridge | +0.4037% | 4/5 | -0.1597% | 1/5 |
| neutral | HGB | +0.3092% | 4/5 | +0.7817% | 4/5 |
| A6 | HGB | +0.4406% | 5/5 | +0.0620% | 2/5 |

HGB在validation forward split中可以识别additive之外的interaction，但迁移到official test后neutral从`+0.3092%`
缩到`+0.0347%`，A6从`+0.4406%`反转为`-0.0069%`。因此失败不是简单的linear head过弱；核心是interaction
specificity不具备论文问题要求的跨split稳定性。A6 validation上interaction相对region也只有2/5为正，说明即使
同split内也不是所有control都稳健。

## 6. Four-layer evidence

### Layer 1 — paper-facing effectiveness

D21不是forecast method，故不比较A6 paper-facing MSE。problem-facing official-test gate完整但失败：两种readout
都没有满足全部preregistered conditions。

### Layer 2 — matched problem attribution

Permuted-history controls通常被真实history击败，说明存在past signal；但additive control解释了几乎全部可学习收益。
multi-horizon-specific non-separability没有获得稳定独立增量。

### Layer 3 — internal health

Oracle headroom、finite outputs、descriptor export和policy fitting均健康。Internal opportunity存在，但learned
interaction没有兑现；不能用oracle替代realized transfer。

### Layer 4 — failure attribution

- `hypothesis_false`：对精确定义的“material且split-stable EVS interaction”，是；
- `intervention_point_wrong`：本次没有paper intervention，不适用；
- `readout_or_head_design_wrong`：不是主要解释，nonlinear readout在validation forward上已有signal；
- `optimization_or_numeric_pathology`：否；
- `capacity_control_explains`：permutation没有解释收益，但更简单的additive hypothesis解释了大部分收益。

Primary attribution：`split_stability_clause_not_supported + simpler_additive_explanation`。Broader representation-level
interaction保持`unresolved`，但没有资格继续占用当前论文slot。

## 7. Paper-story consequence

EVS原本能把两个contributions串成“projective route-validity operator + same-forward credit”。D21表明这条chain的
核心必要性没有跨split成立。若继续实现operator，只会把一个局部、弱、generic sample-adaptation signal堆成复杂
architecture，并面临TimeFuse/TimeRouter/Synapse overlap。

因此：

1. `SC-D21-EVS`关闭为`diagnostic_only_closed_split_unstable`；
2. 不运行seeds2022/2023；
3. 不进入EVS Contribution 1/2 Step4；
4. D14 oracle/crossing保留为historical problem evidence，但不能再自动推出adaptive multi-horizon mechanism；
5. 论文主线回到joint Step2，重新寻找同时具备multi-horizon necessity、test-stable evidence与two-contribution causal
   chain的问题。

## 8. Decision

`close_exact_evs_problem_split_stability_failed_return_step2`。

下一步不是用learned encoder representation救D21，也不是调descriptor/readout。除非新的独立证据先证明interaction
具有跨split稳定性，否则representation-level rescue只属于围绕failed gate继续调试。
