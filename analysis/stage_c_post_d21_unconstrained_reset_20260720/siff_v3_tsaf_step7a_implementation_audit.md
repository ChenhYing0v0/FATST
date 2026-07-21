# SC1-SIFF-v3-TSAF Step7A Local Implementation Audit

## 1. Decision

| Field | Content |
| --- | --- |
| `current_step` | Step7A local implementation complete |
| `problem` | TSAF narrative能否落到production SIFF path，且不偷渡history/requested-H conditioning？ |
| `existence_evidence` | Step4-6 conditional narrative pass；frozen SIFF-v2 fusion calibration evidence |
| `idea` | shared target-coordinate × ordered-log-scale scorer replaces direct history-conditioned policy |
| `theory_check` | allocation sample/channel invariant；arms仍history-dependent；full-domain crop invariant |
| `design` | three TSAF policy modes + production wiring + 24-case local gate |
| `narrative_gate` | inherited conditional pass |
| `effectiveness_gate` | pending；没有train/validation/test result |
| `artifacts` | `siff_v3_tsaf_step7a_local_gate/cases.csv`、`summary.json` |
| `decision` | `tsaf_step7a_local_pass_step7b_prelaunch_next` |

[Fact] `r2026-fsa`环境中26/26 local cases通过。没有加载dataset、checkpoint或test label，没有运行training，也没有
remote process。

## 2. Production tensor path

保留SIFF-v2 arm generator：

1. Encoder输出`hidden [B,C,R]`；
2. SIFF生成`components [B,C,Q=2,D,K]`；
3. ordered scale basis生成`modes [B,C,S=5,D,K]`；
4. scope synthesis生成`arms [B,C,S,T=720]`。

TSAF allocation path为：

1. 既有DCT coordinate buffer：`target_features [T,D]`；
2. normalized log-scale及centered quadratic：`scale_features [S,2]`；
3. 两个linear projections广播相加，经shared GELU/scalar scorer得到`logits [T,S]`；
4. scale-softmax后广播为`weights [B,C,T,S]`；
5. `arms [B,C,S,T] × weights [B,C,T,S] -> full [B,C,T]`；
6. 只在最后输出`forecast [B,H,C]`。

candidate没有`history_projection/policy_hidden/policy_output` direct-policy parameters。history仍通过arms影响forecast，
但不能影响同一$t,s$的allocation。

## 3. Implemented modes and controls

- `target-scale-field`：正确future-coordinate与ordered scale features；
- `target-scale-field-permuted`：只反转allocation的scale-coordinate mapping；
- `target-scale-global`：清零target state，保留learned global scale allocation；
- existing `static-target`继续作为categorical target-only capacity control；
- PCSD base显式拒绝TSAF policy semantics；只有SIFF readout可使用。

三个新mode均可与既有`equal_skill` scope-credit training path组合；未增加或修改loss公式。

## 4. Parameter and initialization contract

TSAF删除direct policy：

$$
R H_h+H_h+(H_h+D)H_p+H_p+H_pS+S,
$$

替换为：

$$
DH_p+2H_p+H_p+H_p+1.
$$

synthetic contract中实际/公式参数均为`13,425`，低于同设置direct SIFF的`13,656`。final scalar scorer为zero
initialization，因此初始allocation严格uniform；这不是从trained parent复制权重，也不声称function-preserving
warm start。

## 5. Local gate

`cases.csv`每列定义：

- `category`：shape/numeric/invariance/structure/parameters/semantics/control/gradient/health/guard/production；
- `case`：唯一测试名称；
- `passed`：是否满足冻结threshold；
- `value`：实际shape、gap、count、gradient norm或mode；
- `threshold`：预期值或不等式。

通过项包括：

- shapes、simplex与strict uniform initialization；
- allocation对history/sample/channel严格不变；
- full-domain prefix gap为0；
- 参数公式精确且少于direct policy；
- target variation、permuted-scale sensitivity与global no-target control；
- SIFF mode、target projection、scale projection、allocation output及history-through-arms gradients均finite/nonzero；
- PCSD guard；
- production `TimeAlign.Model` output `[1,720,2]`及H96 crop exact。

初次checker为20/21，因为deterministic witness把target与scale投影到互不重叠hidden dimensions，softmax消除了
target-only additive term。修正witness使两类坐标进入共享dimensions，并补齐production parameter/hash后为26/26。
该问题属于test witness design，不是production model numeric或gradient failure。

## 6. Code-theory consistency

- [Fact] intended target-scale shared scorer已实现；allocation不读取hidden或requested H；
- [Fact] forecast仍通过SIFF arms读取history，避免退化为history-free predictor；
- [Fact] permuted/no-target controls改变且只改变冻结语义维度；
- [Proxy] `[z,z^2] + MLP`只是一种低容量continuous-scale parameterization，不证明真实scale preference平滑；
- [Uncertainty] zero-output initialization导致首个backward主要激活final scorer，深层allocation projections在随后
  optimizer steps才获得信号；local witness证明非零状态下完整gradient存在，remote smoke仍需审计早期轨迹；
- [Falsification] validation/test若categorical target-only优于TSAF，则ordered scale geometry没有带来增量；若两者均
  低于direct parent，则history-free allocation hypothesis不成立。

## 7. Regression boundary

- frozen SIFF/PCSD algebra-production checker：36/36 pass；
- historical SIFF_EQUAL Step7A checker：model/CLI/parameter/gradient/artifact 11个技术categories全部pass；
- 该历史checker总结果为11/13，因为它硬编码要求2026-07-18 remote/test authorization仍为false，而历史config现已
  记录当日正式授权与执行。失败项仅为`remote_runner_authorization_guard`和`authorization_boundary`，不是本轮
  model regression；本报告不把它写成13/13 pass。
- `ruff`未安装于`r2026-fsa`，故使用`py_compile`、repo-native gates与`git diff --check`作为最小诚实验证。

## 8. Authorization

`local_implementation=true / remote_training=false / official_test=false / confirmation=false`。

下一步只能进入Step7B prelaunch：冻结CLI matrix、reference reuse hashes、new-arm initialization matching、artifact
schema、early-gradient/resource smoke与authorization guard。Step7B通过前不得launch。
