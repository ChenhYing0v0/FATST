# SC-D22-HFA D22-C：target-coordinate raw-history access prelaunch audit

## 1. 11-step record

| Field | Current Record |
| --- | --- |
| `current_step` | Step 2/3 problem existence diagnostic；static/prelaunch complete |
| `problem` | fixed past与same information set下，不同future coordinates是否需要对raw history作不同evidence retrieval？ |
| `existence_evidence` | D22-B finite-capacity frontier不支持；D14 dual-carrier crossing/oracle仅提供条件性headroom |
| `idea` | 用完全同参数的六臂neutral raw-history模型，只改变coordinate-specific retrieval是否成立 |
| `theory_check` | 不输入requested H；coordinate query只组织finite computation，不改变Bayes information set |
| `design` | five datasets × six arms × seed2021；validation选checkpoint；official test一次性完整审计 |
| `narrative_gate` | diagnostic-only pass；query/cross-attention primitive prior-covered，不能作为novelty |
| `effectiveness_gate` | not applicable；本轮只做problem gate |
| `artifacts` | config、runner、analyzer、remote/sync scripts；local synthetic smoke |
| `decision` | `d22c_prelaunch_pass_remote_test_problem_gate_authorized` |

## 2. 为什么该问题仍不同于requested-H adaptation

对fixed history $X=x$、coordinate $t$和pointwise MSE，Bayes action仍为

$$
f_t^*(x)=\mathbb{E}[Y_t\mid X=x].
$$

D22-C没有把requested horizon $H$加入condition。它检验的是finite model中，计算
$f_t(x)$时是否需要由$t$选择不同history evidence。若ordered arm胜出，得到的是
`target-coordinate-specific retrieval necessity`，不是“Bayes mean依赖H”。

反方解释同样明确：若global/pooled/generic control已达到同等效果，则coordinate-specific retrieval不是该
benchmark边界下的必要计算结构；即使attention正常、entropy非零也不能救回hypothesis。

## 3. 2026-07-20 primary-source/code audit

| Source | Primary evidence | 对D22-C的约束 |
| --- | --- | --- |
| [TimePerceiver paper](https://arxiv.org/abs/2512.22550) | target-timestamp queries读取input latents；任务包含任意input/target segments | target query不是新primitive；必须证明fixed-past LTSF中的task-specific necessity |
| [TimePerceiver official code](https://github.com/efficient-learning-lab/TimePerceiver/blob/main/models/TimePerceiver.py) | patch positional embedding、target-position query与query-to-input cross-attention共同形成decoder | D22-C采用source-informed tensor contract，但不复制upstream module或claim |
| [TimePerceiver official repository](https://github.com/efficient-learning-lab/TimePerceiver) | cross-attention分析显示query对resolution/periodicity相关input regions形成对齐 | attention pattern只作internal health；不能替代matched performance control |
| [MQTransformer](https://arxiv.org/abs/2009.14799) | forecast context驱动history attention alignment | context-dependent retrieval已有直接prior；generic matched control必需 |

检索范围：`target-timestamp query`、`context-dependent history attention`、official implementation；source type为
arXiv、NeurIPS official repository与official code。Zotero覆盖不作为freshness或novelty证据。本轮没有导入
`R_2026_FSA`代码、配置或artifact。

## 4. Frozen tensor contract

每个window/channel独立形成一行：

- raw history：$x\in\mathbb{R}^{720}$；
- per-window RevIN：$\tilde{x}\in\mathbb{R}^{720}$；
- non-overlap patches：`[N,24,30]`；
- shared patch encoder：`[N,24,30] -> [N,24,32]`；
- fixed sinusoidal target coordinates：`[720,32]`；
- shared four-head cross-attention：query `[N,720,32]`，memory `[N,S,32]`；
- shared fusion：`[query, context] -> [N,720,32]`；
- shared scalar projection：`[N,720,32] -> [N,720]`。

所有六臂实例化同一个`NeutralTargetAccess`，共享相同parameter names、shapes与matched seed2021 initial state。
arm只改变forward中的memory/query construction；fixed sinusoidal buffers和permutation banks不是trainable
parameters。

## 5. Arms与可归因差异

| Arm | Memory access | Isolated question |
| --- | --- | --- |
| `GLOBAL_COMPRESSED` | raw patches先平均，再编码为一个token | fixed-dimensional global summary是否足够 |
| `POOLED_MEMORY` | 各patch编码并加position后平均为一个token | patch nonlinear statistics是否足够 |
| `ORDERED_TARGET_ACCESS` | canonical target query读取ordered patch tokens | primary intervention |
| `ORDER_SHUFFLED` | 每个stable window/channel id选择固定随机patch permutation，再绑定canonical positions | ordered value-position relation是否必要 |
| `TARGET_SHUFFLED_QUERY` | 每个stable row使用不同固定target permutation；labels保持canonical | coordinate identity是否必要 |
| `GENERIC_MATCHED` | 所有coordinates用同一个query读取ordered memory；canonical coordinate只进入后续readout | retrieval必须由coordinate控制，还是generic memory read + coordinate head已足够 |

`ORDER_SHUFFLED`和`TARGET_SHUFFLED_QUERY`的permutation由source window id与channel id确定，在train/validation/test
内可复现且不读取labels。它们不跨split共享样本，也不根据结果选择permutation。

## 6. Split、selector与test角色

- train：每个dataset最多4,096个evenly spaced windows，拟合六臂；
- validation：最多1,024个evenly spaced windows，只以
  `{96,192,336,720}`四个prefix MSE平均选择checkpoint；
- official test：完整windows、五datasets、六arms、20个dataset-horizon cells；一次性`test_informed`
  problem gate；
- 同一dataset的六臂共享window selection、batch order、initial state、optimizer、epoch budget与selector；
- 不做dataset/horizon/cell tuning，不以test选择epoch；
- seed2021是预注册preliminary problem gate，不支持post-result seed rescue。若positive，后续method仍需独立
  E2E multi-seed formal evaluation。

## 7. Statistics与gate

每个control的cell gain定义为

$$
g_{d,h}^{(c)}
=
\frac{\operatorname{MSE}_{d,h}^{(c)}
-\operatorname{MSE}_{d,h}^{(\text{ordered})}}
{\operatorname{MSE}_{d,h}^{(c)}}.
$$

同时报告MAE gain、dataset/horizon macro、五个coordinate bins、attention entropy、target dispersion、
prediction coordinate dispersion、checkpoint hash与parameter audit。

冻结gate：

1. ordered相对每个control的20-cell macro MSE至少`+0.3%`；
2. 相对`GLOBAL_COMPRESSED`与`GENERIC_MATCHED`至少`+0.5%`；
3. 两个key controls各至少11/20 cells、3/5 datasets、3/4 horizons正向，MAE macro非负；
4. key comparisons的validation/test macro同号；
5. trainable parameter gap不超过1%；不得出现non-finite或单cell超过100%退化；
6. 必须同时超过五个controls才返回Step4。

## 8. Static/data-leakage/matched-capacity audit

- [Pass] config可JSON解析，runner/analyzer可`py_compile`；
- [Pass] synthetic smoke完成六臂forward/backward、validation checkpoint、test evaluation与artifact写出；
- [Pass] smoke六臂parameter count均为777，relative gap为0；
- [Pass] static tensor gate确认六臂输出均为`[N,T]`，ordered与五个controls在随机输入上均非恒等；
- [Pass] analyzer对完整synthetic matrix生成cell/bin/aggregate/health/decision artifacts；
- [Pass] test labels不进入optimizer、checkpoint selection、permutation或hyperparameter选择；
- [Pass] model不输入requested H、future labels、time marks或A6 representation；
- [Pass] A6 sensitivity与paper method implementation仍未授权；
- [Pass] remote runner先记录commit、GPU、environment、split role与output path。

## 9. Failure attribution与scope decision

- 若gate通过：只建立problem evidence，返回Step4设计`lead-time-conditioned evidence operator`；不把
  `ORDERED_TARGET_ACCESS`直接升级为method。
- 若protocol有效但不超过shuffles：`hypothesis_false_exact_target_order_access_protocol`。
- 若超过shuffles但不超过`GENERIC_MATCHED`：
  `capacity_control_explains_or_intervention_point_wrong`。
- 若出现non-finite、matrix缺失、>100%退化或parameter mismatch：
  `diagnostic_invalid_for_direction_rejection`，只允许修复一次protocol。

用户在2026-07-20明确要求当前项目不因一次negative result转出
`deterministic-MSE fixed-past architecture search`。因此该项目级scope决定覆盖上一版“D22-C有效失败即停止整个
search”的默认rollback：有效失败只关闭D22-C exact protocol，并回joint Step2/3寻找不同的falsifiable problem；
不得恢复D17-D21、做D22-C seed/readout/width rescue，亦不得预设第二loss/router。

## 10. Authorization

`diagnostic_implementation_authorized=true`；
`remote_training_authorized=true_after_commit_push_and_gpu_preflight`；
`official_test_access_authorized=true_for_frozen_complete_matrix`；
`paper_method_implementation_authorized=false`；
`a6_sensitivity_authorized=false`；
`contribution_2_design_authorized=false`。
