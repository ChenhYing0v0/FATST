# SC-D23-FCMI Step7A Local Implementation Audit

## 1. Decision

`SC-D23-FCMI-v1-step7a`完成production readout、matched controls、CLI接线与local synthetic gate。
machine decision为：

```text
step7a_local_pass_step7b_design_freeze_next
```

11/11 gates通过。该结果只证明实现与Step4-6合同一致，不构成method effectiveness、paper-core promotion、
remote training或official-test授权。

## 2. 实现范围

新增六个FCMI family readouts：

| Arm | Production mode | Local role |
| --- | --- | --- |
| `STANDARD_QUERY` | `fcmi-standard-query` | lower-parameter source-family control |
| `STANDARD_DUAL_MATCHED` | `fcmi-standard-dual-matched` | same-parameter multi-branch control |
| `GENERIC_DUAL_MATCHED` | `fcmi-generic-dual-matched` | generic contained control |
| `FCMI` | `fcmi` | candidate |
| `FCMI_ORDER_SHUFFLED` | `fcmi-order-shuffled` | value-position binding control |
| `TARGET_SHUFFLED_QUERY` | `fcmi-target-shuffled-query` | local sanity only |

所有dual arms实例化相同的query MLP、cross-attention、main/interaction projections与shared output。
`A6_MEASURE`继续作为strong carrier/objective control。当前没有增加H embedding、router、第二loss或remote runner。

## 3. Tensor contract

ETTh2 natural profile的synthetic audit使用$B=2,C=3,P=12,D=64,T=720$：

```text
memory              [2, 3, 12, 64]
query               [6, 720, 64]
context S           [6, 720, 64]
main mean(S)         [6, 1, 64]
interaction Delta   [6, 720, 64]
state               [6, 720, 64]
output              [2, 720, 3]
```

readout总是先生成$T=720$，再按`target_prefix`截取；requested horizon不进入query或operator。

## 4. Morphism与identifiability

`main_projection`与`interaction_projection`必须是无bias线性映射。若使用bias，
$W\bar S+W\Delta$会重复加入bias，不能精确恢复$WS$。最终实现显式设置`bias=False`，并在初始化时将
`interaction_projection`复制为`main_projection`。

local结果：

- $\max|\operatorname{mean}_t\Delta_t|=1.8196\times10^{-7}$；
- FCMI与`STANDARD_DUAL_MATCHED` initial output最大差
  $6.3330\times10^{-8}<10^{-6}$；
- generic control对任意zero-mean context perturbation的state最大差
  $1.1921\times10^{-7}$，且`interaction_used=false`。

首次checker执行为9/11：checker未切换`eval()`使encoder dropout masks不同，同时带bias的branch违反exact
morphism。两项均属于`design_fault_suspected`，不是hypothesis failure；修正理论合同和evaluation mode后不放宽
阈值，11/11通过。

## 5. Gradient与order audit

synthetic scalar loss下四条关键gradient norm均finite/nonzero：

| Path | Gradient norm |
| --- | ---: |
| main | 0.0384723 |
| interaction | 0.000122619 |
| query | 0.0211355 |
| output | 0.0561015 |

order control只将memory content置换到固定positional slots：

- sorted value marginal最大差：0；
- attended value-position binding最大差：4.18426；
- output最大差：0.00261251；
- target-shuffle output最大差：0.126933。

因此shuffle没有改变参数量或memory value marginal，但确实改变binding和forecast。

## 6. Parameter audit

五个natural profiles中，所有dual arms的active/decoder parameter counts严格相等。FCMI相对A6 active
parameters的差异却很大：

| Dataset | A6 active | FCMI active | Relative gap |
| --- | ---: | ---: | ---: |
| Weather | 419216 | 70529 | 83.1760% |
| ETTm1 | 391408 | 17921 | 95.4214% |
| ETTh1 | 613904 | 68609 | 88.8241% |
| ETTh2 | 419216 | 70529 | 83.1760% |
| ETTm2 | 1006160 | 67649 | 93.2765% |

这不是local gate failure：FCMI与核心attribution controls已经exact matched。但若下一步冻结formal matrix，
FCMI与A6之间的比较存在显著capacity difference，必须预注册`DENSE_DUAL_MATCHED`或等价capacity control；
不得把FCMI对A6的正负差异直接归因于main–interaction decomposition。

## 7. Failure attribution与下一步

- `hypothesis_false`：未触发；尚无E2E effectiveness结果。
- `intervention_point_wrong`：未观察到local证据；memory到future state路径完整。
- `readout_or_head_design_wrong`：首轮bias问题已修复，最终gate通过。
- `optimization_or_numeric_pathology`：local forward/backward未出现non-finite；remote未运行。
- `capacity_control_explains`：尚未检验；A6–FCMI大参数差使未来dense control成为mandatory。

下一步只能进入Step7B design/prelaunch freeze：明确formal arms、dense capacity control、validation/test角色、
四层effectiveness gates与rollback。remote/test仍为false，Contribution 2继续open。

## 8. Artifacts

- machine gate：`d23_fcmi_step7a_gate.json`
- frozen config：`configs/stage_c_d23_fcmi_step7a.json`
- checker：`scripts/check_stage_c_d23_fcmi_step7a.py`
- production layer：`baselines/timealign_official/layers/FCMI.py`
