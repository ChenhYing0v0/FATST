# StageC PCSD / PCC / SIFF 公平重评估预注册

## 1. 当前步骤与目的

- `current_step`: `SC-RETRO-FAIR-v1 Step7A local gate passed；Step7B remote next`
- `role`: `paper-facing retrospective effectiveness audit`
- `CTD`: 按用户决定暂停；`SC-D16-CTD`设计保留但不继续实现
- `test_informed`: `true`

本次不是提出新方法，而是把A6、PCSD、PCC与SIFF放回同一训练、checkpoint与test scorecard下，
回答过去的negative/positive结果有多少能在当前论文规则下成立。旧runs只按validation H720选择checkpoint，
因此不能与新规则直接混入主比较；所有70个arms均从头E2E重训。

## 2. 为什么修改split职责

[Fact] 项目内部已出现validation/test排序反转：PCSD-CF DIRECT相对dense control在validation为正、在test为负。
因此validation gain不足以代表当前chronological test distribution上的论文性能。

[Decision] validation只负责early stopping、checkpoint选择、普通超参数选择、debugging与解释性diagnostic；
正式机制有效性、main result和formal ablation统一由official test的
$H\in\{96,192,336,720\}$ MSE/MAE决定。

[Risk] 反复根据test结果提出后续candidate会使test成为adaptive benchmark，而不是untouched holdout。
Dwork等人的adaptive data analysis结果明确指出，重复、自适应使用holdout可能对holdout本身过拟合；
Wild-Time也说明temporal distribution shift会显著改变out-of-distribution表现。因此本项目不再声称official
test完全untouched，而是明确标记后续candidate为`test_informed`，并用完整冻结矩阵、negative-result reporting、
matched controls与multi-seed confirmation降低风险。

外部检索记录：

- 检索日期：2026-07-17；
- 范围：adaptive holdout reuse、temporal distribution shift、time-series benchmark split；
- 来源：外部primary sources；Zotero未作为完整性判断依据；
- 关键来源：[Dwork et al., NeurIPS 2015](https://papers.nips.cc/paper_files/paper/2015/hash/bad5f33780c42f2588878a9d07405083-Abstract.html)；
  [Wild-Time, NeurIPS 2022](https://openreview.net/forum?id=F9ENmZABB0)。

## 3. Checkpoint规则

每个epoch在validation上计算：

$$
S_{\mathrm{val,std}}
=\frac14\left(
L_{\mathrm{val}}(96)+
L_{\mathrm{val}}(192)+
L_{\mathrm{val}}(336)+
L_{\mathrm{val}}(720)
\right).
$$

选择$S_{\mathrm{val,std}}$最小的checkpoint。四个requested horizons作为四个任务等权进入selector；
不再只选H720。test label不参与epoch或checkpoint选择。所有arms共享epochs=20、patience=5、batch size=32、
learning rate=$10^{-4}$与对应dataset的frozen natural profile。

## 4. 公平矩阵

五个数据集为Weather、ETTm1、ETTm2、ETTh1、ETTh2；Phase A固定seed2021。14个arms为：

1. `a6_full`；
2. `pcsd_direct`；
3. `dense_measure`；
4. `pcsd_measure`；
5. `pcsd_equal`；
6. `pcsd_prior`；
7. `pcsd_pcc`；
8. `siff_equal`；
9. `siff_prior`；
10. `siff_pcc`；
11. `siff_constant_pcc`；
12. `siff_permuted_pcc`；
13. `siff_q1_wide_pcc`；
14. `siff_independent_pcc`。

总计$5\times14=70$次from-scratch joint Encoder–Decoder训练，test产生
$70\times4=280$个standard-horizon cells。`A6→PCSD`使用相同plain full-T L1；
`PCC`必须同时超过equal与prior controls；`SIFF`在equal/prior/PCC三个objective下与PCSD匹配，
并在PCC下比较constant、permuted、Q1-wide与independent controls。

## 5. 指标与gate

对candidate $A$和reference $B$：

$$
G(A,B;d,H)=100\left(1-\frac{\mathrm{MSE}_A(d,H)}
{\mathrm{MSE}_B(d,H)}\right).
$$

正值表示candidate更好。每个primary effect需要同时满足：

- 20 cells equal-weight macro gain不低于0.3%；
- 至少3/5 dataset wins；
- 至少3/4 horizon wins；
- 至少11/20 cell wins。

Phase A正向只代表需要补seed2022/2023 confirmation，不直接形成paper claim。负向结果保留为该exact
mechanism在公平协议下的失败证据。

## 6. Step7A本地gate

`scripts/check_stage_c_fair_reaudit_v1.py`已完成：

- 70/70 CLI contracts；
- 40个unique dataset/readout/rank model constructions；
- 全部full-prefix identity误差低于$2\times10^{-5}$；
- 每个dataset的14 arms具有相同Encoder initialization hash；
- profile hash、comparison references、checkpoint rule与test authorization全部通过。

machine-readable结果：
`analysis/stage_c_fair_reaudit_v1_20260717/prelaunch/local_gate.json`。

## 7. 失败归因与决策边界

本次结果可以判断exact PCSD/PCC/SIFF在统一paper-facing协议下是否有效，但不能自动否定更大的
coupling-spectrum、credit assignment或scale-coordinate研究方向。若出现数值病态、checkpoint hash变化、
matrix缺失或protocol mismatch，结论为artifact/protocol invalid；只允许修复相同version，不允许挑选正向cells。

`next_action`: commit/push后检查3090 GPU，执行resource smoke，再启动70-run dataset-major remote matrix。
