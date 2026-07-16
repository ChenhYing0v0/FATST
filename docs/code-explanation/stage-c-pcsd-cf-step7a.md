# StageC PCSD-CF Step 7A 代码说明

## 1. Scope And Status

本版本把`SC1-PCSD-CF-v1`接入A6-natural Encoder后的active readout path，并实现D15-A Step7A local gate。
它包含五个共享parameter-field scope arms与`direct/equal/static-target/fixed`四类policy control，但不包含
remote runner、dense capacity control、SC2 loss或test访问。

Step7A通过只说明代码符合冻结的shape、projectivity、A6 containment、sharing topology与数值可微契约；不说明
PCSD-CF优于A6，也不授权paper effectiveness claim。

## 2. Forward Computation Flow

### 2.1 A6-natural history carrier

`TimeAlign.Model`保留dataset-aware A6-natural Encoder：

```text
x [B,720,C]
  -> RevIN normalization
  -> PatchEmbed + 2-layer token MLP
  -> memory [B,C,P,De]
  -> flatten(P,De)
  -> z [B,C,R], R=P*De
```

五个profile的$R$分别是ETTh1 `1536`、ETTh2 `768`、ETTm1 `768`、ETTm2 `3072`、Weather `768`。
`flatten`只reshape，不做pooling；PCSD-CF没有冻结或替换Encoder，后续训练必须from-scratch end-to-end。

### 2.2 Fixed target coordinate field

`PCSDCouplingFieldReadout.coordinate_field`保存`Q [T=720,Dq=4]`：第0维恒为1，其余三维是显式去均值的
low-order DCT coordinates。对每个$s\in\{1,48,144,360,720\}$，fixed `group_indices_s [G_s,s]`
定义canonical或random partition，并预计算：

$$
\bar Q^{(s)}_g=\frac1s\sum_{\tau\in g}Q_\tau.
$$

这些tensor均为buffers，不是trainable parameters。canonical/random构造使用相同parameter initialization；只替换
group index与pooled-coordinate buffers。

### 2.3 Shared history-to-future mode field

`history_modes`执行：

```text
z [B,C,R]
  x mode_weight [Dq,R,K=256]
  + mode_bias [Dq,K]
  -> Z [B,C,Dq,K]
```

五个arms不各自保存`Linear(R,K)`。`_scope_forecast`用相应$\bar Q^{(s)}$对同一个$Z$做contract：

Step7B prelaunch audit纠正了首版初始化的tensor方向：`mode_weight [Dq,R,K]`不能reshape成`[Dq*R,K]`
再使用Kaiming，否则fan-in会被误判为$K$。当前代码直接采用与`Linear(R,K)`一致的
$\mathcal U[-R^{-1/2},R^{-1/2}]$，`mode_bias`使用同一bound。该修正不改变operator、containment或projectivity，
但避免跨dataset state width的激活尺度错误。

```text
Z [B,C,Dq,K] x pooled_coordinates_s [G_s,Dq]
  -> group state A_s [B,C,G_s,K]
```

因此$s=1$有720个target-specific state，$s=48$有15个group states，$s=720$只有1个global state；变化的是
future-output state sharing topology，不是五套decoder parameters。

### 2.4 Shared target synthesis

每个group state同时经过identity与GELU feature lift，并只与属于该group的共享target rows相乘：

```text
A_s [B,C,G_s,K]
  x identity_synthesis[target] [G_s,s,K]
  + GELU(A_s) x nonlinear_synthesis[target] [G_s,s,K]
  + temporal_bias[target]
  -> arm_s [B,C,T]

stack five scopes -> arms [B,C,5,T]
```

`identity_synthesis/nonlinear_synthesis [T,K]`和`temporal_bias [T]`跨所有scopes共享。为避免$s=1$显式物化
Weather batch上的`[B,C,720,256]`大激活，实现按最多64个groups分块，再用fixed indices一次scatter回时间顺序。
该chunking与完整einsum数学等价，不改变parameter field或输出。

### 2.5 Direct policy and projective output

`direct` policy执行：

```text
z [B,C,R] -> history_projection [B,C,32]
Q_tau [T,4]
concat per target -> GELU Linear(...,64) -> Linear(64,5)
softmax(scope) -> weights [B,C,T,5]

arms [B,C,5,T] x weights -> full forecast [B,C,T]
crop [:H] -> [B,H,C] -> RevIN denormalization
```

最后一层policy logits的weight/bias全零初始化，所以初始weights严格为`0.2`。这会使上游policy projection第一步
gradient为零；第一步更新`policy_output`后，第二步history/target policy path获得非零gradient。Step7A因此使用
two-step gradient audit，而不是把预期的first-step zero gradient误判为dead path。

`target_prefix`在arms、policy与full fusion完成后才crop。requested $H$不进入coordinate、pooling、mode maps、
synthesis或policy，因而任意native horizon都严格等于同一full-H720 output的prefix。

## 3. Implemented Controls

- `direct`：history projection与natural target coordinate共同决定per-target scope weights；
- `static-target`：将history projection置零，只保留target-coordinate policy；
- `equal`：固定五个scopes各`0.2`，router parameters不进入active forward；
- `fixed`：固定选择`1/48/144/360/720`之一，其他scope仍由同一field定义；
- `partition=random`：只随机重排intermediate scope groups，$s=1/T$端点不作无意义重排。

`map_a6_parameters_`是Step7A/未来M0 control使用的constructive witness，不是warm-start训练接口。它将任意A6
coefficient map写入constant mode、将A6 basis写入identity synthesis，并把nonconstant modes与nonlinear
synthesis置零；这时五个arms完全相同且等于A6。

Step7B production controls另增加：

- `PCSDM0Readout`：按与A6完全相同的parameter creation/init顺序实现`Linear(R,256) × basis [720,256]`；
  相同seed下A6/M0 operator hash与初始输出严格一致，用于排除runner或morphism差异；
- `PCSDDenseMatchedReadout`：`Linear(R,W) -> GELU -> Linear(W,720)`，按每个profile的PCSD总decoder参数自动
  选择最接近的整数$W$；五profile gap均低于`0.1%`；
- fixed policy的training forward只计算被选scope；`forward_with_diagnostics`仍计算全部arms。二者输出严格一致，
  避免五个fixed controls无意义支付five-arm FLOPs。

## 4. Training Adapter And Artifacts

`train_repo.py`已注册`pcsd-coupling-field`为active prefix readout，并把coordinate/rank/policy/partition/chunk
参数写入effective config。`initialization_contract`记录Encoder、PCSD parameters、coordinate与partition hashes，
以及initial policy entropy/usage；`model_diagnostics`区分coupling-field、policy、active-forward和legacy unused
`proj_x` parameters。

`scripts/check_stage_c_pcsd_cf_step7a.py`生成：

- `shape_prefix_checks.csv`与`model_integration_checks.csv`：direct readout及真实A6-natural forward shapes；
- `containment_checks.csv`：三个unique state widths的float32/float64 arbitrary-A6 mapping error；
- `topology_checks.csv`：target group-state对history modes的Jacobian-sharing classes；
- `separation_checks.csv`与`partition_checks.csv`：arm disagreement、equal initialization及partition-only变化；
- `gradient_checks.csv`：五profile module two-step backward与ETTh2真实Encoder-PCSD E2E backward；
- `accounting.csv`：decoder parameter/DoF、static multiply-add FLOP与chunk activation估算；
- `protocol_contract_checks.csv`：CLI propagation与remote/test/SC2/requested-H exclusion；
- `step7a_local_gate.json`与`step7a_local_gate_report.md`：machine-readable gate和research decision。

Step7B新增：

- `check_stage_c_pcsd_cf_step7b.py`：60个dataset-arm contracts、A6/M0 exact pairing、PCSD pairing、fan-in
  initialization、fixed fast path、dense parameter matching与validation-only protocol；
- `evaluate_stage_c_pcsd_cf_checkpoint.py`：按sequential validation rows保存fused/arm/persistence bin losses、
  per-bin policy usage与probe predictions，并复核trained prefix/config/checkpoint invariants；
- `analyze_stage_c_pcsd_cf_step7b.py`：冻结dense-H1..720 MSE AUC comparison、same-run arm/oracle/policy statistics、
  capacity/random specificity gates与failure attribution；
- `remote/run_stage_c_pcsd_cf_step7b.sh`：60-job resumable/status/dry-run/resource-smoke runner，输出根目录固定在
  repo外`/home/yingch/exp_outputs/r-2026-fatst/stage_c_pcsd_cf_step7b`。

## 5. Code-Theory Consistency Evaluation

| Intended theory | Code realization | Remaining proxy / limit |
| --- | --- | --- |
| one shared coupling field | 单一`mode_weight/mode_bias`供五个pooling operators复用 | field rank固定256，尚未验证trained arm skill |
| scope改变output sharing | `P_sQ`产生720/15/5/2/1 group-state Jacobian classes | fixed scales尚未证明是最佳离散化 |
| exact A6 subspace | constant mode + identity synthesis constructive mapping | containment不等于from-scratch optimizer会找到该subspace |
| nonlinear scope differentiation | shared identity+GELU synthesis | 增益可能仅来自约3.10-3.72x总decoder params，必须有dense control |
| sample/target adaptation | history × natural coordinate direct policy | equal-zero起点可能有optimization lag，需Step7B监测collapse |
| exact projectivity | full-T arms/policy/fusion后只执行prefix crop | 当前T固定720，不claim跨maximum-domain extrapolation |
| memory-bounded execution | group/target chunking保持同一einsum contract | CSV是static estimate，remote前仍需GPU profiler/smoke |

可证伪边界：local invariant失效回滚Step5/6；M0有效但所有fixed arms无skill属于
`readout_or_head_design_wrong`并只否定v1；收益被dense/random解释属于`capacity_control_explains`；只有matched
E2E Step7B通过A6/equal/static/dense/random gates，PCSD-CF才可升级为paper-core effectiveness candidate。
