# Stage C D23 FCMI Step7B 代码说明

## 1. 新增生产路径

Step7B新增`fcmi-dense-capacity-matched`，并补齐checkpoint diagnostics、formal analyzer、remote runner与
prelaunch checker。当前外部执行权限仍为false。

## 2. Dense control forward

原有standard-dual path：

```text
memory                    [B,C,P,D]
context                   [B*C,720,D]
standard_dual_state       [B*C,720,D]
standard_output           [B*C,720]
```

dense residual path：

```text
flatten(memory)           [B*C,P*D]
dense_coefficient         [B*C,Rd]
dense_temporal_basis      [720,Rd]
dense_residual            [B*C,720]
output = standard_output + dense_residual
```

`dense_coefficient.weight/bias`与`dense_temporal_bias`zero-init，故initial residual为0；
`dense_temporal_basis`随机初始化，使coefficient在第一步获得非零gradient。coefficient更新后basis在第二步获得
非零gradient。该设计避免把未参与forward的parameter padding误写为capacity control。

## 3. Profile-derived rank

对每个冻结profile，令A6 active count为$N_{A6}$，standard-dual active count为$N_s$，
readout width为$R_h=P D$。dense residual新增参数为：

$$
N_d=(R_h+1+720)R_d+720.
$$

$R_d$取使$|N_s+N_d-N_{A6}|/N_{A6}$最小的整数。最终ranks为
Weather/ETTm1/ETTh1/ETTh2/ETTm2=`234/250/241/234/247`，参数差均小于0.14%。
rank不由任何validation/test metric选择。

## 4. Checkpoint diagnostics

shared evaluator新增FCMI reduced tensors：

- `context_coordinate_std`；
- `main_rms`与`interaction_rms`；
- normalized attention entropy与target dispersion；
- FCMI within-model main-only output及interaction prediction contribution；
- dense residual。

只保存256个probe rows的reduced arrays，不保存完整所有batch attention activation。test evaluator仍校验
checkpoint hash、prefix、finite、training contract、readout contract与test authorization。

## 5. Runner与analyzer

runner冻结40个training jobs：

- 每个run先在validation生成四个standard horizons；
- evaluator对所有runs生成validation invariant和health artifacts；
- 8个arms均进入完整official-test control matrix；
- test前后checkpoint hash必须相同；
- 当前authorization=false时normal launch在任何训练前exit 3。

analyzer检查160个test cells和160个validation cells，并分别输出cell comparisons、summary、run audit、
internal health与machine decision。任何缺失arm/dataset/horizon都会阻止decision。

## 6. Code-theory consistency

- Intended theory：FCMI收益必须同时超过A6、standard/generic/order controls与A6-capacity-matched generic
  trajectory path。
- Code realization：candidate与核心dual controls exact parameter matched；dense control与A6总active parameters
  匹配；所有arms共享objective、selector和profiles。
- Proxy boundary：dense residual使用global trajectory basis，只是capacity explanation control，不是新method或
  Contribution 2。
- Falsification：dense control解释收益时映射`capacity_control_explains`；interaction/order/target controls失败时
  关闭对应机制claim；internal path inactive时不能用aggregate performance promotion。
