# Stage C D23 FCMI Step7A 代码说明

## 1. 实现边界

本次只实现`SC-D23-FCMI-v1`的local production path与matched controls。没有实现remote runner、checkpoint
evaluator、official-test analyzer、H embedding、router或第二loss。

主要文件：

- `baselines/timealign_official/layers/FCMI.py`：FCMI operator与controls；
- `baselines/timealign_official/models/TimeAlign.py`：readout family接入；
- `baselines/timealign_official/train_repo.py`：production CLI、initialization与parameter diagnostics；
- `scripts/check_stage_c_d23_fcmi_step7a.py`：shape/morphism/gradient/parameter/CLI gate。

## 2. Forward tensor flow

TimeAlign encoder先对RevIN-normalized history编码：

```text
x                    [B, 720, C]
memory               [B, C, P, D]
memory_content       [B*C, P, D]
memory_position      [1, P, D]
attended_memory      [B*C, P, D]
target_position      [720, D]
query                [B*C, 720, D]
context S            [B*C, 720, D]
main                 [B*C, 1, D]
interaction Delta    [B*C, 720, D]
state U              [B*C, 720, D]
normalized forecast  [B, 720, C]
forecast             [B, H, C]
```

`target_position`是fixed sinusoidal coordinate，经shared `query_encoder`得到query；它不读取requested H。
cross-attention以query读取`memory_content + memory_position`。

## 3. FCMI与controls

FCMI state为：

$$
U_t=W_{\rm main}\bar S+W_{\rm int}(S_t-\bar S)+E(q_t).
$$

两条branch均使用`bias=False`，否则exact standard morph会重复计算bias。

controls：

- `STANDARD_QUERY`：单个$WS_t$，参数较少；
- `STANDARD_DUAL_MATCHED`：$\frac12(W_1S_t+W_2S_t)$；
- `GENERIC_DUAL_MATCHED`：$\frac12(W_1\bar S+W_2\bar S)$；
- `FCMI_ORDER_SHUFFLED`：先置换memory values，再加固定slot positions；
- `TARGET_SHUFFLED_QUERY`：只用于local query-coordinate sanity。

所有dual arms构造完全相同的modules。`W_int`复制`W_main`初值，所以FCMI与standard dual在初始化时
function-identical；这只是random initialization morph，不是trained capacity preservation。

## 4. Training与CLI contract

FCMI modes属于unified prefix readout：

- future reconstruction/alignment branches关闭；
- production objective可复用`measure_only` harmonic prefix L1；
- full trajectory先生成，再prefix crop；
- CLI冻结`fcmi_n_heads=8`、`fcmi_dropout=0`、permutation seed 20260720；
- five-dataset natural profile继续决定$P,D,d_{ff}$。

Step7A checker通过实际`train_repo.parse_args()`与`build_official_args()`构造35个dataset-arm cases，再用
synthetic tensor执行production model，不读取official test。

## 5. Code-theory consistency

- Intended theory：query context可精确分成trajectory main与zero-mean coordinate interaction，并包含generic和
  standard query functions。
- Code realization：`context.mean(dim=1)`与subtraction实现exact decomposition；无bias dual projections实现
  standard morph；generic composition完全不读取interaction。
- Proxy boundary：fixed sinusoidal query、单层cross-attention和A6 token memory只是v1实现选择，不证明它们是
  唯一或最优实现。
- Falsification：formal E2E结果若不超过dual/generic/order/capacity controls，则分别映射到
  `capacity_control_explains`、coordinate interaction无效或order specificity不足；local gate不能替代这些结果。

## 6. Local verification

最终Step7A为11/11 pass：

- 35个production CLI cases；
- five-profile initialization与parameter matching；
- shape与zero-mean invariant；
- standard exact morph；
- generic interaction exclusion；
- main/interaction/query/output gradients；
- order/target shuffle sanity。

A6与FCMI active parameter gap为83%–95%，因此未来formal matrix必须增加dense capacity-matched control。
