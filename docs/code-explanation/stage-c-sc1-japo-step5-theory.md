# SC1-JAPO Step 5 Theory Checker

## Purpose

`scripts/check_stage_c_japo_step5_theory.py`是`diagnostic_only` checker。它不实现可训练JAPO模型，不读取
dataset，也不评估prediction performance；它只把Step 5的代数、autograd、control与statistic contracts落成
可复算artifacts。

## Functional Flow

1. `restricted_global_nested_basis(T, r_g)`复用已验证的RGNB构造，得到`synthesis: [T,T]`与`atoms: [T]`；
2. `atom_descriptors`从每个atom的support生成`descriptors: [T,4]`，字段为center、width、depth与global flag，
   不包含requested $H$；
3. `joint_gate`把`history: [B,C,R]`投影为`[B,C,S]`，把descriptors投影为`[T,F]`，经
   `interaction: [E,S,F]`得到`logits: [B,C,T,E]`，只在expert axis做softmax；
4. containment audit把任意A6 temporal basis左乘$Q^T$，复制为相同expert maps，再验证
   `synthesis @ mixed_coefficients`与A6 output相等；
5. projectivity audit分别用full atoms与`active_indices(atoms,H)`重算gate，验证shared coefficients与prefix
   outputs相等；
6. geometry-only collapse与joint non-collapse分别验证旧shortcut的no-go和JAPO新增function class；
7. continuity、specialization statistics与initialization symmetry使用synthetic tensors和autograd检查定义与
   gradient contract；
8. `main`写出CSV/JSON，并只授权`pass_step6_design_only`。

## High-Risk Logic

### Active-only equality

`joint_gate`不得看active atom set，softmax axis必须固定为`E`。若未来代码把softmax或mean改到atom axis，
requested $H$会改变shared coefficients，projectivity theorem立即失效。

### Containment versus initialization

containment构造通过把所有expert maps设为相同函数证明function-class inclusion。但symmetry audit同时证明，若
训练初始化也这样做，router gradient为0且uniform experts梯度相同。因此Step 6必须使用independent
from-scratch initialization；checker中的复制操作不能被机械移植到model initialization。

### Statistic boundary

`normalized_gate_entropy`、usage、disagreement、history/geometry sensitivity、interaction residual与routing
effect只定义可观测量。synthetic值非零只说明实现可计算，不是learned specialization evidence，也没有“越低/越
均匀越好”的预注册方向。

## Artifact Effects

- `containment_checks.csv`：每个$T$的A6 embedding、permutation与prefix maxima；
- `prefix_projectivity_checks.csv`：每个$H$的active count和两类equality；
- `control_matrix.csv`：Step 6不得删除的same-bank attribution arms；
- `metric_definitions.csv`：每个新统计的source、computation与meaning；
- 五个JSON：no-go、non-collapse、continuity、symmetry与final gate。

## Code-Theory Consistency

- intended theory：JAPO在无explicit $H$、无dense bypass时包含A6，保持exact prefix projectivity，并严格超出
  fixed affine PAF；
- code realization：float64 coordinate transform、active-only recomputation、constructive scalar witness与
  autograd symmetry test；
- remaining proxy：continuity数值扰动只是sanity check，正式保证来自连续函数复合；synthetic specialization
  statistics不代表真实训练；
- falsification：任意containment/projectivity gap超过`1e-10`、joint witness可被fixed affine map表示，或Step 6
  必须引入$H$/atom-axis normalization才能工作，都将否定当前contract。
