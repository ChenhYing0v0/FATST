# ISCF-v1-CPSI Step7B Tooling 说明

## 1. Runner

`scripts/remote/run_stage_c_iscf_v1_cpsi.sh`从Step7B JSON生成25个new-training jobs。每行包含dataset、arm、
`readout_mode`、direct policy、equal-skill objective、dataset-matched ISCF rank与natural profile。训练固定输入
`[B,720,C]`，Encoder输出经CPSI/controls生成`[B,720,C]`，validation selector只读取H96/192/336/720。

runner把training与formal test拆成两个互斥阶段。普通模式只写checkpoint、training log、validation metrics与
contracts；`FORMAL_TEST_ONLY=1`先逐run检查六类training artifacts，并在25/25齐全后调用checkpoint evaluator。
evaluation前后计算checkpoint SHA256，任何变化都会使job失败。

resource smoke日志扫描先检测`rg`是否存在；精简remote image无`rg`时改用`grep -Ein`。两条路径共用同一
Traceback/OOM/NaN/Inf pattern，避免negated missing command产生false pass。

## 2. CPSI evaluator diagnostics

`cpsi_tensors()`先执行`encode_history(batch_x)`，得到memory并flatten为SIFF/ISCF hidden，再调用readout的
`interaction_diagnostics(hidden)`。pre-synthesis arms内部为`[B,C,S,D,K]`，展平interaction width为
$L=D K$；POST内部为`[B,C,S,T]`。evaluator不保存完整高维tensor，而对最后两个维度计算RMS，得到每个
`[B,C]` row的common/private/left/right/latent/message scalar，再只保留固定前256 rows。

readout contract要求CPSI parent/input initialization hashes存在、interaction rank为32、interaction params为正且
六项probe均落盘。该分支先于generic `pcsd_readout` contract，避免把CPSI误判为普通PCSD。

## 3. Analyzer

`scripts/analyze_stage_c_iscf_v1_cpsi.py`把25个new runs与10个historical references映射到统一
`dataset × arm × horizon` table。每个comparison的gain定义为
$100(1-\mathrm{candidate}/\mathrm{reference})$，分别汇总macro gain、cell wins、dataset wins与最大dataset退化。

decision先检查35-run protocol completeness，再独立计算：paper-facing effectiveness、matched control attribution、
internal health和failure attribution。controls不会在validation阶段删除candidate；只有完整test table后才影响机制
claim。synthetic smoke构造CPSI严格优于其他arms的矩阵，验证240 comparison cells和positive decision path。

NPZ同时包含numeric diagnostics与string `bin_names` metadata。finite audit只对numeric dtype执行`np.isfinite`，
string metadata仍保留但不进入numeric health；synthetic smoke包含mixed-dtype regression，防止再次误判。

`arm_reference_context.csv`对每个new arm分别在validation/test上比较ISCF-v0与A6_FULL。每行的
`macro_gain_percent`是20个dataset-horizon cells的$100(1-candidate/reference)$算术平均；`cell_wins`统计正值cell，
`dataset_wins`先在dataset内平均四个horizons再统计正值，`max_dataset_degradation_percent`取最差dataset均值的负部。
这些context rows不改变预注册CPSI gate，只用于解释controls是否揭示替代路径或validation-to-test reversal。

## 4. Code-theory consistency

理论要求是“test-first effectiveness + diagnostic-only controls”，代码通过全量manifest与test前25/25 completeness
check实现。CPSI health tensors只证明interaction path活跃，不能证明收益属于common-private mechanism；最终仍需要
CPSI相对ISCF的test MSE/MAE和四个matched controls。若message为零或nonfinite，只能判定exact run/design无效；若
性能正但control tie，则判定attribution blocked，不把性能证据删除。
