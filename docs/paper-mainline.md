# Paper Mainline

## Current Position

| Field | Content |
| --- | --- |
| `paper_target` | 高水平 SCI 期刊时间序列预测论文 |
| `working_title` | Beyond a Fixed Forecasting Strategy: Coupling-Adaptive Decoding for Unified Multi-Horizon Forecasting |
| `current_stage` | `StageC-UVHF` active；StageB 已归档 |
| `current_11_step` | PCSD-CF Step7B 60-run prelaunch gate passed；seed2021 validation-only remote screen authorized |
| `source_evidence` | A6-LBF-r256 historical/source-faithful performance |
| `mechanism_control` | same-run end-to-end A6；frozen A6仅作reference/conditional diagnostic |
| `test_reference` | 3 datasets × 3 seeds × 8 horizons，72/72 complete |
| `future_validation_suite` | ETTh1/ETTh2/ETTm1/ETTm2/Weather；five natural profiles frozen |
| `active_ledger` | `docs/stage-ledgers/stage-c-unified-forecasting-redesign.md` |
| `paper_core_status` | PCSD-CF narrative/local-contract ready但effectiveness-unready；CCRL retired；Contribution 2 problem unverified；test held |

## Research Thesis

论文研究对象固定为`fixed-past unified multi-horizon generation`，但核心问题不再是Encoder–Decoder接口：

> 给定同一段fixed past，一个共享模型不仅要生成任意requested horizon的exact future prefix，还应决定不同
> future targets以多大范围共享predictive structure。现有Direct、AR、MIMO、DIRMO与future-query decoders
> 通常固定一种output coupling granularity，或在模型外按dataset/horizon选择strategy；这并不是真正统一的
> forecasting strategy。

A6已经满足one-model、domain-only crop与exact prefix equality；它的`global_coeff [B,C,256] × basis[:H]`
属于global low-rank/MIMO-like coupling endpoint。CATS-like independent future queries代表point/direct endpoint，
DIRMO则以固定block size位于两者之间。新主线研究一个multi-horizon任务独有的问题：point、block与global
sharing scope是否应在同一projective decoder中共存并由数据选择。

[Theory Boundary] 对deterministic separable MSE，显式future dependency不是Bayes point predictor的必要条件。
本论文只claim有限样本、有限capacity下parameter sharing的bias–variance–flexibility trade-off，不能把
probabilistic trajectory dependency直接写成point-MSE必然收益。

requested horizon 在当前主线中只定义输出域与计算域，不作为 learned semantic feature。禁止将离散
horizon ID、benchmark-specific embedding、per-horizon expert 或 per-horizon hyperparameter 作为核心机制。

forecast-revision surface已转移到根目录`New-idea.md`，状态`deferred_next_paper`；它不再是当前论文问题。

## Contribution Slots

[Decision] `CADMO/CPGA`因问题范围只落在history-interface而退出active slots；它们不是实验方向级失败，而是
`rejected_by_narrative_scope`。ordered patch memory只保留为`D14-P auxiliary_interface_probe`，不决定论文主线。

两个active slots曾由`PCSD/CCRL` provisional占用；2026-07-16 training-consistency audit已将CCRL降为
`diagnostic_only_not_scheduled`，第二slot重新开放。D14-A0的neutral linear RRR gate已返回：stable crossing
0/5、sample × bin oracle macro 0.0586%、canonical-vs-random -0.1427%。但其factor params并不等于
rank-manifold effective DoF，且scale risks最多只相差0.04036%，所以方向级拒绝无效。A1随后以matched grouped
nonlinear heads强制不同sharing topology，且全部scales包含full-affine map；neutral raw-history作为primary gate，
A6-natural作为from-scratch E2E sensitivity carrier。

neutral seed2021现已返回：40/40 complete，function separation/carrier skill/crossing均5/5，sample × bin oracle
macro 7.6753%，canonical-vs-random 0.8945%且5/5正。train-only fixed scale跨datasets落在48/360/720，说明一个
固定scope没有统一支配。该结果是problem evidence，不是PCSD performance，随后已由A6与multi-seed gate复核。

A6-natural也已返回5/5 crossing。three-seed confirmation进一步确认：neutral/A6均为5/5 stable crossing；扣除
validation-best fixed scale后的strict oracle为7.1107%/9.1259%，再扣除每个future bin固定scale后的
sample-specific headroom仍为6.7948%/8.5990%。因此adaptive coupling problem已从single-seed clue升级为
dual-carrier、three-seed direct evidence。contiguity仅在两carrier各4/5 datasets稳定，不能claim universal
temporal grouping law。另一方面，GroupedMLP相对A6-LBF H720仍落后2.6886%，所以D14-A确认的是研究问题，
不是Contribution 1 method performance。

### Contribution 1 Candidate: PCSD-CF

`PCSD-CF`（Projective Coupling-Spectrum Decoder with a Coupling Field）在固定future domain上表示多个
output-sharing scopes：

$$
\mathcal S=\{1,b_1,b_2,\ldots,T\},\qquad
\hat y_\tau=\sum_{s\in\mathcal S}\alpha_s(X,\tau)\hat y_\tau^{(s)}.
$$

$s=1$接近Direct/independent query，$s=T$包含A6 global MIMO-like arm，中间$s$是parallel block scopes。
它不是五个完整models：同一history state先生成一个shared mode field，future-coordinate descriptors再按scope
pooling成point/block/global predictive states，并共享target synthesis rows。requested $H$不进入operator或
policy，只执行$F_H=\mathcal R_HF_T$。

固定coordinate field的constant mode与zero-mean modes允许构造任意A6 mapping：令nonconstant mode maps及
nonlinear synthesis weights为零，global pooling即精确退化为`coeff [B,C,256] × basis [T,256]`。这是
function-class containment，不是warm-start或learned-capacity preservation。首个control只使用actual fused
forecast loss端到端训练，不加入risk、oracle、balance、diversity或counterfactual auxiliary loss。

[Novelty Boundary] Direct/MIMO/DIRMO/Stratify已覆盖固定strategy与block-size continuum；CATS、MQTransformer、
TimePerceiver已覆盖future/target queries；Implicit Forecaster已覆盖global wave decoding。因此PCSD只能claim完整
链条：`one projective parameter field -> scope pooling changes future-output state sharing -> simultaneous
point-to-global operators -> exact A6 subspace -> sample/target policy -> no requested-H semantics`。
DeepONet/PoU-MoE已覆盖coordinate synthesis与local operator mixture，因此这些primitive不计novelty。
Step7A production implementation现已通过全部9类local gates：five-profile shape/integration、65个dense/arbitrary
prefix cases、float32/float64 A6 containment、720/15/5/2/1 Jacobian-sharing topology、random-parameter arm
separation、partition-only parameter equality、module与真实Encoder-PCSD two-step gradients、static accounting及
protocol exclusion均通过。float32 containment maximum output gap为`3.815e-6`，float64为`3.109e-15`；
修正fan-in初始化后canonical/random minimum pairwise arm NRMSE为`0.131493/0.023079`。PCSD field core相对
A6 decoder参数为
`3.0291-3.6184x`，含policy为`3.1006-3.7224x`，所以dense capacity control仍是Step7B硬要求。
status=`pcsd_cf_step7a_local_pass / effectiveness_unready`。

Step7B production/prelaunch现已冻结：A6、exact-paired M0、five fixed scopes、equal、static-target、direct、random
partition与dense nonlinear matched共12 arms × 5 datasets。A6/M0相同seed的operator hash与初始输出gap均为`0`；
full PCSD arms共享完全相同的trainable initialization；dense control参数gap低于`0.1%`。primary metric在结果返回前
固定为validation dense-H1..720 MSE AUC，test=false。用户已授权seed2021 screen，但effectiveness仍未观察。

### Retired Core Candidate: CCRL

`CCRL`（Cross-fitted Coupling-Regret Learning）不期待ordinary mixture loss自动产生expert specialization。它在
train split内chronological cross-fit每个coupling arm，对held-out training sample与target region构造centered
relative risk：

$$
r^{cf}_{i,b,s}=L^{cf}_{i,b,s}-\frac1{|\mathcal S|}\sum_jL^{cf}_{i,b,j}.
$$

policy仅由inference可见history与target coordinate预测relative risk。Step4-6审计已纠正原soft-label解释：对MSE，
weighted expert risk只是mixture loss上界，不能把$\operatorname{softmax}(-r)$称为optimal fusion。因此候选training
principle固定为actual fused forecast loss + auxiliary cross-fitted risk distillation；matched direct-fusion是mandatory
control。

[Retirement Reason] FFORMA/TimeFuse已覆盖feature-based sample fusion；TimeRouter已覆盖oracle labels、context/CV/
forecast features与nonlinear routing。更关键的是，CCRL需要独立fold experts生成稀疏OOF labels，再监督共享
representation的最终PCSD；teacher/student architecture不一致，labels会随joint arms更新而stale，额外工程不属于
最终推理图。它技术上可作辅助loss，但不是统一的single-run end-to-end training principle。

[Decision] status=`diagnostic_only_not_scheduled`；D14-B1在Step7A前取消。旧source/theory/config保留为历史control，
不得继续实现或作为Contribution 2 claim。

### Open Contribution 2 Slot

第二个contribution必须原生依赖PCSD的same-run arms，并直接服务最终fused forecast。当前只保留working
hypothesis `SC2-ICC`（same-forward interventional coupling credit），但尚未证明direct PCSD存在credit-assignment
failure，因此不冻结名称、loss或claim。

只有D15-A同时证明“same-run arms有skill与marginal/oracle headroom”以及“direct history × target policy没有利用
这些headroom”，并排除arm under-training、capacity与numeric explanations，SC2才允许进入Step2-4。否则不为填满
第二slot而添加training mechanism。

### Joint Story

PCSD-CF回答“一个unified decoder如何用同一parameter field表示不同future-output sharing strategies”。
Contribution 2未来只能回答“在same-run joint decoder中，若ordinary task loss无法正确分配scope credit，应如何
用同一次forward的forecast evidence修复”，而不能再依赖外部teacher pipeline。

[Execution Order] D14-A problem confirmed -> CCRL consistency audit and retirement -> PCSD-CF Step4-6 conditional
pass -> D15-A Step7A local invariants passed -> Step7B prelaunch passed -> GPU resource smoke -> authorized
five-dataset seed2021 direct-control screen。只有direct-control screen形成可归因的credit failure，SC2才回到
Step2-4。当前test、seeds2022/2023与Contribution-2 implementation均false。

### Closed Candidate: PRISM Decoder

`PRISM`（Prefix-Risk Isometric Synthesis Module）保留A6的free coefficient path：

$$
M[B,C,P,D]\rightarrow a[B,C,256]\rightarrow U_\mu[:H,:][H,256]\rightarrow\hat y_H[B,C,H].
$$

$U_\mu^TW_\mu U_\mu=I$，并用prefix family诱导的
$\mathbb E_H\|\operatorname{offdiag}(U^TW_HU)\|_F^2$控制short-prefix locality与global compaction的
Pareto tradeoff。$H$只crop rows，不进入learned path。D12-A未直接证伪其locality hypothesis，但其前置joint
problem gate失败，D12-B取消；status=`retired_without_effectiveness_test`。D6 crossing只保留为历史evidence。

### Closed Candidate: CAPE Frame Learning

`CAPE`（Cross-fitted Adaptive Predictable-Energy frame learning）不再修改future-step loss weights，而是用
train-only out-of-fold predictions估计$\Sigma_m=\operatorname{Cov}(\mathbb E[y\mid x])$。rank-limited frame
应最大化$\operatorname{tr}(U^TW_\mu\Sigma_mW_\mu U)$，避免raw-label PCA把capacity分配给不可预测noise。

`localization on/off × predictable/raw covariance`形成预注册`2x2` factorial。旧`MIPR`因D11不支持conflict、
benchmark measure headroom弱且与Time-o1/QDF/Loss Shaping邻近，降为`retired_as_core_candidate`；
$W_\mu$只保留为exact risk protocol/control。

D12-A-v1因uniform normalized risk mismatch不能方向级否定；修复后的v2复用相同pilots并与raw MSE对齐，
最终仅ETTh1支持，`1/5 < 3/5`。ETTm1/ETTm2/Weather的A6 raw gap@256仅`0.18%-0.34%`；CAPE
status=`failed_problem_gate / closed_as_core_candidate`。

完整source/theory audit与D12 gates见
`analysis/stage_c_d12_predictable_frame_feasibility_20260715/d12_final_result_and_rollback.md`。

### Historical Contribution 1 Record: Projective Forecast Operator Redesign

历史`narrative_ready`候选为`PMFO-RCT`。它从A6 history memory建立future interval tree，按
`90 -> 30 -> 10 -> 5 -> 1`逐层生成scaling/detail coefficients，并用fixed orthogonal contrast保证fine
detail不能改写parent coarse projection。目标性质：

- exact refinement recovery与nested-prefix consistency；
- $H$ 只prune与prefix相交的tree nodes，不进入learned state/query/router；
- parent-to-child shared state transition + orthogonal detail complement + local support；
- contribution来自future-side refinement conservativity与domain execution，不是“又一个wavelet/continuous
  basis decoder”。

PMFO-RCT v1已完成其falsification职责：theory/local invariants成立，但Step 7B相对A6的dense-MSE macro为
`-1.0955%`，三dataset均退化，故不能成为paper core。组件归因并不相同：conservative synthesis相对
no-conservation在三dataset一致改善（macro `+2.3393%`），保留为redesign证据；recursive transition相对
no-transition仅`+0.0486%`且跨dataset不一致，v1 claim撤回；structured decoder相对matched dense的
`+0.7193%`只是弱信号。

[Decision] 关闭范围仅是固定`90/30/10/5/1` mixed-radix partition、v1 state transition和整体替换A6
readout的组合。Contribution 1 slot与projectivity/conservation问题仍开放；回到Step 4重审function-class
containment、future partition与history-to-node interface。Step 7B没有操纵Encoder，不能据此认定Encoder不足。

Step 4 redesign audit已进一步确认：PMFO v1 readout有`212,010` parameters，而覆盖A6 rank-256 affine
operator family至少需要`316,112`维；相同256维latent不能称为capacity preservation。A6 effective operator
在fixed 90/30 boundaries上的jump ratio约`0.989-1.009`，8个PMFO root nodes的history-patch profile
cosine为`0.936-0.994`。因此function-family restriction、unsupported factorization与weak scale-native
interface均进入v1 failure attribution。

新provisional candidate为`Function-Preserving Multiresolution Operator Morphism (FPMO)`：把整个A6 future
operator改写到perfect-reconstruction multiresolution coordinates中，参数空间必须显式包含A6；ordered
memory直接进入scale coefficients，不经过shared recursive state作为唯一history path；$H$只选择与prefix
相交的supports。该候选不是“A6 output + residual patch”，也不能以tree、wavelet、lifting或network morphism
单项作为novelty。

[Fact] Step 5已构造任意正整数$T$的orthonormal interval transform，并在9个$T$、53个prefix cases上验证
exact A6 embedding、perfect reconstruction与native restriction，max algebraic gap=`5.329e-14`。因此
`FPMO-M0`可以在无dense bypass下完全复现A6；但它与A6只是bijective coordinate transform，只能作control。

要让history-to-scale path真正不同，`FPMO-DS`为各tree depth设置独立history factors。该class包含A6，
但T720下各group rank caps之和为720，等价于full affine readout。由此得到no-go boundary：exact包含全部
A6、independent scale states与总latent budget 256不能同时满足。params差异不用于否定方法，但full-affine
capacity必须由同function-class `FPMO-DA` control隔离。

[Fact] Step 6进一步证明，T720下linear DS与DA不仅capacity相近，而是拥有完全相同的full-affine
function class；对任意orthogonal coordinates与任意row grouping也成立。DS增加的是non-identifiable
deep-linear factorization。已有matrix-factorization工作说明这可能改变implicit optimization bias，但该
差异不是future-scale专属机制，也不能直接从其GD理论外推到当前Adam + L1 joint training。

[Decision] `FPMO-DS rejected_by_narrative_gate`。M0、DA与DS-L只保留为control/diagnostic artifacts，不进入
Step 7。普通per-scale nonlinear extension会破坏automatic exact A6 containment，并引入新的activation、
capacity与prior-art问题，必须作为新候选重新通过Step 2-5，不能事后挽救DS。Contribution 1 slot保持开放；
当时的cursor回到Step 2/3，以`SC1-D2`分离rank expansion、generic nonlinearity与true-scale alignment。

[Diagnostic] D2 formal5已完成165/165 frozen-memory fits且invariants pass。true interval basis相对random basis
macro `+3.0635%`，5/5 datasets、15/15 seeds为正；但true depth grouping相对same-basis random grouping只有
`+0.0947%`，仅2/5 datasets达到2/3 seeds为正，未过mandatory gate。因此精确的scale-grouping problem关闭，
rollback Step 2。basis signal当时尚未由完整$2\times2$ factorial识别为独立main effect，因此只授权了
`SC1-D3 crossed basis-group diagnostic`，而未升为decoder contribution。

[Fairness Boundary] 上述“关闭”仅针对frozen A6 representation上的final-head grouping设计，不能作为所有
end-to-end scale grouping的方向级否定；当前PLGO也不依赖该已测试grouping。

[Diagnostic] D3已补齐`random basis × random group` cell并形成15个dataset-checkpoint primary units。
basis main MSE reduction为`+2.9174%`，在true groups与random groups下分别为`+3.1164%/+2.7181%`；
5/5 datasets均通过方向一致性与interaction guard，MAE为`+2.3098%`。因此basis geometry作为独立probe
main effect获得支持，并授权返回Step 4。但exact depth grouping仍为false；balanced interval basis本身也受
Haar/wavelet/whitening prior art约束，尚不是Contribution 1。当前问题转为识别conditioning、energy
compaction、local-support或prefix compatibility中的真实机制，并证明其原生服务unified horizons。

[Diagnostic] D4已完成315/315 frozen-memory fits。balanced相对permuted interval的八horizon macro为`+1.6324%`，说明
contiguous locality在A6 representation/probe family中形成稳定conditional signal；但相对DCT-II与fit-only PCA分别为`-0.8609%/-1.5050%`，且相对random
interval tree仅`+0.2742%`。因此exact midpoint balancing与best-accuracy claim在该conditional probe中不成立，decision=
`standard_structured_basis_explains_gain_return_step2`。这不否定把balanced interval basis用于forecast
generation的组件级创新，而是要求paper-core novelty来自更完整的组合：future-prefix local support、
horizon-agnostic restriction、predictive conditioning与实际selective synthesis共同成立。

[Current Problem] 新问题暂命名`SC1-CLG`（Conditioning-Locality Gap）。D4的descriptive geometry显示
log MSE与coefficient covariance off-diagonal ratio的平均Spearman为`+0.8405`，与top-16 energy capture为
`-0.8357`；interval basis用约55个active atoms覆盖H48，而DCT/PCA需720个。下一步先判断local-support
orthogonal family能否在conditioning与prefix sparsity之间形成稳定Pareto improvement，而不是直接实现新head。
该problem现由`SC1-D5`检验：在同一frozen-memory head下，以fit-only geometry从预注册的block-local
DCT/PCA families选择`H48 active atoms <= 96`的basis，并对balanced、global DCT与global PCA做五dataset、
三checkpoint、八horizon比较。D5是problem diagnostic，不是新decoder候选。

[Diagnostic] D5 primary selector在15/15 units选择b96 PCA，但相对balanced仅`+0.0322%`，primary gate fail。
然而预注册的`block_dct2_b144`相对global DCT在short horizons约`+1.05%`、long horizons约`-1.15%`，
11/15 primary units同时short-positive与long-negative；其H48 active atoms为144而非720。因此按failure
attribution rule，D5只能否定`<=96 + offdiag selector`，不能否定local/global co-design。当前由D6在未使用的
validation batches 8-15确认support-scale × horizon interaction；该确认现已完成，但仍无已通过theory gate的
paper-core method。

[Strong Evidence] D6在disjoint validation window完成225/225并通过全部gates：b144相对global DCT的short
MSE为`+1.1964%`、long MSE为`-1.2675%`，12/15 primary units crossing；short-positive与long-negative
分别覆盖4/5和5/5 datasets。该结果支持A6 representation下存在support-scale interaction，并据此将SC1
problem收紧为：同一future function需要local-prefix synthesis与global-domain coherence，但requested H只
定义domain，不能成为learned semantic condition；它本身不证明任意end-to-end decoder都存在同样强度。

[Provisional Candidate] `SC1-PLGO`（Projective Local-Global Operator）通过Step 4 conditional narrative gate。
它不是“首次basis/wavelet forecast”：N-BEATS、N-HiTS、BasisFormer、FBM、WaveToken、Implicit Forecaster与
FlowState已占据相关单项。可辩护边界是global smooth atoms、interval-local supports、domain-only restriction与
selective synthesis的组合。balanced interval basis保留为local support scaffold；Step 4当时只授权进入
Step 5 stable reconstruction/function-class/capacity no-go audit。

[Fact] PLGO Step 5已构造Restricted-Global Nested Basis (`RGNB`)：root保留global DCT subspace，每个balanced
interval的detail是children scaling union相对parent的orthogonal complement。stable local Chebyshev chart修复了
raw restricted-DCT最高`3.110e17`的conditioning pathology；12个$(T,r_g)$、101个selected prefixes与
3,731个all-$H$ active-bound cases通过，max algebraic gap=`2.141e-13`。因此stable global-local synthesis、
arbitrary-prefix restriction与无dense bypass的A6 morphism均可行。

[Decision] 该结果只让mathematical scaffold通过。square `PLGO-ONB-M0`与A6是isometric
reparameterization；direct global/local frame有$r_g$维coefficient kernel；independent support-group maps在
T720的rank caps sum=720并退化为full affine。三者分别只能作为control、overcomplete control与rejected
capacity-confounded design。support pruning也不等于generator-level speedup，效率claim继续撤销。

[Hypothesis] Step 6唯一保留的问题是：一个不读取$H$的shared atom-conditioned generator，能否利用
support/scale/global-local descriptors原生生成active coefficients，并超过matched dense与random-descriptor
controls。prior-art primitive overlap只收紧claim，不自动否决该task-specific组合；但generator尚未通过
descriptor attribution与capacity controls，故PLGO为conditional narrative candidate，禁止直接训练method。

[Fact] Step 6进一步审计后，atomwise PAF tensor contract本身成立：33个prefix cases的coefficient subset、
prefix synthesis与paired-order invariance max gap=`4.547e-13`，且effective rank不超过256。external
source audit显示generic primitives已有DeepONet branch-trunk、NOMAD nonlinear decoder、BasisFormer basis
coefficient attention、TimePerceiver target queries与FlowState functional basis先例；这些先例用于约束
component claim和mandatory comparisons，不自动否决完整contribution。

[Strong Evidence] internal evidence也不允许直接实现：B11 basis-conditioned field的收益被no-basis与
constant-slot controls解释；B14 model-independent retrieval-demand只有`1/6` settings、`0/3` datasets通过。
因此atom-to-memory retrieval被明确删除，narrowed PAF只能读取与A6一致的shared flattened memory。

[Decision] `SC1-PLGO-PAF` Step 6按完整`problem-constraint-mechanism-implementation-claim`链条获得
conditional narrative pass。不把generic atom query、branch-trunk或HyperNetwork单独写成创新；Contribution 1
的候选边界是multi-horizon projective contract、RGNB local/global support geometry与atomwise generation的
组合。D7 frozen-memory diagnostic只负责conditional geometry attribution，不能替代Step 7 end-to-end gate。

[Strong Evidence] D7完成105/105 frozen-memory fits并使用fresh validation batches16-23。canonical RGNB
descriptors相对PERM/RANDOM在compact/matched widths分别提升MSE `+13.8034%/+12.8418%`、MAE
`+9.8581%/+9.3269%`，两个width均覆盖5/5 datasets；gain在H48最强并随horizon增长减弱，与D6 short-prefix
local-support evidence一致。

[Protocol Correction] D7相对free-M0的`-37.3836%/-39.1031%`是frozen A6 representation上的compatibility
gap，不是method-readiness gate。A6 Encoder由A6 decoder共同塑造，free-M0天然兼容该representation；PAF
replacement head没有机会反向塑造Encoder。raw metrics不变，但原“exact PAF v1失败并返回Step4”结论撤销。

[Decision] PAF恢复为`narrative_ready`，下一步是`SC1-D8-E2E`：A6、GEO、PERM、RANDOM compact/matched
七arms全部from scratch端到端joint training，五dataset seed2021先screen，再对decisive arms做三seed确认。
只有stable E2E PAF仍失败，才返回Step4 capacity-preserving redesign；frozen cross-swap不再作为primary gate。

[Fact] D8 Step7A已通过：五profiles × 七arms的210个shape-prefix与35个gradient cases全部通过；full-prefix
max gap=`2.384e-6`，flatten/patch-block-sum max gap=`5.722e-6`。runner dry-run生成35 jobs，analyzer
synthetic gate通过，method screening在CLI层禁止test。该证据只授权Step7B，不构成effectiveness结果。

[Strong Evidence] D8 Step7B完成35/35 validation-only runs。GEO-c256相对same-run A6 dense MSE macro
`-28.10%`、5/5 datasets均负，MAE macro `-20.54%`，因此exact shared-latent PAF不能成为paper-core方法。
但GEO相对PERM/RANDOM median为`+14.33%`、5/5为正；m694下geometry effect仍为`+14.71%`，而width
扩展只比c256回收`+0.58%`。geometry mechanism成立，width/capacity不是主要失败原因。

[Failure Attribution] GEO五dataset均hit epoch cap，但最后5 epochs validation只改善`0.02%–0.49%`；没有
divergence、NaN或>100% degradation。patch entropy在4/5 datasets下降，但ETTh1 entropy几乎不变仍退化
`35.62%`。因此关闭的是$\alpha_j=\psi(d_j)^TAh$这一exact shared/separable readout，而非RGNB、
projectivity或PLGO方向。当前回Step4审计patch-level intervention与readout function class；不做三seed或
无边界longer-epoch sweep，SC2继续held。

[Tensor Boundary] `memory [B,C,P,D] -> hidden [B,C,R]`是$R=PD$的bijective flatten，不是pooling；patch
identity没有在这一步丢失。A6与PAF都随后执行$R\rightarrow256$投影。PAF的真实风险是shared latent与
descriptor-generated atom map构成的separable history-atom interaction，而不是shape从四维变三维本身。D8
强制报告patch-block contributions与atom-patch Jacobian；B14未支持的atom retrieval不会未经新Step4-6直接加入。

[Step 4 Redesign Decision] source-informed audit进一步证明，单纯增加geometry-only branches不能解除该失败：
任何

$$
\alpha_j=\sum_e\pi_e(d_j)\psi_e(d_j)^TA_eh
$$

都可把加权trunks与branch matrices拼接为一个更宽的PAF。固定总rank时function class不变；扩rank时收益可由
capacity解释。因此“scale/atom experts”本身不进入paper core。直接atom-to-patch cross-attention也不推进：
flatten是bijective reshape，且history patch与future atom没有已证实的canonical alignment，B14与
OFormer/GNOT/BasisFormer/TimePerceiver均对该shortcut形成压力。

[Method Candidate] 当前只保留`SC1-JAPO`（Joint Atom-History Projective Operator）进入Step8 effectiveness screen。
它用free RGNB expert maps生成atom coefficients，但gate必须同时依赖history context与atom support geometry；
requested $H$仍只选择active atoms，不进入router。与geometry-only mixture不同，history-dependent gate不能吸收为
fixed temporal table，因此有机会解除D8的fixed separability。令所有experts表示同一A6-equivalent RGNB map时，
任意convex gate仍复现A6，故候选原则上无需dense bypass即可包含A6。

[Theory Result] Step5已在4个$T$、22个prefix cases上验证A6 containment与exact projectivity，最大误差分别为
`1.137e-13/1.172e-13`；constructive $f(h)=h\tanh(h)$证明joint gate严格超出fixed affine PAF，而
geometry-only mixture仍collapse为fixed operator。requested $H$不进入learned path。

[Optimization Boundary] exact containment只是function-class guarantee。identical experts会令router gradient为0并
保持expert symmetry，因此首版必须independent from-scratch initialization，不能把containment构造用作复制初始化。

[Step6 Design] JAPO固定为两个independent rank-256 RGNB coefficient experts，history与8维atom geometry分别
投影到$G=32$后以multiplicative feature形成expert-only softmax。identical copy、warm-start、hard top-k、active-atom
normalization、explicit H和auxiliary routing loss均禁止。五profiles design checker的projectivity最大误差
`3.331e-16`，initial entropy最低`0.999855`，所有joint gradient paths通过。

[Step7A Implementation] production `JAPOReadout`、six same-bank modes、checkpoint invariants与validation-only
runner/analyzer已落地。五profiles × 七arms的210/210 prefix与35/35 gradient cases通过；最大prefix gap
`4.768e-7`，patch-block rewrite gap `5.722e-6`；Encoder与expert-bank paired initialization hashes通过。

[Step8 Evidence] seed2021的35/35 validation-only runs、paired from-scratch initialization与全部artifact/invariant
audit均通过。JOINT相对A6的dense MSE macro为`-1.3754%`、0/5 datasets正向；相对same-bank control median为
`-0.0780%`、2/5正向。该结果没有达到严重失败阈值，也没有达到provisional-pass阈值，因此是
`inconclusive`而不是pass或方向级fail。router normalized entropy最低`0.993263`，提示当前训练可能未形成明显
expert specialization，但单seed不能区分optimization variance与exact design weakness。

[Step8 Decision] seed2022 unchanged matrix完成后，70/70 artifacts与paired contracts全部通过。two-seed mean下
JOINT相对A6为`-1.2435%`、0/5，且相对same-bank median为`-0.1175%`、仅1/5；UNIFORM/HISTORY/ATOM均在
macro上优于JOINT，触发`capacity_control_explains` hard gate。两个seed的router entropy都接近1，说明weak
specialization可重复。`SC1-JAPO exact v1`因此降为`failed_as_core_candidate`，seed2023停止。

[Rollback Boundary] 本结果否定当前`two free RGNB experts + factorized softmax weak mixing`作为paper-core实现，
不否定A6 containment、RGNB projectivity、canonical geometry的PERM/RANDOM小幅正向信号，亦不否定conditional
projective operator方向。Contribution 1回Step4 source-informed redesign；新candidate过Step4-6前不实现，test、
SC2-MIPR与joint factorial继续held。

[Systematic Review] 全阶段证据支持的不是“再加一个basis/router”，而是更窄的问题：A6已具有强free operator与
domain-only prefix consistency，RGNB提供future-side local/global support坐标；尚未证明的是history memory中是否
存在与这些support尺度可识别对应的operator structure。flatten `[B,C,P,D] -> [B,C,PD]`为bijective reshape，
所以当前不把信息压缩当作失败原因，而把“multiscale structure是否可访问、是否值得显式建模”作为待检验问题。

[Next Diagnostic] `SC1-D9 History-Support Operator Evidence Audit`预注册为`diagnostic_only`。D9-A先精确恢复
A6 memory-to-future operator，把history侧分成global/coarse/mid/local scale coordinates，把future侧分成global
root与local support/detail coordinates，并与atom-label permutation和random orthogonal history bases作matched
comparison；只有A通过才做sample-dependent input-Jacobian确认。D9通过只说明新local-global operator具有
existence evidence，不能证明method effectiveness；失败则回Step2/3，而不是继续叠加MoE、router或training loss。详细复盘见
`analysis/stage_c_sc1_post_japo_systematic_review_20260715/systematic_stage_review.md`。

[D9-A Result] 15/15 exact operator audits与Parseval invariant通过，但ordered scale hypothesis未过gate：
five-dataset macro rho=`0.173810`，positive effect datasets=`2/5`，atom-label permutation=`1/5`，
random-history-basis=`0/5`。details相对global root整体更偏高频在15/15 units出现，但details depth 0-5内部不
单调；该binary现象是post-hoc clue，不能挽救primary result。Contribution 1因此回Step2/3，D9-B取消。

[New Problem Boundary] 下一步只设计`SC1-D10 Raw History–Future Scale Identifiability`，在独立raw-data evidence
上区分binary global/detail、monotone multiscale与no-scale三种hypotheses。只有problem存在性与matched controls
通过后，才允许重新形成Step4 architecture candidate；当前new model、test、SC2与factorial均不授权。

[D10 Frozen Design] history使用七个DCT frequency bands，future使用RGNB global root与六层details；两侧group
sizes天然相同，但每个cell仍whiten并固定为16→16 sketched ridge以排除capacity差异。binary gate使用独立2×2
global/detail interaction，monotone gate只检查details内部6×6 diagonal，防止global/detail粗二分伪造多尺度证据。
fit/holdout按train时间区间隔离，final evidence只用official validation；paired history/future permutations为mandatory
controls。当前授权仅限diagnostic，不是model implementation。

[D10 Result] binary hypothesis所有五项gate均失败：effect/control只有2/5，两个directional selectivities同时为正
为0/5。detail-only diagonal相对median与paired controls在4/5 datasets为正，但canonical band从未在任何dataset的
至少4/6 rows成为最佳，6! mapping也仅2/5通过。ETTh1/ETTh2、ETTm1/ETTm2呈现不同off-diagonal patterns，
Weather近零；不存在可支撑unified method的cross-dataset mapping。decision为
`raw_aligned_scale_not_supported_rollback_step2`。

[Boundary Reset] D9 learned-operator层与D10 raw-data层共同关闭history-scale aligned routing。future-side RGNB、
projectivity与D6 horizon-support crossing仍保留。下一Step2问题暂定为future global/local components在不同prefix
losses下的error/gradient responsibility；在D11 source/theory audit前，不恢复SC2，也不实现adaptive router、new
decoder或loss。

[D11 Step2/3] source audit确认Time-o1已经直接提出transformed label alignment、label autocorrelation与
forecast-step task overload；FreDF、DBLoss及withdrawn Hybrid Loss进一步覆盖frequency/component loss与动态调权。
因此论文不能把“分解future再加component loss”作为创新。D11将边界收紧为prefix measure下的exact
future-component gradient responsibility与intervention-point diagnosis。

[D11 Frozen Diagnostic] 对complete orthogonal projectors有
$\sum_gJ^TP_gv=J^Tv$，故可在不假设prefix mask与basis commute的情况下，把MSE/L1 output gradient精确归因到
RGNB groups。五dataset × 三A6 checkpoints使用train/validation replication，比较RGNB、DCT与三个random bases，
并分离strict negative conflict、low positive alignment、norm imbalance与coordinate artifact。任何positive结果只
返回Step4，不直接授权decoder、loss、optimizer或SC2。

[D11 Result] accepted v2完成五dataset × 三checkpoints。strict short/long directional conflict为`0/5`，所有
validation MSE total paths/batches均为positive dot；support-specific component gate仅`2/5`，且同一component跨
short/long的negative fraction为0。formal decision=`transform_generic_pressure_sc2_only`，含义不是SC2通过，
而是SC1 conflict-aware decoder问题关闭并回Step2暂停。

[Coverage Boundary] RGNB responsibility distribution在3/5 datasets随prefix measure变化；short measure对最后两个
projective groups严格zero-gradient，而long measure对二者的平均share约为`0.064107/0.020441`。这只建立
`nested support -> unequal update opportunity` observation。Time-o1、per-step loss shaping与generic task
weighting/sampling已有直接邻近工作，所以下一步仅允许Contribution 2 Step1-3 external novelty/problem audit；
`SC2-MIPR`、coverage normalization、PCGrad、new loss、test与joint factorial仍不授权。

[Narrative Boundary] nonlinear decoder、operator MoE、geometry gating、structure-guided time-series MoE与
step-specific representation均已有直接prior art。可辩护边界只能是joint history-atom conditional operator、
RGNB exact projectivity与multi-horizon domain-only execution的完整组合。JAPO status更新为`narrative_ready`；
UNIFORM/HISTORY/ATOM/PERM/RANDOM same-bank controls与staged three-seed gates已冻结。当前只授权不改变design的
seed2022五dataset × 七arm validation-only confirmation；two-seed mean未过gate则停止exact JAPO并归因，只有
通过才授权seed2023。test与SC2-MIPR仍不授权。

### Historical Contribution 2 Record: Measure-Induced Projective Risk

SC2保留`PIR` slot ID，formal objective收紧为`MIPR`。raw horizon measure的exact risk为
$e^TW_\mu e$；MIPR定义$\widetilde W_\mu=\sum_lQ_lW_\mu Q_l$，在PMFO refinement blocks上保留
within-scale weighting并删除cross-scale coupling。它是decoder-aligned structured surrogate，不是比raw
risk“更measure-aligned”的等价改写。

历史状态曾为`narrative_ready / effectiveness_pending / held_after_SC1_rollback`；post-D11现已更新为
`retired_as_core_candidate`。L2下quadratic algebra成立；
Huber/L1没有exact block-metric等价，首轮不实现。`log_uniform_h` off-block energy为`0.205154`，
`uniform_h/benchmark_h`只有`0.003456/0.002480`，因此贡献主场景必须是continuous dense-horizon
deployment，不能只靠四个benchmark horizons。

[Diagnostic status] D1-v2 aggregate PIR problem gate通过，但证据具有measure boundary：log-uniform强、
uniform弱而跨dataset、benchmark projected excess 0/3。该历史边界已在Step4-6收紧为MIPR与
same-measure raw control。

## Frozen Baseline Evidence

natural profile：

- Weather: `patch_num=12, d_model=64, d_ff=128`；
- ETTm1: `patch_num=24, d_model=32, d_ff=64`；
- ETTh2: `patch_num=12, d_model=64, d_ff=128`。

contract hash:
`254d85d47a9e5b7c212f8a8b88decf17a0328a1ea1df324c9cc65be4c672a50c`。
profile 与 checkpoint 均由 validation 预先冻结，test 不参与选择。完整表见
`analysis/stage_c_natural_baseline_test_20260713/natural_baseline_test_report.md`。

## Contribution Boundary

[Current Boundary] 新主线不claim首次提出Direct/MIMO/DIRMO continuum、future query、block-wise output、
multi-scale decoder、dynamic ensemble、regret supervision或cross-validation。Stratify、CATS、MQTransformer、
TimePerceiver、Implicit Forecaster、MQF2、Multi-output Ensembles与TimeRouter均为mandatory controls。

[Current Boundary] 可探索的贡献单位仅是完整
`fixed past -> exact-prefix neural decoder -> point-to-global coupling spectrum -> sample/target-region
counterfactual policy -> no requested-H semantics -> no external strategy search`链条。

[Theory Boundary] 对deterministic separable MSE，future covariance不是Bayes point predictor的必要输入。
coupling只作为有限样本、有限capacity下的parameter-sharing inductive bias；不得宣称joint output具有
population-risk必然优势。

[Fact] A6先生成`coeff [B,C,256]`，再使用`basis[:H] [H,256]`直接计算H步输出；它已经满足domain-only
horizon、exact prefix equality与output-side $O(HK)$ computation。A6在新mainline中是global MIMO-like
coupling endpoint，不是待替换的弱head。

[Current Boundary] D6 short/local与long/global crossing属于basis-support evidence，只能间接提出output-coupling
hypothesis。新D14必须用neutral carrier、matched point/block/global heads、random partition与capacity controls
直接检验；frozen A6只作sensitivity。

[Decision] ordered patch memory降为optional Encoder–Decoder interface ablation。未来若PCSD需要更丰富history
access，可运行`D14-P`；其positive/negative均不能单独通过或拒绝PCSD-CF，也不能建立当前open SC2 slot。

[Decision] 旧 StageB coefficient conditioning、STBO、GRU future composition、unit-specific retrieval 与
encoder repair 均不再是 active candidate。历史失败只按各自 failure attribution 使用，不能被扩大为未经
测试的方向级结论，也不能因为 archive 中代码仍存在而自动复活。

[Decision] Step 7B将“结构正确”与“预测有效”明确分离：15/15 trained invariants通过说明实现与algebra无误，
但不补偿三dataset performance gate失败。当前归因为exact v1 `readout_or_head_design_wrong`，而非
`optimization_or_numeric_pathology`、Encoder方向失败或conservation方向失败。

[Decision] Step 4 source audit排除了三条捷径：不采用LeapTS式learned horizon/scale scheduling，不采用
PRISM式history tree + fixed-H dense heads，不采用Asymmetric-MMF式global low-rank + hierarchy residual作为
paper core。lifting、nested basis与network morphism只作为构造和proof evidence。

[Decision] Step 5进一步排除“function-preserving transform本身就是创新”：M0没有新function，direct atom
版本与dense affine正交等价，DS则有capacity expansion。Contribution 1必须在Step 6给出并验证
`DS > matched DA`所对应的scale-native inductive bias，否则FPMO不能成为paper core。

[Decision] Step 6 narrative audit已关闭该路径：DS与DA的function class相同，且factorization对random
orthogonal/group controls同样成立；requested prefix虽可少生成inactive coefficients，但dense $D_l$仍要求
先生成全部720维scale latents。由此否决的是当前linear DS design，而不是“future multiscale structure不存在”。
后续D2/D3 frozen-memory diagnostics在A6 representation上不支持depth grouping、支持basis main effect；它们
已经推动Step4机制审计，但不能作为end-to-end grouping direction的普遍否定。

[Decision] 新PLGO Step5没有重复旧FPMO结论，而是把global smooth root与interval-local complements组成了
stable square basis；但同样确认“fixed invertible transform本身不是method”。Contribution 1的新增机制必须
位于coefficient generation path，并由matched dense/random-descriptor controls隔离。Step6若无法做到，回滚
Step4，不以训练性能包装RGNB。

## Main Experiment Logic

1. 固定 natural A6 baseline 与 test reference；
2. D1-A验证label/residual nested structure，D1-B验证当前A6 memory存在可访问forecast information，D1-C验证
   learned basis geometry，同时审计measure/projected gradients；
3. PMFO-RCT与MIPR曾分别通过初版Step 4-6 narrative/theory gate；
4. Step 7A local invariants通过；Step 7B使用固定full-H720 pointwise L1、所有model parameters端到端训练，
   完成15-run architecture controls；
5. PMFO-RCT v1 effectiveness失败，回滚Step 4；MIPR、factorial与full matrix全部暂停；
6. Step 4 redesign audit已解释A6 function class、fixed partition与interface问题，并只把FPMO推进到Step 5；
7. FPMO Step 5 embedding/restriction通过但capacity no-go使其仅partial pass；
8. Step 6已判定DS claim无法脱离full-affine factorization解释，故FPMO不进入实现；
9. SC1-D2 core3 partial只支持basis geometry、不支持depth grouping；先冻结ETTh1/ETTm2 profile，再以拆分的
   random-group/random-basis controls完成formal5；
10. SC1-D3已确认basis main effect但否定grouping叙事；先以structured-basis/whitening controls和external
    prior art完成Step 4 mechanism audit；
11. SC1-D4确认locality但由DCT/PCA解释accuracy，故回Step 2/3诊断SC1-CLG；只有新SC1重新通过Step 4-6并完成screening后，才恢复MIPR、
    `2x2` factorial与3-seed full matrix；第二 backbone与official native baselines最后做generality gate。
12. SC1-D5 primary selector失败但b144出现short/long crossed interaction；D6使用disjoint validation window确认，
    pass只返回Step 4，不直接实现operator。
13. D6全部gate通过；PLGO在external source audit后conditional进入Step5，method implementation仍false。
14. PLGO Step5通过RGNB algebra/prefix/A6 morph，但ONB、frame与independent-group variants均被function/control
    no-go限制；只进入Step6 generator design，method implementation仍false。
15. PLGO Step6的PAF tensor/rank gate通过；external primitive overlap不再自动否决task-specific贡献，
    B11/B14促成D7 conditional attribution；D7现已完成并通过geometry gate。
16. D7在frozen A6 memory上确认conditional geometry effect；free-control gap因Encoder-Decoder co-adaptation
    不能判定method readiness。PAF重新开放。
17. D8-E2E Step7A已通过七arms、五profiles、projectivity、gradient与patch-interface gates；Step7B固定为
    35-run validation-only screen，结果返回前不启动三seed或MIPR。
18. D8 Step7B已完成：exact PAF effectiveness fail、geometry attribution pass、m694不能救回A6 gap；
    Contribution 1回Step4 redesign，三seed/MIPR/joint factorial继续暂停。
19. Step4排除flatten压缩、patch retrieval与geometry-only expert shortcuts，只保留joint history-atom JAPO。
20. JAPO Step5通过A6 containment、exact projectivity与strict non-collapse；identical initialization被symmetry
    audit禁止。
21. JAPO Step6已冻结E2/K256/G32、independent initialization、seven-arm matrix与staged seeds；candidate成为
    `narrative_ready`。
22. JAPO Step7A通过210 prefix、35 gradient、paired hashes与runner/analyzer gates；
23. JAPO seed2021完成35/35且无protocol/numeric pathology，但vs A6 macro `-1.3754%`、0/5，仅构成
    stable/inconclusive evidence；按冻结gate只补seed2022，不调architecture/hyperparameters，test/SC2仍暂停。
24. JAPO two-seed 70/70 gate最终失败：vs A6 `-1.2435%`、0/5，same-bank hard gate触发；exact v1关闭，
    seed2023停止，Contribution 1回Step4 operator-intervention redesign，projective direction本身不作否定。
25. post-JAPO系统复盘完成；下一步只执行SC1-D9 history-support operator diagnostic。D9过gate后才允许形成新的
    Step4-5 candidate，当前不授权model implementation、test、MIPR或joint factorial。
26. D9-A exact operator gate失败且无numeric/protocol pathology；ordered history-scale alignment关闭，D9-B取消，
    rollback Step2/3。binary global/detail只作D10 hypothesis，不作Contribution 1 evidence。
27. D10 Step2/3 design冻结binary、detail-monotone与no-aligned-scale三选一gate；通过只返回Step4，失败则继续
    rollback Step2，不允许从exploratory off-diagonal matrix事后生成method。
28. D10 primary gate失败且protocol有效；history-scale routing从Contribution 1 mainline关闭。下一步只审计
    future-component responsibility problem，不把generic adaptive multiscale mixing改名为新贡献。
29. D11 strict future-component conflict为0/5，support-specific gate为2/5；conflict-aware decoder/loss关闭，
    只把projective coverage observation交给Step1-3 problem audit。
30. D12 risk-aligned predictable-frame support仅1/5；CAPE关闭、PRISM joint route retired、D12-B取消，
    两个contribution slots回到Step2。
31. post-D12系统复盘提出NIFRO/IARL：基本对象由single forecast row改为nested-information
    forecast-revision surface；Forking-Sequences与generic stability loss被列为mandatory prior-art controls。
32. 用户确认forecast revision应作为下一篇独立SCI问题；已转移到`New-idea.md`，D13改为
    `deferred_next_paper`，不再是当前active cursor。
33. CADMO/CPGA曾作为fixed-past compression pair提出，但用户指出ordered patch memory只属于decoder interface，
    不能服务multi-horizon核心叙事；两项改为`rejected_by_narrative_scope`，未进入method implementation。
34. 新主线把multi-horizon矛盾定义为future-output coupling strategy：Direct/query、block-MIMO与global MIMO
    固定不同sharing scopes，而unified model不应依赖per-dataset/horizon external strategy selection。
35. provisional `PCSD`在一个exact-prefix decoder内表示point-to-global coupling spectrum；provisional `CCRL`
    用train-OOF sample × target-region regret监督coupling policy。primitive-level DIRMO/MoE/regret不计创新。
36. 下一步只执行新D14-A/B：A先验证matched coupling-scale crossing与oracle headroom；A pass后B才验证
    history+target regret predictability。neutral raw-history carrier为primary，frozen A6只作sensitivity；test=false。
37. D14-A fail关闭pair；A pass/B fail只让PCSD回Step4并重找SC2；A/B pass也只返回formal Step4-6，不能直接
    实现method或启动remote training。
38. D14-A最终通过；D14-B1在implementation前因cross-fit teacher/student mismatch退出paper core。PCSD-CF
    已完成native Step4-6、D15-A Step7A与Step7B prelaunch gate；seed2021 remote screen已授权，SC2 slot保持
    open，test=false。

未来candidate screening固定扩展到ETTh1、ETTh2、ETTm1、ETTm2、Weather。五dataset用于cross-dataset
generality，seeds2021/2022/2023用于stochastic confirmation；两者不能互相替代。ETTh1/ETTm2必须先完成
validation-only natural profile freeze。

任何 candidate 若在 problem或narrative gate失败，回滚 Step 2/3；不得通过叠加 Encoder、MoE、auxiliary
loss 或更多 tuning 来掩盖失败。

## Canonical Active Artifacts

- `analysis/stage_c_multi_horizon_coupling_mainline_reset_20260715/multi_horizon_coupling_mainline_reconstruction.md`
- `Papers/multi-horizon-output-coupling-audit.md`
- `docs/experiments/stage-c-d14-output-coupling-granularity.md`
- `docs/stage-ledgers/stage-c-unified-forecasting-redesign.md`
- `docs/research-roadmap.md`
- `analysis/stage_c_fixed_past_mainline_reset_20260715/fixed_past_mainline_reconstruction.md`（superseded CADMO/CPGA record）
- `docs/experiments/stage-c-d14-conditional-patch-memory-headroom.md`（D14-P auxiliary，not scheduled）
- `docs/experiments/stage-c-pmfo-pir-problem-diagnostic.md`
- `analysis/stage_c_contribution_research_reset_20260713/stage_c_contribution_deep_audit.md`
- `analysis/stage_c_natural_baseline_test_20260713/natural_baseline_test_report.md`
- `analysis/stage_c_d1_pmfo_pir_offline_20260713/`（v1 invalid audit evidence）
- `analysis/stage_c_d1_pmfo_pir_offline_v2_20260713/research_interpretation.md`
- `analysis/stage_c_step46_pmfo_pir_theory_gate_20260713/step46_design_and_prior_art.md`
- `analysis/stage_c_step7a_pmfo_rct_local_20260713/step7a_local_gate_report.md`
- `analysis/stage_c_step7b_pmfo_rct_20260713/step7b_screening_report.md`
- `analysis/stage_c_step7b_pmfo_rct_20260713/failure_attribution_addendum.md`
- `analysis/stage_c_step4_source_informed_redesign_20260713/step4_source_informed_redesign_audit.md`
- `analysis/stage_c_step5_fpmo_theory_20260713/step5_theory_feasibility.md`
- `analysis/stage_c_step6_fpmo_narrative_control_20260713/step6_narrative_control_gate.md`
- `analysis/stage_c_sc1_d4_structured_basis_20260714/research_interpretation.md`
- `analysis/stage_c_sc1_d5_conditioning_locality_20260714/research_interpretation.md`
- `analysis/stage_c_sc1_d6_horizon_support_interaction_20260714/research_interpretation.md`
- `analysis/stage_c_sc1_step4_projective_local_global_audit_20260714/source_informed_audit.md`
- `analysis/stage_c_sc1_plgo_step5_theory_20260714/step5_theory_feasibility.md`
- `analysis/stage_c_sc1_plgo_step6_design_20260714/step6_design_gate.md`
- `analysis/stage_c_sc1_d7_descriptor_sufficiency_20260714/research_interpretation.md`
- `analysis/stage_c_sc1_plgo_step4_redesign_20260714/step4_source_informed_redesign.md`
- `analysis/stage_c_sc1_japo_step5_theory_20260714/step5_theory_feasibility.md`
- `analysis/stage_c_sc1_japo_step6_design_20260714/step6_method_control_design.md`
- `analysis/stage_c_sc1_japo_step7a_local_20260714/step7a_local_gate_report.md`
- `analysis/stage_c_sc1_post_japo_systematic_review_20260715/systematic_stage_review.md`
- `analysis/stage_c_sc1_d9_history_support_operator_audit_20260715/d9_result_and_rollback.md`
- `analysis/stage_c_sc1_d10_raw_scale_identifiability_20260715/d10_step23_diagnostic_design.md`
- `analysis/stage_c_sc1_d10_raw_scale_identifiability_20260715/d10_result_and_rollback.md`
- `Papers/stage-c-external-decoder-objective-audit.md`
- `docs/experiments/stage-c-five-dataset-validation-policy.md`
- `docs/code-explanation/stage-c-pmfo-rct-step7a.md`
- `docs/code-explanation/stage-c-sc1-japo-step5-theory.md`
- `docs/code-explanation/stage-c-sc1-japo-step6-design.md`
- `docs/code-explanation/stage-c-sc1-japo-step7a.md`

2026-07-13 reset 前主线完整 snapshot 位于
`docs/archive/pre-stage-c-reset-20260713/`，仅作历史审计。
