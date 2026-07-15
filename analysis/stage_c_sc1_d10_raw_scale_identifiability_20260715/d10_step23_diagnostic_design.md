# SC1-D10 Raw History–Future Scale Identifiability

## Decision Summary

| Field | Pre-Registered Decision |
| --- | --- |
| `current_step` | Contribution 1 rollback Step 2/3 |
| `problem` | raw history→future relation支持binary global/detail、detail-level monotone alignment，还是不支持aligned-scale routing？ |
| `existence_evidence` | D9 post-hoc binary clue；ordered A6 operator scale hypothesis failed |
| `idea` | capacity-matched sketched ridge matrix over raw normalized history DCT bands × future RGNB groups |
| `theory_check` | exact orthogonal transforms；七组维数天然匹配；所有probe输入/输出固定16维 |
| `design` | official train fit + separated train holdout + official validation；5 datasets；3 sketch seeds × 3 fixed ridge lambdas |
| `narrative_gate` | not required；`diagnostic_only` |
| `effectiveness_gate` | not applicable；不训练forecast model、不读取test |
| `artifacts` | seven-by-seven matrices、binary 2×2 tables、control families、split/invariant metadata、gate report |
| `decision` | monotone / binary / no-aligned-scale三选一；任何pass都只返回Step4，不授权method training |

## 1. Why D10 Is Needed

D9-A只分析A6已学到的memory-to-future operator。它没有发现future detail depth 0-5与history frequency逐层
单调对应，但15/15 checkpoints出现了post-hoc global-root vs all-details二分。D10不再查看model weights，而是
直接问raw data中是否存在这种predictive structure。

这个问题不能从“多尺度模型通常有效”直接推出。外部primary sources给出三种彼此不同的先例：

1. [TimeMixer](https://arxiv.org/abs/2405.14616)同时进行fine-to-coarse与coarse-to-fine mixing，并把不同尺度
   视为complementary predictors；它不支持简单的一一对角假设。
2. [Pathformer](https://arxiv.org/abs/2402.05956)强调不同series/input dynamics可能偏好不同尺度，因此固定
   dataset-level scale mapping可能过强。
3. [MultiWave](https://proceedings.mlr.press/v209/deznabi23a.html)把不同frequency bands交给不同components，
   说明band-specific predictability可以成为可检验对象，但其任务与本项目不同，不能当作本地existence evidence。

[Decision] D10必须把binary、details内部monotone与random/no-scale controls同时预注册；若只看到稳定的
off-diagonal matrix，先记为exploratory cross-scale structure，不在同一批数据上事后升级为新hypothesis。

## 2. Raw Sample Contract

每个window包含`history [720,C]`与`future [720,C]`。沿用A6 instance normalization：对每个sample-channel以
history mean/std归一化history，并用同一mean/std得到future deviation。随后把channel展开为独立probe rows；
probe不做cross-channel mixing。

- fit：official train split前60%候选区间，均匀抽取256 windows；
- inner holdout：official train split后20%候选区间，均匀抽取128 windows；
- 中间20% train windows作为temporal gap，不参与fit或holdout；
- final evidence：official validation split均匀抽取256 windows；
- test split：禁止读取；
- 五个datasets分别独立fit，结果先在dataset内部聚合，再做cross-dataset gate。

fit、holdout与validation index写入metadata。相邻window可能共享部分observations，因此不把row-level classical
$p$值当作证据；gate使用dataset replication、fixed controls与mapping permutation。

## 3. Scale Coordinates

### 3.1 History groups

对normalized history使用720维orthonormal DCT-II。七组frequency index固定为：

$$
[0,16),[16,32),[32,64),[64,128),[128,256),[256,512),[512,720).
$$

group sizes为`16,16,32,64,128,256,208`。

### 3.2 Future groups

对future deviation使用`RGNB(T=720, global_rank=16)`，得到global root与detail depths 0-5；其group sizes恰好
也是`16,16,32,64,128,256,208`。维数相等只是建立matched coordinates，不被当作scale alignment证据。

### 3.3 Capacity-matched sketch

每个history/future group先用fit rows估计逐coordinate mean/std并whiten，再乘固定的orthonormal random sketch
到16维。16维group只做orthogonal rotation；更大group压缩到16维。三个sketch seeds固定为
`20260715/20260716/20260717`。

这一步确保每个cell都是`16 -> 16` linear ridge，参数量与optimization class完全一致。random sketch只用于
capacity matching，不claim完整保存某个group的全部predictive information；因此三seed稳定性是mandatory。

## 4. Probe And Controls

所有cells使用同一closed-form ridge：

$$
\widehat W=(X^TX/n+\lambda I)^{-1}X^TY/n,
$$

固定$\lambda\in\{10^{-3},10^{-2},10^{-1}\}$全部报告，不按cell选择。primary statistic对3 sketch seeds ×
3 lambdas共9个replicates取mean；inner holdout只检查sign stability，final gate使用validation $R^2$。

三种paired coordinate families：

1. `canonical`：ordered history DCT groups × canonical future RGNB groups；
2. `history_perm`：保持DCT coefficients与group sizes，但随机打乱mode-to-group assignment；
3. `future_perm`：保持RGNB coefficients与group sizes，但随机打乱atom-to-group assignment。

permutation与sketch在replicate内paired。所有family的数据、probe width、lambda与row count相同。

## 5. Primary Statistics

### 5.1 Binary global/detail

另建capacity-matched 2×2 probe：history global为DCT modes 0-15，history detail为其余704 modes的16维sketch；
future global为16个root atoms，future detail为其余704 atoms的16维sketch。

定义：

$$
\Delta_G=R^2(G_f\leftarrow G_h)-R^2(G_f\leftarrow D_h),
$$

$$
\Delta_D=R^2(D_f\leftarrow D_h)-R^2(D_f\leftarrow G_h),
$$

$$
I_{bin}=\frac{\Delta_G+\Delta_D}{2}.
$$

只有两个directional selectivities同时为正，binary interaction才有机制意义。

### 5.2 Detail-level monotone alignment

从7×7 matrix中排除global row/column，避免binary separation伪造multiscale diagonal。对future detail group
$j\in\{1,\ldots,6\}$：

$$
g_j=R^2_{j,j}-\operatorname{median}_{i\ne j,\ i\ge1}R^2_{j,i},
$$

并定义$G_{mono}=\frac{1}{6}\sum_jg_j$。同时报告六个future detail groups中canonical history group获得最佳
$R^2$的数量，以及6! exact history-label permutation one-sided empirical p-value。

### 5.3 Exploratory matrix

记录每个future group的best history group及9 replicates的一致率，用于发现non-monotone cross-scale coupling。
它没有primary gate；本轮不能据此提出或通过新method。

## 6. Pre-Registered Gates

### Binary pass

同时满足：

1. 至少4/5 dataset的validation mean $I_{bin}\ge0.01$；
2. 至少4/5 dataset的validation mean $\Delta_G>0$且$\Delta_D>0$；
3. 至少4/5 dataset的canonical $I_{bin}$比`history_perm/future_perm`较强者高至少`0.005`；
4. 45个dataset-replicates中至少36个canonical $I_{bin}>0$；
5. 至少4/5 dataset的holdout与validation mean interaction同为正。

### Detail-monotone pass

同时满足：

1. 至少4/5 dataset的validation mean $G_{mono}\ge0.01$；
2. 至少4/5 dataset的canonical-best count不少于4/6；
3. 至少4/5 dataset的6! permutation $p\le0.05$；
4. 至少4/5 dataset的canonical $G_{mono}$比`history_perm/future_perm`较强者高至少`0.005`；
5. 45个dataset-replicates中至少36个$G_{mono}>0$。

### Decision table

| Detail-monotone | Binary | Decision |
| --- | --- | --- |
| pass | pass/fail | `raw_detail_monotone_supported_return_step4` |
| fail | pass | `raw_binary_global_detail_supported_return_step4` |
| fail | fail | `raw_aligned_scale_not_supported_rollback_step2` |

任何pass只证明problem existence并授权Step4 source-informed candidate design；不授权forecast model implementation、
test、SC2或joint factorial。

## 7. Invariants And Failure Attribution

Mandatory invariants：DCT/RGNB orthogonality、sketch column orthogonality、group coverage/non-overlap、2×2与7×7
probe width equality、fit-only normalization、split indices、no test、finite ridge solution、artifact completeness。

- invariant或numeric失败：`diagnostic_invalid_for_direction_rejection`；
- canonical与controls都弱：`hypothesis_false_at_raw_aligned_scale_level`；
- canonical有信号但control同样强：`capacity_or_coordinate_control_explains`；
- fit/holdout正而validation失败：`distribution_or_probe_generalization_failure`；
- binary pass、monotone fail：只允许two-block problem，不得恢复seven-level scale-matched decoder；
- monotone pass：仍需Step4-6证明新operator不是TimeMixer/Pathformer/MultiWave primitive的重命名。
