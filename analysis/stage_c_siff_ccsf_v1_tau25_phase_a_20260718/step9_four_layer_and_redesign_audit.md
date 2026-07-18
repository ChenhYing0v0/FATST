# CCSF tau0.25 Phase-A Step9–10深度复盘与回滚设计

## 1. 本轮测试什么

formal candidate为`SC1-SIFF-v2-CCSF-v1-tau25`。矩阵固定为10 arms × 5 datasets × seed2021，共50 runs、
200个official-test H96/H192/H336/H720 cells。所有arms从相同初始化类别端到端训练；checkpoint只由validation
四horizon mean MSE选择，test不改变checkpoint。

本报告按四层证据回答：

1. paper-facing effectiveness是否成立；
2. gain能否归因于contrast architecture、RELCAL objective及二者interaction；
3. arms、policy与correction内部是否按理论工作；
4. 失败属于hypothesis、intervention/readout、optimization还是capacity control。

## 2. Artifact与协议完整性

[Fact] 50/50 runs、200/200 test cells完整；50个invariants均通过，test access date为`2026-07-18`，
checkpoint retrained=true、test前后hash不变、Encoder initialization按dataset matched、prefix gap为0、所有值finite。
训练无NaN/Inf或divergence，因此不属于`optimization_or_numeric_pathology`。

正式analyzer输出位于`returned/_analysis_seed2021/`；post-E2E diagnostic位于`post_e2e_diagnostic/`；validation/test
与checkpoint轨迹复核位于`deep_audit/`。

## 3. Paper-facing effectiveness

### 3.1 两个主比较

| Comparison | Test MSE gain | MAE gain | MSE cells | datasets | horizons | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CCSF_RELCAL vs A6_MEASURE | -0.8567% | -0.7251% | 8/20 | 2/5 | 0/4 | fail |
| CCSF_RELCAL vs SIFF-v1 EQUAL | -0.6159% | -0.3262% | 6/20 | 1/5 | 0/4 | fail |

dataset-level MSE gain中，full CCSF相对A6只在ETTm1 `+0.9320%`、ETTh2 `+1.0124%`为正；Weather
`-1.6741%`、ETTm2 `-1.6851%`、ETTh1 `-2.8688%`。相对v1只ETTm1为正。

[Decision] paper-facing effectiveness=false。exact candidate不能进入confirmation，也不能以内部oracle信号补救。

### 3.2 Validation不是主要失败解释

full CCSF vs A6在validation已经是`-0.3404%`，test扩大到`-0.8567%`，方向一致。full CCSF vs v1在validation
为`+0.1003%`，test反转为`-0.6159%`；但即使忽略该反转，A6 hard baseline仍明确失败。因此这不是单纯的
checkpoint假失败或test distribution shift误杀。

## 4. Matched mechanism attribution

| Frozen comparison | Test MSE gain | Interpretation |
| --- | ---: | --- |
| CCSF_EQUAL vs SIFF-v1 EQUAL | -0.4802% | architecture-only不成立 |
| CCSF_EQUAL vs zero-contrast capacity | -0.2412% | true contrast不及同参数zero input |
| CCSF_RELCAL vs CCSF_EQUAL | -0.1386% | objective main effect不成立 |
| full vs SIFF-v1 RELCAL | -0.0069% | architecture在RELCAL下接近零，不是净增益 |
| RELCAL vs standardized teacher | -0.0132% | relative teacher无specificity |
| true contrast vs zero contrast | -0.1619% | contrast没有净收益 |
| true contrast vs permuted contrast | +0.3568% | correct semantics比wrong semantics好 |
| ordered vs independent | -0.4582% | ordered field specificity失败 |

architecture-objective interaction为`+0.4754%`并通过，但其来源是：

$$
(-0.1386\%) - (-0.6140\%) = +0.4754\%.
$$

即RELCAL在plain SIFF上伤害更大，CCSF只是减少了这部分伤害；architecture与objective两个main effects都为负。
因此positive interaction不能被叙述成两项贡献协同成功。

最关键的contrast结果是“true > permuted，但true < zero”。它证明模型能感知contrast语义排列，却没有证明
contrast是有用干预。正确结论是`semantic sensitivity without net utility`。

## 5. Internal mechanism health

### 5.1 仍然成立的内部正证据

- oracle headroom五dataset均为正：5.30%–12.50%；
- pairwise arm NRMSE为0.084–0.315，arms没有collapse；
- final pointwise best-arm accuracy相对base policy提高6.29–15.73 percentage points；
- correction RMS非零且prefix projectivity严格保持。

因此失败不是“没有不同arms”或“correction完全没训练”。

### 5.2 真正失败的内部路径

- policy entropy在ETTm1/ETTm2/ETTh1/ETTh2为0.952–0.994，接近uniform；
- final allocation相对uniform mixture五dataset全部为负：`-0.0255%`至`-0.7751%`；
- policy-skill centered alignment除Weather外均为负，macro不通过；
- region/bin尺度上，final policy相对base policy的expected arm MSE为`-0.2507%`，mixture MSE为`-0.0095%`；
- learned correction与region skill的centered alignment为`-0.3416`。

“best-arm top-1命中提高”与“整体allocation变差”并不矛盾：argmax只关心第一名，而soft mixture performance还取决于
全部五个weights、误差幅度和forecast cross terms。当前correction偶尔把第一名排对，却把其余概率质量分配到更差
arms，且高误差区域影响MSE更大。

## 6. Post-E2E sufficiency diagnostic

为区分`hypothesis_false`与`readout/objective wrong`，冻结了test-derived、row-cross-fit diagnostic。它只在一半
rows拟合multinomial probe，在另一半评估，然后交换；不能pass方法或选择超参数。

| Comparison | expected arm MSE gain | positive folds | mixture MSE gain | positive folds |
| --- | ---: | ---: | ---: | ---: |
| contrast+base vs base features | +1.5462% | 9/10 | +0.7512% | 6/10 |
| contrast+base vs shuffled | +1.4443% | 9/10 | +0.8827% | 8/10 |
| post-hoc vs learned final policy | +6.1215% | 10/10 | +2.7067% | 9/10 |

8/8 diagnostic gates通过。ETTm1 second fold是明确例外，说明信号不是无条件稳定；但跨dataset总体证据足以否定
“训练后contrast信息已经消失”这一解释。

[Strong Evidence] exact failure主要是`intervention_point_wrong + readout_or_head_design_wrong`，并具体表现为
policy/supervision granularity mismatch：

- production teacher以每个future coordinate的instantaneous absolute arm error监督；
- 单点winner高度受不可约噪声和偶然误差影响；
- 真正可预测的relative competence在region/bin聚合后更明显；
- pointwise KL把有结构的region signal压成高噪声逐点责任，最终correction与region skill反向。

## 7. Source-informed redesign audit

检索日期为`2026-07-19`，来源以外部primary sources为主，Zotero不作为覆盖完整性依据。

- [MoLE, AISTATS 2024](https://proceedings.mlr.press/v238/ni24a.html)已覆盖forecast experts与router的端到端自适应组合；
- [Learning in Gated Neural Networks, AISTATS 2020](https://proceedings.mlr.press/v108/makkuva20a.html)明确表明expert与gate可能需要不同loss，支持“joint gradient并不自动恢复正确routing”；
- [GateTS, arXiv 2025](https://arxiv.org/abs/2508.17515)已覆盖attention-inspired forecasting gate及减少auxiliary routing loss；
- [Fast Training of MoE via Expert Loss Integration, arXiv 2026](https://arxiv.org/abs/2605.10330)已覆盖将expert-specific errors纳入forecast training；
- [Temperature in Softmax Gaussian MoE, ICML 2024](https://proceedings.mlr.press/v235/nguyen24a.html)说明temperature与softmax gate参数存在非平凡估计交互；
- [Implicit Forecaster, NeurIPS 2025](https://openreview.net/forum?id=gqoeQPhQcE)已把forecast decoding阶段与global wave synthesis作为独立问题。

因此下一步不能claim generic MoE、expert-loss supervision、temporal smoothing、attention gate或temperature本身。
可保留的完整问题链是：

> unified multi-horizon decoder中的experts不是任意pattern experts，而是不同output-coupling scopes；它们的
> relative competence在单点上不可稳定识别，却在horizon-agnostic future regions上形成可预测统计结构。policy
> representation与training responsibility必须在同一region granularity上共同设计，同时保持full-domain
> projectivity。

## 8. D2：Region granularity sufficiency

D2固定full-domain连续partitions，widths为`{1,48,144,360,720}`，不使用benchmark horizon ID。每个width均重新执行
row two-fold classifier，并比较true contrast、base-only与row-shuffled contrast。

| Width | true vs shuffled expected-arm gain | true vs shuffled mixture gain | positive datasets | Width gate |
| ---: | ---: | ---: | ---: | --- |
| 1 | -0.2909% | +0.9604% | 1/5 | fail |
| 48 | +1.2905% | +0.8882% | 5/5 | pass |
| 144 | +1.8657% | +1.1082% | 5/5 | pass |
| 360 | +1.5793% | +1.2005% | 5/5 | pass |
| 720 | +1.4827% | +0.9796% | 5/5 | pass |

[Strong Evidence] aggregation确实提高了“哪个arm单独更好”的可辨识性，说明pointwise teacher存在噪声；但best native
width相对pointwise的mixture增量只有`+0.1478` percentage points，未达到冻结的`+0.3`门槛。因此D2只通过2/3
diagnostic gates。region是有效分析尺度，但不足以单独成为Contribution或授权新method。

## 9. D3：Best-arm risk与fused-mixture risk分解

D3对每个sample-region计算两个target-label oracle：

$$
L_{\text{best}}=\min_s\frac{1}{|R|}\sum_{\tau\in R}e_{s,\tau}^2,
\qquad
L_{\Delta}=\min_{w\in\Delta^{S-1}}\frac{1}{|R|}
\sum_{\tau\in R}\left(\sum_s w_se_{s,\tau}\right)^2.
$$

后者包含arm residual cross terms，用来检查当前teacher忽略误差互消是否为主矛盾。

| Width | best arm vs uniform | simplex vs uniform | simplex vs best arm | simplex vs learned final |
| ---: | ---: | ---: | ---: | ---: |
| 48 | +8.4963% | +9.7446% | +1.3590% | +10.0446% |
| 144 | +5.8942% | +7.2042% | +1.3824% | +7.5115% |
| 360 | +4.2105% | +5.5029% | +1.3374% | +5.8157% |
| 720 | +2.6922% | +4.0467% | +1.3797% | +4.3638% |

simplex相对best-arm只有约`1.34%–1.38%`额外空间，且按预冻结dataset gate没有任何width通过；相反，learned final
距离best-arm/simplex仍有显著空间。[Decision] residual covariance不是当前首要失败原因。不能据此转向
covariance-aware combination；该方向既缺乏问题强度，也与classical forecast combination/stacking prior高度重叠。

## 10. D4：Readout sharpness

D4对同一cross-fit probability同时评估exponents `{0.5,1,2,4,8}`与hard argmax。所有arms为同时诊断项，不进行
test-based production temperature selection。

- scope-native widths中的最佳arm为`width=48, exponent=2`，相对原soft probability仍为`-0.0186%`，
  仅1/5 datasets为正；
- hard routing在width 48/144/360分别为`-1.3385%/-0.9831%/-0.7863%`；
- 只有global width720的smoothing `exponent=0.5`有`+0.2233%`，低于门槛且不属于scope-native route。

[Decision] softness/temperature不是主要瓶颈。contrast probe并非“已经正确识别，只需更果断地选arm”；它保留的是
弱、统计性的soft allocation信息，无法通过简单readout重标定变成paper-level gain。

## 11. Source-informed closure boundary

新增primary-source audit进一步表明：

- [MoLE](https://proceedings.mlr.press/v238/ni24a.html)已经覆盖forecast experts与end-to-end router；
- [Fast Training of MoE via Expert Loss Integration](https://arxiv.org/abs/2605.10330)已经直接覆盖forecast
  expert-error-aware objective；
- [Coupling Experts and Routers via an Auxiliary Loss](https://arxiv.org/abs/2512.23447)把expert-router capability
  coupling作为显式auxiliary-loss问题；
- [Advancing Expert Specialization for Better MoE](https://openreview.net/forum?id=iydmH9boLb)已经讨论
  overly-uniform routing、specialization与auxiliary objectives；
- classical forecast combination已覆盖error-covariance-aware convex weighting。

所以继续沿`contrast descriptor + competence teacher + temperature/sharpening`微调，不仅实验证据不足，novelty
空间也已被显著压缩。D2的region结果只能作为future decoder responsibility的诊断证据，不能被重新命名为新贡献。

## 12. 最终决策

- exact `SC1-SIFF-v2-CCSF-v1-tau25`：`failed_effectiveness_and_attribution`；
- confirmation seeds：不启动；
- broad multi-scope specialization：仍有oracle证据；
- target-free contrast-policy route：D2/D3/D4后关闭，不再做region、covariance或temperature修补；
- pointwise CCSF correction与pointwise RELCAL：关闭；
- failure attribution：exact v1最初为`intervention_point_wrong + readout_or_head_design_wrong`；扩大诊断后收紧为
  `hypothesis_false_for_contrast_policy_as_core`，同时保留`multi-scope competence exists`；
- `SC1-SIFF-v2-EQ-ATTR-v1`继续冻结为当前性能最接近发表水平的candidate，但不能把CCSF作为其归因修复；
- rollback：整个Contribution 1回Step2/4，Contribution 2回Step2；下一步必须重新定义“unified multi-horizon
  generation中值得解决的decoder contract”，不得继续堆叠router auxiliary losses。

[Self-critique] D2-D4均使用test-derived probe rows，只能完成failure attribution和路线关闭，不能证明任何替代方法。
D3 oracle还使用future labels，量化的是上限而非可部署能力。上述边界使我们不能从“oracle很大”直接推出“再设计一个
router就会成功”，也不能以test-derived最佳width或exponent启动训练。
