# StageC D14-A Source-Informed Diagnostic Design

## 0. Status

| Field | Value |
| --- | --- |
| `current_step` | Step 2-3 problem verification |
| `role` | diagnostic_only；neutral-carrier D14-A0 |
| `problem` | point/block/global output-sharing scopes是否存在稳定crossing与可兑现oracle headroom |
| `method_training` | false |
| `test_access` | false |
| `D14-B` | held；A通过前不实现 |
| `rollback` | valid A fail -> close current PCSD/CCRL problem route；invalid -> redesign diagnostic |

## 1. Source Findings

本轮external primary-source search日期为2026-07-15。Zotero仅作seed，不用于完整性判断。

- [Stratify](https://link.springer.com/article/10.1007/s10618-025-01135-1)把multi-output size写成
  forecasting strategy的bias、variance、flexibility与computation trade-off，并指出没有普遍最优strategy；
- [Low Rank Forecasting](https://arxiv.org/abs/2101.12414)把rank-$r$ forecaster写成past-to-latent与
  latent-to-future两步factorization，说明output rank是可审计的shared latent-state constraint；
- [Reduced Rank Ridge Regression](https://pmc.ncbi.nlm.nih.gov/articles/PMC3444519/)与multi-task reduced-rank
  literature说明low-rank coefficient matrix是跨multiple responses/tasks共享信息的经典手段；
- 因此reduced rank不是本论文创新。D14只借它构造一个closed-form、低optimization-confound的coupling probe。

## 2. Why Ordinary Ridge Is Invalid For This Question

对$T$个outputs的separable squared loss与相同ridge penalty，分别拟合$T$个scalar ridge，与一次拟合
$W\in\mathbb R^{d\times T}$的multi-output ridge具有相同normal equations。把outputs写成一个matrix并不会
自动产生coupling；若用这种probe比较Direct与MIMO，会得到伪问题。

D14-A0改用blockwise rank constraints：

$$
\hat Y_{B_j}=XW_j,\qquad \operatorname{rank}(W_j)\le r_s,
$$

其中$B_j$是size-$s$的future block。$s=1$时每个target拥有独立full linear map；$s=720$时全部targets共享
一个rank-60 latent map；中间scales只在block内部共享latent directions。

## 3. Matched Parameter Construction

neutral carrier width固定$d=64$。point endpoint的factor count为：

$$
P_1=T(d+1)=720\times65=46800.
$$

对scale $s$，选择同一block rank$r_s$，使

$$
P_s=\frac{T}{s}r_s(d+s)
$$

最接近$P_1$，并满足$r_s\le\min(d,s)$。冻结结果：

| scale | blocks | rank/block | factor params | relative gap |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 720 | 1 | 46800 | 0.000% |
| 48 | 15 | 28 | 47040 | +0.513% |
| 144 | 5 | 45 | 46800 | 0.000% |
| 360 | 2 | 55 | 46640 | -0.342% |
| 720 | 1 | 60 | 47040 | +0.513% |

所有arms共享720维intercept，该common term不计入relative budget；最大gap必须小于1%。params差异不参与
profile选择，但该matched construction用于排除capacity解释。

## 4. Neutral Carrier

每个window/channel独立作history mean/std normalization：

$$
x'=(x-\mu_x)/(\sigma_x+\epsilon),\qquad
y'=(y-\mu_x)/(\sigma_x+\epsilon).
$$

只用当前fold fit rows对$x'$拟合target-agnostic PCA-64；calibration与official validation只transform。PCA score
再按fit mean/std标准化。该carrier不与任何output head co-adapt，也不读取future labels来选择features。

为了防止“所有heads都不会预测”产生无意义negative，加入carrier-skill gate：train-selected best fixed scale必须
在至少3/5 datasets上相对train-target-mean baseline改善至少0.5%。若失败，D14标记
`diagnostic_invalid_for_direction_rejection`，不能关闭PCSD方向。

## 5. Closed-Form Reduced-Rank Fit

先在centered fit rows上计算numerically stabilized OLS：

$$
W_{ols}=(X^TX+10^{-8}I)^{-1}X^TY.
$$

对每个output block，令$G_j=W_j^TX^TXW_j$，取其top-$r_s$ eigenvectors $V_{j,r}$：

$$
W_{j,r}=W_jV_{j,r}V_{j,r}^T.
$$

这是standard reduced-rank least-squares的output-subspace projection。所有scales共享同一个$W_{ols}$、fit rows、
PCA carrier与numeric ridge，因此差异只来自output sharing scope。

## 6. Partitions And Controls

每个fold同时建立：

1. `canonical_s*`：从future origin开始的contiguous blocks；
2. `shifted_s*`：boundary平移$s/2$的circular contiguous blocks，检查固定边界偶然性；
3. `random_s*`：相同block sizes与rank、随机打散target coordinates；
4. `train_mean`：无history信息；用于carrier skill；
5. `persistence`：history last value重复到future；仅作forecast context；
6. `equal_canonical`：五个canonical scales等权；
7. `train_selected_best`：只按train calibration full-domain MSE选择一个canonical scale。

point arm与unstructured full-affine是同一function class；代码必须验证二者prediction equality。global arm的
rank-60约束代表full-domain sharing，不使用A6 checkpoint。

## 7. Chronological Folds

三fold分别标记2021/2022/2023，但它们不是随机model seeds，而是预注册rolling train regions。每fold使用512
fit windows、128 calibration windows；fit/calibration observation gap至少$2T$。official validation均匀取256
windows。ETTh1、ETTh2、ETTm1、ETTm2、Weather使用相同fractions与window counts。

## 8. Exact Statistics And Gates

future bins固定为`short=[0,144)`、`mid=[144,360)`、`long=[360,720)`；另报告八个dense prefixes。

pair $(s_a,s_b)$在一个fold内crossing，当其bin-wise relative gain同时满足：

$$
\max_b g_{a>b,b}\ge0.1\%,\qquad \min_b g_{a>b,b}\le-0.1\%.
$$

dataset stable crossing要求同一pair至少2/3 folds crossing。总体D14-A0 pass还要求：

1. carrier skill至少3/5；
2. stable crossing至少3/5；
3. sample × bin canonical oracle相对train-selected fixed scale的five-dataset macro gain至少0.5%；
4. canonical oracle相对random-partition oracle至少3/5为正，macro gain至少0.1%；
5. parameter、PCA orthogonality、condition number、finite与split invariants全部通过；
6. 任一dataset不得出现超过100%的severe degradation。

shifted partition只检查contiguous-boundary robustness，不要求canonical固定起点优于shifted。

## 9. Failure Attribution Boundary

- carrier skill fail：`intervention_point_wrong / diagnostic_invalid_for_direction_rejection`；
- parameter、condition、finite fail：`optimization_or_numeric_pathology`；
- random partitions解释收益：`capacity_or_generic_partition_explains`；
- skill与invariants通过，但无crossing/oracle headroom：只关闭当前`neutral PCA64 + linear RRR coupling`
  problem evidence；是否扩大到PCSD方向必须结合A6 sensitivity与formal failure audit；
- positive只授权返回Step 4-6，不证明PCSD E2E effectiveness。

## 10. Decision

[Decision] source-informed family通过Step 5 diagnostic feasibility：它确实改变output sharing、parameter budget
近似相等、closed-form fit不依赖neural optimizer。授权实现与运行D14-A0；D14-B、PCSD method与test仍false。
