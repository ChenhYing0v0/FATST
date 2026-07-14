# SC1-D4 Structured-Basis Mechanism Diagnostic: Research Interpretation

## 1. Decision

`decision = standard_structured_basis_explains_gain_return_step2`。

[Strong Evidence] balanced-interval basis相对random/identity/permuted controls有真实正效应，但它的accuracy
优势不具有balanced-specific或best-structured-basis特异性：DCT-II和fit-only PCA在相同head、split、optimizer
下跨八个horizons整体更优；random interval trees基本复现balanced midpoint tree。

因此：

- exact balanced midpoint basis不能直接进入Step 5；
- “basis用于forecast generation”保留为组件级创新空间；
- 当前更真实的问题是**predictive conditioning与prefix-local support之间的Pareto gap**；
- method implementation、Encoder、MIPR和MoE继续未授权。

## 2. Validity Audit

| Check | Result |
| --- | --- |
| Fits | 315/315 complete |
| Metadata | 15/15 complete |
| Orthogonality | all bases `<=2e-5` |
| Test / forecast update | no test；A6 frozen |
| PCA leakage | covariance只使用fit targets |
| Numeric stability | 315/315 finite |
| Best epoch | 15–76；0/315触及120上限 |
| Local recomputation | 与remote decision一致 |

不存在`optimization_or_numeric_pathology`，所以本轮可以否定exact hypothesis，而不是只否定代码实现。

## 3. Main Results

以下均为balanced相对control的八horizon平均log-effect换算的MSE reduction；positive表示balanced更好：

| Control | Macro MSE reduction | Gate | Interpretation |
| --- | ---: | --- | --- |
| identity | +3.5292% | pass | 只在time-point coordinates预测明显更弱 |
| random orthogonal | +2.8103% | H720 +2.7181%，pass | D3 basis signal精确复现 |
| permuted interval | +1.6324% | pass | contiguous temporal placement具有真实作用 |
| random interval tree | +0.2742% | fail | exact midpoint balancing不特异 |
| DCT-II | -0.8609% | fail | fixed global smooth basis整体优于balanced |
| fit-only PCA | -1.5050% | fail | data-adaptive decorrelation/compaction进一步优于balanced |

### Cross-dataset boundary

- DCT-II在ETTh1/ETTh2/ETTm1/ETTm2均优于balanced；Weather基本持平；
- PCA在四个ETT datasets优于balanced，仅Weather由balanced领先；
- random interval tree只在ETTh1/ETTh2由balanced明显领先，在ETTm1/ETTm2持平或反向；
- locality gate在4/5 datasets通过，ETTh2只有1/3 checkpoints为正，但macro仍为正。

### Dense-horizon boundary

balanced相对DCT/PCA在8/8 horizons都未达到`-0.25%` noninferiority。balanced-vs-DCT从H48的
`-1.4764%`逐渐缩小到H720的`-0.2918%`；balanced-vs-PCA为`-1.6700%`至`-0.5095%`。

相反，balanced-vs-permuted interval在8/8 horizons均为正，且H48最大`+4.2040%`、H720为`+0.5195%`。
[Inference] contiguous interval locality对短prefix最重要，这与unified multi-horizon叙事相关；但它尚不足以
抵消DCT/PCA更好的predictive conditioning。

## 4. Geometry Attribution

| Family | Off-diagonal ratio | Top-16 variance capture | H48 active atoms |
| --- | ---: | ---: | ---: |
| PCA-fit | ~0 | 0.8400 | 720 |
| DCT-II | 0.5234 | 0.7790 | 720 |
| balanced interval | 0.5513 | 0.7274 | 55 |
| random interval tree | 0.5571 | 0.7232 | 54.7 |
| permuted interval | 0.6263 | 0.4775 | 206.3 |
| identity | 0.9804 | 0.0733 | 48 |
| random orthogonal | 0.9922 | 0.0932 | 720 |

每个dataset/checkpoint内以七个families计算的descriptive Spearman correlations：

- mean `corr(log MSE, covariance off-diagonal ratio) = +0.8405`；
- mean `corr(log MSE, top-16 variance capture) = -0.8357`；
- mean `corr(log MSE, top-64 variance capture) = -0.8190`；
- mean `corr(log MSE, H48 active atoms) = -0.4126`。

[Strong Evidence] 在当前GroupedNonlinearHead中，accuracy排序主要随decorrelation与energy compaction变化；
更少的active atoms并不自动带来更低MSE。identity最local却很弱，DCT/PCA全局却最好，证明“locality alone”
不是充分机制。

[Boundary] 每个correlation只有七个family points，是mechanism描述而非显著性检验；不能据此证明causality。

## 5. Innovation Judgment

对用户提出的“将balanced interval basis用于预测生成也有创新”作如下收紧：

1. [Fact] 用basis/wavelet coefficients生成forecast已有N-BEATS、BasisFormer、FBM、WaveToken等直接prior art；
2. [Supported] 本项目的future-domain interval supports、H只作domain restriction、同一operator服务dense horizons
   的组合仍有可辩护空间；
3. [Not Supported] 当前不能把“balanced midpoint basis本身更适合预测”作为核心claim；DCT/PCA更准，
   random interval tree与它近似等价；
4. [Retained Direction] interval-local generation应作为结构约束，与predictive conditioning共同设计，而不是
   把现有balanced basis直接包装成Contribution 1。

所以本轮不是否定你的判断，而是把创新单元从“一个固定basis”升级为：

> 一个同时保留future-prefix local support、又主动优化coefficient predictability/conditioning的
> horizon-agnostic generation operator。

## 6. Failure Attribution

- `hypothesis_false`：exact balanced midpoint basis具有超越standard structured bases的特异accuracy优势；
- `hypothesis_supported`：contiguous interval locality优于同atom的time permutation；
- `capacity_control_explains`：否，所有basis使用相同head parameters与groups；
- `optimization_or_numeric_pathology`：否；
- `remaining_untested`：局部正交basis能否通过data-adaptive construction接近DCT/PCA conditioning，同时保持
  prefix support；active atoms是否能转化为实际selective coefficient computation。

## 7. Rollback And Next Step

按预注册gate返回Step 2/3，建立新problem `SC1-CLG`（Conditioning-Locality Gap）：

1. 判断local-support orthogonal family中是否存在足够conditioning headroom，而非先设计network；
2. 以大量interval trees / richer local packets离线测量可达到的off-diagonal ratio、energy compaction与
   prefix active-set；
3. 若local family无法接近DCT/PCA，basis只作工程component，不占Contribution 1；
4. 若存在稳定Pareto improvement，再返回Step 4审计learned lifting/wavelet packet/local operator prior art，
   提出新candidate；
5. 必须证明coefficient head可按active supports选择性执行，否则active-atom count不能转成efficiency claim。

SC2-MIPR继续held。
