# StageC SIFF/MCCA Step 7A Code Explanation

## 1. SIFF Forward Flow

入口为`SIFFCouplingFieldReadout`。Encoder保持A6-natural carrier不变，其输出memory先flatten为
`hidden [B,C,R]`，其中$R=P\times d_{model}$。

1. `mode_weight [Q,D,R,K]`与`mode_bias [Q,D,K]`把history映射为
   `component_modes [B,C,Q,D,K]`。
2. 固定`scale_basis [S,Q]`将component组合为`scale_modes [B,C,S,D,K]`。
3. 第$s$个mode `scale_modes[:,:,s] [B,C,D,K]`进入原PCSD的`_scope_forecast`；既有canonical grouping、
   pooled target coordinates、`identity_synthesis/nonlinear_synthesis [T,K]`和`temporal_bias [T]`均不变。
4. 五个arm堆叠为`arms [B,C,S,T]`，既有policy产生`weights [B,C,T,S]`并融合为
   `full [B,C,T]`。
5. `target_prefix`只在最后裁剪，因此任意requested horizon不改变component、scale mode、arm或policy computation。

Production candidate固定$Q=2,D=4,K=256$。第一列scale basis恒为1；第二列由五个coupling scales的
log-coordinate做centered unit-RMS normalization。

### Controls

- `siff-constant-control`：两列均为constant，代数上collapse为single field，但保留Q2 parameter storage；
- `siff-permuted-scale-control`：保留同一coordinate value set，反转scale-to-coordinate对应；
- `siff-q1-wide-control`：Q1、dataset-specific rank，用于generic width attribution；
- `siff-independent-scope-control`：`scale_basis=I_5`，用于independent expert attribution；
- `siff-dense-nonlinear-matched`：按SIFF decoder parameter count选择dense hidden width。

## 2. MCCA Training Flow

MCCA复用PCC的raw-scale L1、dense-prefix risk、standardization与harmonic transport。输入均来自同一次forward：

- fused forecast `[B,T,C]`；
- arm forecasts `[B,C,T,S]`；
- policy `[B,C,T,S]`；
- target `[B,T,C]`。

令$N=BC$，每个row $(n,t)$ 的mass为$a_{nt}=\omega_t/N$。ramped capability为

$$
\bar c=(1-\alpha)U+\alpha c,
$$

reference measure为$a\bar c$。MCCA的column marginal固定为

$$
\rho_s=0.8\sum_{n,t}a_{nt}\bar c_{nts}+0.2/S,
$$

它与同progress PCC credit的总scope mass完全一致。`log_i_projection`用64次log-domain Sinkhorn求满足row/column
marginals且最接近reference的allocation；再除row mass得到`credit [B,C,T,S]`。credit stop-gradient，skill loss和
route KL仍向arms与policy传递梯度。inference时MCCA完全不存在。

### Training Controls

- `mcca_pointwise_full`：不做prefix-risk transport，只用pointwise capability；
- `mcca_uniform_balanced`：保留OT solver但column marginal固定uniform，隔离generic balanced OT。

CLI继续沿用历史名称`--pcc-objective-mode`以保持artifact compatibility，但其choices现包含PCC和MCCA；
`effective_config.json`明确记录实际mode、solver、iterations与kernel floor。

## 3. Dense Measure-Only Compatibility

`siff-dense-nonlinear-matched`没有scope arms。训练adapter在`measure_only + non-coupling readout`时直接对fused
forecast计算相同projective target measure L1，避免用普通full-point L1造成objective confound；其他credit mode仍强制要求
PCSD/SIFF coupling readout。

## 4. Code-Theory Consistency Audit

[Fact] code实现了Step6冻结的`history -> Q components -> scale basis -> S modes`路径；没有horizon embedding、
forecast residual或separate decoder per scale。

[Fact] Q1与PCSD、Q2 constant collapse、A6 subspace均有exact local witness；float64/float32 MCCA marginals和
same-mass identity均通过冻结threshold。

[Proxy] reversed coordinate用于permuted control，是固定错误配对的一个代表，不等价于穷举所有permutations。

[Falsification] 若ordered SIFF不超过constant/permuted/Q1/independent/dense controls，architecture claim失败；若MCCA
不超过same-mass PCC与uniform OT，training claim失败；若只有joint gain而factorial main effects不成立，不得宣称两个
contribution均被验证。
