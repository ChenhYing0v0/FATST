# Post-PCC SIFF/MCCA Step5 Theory Feasibility

## Decision

| Field | Value |
| --- | --- |
| `current_step` | Step5 complete；Step6 source-informed method/control design next |
| `architecture_candidate` | `SC1-SIFF`：Scale-Indexed Forecast Field |
| `training_candidate` | `SC2-MCCA`：Measure-Constrained Competitive Assignment |
| `local_gate` | 10/10 pass |
| `decision` | `step5_theory_pass_step6_source_design_next` |
| `implementation/remote/test` | false / false / false |

本结果证明provisional pair在代数上不是立即矛盾，并可进入Step6；不证明production architecture有效、novel或性能
可提升。

## SC1-SIFF Feasibility

SIFF把internal coupling scale $s$映射为$z_s=\log s/\log T$，再用连续basis $\phi_q(z_s)$生成
scope-conditioned history modes：

$$
M_s(h)=\sum_{q=0}^{Q-1}\phi_q(z_s)M_q(h).
$$

在checker中，$Q=1$、$\phi_0=1$且其余components为零时，SIFF相对current PCSD shared field的max absolute
gap为`3.5527e-15`；full-output-then-crop prefix gap为`0`。

constant-coordinate witness中，current field的两个scope gap严格为`0`，因为它们读取同一history mode；SIFF加入
linear scale component后gap为`1.0`。因此SIFF严格超出**current Q=1 field在该受限construction上的function
class**。

[Boundary] 该witness不证明SIFF超出任意generic wider decoder，也不证明额外parameters不是增益原因。Step6必须冻结
Q1-wider、dense-matched和independent-scope controls，并给出production $Q/rank$ budget。

## SC2-MCCA Feasibility

MCCA以projective target measure $\omega\in\Delta^T$作为row marginal，以scope skill budget
$\rho\in\Delta^S$作为column marginal，对positive capability构造entropic assignment $A$：

$$
\sum_sA_{ts}=\omega_t,
\qquad
\sum_tA_{ts}=\rho_s.
$$

synthetic crossed-capability case得到：

| Check | Value |
| --- | ---: |
| max row-marginal gap | `6.2450e-17` |
| max column-marginal gap | `1.1102e-16` |
| best-scope mass gain over uniform | `0.6667` |
| dominant-arm case minimum scope mass | `0.2` |
| minimum skill-gradient L1 norm | `0.07643` |
| router-gradient norm | `0.31104` |

这说明双marginal约束可以同时做到：target risk measure不丢失、每个scope总体不饿死、不同targets仍可获得非均匀
competitive assignment。skill与router paths在stop-gradient assignment下均finite/nonzero。

[Boundary] balanced assignment/entropic Sinkhorn已有BASE与SSR等prior art；本地case也使用人为清晰的crossing。
MCCA能否在真实PCSD/SIFF capability上避免“强制无用arm”或mini-batch moving target，仍未知。

## Code-Theory Consistency

- Intended theory：SIFF提供training可利用的scope-identifiable degrees，MCCA以有限skill mass避免uniform per-target
  floor导致的homogenization；
- Realized check：float64 containment/projectivity/contrast，positive Sinkhorn marginals，crossed specialization与
  skill/router gradients；
- Proxy boundary：checker没有实例化production PCSD synthesis/pooling kernel，没有真实dataset、optimizer或
  checkpoint；
- Falsification：Step6若证明SIFF可被generic width吸收、无法公平控制参数，或MCCA退化为generic load balance且无
  task-specific必要性，则回Step4；不得因10/10 synthetic pass直接实现。

## Step6 Requirements

1. 冻结production SIFF tensor contracts、$Q/rank$与parameter controls；
2. 明确$\rho_s$来自何处，禁止dataset/horizon tuning与future-label offline teacher；
3. 对BASE/SSR、orthogonality/variance、Expert Loss Integration、AME-TS与heterogeneous expert prior art逐项收紧claim；
4. 冻结`PCSD/SIFF × EQUAL_SKILL/MCCA`的$2\times2$ factorial，并加入Q1-wider、generic OT与current PCC controls；
5. narrative gate与code-theory audit同时通过后，才允许Step7A local implementation。

decision=`step5_theory_pass_step6_source_design_next`。
