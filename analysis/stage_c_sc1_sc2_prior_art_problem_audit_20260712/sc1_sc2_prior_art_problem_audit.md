# StageC SC1/SC2 Prior-Art And Problem Audit

## Audit Question

SC0 carrier冻结后，本审计判断两个paper slots是否已经有足够清晰的问题与novelty boundary进入Step 4-6。
结论是：两者都尚未method-ready，而且原始表述需要收紧。

## Primary-Source Matrix

| Work | Decoder / training claim | Direct novelty pressure |
| --- | --- | --- |
| ElasTST, NeurIPS 2024 | future placeholders、structured mask、horizon invariance、horizon reweighting | exact prefix invariance与uniform-horizon reweighting不能作为本项目独占创新 |
| TimePerceiver, NeurIPS 2025 | target timestamp queries + generalized target-set training | “requested target进入decoder”本身不新 |
| Implicit Forecaster, NeurIPS 2025 | 以frequency/amplitude/phase隐式预测waves，再合成forecast | global structured/implicit decoder已是拥挤方向 |
| FlowState, 2025 | functional basis decoder支持dynamic horizon与sampling-rate变化 | arbitrary-horizon functional basis与A6 learned basis高度邻近 |
| Shifting Time, ICML 2025 | continuous time-shift neural operator | continuous-time/super-resolution operator已有直接prior art |

Primary sources:

- https://openreview.net/forum?id=RCeZ063p33
- https://openreview.net/forum?id=gqoeQPhQcE
- https://openreview.net/forum?id=R50AT6nAsM
- https://openreview.net/forum?id=emkdmORaj4
- https://arxiv.org/abs/2411.01842

## SC1-PFO Audit

[Fact] A6先计算`Y_hat_full in [B,720,C]`，任意requested horizon只做`Y_hat_full[:,:H,:]`。因此对同一输入，
H1 < H2时两次输出在前H1点完全相同；projective consistency已经由restriction operation保证，而不是一个
待修复的empirical defect。

[Fact] requested horizon没有进入A6 computation graph，且无论H多短都计算完整720-step trajectory。因此
它是`prefix-compatible full-trajectory operator`，不是horizon-adaptive decoder。

[Decision] 原假设“修复projective inconsistency”属于`hypothesis_false_on_current_carrier`；SC1广义表述不
通过narrative gate。可保留的新问题候选是：能否构造一个horizon-adaptive、incrementally refinable的
forecast operator，在只计算所需future extent时仍保持nested-prefix compatibility？但ElasTST/FlowState
已直接施加强novelty pressure，必须先证明本项目的tensor contract与贡献边界不同。

## SC2-HML Audit

[Fact] 历史B7 multi-prefix objective对`{96,192,336,720}`取prefix-mean再平均，使0-96与336-720的平均
step weight比达到14.39×；ETTh2/ETTm1的tail gain相对变弱，但Weather是反例。

[Fact] 当前frozen mechanism-control使用`pred_loss_mode=full`与单一full-720 pointwise L1。它对720个future
steps等权，不存在上述nested-prefix重复覆盖。旧B7证据不能直接证明当前control有optimization pathology。

[Strong Evidence] 若部署目标是随机requested horizon $H\sim\mu$，风险
$E_{H\sim\mu}[H^{-1}\sum_{t\le H}\ell_t]$会诱导step weight
$w_\mu(t)=E[\mathbf{1}(t\le H)/H]$。因此full-720 uniform-step loss只对应特定训练measure，而不是任意
varied-horizon deployment measure。

[Decision] “training risk应匹配声明的horizon measure”是有效problem candidate，但简单harmonic
reweighting与uniform-horizon sampling已受到ElasTST直接覆盖。SC2必须先证明不同measure在frozen carrier
上产生稳定且有意义的gradient/exposure差异，再寻找超越fixed weighting的机制。

## Current Gate And Rollback

- `SC1-PFO`: `problem_redefinition_required`; rollback Step 2。
- `SC2-HML`: `problem_candidate_diagnostic_required`;停在Step 3。
- method implementation: unauthorized。
- next diagnostic: frozen carrier固定train batches，对至少三种horizon measures做loss-unbiased reweighting，
  记录shared decoder/encoder gradient cosine、norm、per-step influence与dataset一致性；不训练新模型。

## Self-Critique

[Uncertainty] FlowState当前证据来自公开submission页面，尚未完成其full paper/code tensor-level审计；SC1
novelty判断应视为保守而非最终拒绝。SC2即使发现gradient差异，也可能只是不同risk定义的必然结果，不能
自动证明需要新算法；必须进一步连接到dense validation trade-off。
