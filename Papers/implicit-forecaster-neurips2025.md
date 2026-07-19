# Implicit Forecaster（NeurIPS 2025）

## Metadata

- Title: *Towards Accurate Time Series Forecasting via Implicit Decoding*
- Venue: NeurIPS 2025
- Authors: Xinyu Li et al.
- Primary paper:
  https://papers.nips.cc/paper_files/paper/2025/hash/0e82ef0c89df6a6eff8734ea7e27c42f-Abstract-Conference.html
- Official code: https://github.com/rakuyorain/Implicit-Forecaster
- Search/verification date: 2026-07-19
- Discovery: external search；未用Zotero覆盖判断新颖性

## Problem And Mechanism

[Fact] 论文指出大量TSF工作集中在history encoder，而forecasting phase仍常用直接output projection独立生成各
future points。其`Implicit Forecaster (IF)`把encoder representation与原始input spectrum结合，预测frequency
pool中各waves的amplitude与phase，再通过iDFT合成future series。

核心路径为：

$$
X\rightarrow X_{\mathrm{enc}},
$$

$$
(X_{\mathrm{enc}}, |\operatorname{DFT}(X)|)
\rightarrow \hat A,
$$

$$
(X_{\mathrm{enc}}, \arg\operatorname{DFT}(X))
\rightarrow (\widehat{\sin\phi},\widehat{\cos\phi})
\rightarrow \hat\phi,
$$

$$
(\hat A,\hat\phi)\rightarrow \operatorname{iDFT}\rightarrow \hat Y.
$$

phase不直接回归角度，而是预测sine/cosine后经`atan2`恢复，以避免$-\pi/\pi$边界不连续。frequency pool可大于
forecast length，从而显式包含更低frequency。

## Critical Controls

论文比较了：

- Linear decoder；
- parameter-matched nonlinear MLP；
- Transformer decoder；
- IF without input-spectrum skip；
- input-spectrum skip only。

[Fact] 这些controls用于区分wave composition、generic nonlinearity、heavy decoder与history spectrum skip。

## Relevance To FATST

[Strong Evidence] IF支持“forecasting phase本身值得研究”，但它也直接占据了
`frequency/amplitude/phase implicit wave decoding`。本项目不能把类似Fourier head换名为创新。

A6与IF的关键差异：

| A6-LBF | IF |
| --- | --- |
| globally learned basis | fixed frequency pool |
| sample-specific coefficients | sample-specific amplitude and phase |
| Encoder fused state为主要输入 | Encoder state + raw input spectrum skip |
| learned basis直接线性合成 | polar spectrum经iDFT合成 |

因此IF最合适的当前角色是`SC-D19-IFC control_only`：在same unified full-$T$、measure objective与
end-to-end protocol下，判断structured wave decoding是否仍比A6 learned-basis generation有headroom。

## Adoption Boundary

- 采用：mechanism contract、phase continuity、frequency-pool与matched decoder controls；
- 不采用：直接复制upstream module、horizon-specific training scripts或原repo数据协议；
- 必须本地实现并适配当前tensor contract；
- full-$T=720$先生成、prefix crop，以保持当前fixed-past exact-projective任务定义；
- IF control通过也不自动成为本项目Contribution 1，只证明trajectory-level decoder问题仍值得继续。

## Official Code Audit（2026-07-19）

official code确认：

1. `spectrum_size=720`为默认，`fourier_norm="ortho"`；
2. 任意`pred_len`均先由`irfft`生成720 points，再在`main.py`裁剪`pred[:, :pred_len]`；
3. amplitude head为two-layer MLP + `ALU(w=0.5)`；
4. phase由两个two-layer MLP预测tanh sine/cosine，再用`atan2`；
5. input skip读取原始history的rFFT amplitude/phase；
6. upstream使用RevIN、Adam $10^{-4}$、MSE、16 epochs、patience 3；
7. upstream benchmark仍对H96/H192/H336/H720分别训练模型，未使用我们的unified prefix-measure objective。

因此，IF synthesis天然满足“full-T generation + crop”结构，但upstream training不是single unified model。
项目中的D19会保留synthesis contract，把training改为same `A6_MEASURE` objective；这属于source-informed
task adaptation，不是exact reproduction。

## Uncertainty

[Boundary] paper与核心official implementation现已核对。upstream lookback=96只作为source default，不作为
本地A6 contract；活动D19 v1.1已将IF skip适配为same 720-point history，并通过Step7A 114/114 local gate。
当前进入Step7B prelaunch；其通过前不得启动远程实验或读取official test。
