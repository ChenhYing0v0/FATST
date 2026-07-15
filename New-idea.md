# New Idea：Causal Forecast-Revision Surfaces

> 独立论文 idea 存档；不是当前 `R_2026_FATST` 的 active mainline。

| Field | Value |
| --- | --- |
| `idea_status` | `deferred_next_paper` |
| `active_in_current_paper` | false |
| `record_date` | 2026-07-15 |
| `provisional_title` | Learning Forecasts That Evolve with Information: Causal Revision Surfaces for Multi-Horizon Forecasting |
| `candidate_1` | NIFRO: Nested-Information Forecast Revision Operator |
| `candidate_2` | IARL: Innovation-Accounted Revision Learning |
| `next_gate` | D13 rolling-origin problem verification |
| `current_owner` | future independent SCI project |

## 1. 一句话问题

通常的multi-horizon model只回答：

> 在当前时刻$o$，未来$H$步分别是多少？

这个idea研究另一个问题：

> 随着$o+1,o+2,\ldots$的新观测不断到来，模型对同一个未来目标$Y_\tau$的预测，应当如何有依据地修订？

它不是当前fixed-past任务的decoder小改造，而是一类“连续预测 / forecast evolution”问题。因此将其从
当前论文主线中剥离，保留为下一篇SCI论文的核心候选。

## 2. 基本预测对象

定义：

$$
F(o,\tau)=\mathbb E[Y_\tau\mid\mathcal F_o],\qquad o<\tau.
$$

所有$(o,\tau)$形成一个上三角forecast surface：

- 固定origin $o$的一行，是通常的multi-horizon forecast；
- 固定target $\tau$的一列，是same-target revision path；
- 部署时只需读取latest-origin row，但训练时可利用整张surface；
- requested horizon只裁剪latest row，不作为learned horizon ID。

这使论文问题从“一次生成一条future vector”扩展为“学习预测如何随information set演化”。

## 3. 为什么值得独立成文

[Fact] rolling deployment中，模型会反复预测同一个未来时点；后一次预测比前一次多看到了近期数据。

[Hypothesis] 现有window-sampled forecasting虽然可在每个origin独立产生预测，但没有保证：

1. 新信息触发的forecast revision确实改善了同一target的accuracy；
2. revision幅度与由新信息换来的risk reduction相匹配；
3. newly arrived patch对哪些future targets负责是可识别的；
4. forecast在origin axis与target axis上同时满足causality/projectivity。

如果该问题跨dataset存在，它连接了forecasting architecture、forecast rationality、stability与online
decision support，叙事深度明显高于普通decoder patch。

## 4. Candidate 1：NIFRO

`NIFRO`（Nested-Information Forecast Revision Operator）不把每个origin视作互不相关的训练样本，而是
显式生成causal `origin × target` surface。

一种provisional tensor contract为：

$$
M\in\mathbb R^{B\times C\times P\times D},
\qquad
\Delta\in\mathbb R^{B\times C\times P\times T},
$$

其中$\Delta_{p,\tau}$表示第$p$个information increment对target $\tau$的revision contribution。沿origin
axis做causal prefix accumulation：

$$
F_{p,\tau}=F_{0,\tau}+\sum_{j\le p}\Delta_{j,\tau}.
$$

### 4.1 必须满足的结构合同

1. `causality`：第$p$行不得读取$p$之后的信息；
2. `origin projectivity`：截断information prefix不改变已有origin rows；
3. `target projectivity`：请求$H_1<H_2$时，前$H_1$个target输出严格一致；
4. `A6 linear containment`：最小linear control必须能重写A6 latest-row linear readout；
5. `direct increment-to-target path`：new patch对其修订的target存在直接gradient path；
6. `no horizon semantic conditioning`：$H$只确定输出域。

### 4.2 不可宣称的内容

- 不能claim首次联合训练多个forecast creation dates；MQ-RNN / Forking-Sequences已有相邻工作；
- 不能claim causal attention、target query或forecast surface本身全新；
- 不能claim generic forecast stability；已有N-BEATS-S等工作；
- 贡献必须落在nested-information factorization、dual projectivity、patch-conditioned revisions与IARL的
  完整任务链上。

## 5. Candidate 2：IARL

`IARL`（Innovation-Accounted Revision Learning）不把所有forecast revisions都压小，而是区分：

- 有新信息依据、并换来accuracy improvement的有用revision；
- revision很大、却没有降低error的无效波动。

对同一target定义：

$$
\Delta=F(o+1,\tau)-F(o,\tau),
\qquad
e_{new}=Y_\tau-F(o+1,\tau).
$$

conditional-mean projection给出理想moment：

$$
\mathbb E[e_{new}\Delta]=0,
$$

以及risk decomposition：

$$
\mathbb E[e_{old}^{2}-e_{new}^{2}]=\mathbb E[\Delta^2].
$$

IARL的目标不是复制这两个经典identity，而是把它们转化为neural forecast surface上的可训练、分距离、
数值稳定的accounting rule。

### 5.1 Provisional loss

$$
\mathcal L
=\mathcal L_{surface\_point}
+\lambda_m\mathcal L_{revision\_moment}
+\lambda_g\mathcal L_{risk\_gain}.
$$

moment应在batch × channel × target-distance bin上归一化计算，避免短期高方差channel主导。具体形式必须在
未来项目的Step 4-6中重新推导，不能直接把当前表达式视为已冻结method。

### 5.2 与generic stability loss的区别

generic stability通常直接惩罚$\|\Delta\|^2$；IARL允许$\Delta$很大，只要求它由新信息带来的accuracy gain
解释。它追求的是“有依据的改变”，不是“不改变”。

## 6. 两项contribution的闭环

1. NIFRO提供可学习的revision surface；
2. IARL定义哪类revision在统计意义上值得保留；
3. 没有NIFRO，IARL容易退化为普通rolling-window regularizer；
4. 没有IARL，NIFRO可能退化为一组相互独立的origin heads；
5. 二者共同形成architecture + training principle，而不是两个松散模块。

## 7. 最重要的理论与协议风险

### 7.1 Fixed-window不是严格nested information

当前A6输入为rolling 720-step window。origin前移时既加入new block，也移除expired block，因此两个effective
inputs并非严格nested。未来实验必须分别保存：

- `added block`；
- `expired block`；
- `shared middle block`。

若forecast revision主要由expired block解释，则不能使用“nested information”强叙事；需要改用
`changing information set`，或者采用long-context/no-expiry control。

### 7.2 Forecast rationality不是模型专属新理论

conditional expectation、martingale revision与orthogonality identity已有长期统计学基础。新颖性只能来自：

- 它们如何约束multi-horizon neural forecast surface；
- architecture如何使new information contribution可识别；
- 与fixed-window、target distance、patch increments的任务耦合；
- 跨dataset performance与revision-efficiency证据。

### 7.3 多origin训练本身已被覆盖

MQ-RNN、MQTransformer与Forking-Sequences等工作构成mandatory baselines。任何未来实现都必须证明收益不是
由“更多training targets / 更多forecast origins”这一capacity或data-augmentation control解释。

## 8. D13：未来项目的第一道problem gate

D13不训练NIFRO/IARL，只利用冻结A6 checkpoints检查问题是否真实存在。

### D13-A：Revision efficiency

对相同target比较old/new origins：

$$
\Delta=\hat y_{new}-\hat y_{old},
$$

$$
R=\mathbb E[\Delta^2],
\qquad
G=\mathbb E[(y-\hat y_{old})^2-(y-\hat y_{new})^2],
$$

$$
C=\mathbb E[(y-\hat y_{new})\Delta].
$$

验证algebra identity：

$$
G-R-2C=0.
$$

重点不是identity是否成立，而是：

1. $G>0$是否跨dataset成立；
2. revision efficiency $G/(R+\epsilon)$是否系统偏离1；
3. train-only scalar calibration能否在validation改善A6；
4. inefficiency是否由expired window artifact解释。

### D13-B：New-patch information

只有D13-A通过后，才比较：

1. old-state-only；
2. new-patch-only；
3. old-state + new-patch；
4. old-state + time-shifted patch。

目标是判断new patch是否含有可预测的ideal correction，而不是直接训练新method。

### 决策

| D13-A | D13-B | Decision |
| --- | --- | --- |
| fail | 不执行 | joint route关闭，重新定义问题 |
| pass | fail | patch-direct NIFRO不成立；IARL仍只是问题候选 |
| pass | pass | 只进入formal Step 4-6，不直接实现method |
| invalid | 不判断 | 修复diagnostic，不得方向级否决 |

完整旧protocol保留在
`docs/experiments/stage-c-d13-rolling-origin-revision-efficiency.md`，旧系统复盘保留在
`analysis/stage_c_post_d12_revision_surface_mainline_20260715/systematic_review_and_mainline_redesign.md`。

## 9. 未来独立论文的建议实验顺序

1. Step 1：更新external primary-source review；
2. Step 2-3：完成D13-A/B problem verification；
3. Step 4：在Forking-Sequences、MQTransformer、stability与rationality文献下重做novelty audit；
4. Step 5：证明causality、dual projectivity与A6 containment；
5. Step 6：冻结surface tensor contract、controls与loss moments；
6. Step 7：local invariant tests；
7. Step 8-10：先5 datasets × 1 seed screen，再3 seeds与second backbone；
8. 只有NIFRO/IARL各自主效应成立，才执行joint factorial。

## 10. 与当前论文的边界

[Decision] 自2026-07-15起：

- 该idea不再占用当前论文的Contribution slots；
- D13不再是当前active protocol；
- 当前项目回到`fixed-past unified multi-horizon generation`；
- 当前项目不使用rolling-origin、same-target revision或newly arrived data作为核心问题；
- 将来若重启该idea，应从本文件与D13 protocol开始，不从当前fixed-past代码分支直接叠加。

## 11. 外部source seed

本条目使用外部primary/official sources作为起点，Zotero不是完整性依据：

- MQTransformer：https://openreview.net/forum?id=rxF4IN3R2ml
- Forking-Sequences：https://arxiv.org/abs/2510.04487
- N-BEATS-S：https://doi.org/10.1016/j.ijforecast.2022.06.007
- On forecast stability：https://doi.org/10.1016/j.ijforecast.2025.01.006
- Multi-horizon rationality bounds：https://doi.org/10.1080/07350015.2012.634337

[Uncertainty] 该idea目前只有强理论吸引力与明确研究对象，没有内部跨dataset practical headroom证据；因此
它适合作为下一篇论文的高潜力问题，不应被描述为已验证贡献。
