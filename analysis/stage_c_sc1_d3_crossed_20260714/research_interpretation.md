# SC1-D3 Crossed Basis-Group Diagnostic: Research Interpretation

## 1. Question And Decision

[Strong Evidence] SC1-D3补齐D2缺失的`random basis × random group` cell后，balanced-interval basis在当前
frozen-memory grouped nonlinear probe family中表现为**独立、跨group context稳定的main effect**，而不是
只在true depth grouping下出现的interaction artifact。

- `decision = basis_main_effect_supported_return_step4`；
- 只授权Contribution 1返回Step 4做source-informed idea design；
- 不授权直接实现decoder、MIPR、MoE或修改Encoder。

## 2. Artifact And Validity Audit

| Check | Result |
| --- | --- |
| Missing-cell fits | 45/45 complete |
| Primary paired units | 15/15：5 datasets × 3 checkpoints |
| D2/D3 metadata | 15 + 15 complete；contract hashes一致 |
| Test split | 未加载 |
| Forecast model | frozen；未更新 |
| Official validation | 仅final evaluation；不参与early stopping |
| Orthogonality/Parseval | pass，threshold `1e-5` |
| Numeric stability | 45/45 finite |
| Best epoch | 15–76；0/45停在`max_epochs=120` |
| Local recomputation | 与remote decision一致 |

[Fact] D3 head params随frozen memory width变化：ETTh1 `564,784`、ETTm2 `1,105,456`，其余三套
`294,448`。params差异不参与profile选择或跨dataset结论；每个dataset内四个factorial cells保持同一
GroupedNonlinearHead capacity与optimization contract。

## 3. Preregistered Factorial Results

| Dataset | Basis main MSE reduction | True-group conditional | Random-group conditional | Basis main MAE reduction | Interaction guard |
| --- | ---: | ---: | ---: | ---: | --- |
| ETTh1 | +3.7081% | +3.5728% | +3.8433% | +2.1638% | pass |
| ETTh2 | +5.3559% | +5.0783% | +5.6326% | +3.3134% | pass |
| ETTm1 | +0.5834% | +0.5758% | +0.5910% | +0.8731% | pass |
| ETTm2 | +3.8430% | +4.9449% | +2.7284% | +3.3109% | pass |
| Weather | +1.0119% | +1.3221% | +0.7008% | +1.8658% | pass |
| **Macro** | **+2.9174%** | **+3.1164%** | **+2.7181%** | **+2.3098%** | **5/5 pass** |

Gate consistency：

- basis main：5/5 datasets均有3/3 checkpoint effects为正；43/45 structure blocks为正；
- true-group conditional：5/5 datasets均3/3 checkpoints为正；45/45 blocks为正；
- random-group conditional：5/5 datasets均3/3 checkpoints为正；43/45 blocks为正；
- MAE guard通过；
- 5/5 datasets满足$|I_{BG}|\le|\Delta_B|$；
- group main effect的macro MSE reduction为`-0.0821%`，与D2“exact depth grouping不成立”的结论一致。

## 4. What The Result Means

[Strong Evidence] D2的basis信号没有被random grouping消除。true basis在true group下带来`3.1164%` MSE
reduction，在random group下仍有`2.7181%`；两者接近，且interaction不主导。因此当前最合理的归因是：
**balanced interval coordinate geometry改变了target readout的可学习性或regularization geometry**，而不是
11个depth groups本身提供了有效conditional computation。

[Fact] 这也进一步排除了一个简单叙事：不能再把收益写成“multi-scale groups分别学习不同future scales”。
group main effect为负且D2 mandatory group gate已失败。保留下来的问题应改写为：

> 为什么prefix-local、balanced-interval coordinates能让同容量nonlinear readout更容易从frozen ordered
> memory学习full future，并且这种优势能否被转化为native unified-horizon operator，而不是静态basis trick？

## 5. Alternative Explanations And Self-Critique

[Hypothesis] 当前basis优势可能来自以下一个或多个因素，D3尚未区分：

1. **target covariance conditioning**：true basis更接近future target的可预测/低相关坐标；
2. **energy compaction**：少数interval coefficients承载更高、也更稳定的predictable energy；
3. **local-support regularization**：coefficient errors映射回time domain时具有局部支持，改善有限样本泛化；
4. **prefix compatibility**：balanced intervals与requested prefix有结构关系，但本轮只训练/评估full H720，
   尚未证明dense-horizon统一预测收益；
5. **random-control difficulty**：random orthogonal coordinates可能是不自然的强负control，胜过它不等价于
   胜过DCT、Fourier、Haar、learned PCA或data-adaptive bases。

[Speculative] 因而“basis本身就是Contribution 1”仍然不成立。balanced interval construction接近已有
Haar/wavelet/tree transform家族，novelty pressure很高；只有找到与`multi-horizon unified forecasting`
直接相连、且超越standard structured bases与matched regularization controls的机制，才有进入Step 5的价值。

另外，D3只使用三组diagonal seed pair，没有跑完整的3×3 random-basis × random-group cross。由于两个
conditional effects在5/5 datasets方向一致且interaction guard全过，这一限制不阻断当前Step 4授权；但若
后续claim依赖精确interaction magnitude，应补full cross或改用更多独立structure draws。

## 6. Failure Attribution Boundary

本轮没有`optimization_or_numeric_pathology`：loss finite、best epochs未触顶、invariants与local recomputation
均通过。D3支持的是probe-family层面的`basis_main_effect`，不是end-to-end method effectiveness。

- `hypothesis_false`：exact balanced-depth grouping，已由D2关闭；
- `hypothesis_supported_for_step4`：balanced interval basis geometry具有独立probe main effect；
- 仍未测试：standard structured-basis controls、mechanism attribution、dense-horizon prefix behavior、
  end-to-end decoder收益、SCI novelty。

## 7. Next Research Step

正式返回**Step 4 source-informed mechanism audit**，不直接进入implementation：

1. external primary-source audit：Haar/unbalanced Haar、wavelet forecasting、functional bases、output
   decorrelation/whitening、multiresolution neural operators与prefix-consistent forecasting；
2. offline mechanism decomposition：比较true basis与DCT/Haar/PCA/learned orthogonal controls的target
   covariance、energy compaction、coefficient predictability与prefix locality；
3. 只有当某一机制同时具有`multi-horizon native relevance + matched-control advantage + novelty boundary`，
   才在Step 4提出新SC1 candidate并进入Step 5 theory feasibility；
4. 若standard structured bases或简单whitening完全解释收益，则回滚Step 2，basis只保留为training/readout
   control，不占Contribution 1。

SC2-MIPR继续held，Encoder与MoE继续冻结。
