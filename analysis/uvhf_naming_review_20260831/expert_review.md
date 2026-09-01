# UVHF/MSD/BCA 命名修改的专家评审

## 结论

**建议：经作者确认后采用。** 相较于原有的 HoriScope/BSCA 命名，修改后的层级关系更加清晰，也更有利于突出论文的核心定位。新方案使标题能够直接指向研究任务，明确区分架构贡献与优化贡献，并统一了 Abstract、Method、Experiments、Figures 和 Appendices 中的术语体系。

## 面向审稿人的优势

1. **任务指向性更强。** 标题直接表明论文研究的是 unified varied-horizon time-series forecasting。审稿人无需先理解一个新造的模型名称，即可快速判断论文主题及其相关性。
2. **贡献层级清晰。** `UVHF = MSD + BCA` 提供了简洁且易于记忆的认知框架：UVHF 表示完整的预测框架，MSD 表示前向推理所使用的 decoder architecture，BCA 表示与之配套的 training strategy。
3. **Claim 与证据层级更加一致。** Method、Ablation 和 Generalization Studies 现在均指向其实际评估的对象。Decoder replacement 被准确描述为使用 BCA 训练的 MSD transfer，从而避免审稿人误认为 BCA 属于 inference-time decoder 的组成部分。
4. **Task 与 model 的语义得到区分。** 通用研究任务始终写作 `unified varied-horizon forecasting`，而 `UVHF` 仅用于指代本文提出的完整框架。这一规则解决了同一缩写同时表示 forecasting setting 和具体模型时可能产生的歧义。
5. **视觉表达保持一致。** 所有包含旧模型名称的图片均基于相同的 source data 重新生成；仅展示组件的图片也通过 caption 与新的术语层级完成了同步。

## 仍需关注的风险与边界

1. **缩写的通用性风险。** `UVHF` 与任务名称天然接近，其清晰性依赖于全文持续遵守术语契约。后续修改中不应重新引入 `the UVHF task`、`UVHF setting` 或 `UVHF workflow` 等表述。
2. **Novelty claim 的边界。** 既有 flexible-horizon 方法，尤其是 ElasTST，使本文不宜直接宣称 UVHF 首次创立了 multi-horizon forecasting paradigm。当前手稿将贡献限定为系统化任务定义、decoder-side problem analysis 和针对性的 forecasting framework，这一边界较为稳妥。
3. **BCA 名称的特异性。** `Balanced Co-Adaptation` 简洁，但脱离上下文时可能显得较为宽泛。因此，其首次定义需要明确指出 co-adaptation 发生在 scope-indexed forecast field 与 allocation process 之间。
4. **MSD 与 multi-scale encoding 的区分。** 审稿人可能会将 MSD 初步理解为传统的 multi-scale history modeling。Related Work 和 Discussion 需要继续保持当前的核心区分：scope 描述 output-side latent-state reuse extent，而 scale 通常描述 input resolution 或 frequency structure。

## 验证摘要

- 原始投稿文件夹已经过 checksum 检查，文件内容未发生变化。
- 修改后的 TeX 和编译 PDF 中均不存在旧手稿术语 `HoriScope`、`ISCF` 和 `BSCA`。
- 未检测到 task/model 语义混用模式。
- 修改稿共包含 94 个蓝色高亮区块。
- 两个修改后的绘图脚本均通过 Nature Figure static preflight。
- 手稿编译结果为 24 页，所有 citations 和 cross-references 均已正确解析，未出现 overfull 或 underfull box warning。
- 已完成全部 24 页的视觉检查；修改后的 labels、tables、captions 和 appendix figures 均不存在裁切、重叠或明显排版问题。
