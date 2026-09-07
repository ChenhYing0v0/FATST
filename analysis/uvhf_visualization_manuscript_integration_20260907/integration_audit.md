# Submission 可视化整合审计

用户授权按已讨论方案直接修改UVHF_KBS_submission。任务为manuscript/experiments+caption、英文期刊写作；使用nature-writing与tsf-writing，PDF技能用于已编译文件的只读渲染检查。没有新训练、test访问或改动冻结图。

## 论证与段落安排

一句话论点：一个选定真实validation窗口可直观展示UVHF的预测准确性和CHPC同时实现，总体效果继续由Main-I/Main-II支持。第一段承接aggregate comparisons、介绍TimeMixer四个独立模型并观察公共前缀；第二段解释反复低谷并指向Appendix C。CHPC、UVHF、TimeMixer、forecast origin、future step等沿用稿件定义。不给一致性与精度之间作因果归因。

新增5.8 Visualization of forecasting performance and prefix consistency，位于Generalization studies后。图为Figure8。caption定义step69的Δ、六对horizon平均分歧、完整horizon MSE降低与选定样本属性。Appendix A Implementation Details补充visualization checkpoint与Main-I来源不同、seed、validation checkpoint选择与后验选样；Appendix C首句回应正文5.8并定位为额外UVHF案例，不冒称跨数据集baseline对比。

## 连贯性与范围

Main-I/Main-II数值、模型与公式、Introduction、原有Figure1–7/C.1内容、Discussion结论均未改动。UVHF与TimeMixer描述为同origin和future targets，未误称相同输入历史。图8与全配套文件通过旧freeze_manifest逐文件SHA256验证。旧冻结记录表示2026-09-06历史状态，本次integration_manifest记录已完成插入的最新状态。

## 编译与版式

最终以submission目录为工作目录运行latexmk -pdf -interaction=nonstopmode -halt-on-error elsarticle-template-num.tex，生成26页PDF。最初新图漂移至Discussion之后；只为相邻Figure6、7和8放宽浮动位置至[!htbp]，并在Discussion前使用FloatBarrier，最终5.8与Figure8连续出现在PDF第17/18页（印刷页16/17），Discussion随后。未缩小既定图，Figure8独占图页以保持标注可读。避免正文小节与Figure8之间插入Discussion。

交叉引用5.8/Figure8/AppendixC已解析；修正了模板附录引用自带Appendix前缀导致的重复。图页与正文页已渲染检查，无裁切、图注溢出或遮挡。最终无Overfull、undefined references或控制序列错误。保留四条hyperref书签数学符号提示，不影响正文。没有新增引用文献。

最终PDF为UVHF_KBS_submission/UVHF_KBS_submission.pdf。latex_build.log保存构建信息，pdf_page_17/18.png为审阅快照，integration_manifest.json为新TeX/PDF/图hash。辅助编译文件已清理；未同步覆盖其他历史稿件目录。

结论：本轮插入及语言、数值边界、引用与版式检查通过；不等同于期刊外审通过。单选案例且TimeMixer该窗口表现弱的事实保持原始审计记录，新增正文未将其提升为代表性或总体效应。
