# SC1-D8 Step 7A Local Gate

- decision: `step7a_pass`
- shape-prefix: `True` (210/210)
- gradient/patch interface: `True` (35/35)
- descriptor/basis: `True`
- horizon path: `True`
- full-prefix max abs: `2.384e-06`
- flatten/block-sum max abs: `5.722e-06`
- float32 basis orthogonality max abs: `5.126e-06`
- test used: `false`
- forecast training run: `false`

数值协议说明：初始1e-6检查只被float32长向量累积误差触发；shape、gradient与float64参考构造均正常。项目既有projectivity协议使用1e-5，因此在remote前统一修正阈值，不涉及训练结果或性能选择。

该gate只证明joint-training implementation、projectivity、patch information path与gradient contract可执行；不构成forecast effectiveness证据。
