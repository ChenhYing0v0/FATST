# StageC SC2-PCC Step5 Checker 说明

## 模块作用

`scripts/check_stage_c_sc2_pcc_step5.py`是`SC2-PCC-v0`的local theory/synthetic checker。它不实现模型训练，
也不读取benchmark数据；作用是把Step 5中的梯度、prefix measure、projectivity与router可恢复性写成可重复检查。

## 输入与输出

脚本无数据输入，默认输出到`analysis/stage_c_sc2_pcc_step5_theory_20260716/`：

- `theory_and_synthetic_cases.csv`：每行一个检查项；
- `local_gate.json`：汇总case数量、gate状态与授权边界。

CSV列定义：

- `case`：检查项稳定标识；
- `value`：由合成tensor或自动微分得到的数值；
- `threshold`：预注册通过条件；
- `pass`：`value`是否满足该条件。

## Forward And Gradient Flow

`gradient_checks`构造`arms [B,T,S]`与`logits [B,T,S]`，其中`B=7`、`T=19`、`S=5`。
`policy=softmax(logits)`，`fused=(policy * arms).sum(-1)`得到`[B,T]`预测。脚本分别对plain fused loss与
PCC objective调用`torch.autograd.grad`，再与解析式逐元素比较最大绝对差。

PCC capability由同一`arms`和`target [B,T]`计算，但在softmax前执行`detach`，因此skill loss只向arms传播，
route KL只把`policy`拉向固定的same-forward capability target。checker验证的是output-level梯度系数，不代表
shared network parameter的梯度不会跨target抵消。

## Prefix And Synthetic Cases

`prefix_measure(T)`返回`omega [T]`，其中

$$
\omega_t=T^{-1}\sum_{H=t}^{T}H^{-1}.
$$

checker在`T=720`上验证dense prefix AUC恒等式、权重归一化与单调性，并直接比较full output的多个prefix crops。

`synthetic_router_recovery`构造`features [96,64,4]`，四个feature依次为history coordinate、target coordinate、
二者乘积与constant。固定teacher产生`capability [96,64,5]`；一个无bias linear policy从零初始化训练1200步。
KL与argmax accuracy检查policy是否能恢复同时依赖history和target的scope choice，crossing statistics则防止
合成teacher退化为只依赖单一axis。

## Code-Theory Consistency

- Intended theory：plain fused loss的arm credit被policy缩放；PCC提供skill floor与capability-aligned router梯度。
- Code realization：float64 autograd与解析式最大绝对差直接对照，15个case全部通过才返回成功。
- Proxy boundary：合成router没有真实forecast noise，且没有shared PCSD parameters；它只能证明代数与可表达性。
- Falsification：任一梯度identity、prefix identity、projectivity或crossed router recovery失败，Step 5即不通过。

该checker通过后只授权Step 6 design，不授权PCC实现、远程训练、test或confirmation seeds。
