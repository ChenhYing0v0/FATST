# Section 3 Problem-Evidence Final Figures 代码说明

## 1. 作用与边界

`scripts/plot_intro_problem_evidence_final.py`只读取冻结的validation source data，
生成两张Section 3问题证据图。它不训练模型、不访问test、不重新选择sample，
也不修改任何forecast values。

```text
ETTh2 prefix source CSV + summary + pair metrics
  -> Figure 2 SVG/PDF/PNG/TIFF

ETTm2 selected region-MSE CSV + summary
  -> Figure 3 SVG/PDF/PNG/TIFF
```

## 2. Figure 2 source tensors and quantities

`selected_forecast_data.csv`每行对应一个relative step：

| Column | Source | Meaning |
| --- | --- | --- |
| `relative_step` | validation window index | history为负，future为1--720 |
| `phase` | artifact role | `history`或`future` |
| `history` | inverse-scaled `history[o,:,c]` | selected observed values |
| `ground_truth` | inverse-scaled H720 target | selected future truth |
| `prediction_hH` | inverse-scaled DLinear prediction | horizon H model output |
| `difference_hH_vs_h720` | prediction H minus prediction H720 | same-step disagreement |

Final Figure 2只展示last-48 history与first-96 future，因为96是四个requested
horizons的共同overlap。panel a把history、ground truth和四条horizon-specific
predictions整合在同一个hero axis中。四条prediction都使用thin solid line，并
同时用固定颜色、不同marker、稀疏错位marker positions和subtle white separation
stroke编码；$H=720$作为略粗但较低z-order的reference，避免遮挡其余horizons。
预测主线宽度为0.82 pt，$H=720$为0.95 pt；marker间隔从12放宽至18个future
steps，white under-stroke仅增加0.38 pt，以降低重叠区的视觉墨量。panel内的
mean $|\Delta|$摘要直接从source predictions相对$H=720$重新计算，不使用图像
像素或手工抄写。

`pair_metrics.csv`的`nchpd_l1`来自所有validation origins、future overlap steps与
channels的standardized mean absolute difference。panel b将六个pair values映射到
紧凑3×3 upper-triangular matrix；blank lower triangle表示未重复展示对称
comparison。最终Figure 2只有两个顶底对齐的panels，不再使用上下堆叠的独立
raw-difference subplot。

## 3. Figure 3 source tensors and quantities

`selected_region_risk.csv`有60行；文件名沿用冻结artifact contract，其中`mse`
column才是manuscript与figure使用的正式统计量：

```text
5 sharing extents × 12 future regions
```

每行`mse`为selected origin上对应scale、60-step region及all channels的mean
squared error。为避免把fixed reference误读为missing data，panel a使用每个
region自身的最低MSE作为视觉参考：

$$
E_{s,b}
=
\frac{\operatorname{MSE}_{s,b}-\min_{s'}\operatorname{MSE}_{s',b}}
{\min_{s'}\operatorname{MSE}_{s',b}}\times100\%.
$$

panel a绘制$E_{s,b}$，outlined square标记
$\arg\min_s\operatorname{MSE}_{s,b}$。因此每列只有
region winner为0；$s=720$只在regions 10--12保持0，而不是像旧版
fixed-reference encoding那样整行恒为0。旧版白色$s=720$行来自
$(\operatorname{MSE}_{720,b}-\operatorname{MSE}_{720,b})/
\operatorname{MSE}_{720,b}=0$，并非missing values。

panel b仍保留sample-best fixed extent $s^\mathrm{fixed}=720$作为业务上更直接的
统一decoder reference，其bar height为：

$$
G_b
=
\frac{
\operatorname{MSE}_{s^\mathrm{fixed},b}
-\min_s\operatorname{MSE}_{s,b}
}{
\operatorname{MSE}_{s^\mathrm{fixed},b}
}\times100\%.
$$

bar与baseline marker颜色来自region winner。winner恰为fixed s720时$G_b=0$，
仍用对应的rose square显示winner identity，避免零高度bar使region 10--12
不可见。虚线表示12个regions的mean gain，即descriptive headroom 8.1%。

## 4. Export and QA

脚本固定：

- Python/matplotlib backend；
- sans-serif publication fonts；
- editable SVG text与PDF Type 42 fonts；
- exact 183 mm double-column output width，不依赖tight-bbox再缩放；
- PNG 300 dpi；
- LZW TIFF 600 dpi；
- source-data/selection/claim boundary写入`figure_manifest.json`。

Nature static source audit为13 PASS、1 WARN、0 FAIL。WARN只因validator不能从变量
表达式静态解析183 mm width；PDF media box实测518.74 pt=183 mm。

## 5. Code-theory consistency

Intended claim：

1. independently optimized horizons可在相同future steps上分歧；
2. matched fixed sharing extents的region-wise MSE ordering可发生明显变化。

Code realization：

- Figure 2把single selected example与all-validation heatmap分离，并在一个
  trajectory panel中保留raw curves与mean-difference summary；
- Figure 3只展示matched neutral decoder region-wise MSE，不出现ISCF/BSCA；
- maximum selection与same-validation descriptive oracle均在manifest和caption
  中公开。

仍然只是proxy：

- selected examples不估计prevalence；
- region oracle不是learned out-of-sample policy；
- DLinear evidence不能代表所有varied-horizon models。

若source alignment、test boundary、CSV completeness、SVG/PDF/TIFF export或
winner reconstruction任一失败，figure package不得进入paper draft。
