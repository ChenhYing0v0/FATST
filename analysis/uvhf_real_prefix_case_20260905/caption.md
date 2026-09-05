# 建议图注（独立候选，未写入冻结稿件）

**Fig. X | Forecast accuracy and cross-horizon agreement on a real trajectory.**
**a**, Four independently trained horizon-specific DLinear models, with
requested horizons of 96, 192, 336 and 720 steps.
**b**, The frozen UVHF model and its four nested forecast endpoints. Both
systems receive the same 720-step history and predict ETTh2 oil temperature
(OT) from the same forecast origin (16 August 2017, 14:00). Only the last
96 observed steps are displayed. Dark curves indicate ground truth; teal
indicates UVHF. Grey shading denotes history and blue shading the common
96-step forecast interval.
**c**, Enlarged common-prefix trajectories (upper) and within-system
differences from the corresponding 720-step forecast (lower). The faint
envelope spans the four DLinear predictions and is not an uncertainty
interval. UVHF requests produce identical overlapping outputs in the
numerical check. **d**, MSE on each complete requested horizon, normalized
using the training-split standard deviation; percentages denote the
reduction relative to the corresponding DLinear model. Mean CHPD averages
absolute forecast differences across all six horizon pairs, each over its
entire shared prefix, in the original temperature scale. All curves are
unsmoothed. This single validation example was selected from 4,322
origin–variable pairs (2,161 origins; HUFL and OT) using a disclosed joint
accuracy-and-disagreement rule. It illustrates a case rather than population
performance. DLinear controls share the input length with UVHF but retain
their own training protocol; this is not a mechanism-attribution ablation.
Source data and the selection audit are provided with the figure.

## 正文引用时的边界

可以写：该真实案例直接展示，horizon-specific DLinear 对同一未来步的预测随模型目标长度变化；UVHF 给出统一前缀轨迹，且该案例在全部四个完整 horizon 上误差更低。

不能写：这一个案例证明 UVHF 在所有数据集均更精确；所有传统模型都有同样大小的前缀不一致；CHPC 本身导致精度改善；CHPC 只有 UVHF 才能实现；这是 untouched test 或随机代表性案例。零 disagreement 也可由单个 H720 模型裁剪实现，精度优势应结合现有 Main-II 与主表证据讨论。
