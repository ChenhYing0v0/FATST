# SC-D19-IFC Step 6 v1.1 History-Contract Repair

## 1. What was found

Step 7A code-theory audit发现，已提交的`SC-D19-IFC-control-v1`把upstream IF默认
`lookback=96`直接写成了本地`history_spectrum_bins=49`。但冻结A6 natural carrier的实际输入是
`seq_len=720`：

$$
X[B,720,C]\rightarrow M[B,C,P,D]\rightarrow h[B,C,R].
$$

若IF skip只读取96 points，而A6 Encoder读取720 points，虽然所有arms仍能运行，但“IF与matched direct读取
same A6 history contract”的表述不成立，且96-point窗口的选择没有本项目证据。

## 2. Failure attribution

| Cause | Judgment |
| --- | --- |
| `hypothesis_false` | no；尚未训练 |
| `intervention_point_wrong` | no evidence |
| `readout_or_head_design_wrong` | no evidence |
| `optimization_or_numeric_pathology` | no；尚未运行 |
| `capacity_control_explains` | not evaluated |
| design/protocol mismatch | **yes**；upstream default被误当作本地matched input contract |

因此v1仅被`superseded_before_implementation`，不构成D19方向或IF control失败。

## 3. v1.1 repair

`SC-D19-IFC-control-v1.1`固定：

1. A6 Encoder、IF skip与matched direct均读取相同$X[B,720,C]$；
2. history rFFT为$361$ bins；
3. future polar spectrum仍为$361$ bins并显式`irfft(n=720, norm="ortho")`；
4. IF/no-skip module shape与parameters完全相同，no-skip只把361-bin amplitude/phase替换为zero；
5. matched direct读取$[h,A_x,\Phi_x]$，逐profile重新匹配参数。

| $R$ | IF params | Direct width | Direct params | Gap |
| ---: | ---: | ---: | ---: | ---: |
| 768 | 9,161,787 | 4,143 | 9,160,893 | 0.00976% |
| 1,536 | 13,880,379 | 4,659 | 13,879,881 | 0.00359% |
| 3,072 | 23,317,563 | 5,164 | 23,316,180 | 0.00593% |

参数量仍不参与dataset profile选择；matched direct继续是capacity/nonlinearity hard control。

## 4. Decision

旧`configs/stage_c_d19_if_control_step6.json`保留为superseded audit artifact，不再进入实现或训练。活动contract
改为`configs/stage_c_d19_if_control_step6_v1_1.json`。

该修复发生在任何D19 training、remote或test access之前，不是观察结果后的调参。Step7A可基于v1.1继续；
remote、official test与paper method仍为false。
