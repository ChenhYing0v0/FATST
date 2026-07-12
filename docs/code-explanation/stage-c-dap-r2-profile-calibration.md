# StageC SC0-DAP-R2 Dataset Profile Calibration

## Purpose

SC0-DAP-R2不再把active parameter matching作为dataset profile选择条件。旧三臂仍保留为
capacity-controlled patch allocation diagnostic，但其人为`d_ff=536/1072`不再决定active carrier。

R2的目标是用有限、自然、validation-only的两阶段grid，为每个dataset选择一次基础结构。该选择不是论文
创新，也不用于比较patch机制；它只建立后续method/control共用的operating point。

## Phase A: Patch Screen

固定`d_model=64, d_ff=128`，比较`patch_num={12,24,48}`。对应history state width和active params可以
自然变化，参数量只记录、不参与排序。

当前真实module graph审计的active-forward params分别为`419,216 / 613,904 / 1,006,160`，明确不是
capacity-matched实验。

每个run仍使用full-720 L1、best-validation checkpoint、max20/patience5，并输出8个dense validation
horizons。对dataset $d$、profile $g$、horizon $h$定义：

$$
r_{d,g,h}=\frac{\mathrm{MSE}^{val}_{d,g,h}}
{\min_{g'}\mathrm{MSE}^{val}_{d,g',h}}-1.
$$

primary score为8个horizon regret的平均；tie依次比较maximum regret、H720 regret和profile name。params、
FLOPs和test均不进入winner排序。

## Phase B: Width Screen

固定各dataset在Phase A选择的patch_num，比较自然width：`D/ff=32/64, 64/128, 128/256`。Phase A的
`D64/ff128` run直接复用，因此只需6个新增run。

`analyze_stage_c_dap_r2b_width_screen.py`同时读取Phase A medium artifacts和Phase B narrow/wide artifacts，
使用相同dense regret selector。Phase B runner从`r2a_summary.json`解析dataset patch mapping，避免shell内
手工重写winner。

## Confirmation Boundary

seed2021只做coarse selection。最终每dataset profile追加seeds2022/2023，只确认finite training和absolute
validation stability，不声称selected-only runs能重新证明relative winner。若不稳定，不根据test或扩大grid
追结果，而是回protocol gate。

## Verification

local checker验证：三个patch均整除720；输出shape为`[B,720,C]`；observed active params确实不同；
synthetic dense selector不读取params；ETTh2 one-batch smoke可early stop并只产生validation artifacts。
