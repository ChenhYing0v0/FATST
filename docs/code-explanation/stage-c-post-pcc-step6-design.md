# SIFF/MCCA Step 6 Checker 代码说明

## Purpose

`scripts/check_stage_c_post_pcc_step6.py`是design checker，不是forecast model。它读取冻结config与five-dataset natural
profiles，验证Step 7A实现前必须成立的数学、shape、数值与control contracts；不读取dataset、checkpoint或test。

## SIFF Tensor And Control Flow

checker按以下contract构造synthetic modes：

```text
hidden [B,C,R]
 -> W [Q,D,R,K], b [Q,D,K]
 -> component modes [B,C,Q,D,K]
 -> phi(scales) [S,Q]
 -> scale-indexed modes [B,C,S,D,K]
```

它检查$Q=1$ containment、constant same-parameter control可合并回Q1、ordered/permuted scale semantics不同，以及
constant/linear basis的归一化。parameter accounting按每个dataset的真实`R=patch_num*d_model`计算SIFF field、Q1-wide
和independent-scope matched ranks；rank由闭式nearest-integer rule得到，不参与profile选择。

## MCCA Numerical Flow

checker把6个synthetic batch-channel instances与$T=720$ targets展开为rows：

```text
capability [N*T,S]
 + projective row mass [N*T]
 -> ramped reference measure [N*T,S]
 -> PCC-equivalent column mass [S]
 -> 64-step log-domain Sinkhorn
 -> allocation [N*T,S]
```

它分别在float64/float32检查row/column marginals、MCCA/PCC column-mass equality、MCCA相对PCC mixed credit的KL
advantage、global starvation floor及stop-gradient skill path的finite nonzero gradient。

## Outputs And Boundary

- `design_cases.csv`：每个数学/contract case的value、threshold与pass；
- `parameter_controls.csv`：每dataset的field parameters、matched ranks与relative gaps；
- `local_gate.json`：22/22 aggregate gate及Step7A authorization。

通过只授权Step7A local implementation。checker不证明source-level novelty、真实optimization稳定性、validation/test
performance或paper contribution成立。
