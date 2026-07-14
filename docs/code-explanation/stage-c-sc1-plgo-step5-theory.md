# SC1-PLGO Step 5 Theory Audit Code Explanation

## Scope

`scripts/check_stage_c_plgo_step5_theory.py`是CPU-only float64 proof harness，不是forecast model。它不读取
dataset或checkpoint，不更新任何model参数，只生成Step 5 algebra/function-class artifacts。

## Forward Construction Flow

1. `dct_prototypes(T, r_g)`生成`G: [T,r_g]`，列为orthonormal DCT-II global modes。
2. `restricted_dct_coordinates(T,s,e,r_g)`在interval $[s,e)$上生成
   `local_coordinates: [|I|, min(|I|,r_g)]`。它利用$\cos(kx)=T_k(\cos x)$，先把$\cos x$局部rescale到
   $[-1,1]$，再构造Chebyshev coordinates；这与restricted DCT具有相同span，但数值更稳定。
3. `build_node(...)`递归生成`V_I`，把children scaling bases block-diagonal成
   `child_scaling: [|I|,d_L+d_R]`，再用`coordinates=child_scaling.T @ scaling`表示parent space。
4. 对`coordinates.T`做full SVD；其right-nullspace生成
   `detail: [|I|,d_L+d_R-d_I]`，即$(V_L\oplus V_R)\ominus V_I$。
5. `collect_details(...)`把root scaling与全部interval details嵌入full domain，得到
   `synthesis Q: [T,T]`和逐column `Atom(kind,depth,start,end)`。
6. `active_indices(atoms,H)`只选择`start < H`的global/local atoms；
   `Q[:H,active] @ alpha[active]`生成`prefix: [H]`。

## A6 Morphism Check

随机但fixed-seed的A6-style path为：

```text
hidden [17]
  -> coefficient_map [K,17]
  -> coefficient [K]
  -> output_basis [T,K]
  -> reference [T]
```

proof harness计算：

```text
transformed_basis = Q.T @ output_basis   # [T,K]
transformed_bias  = Q.T @ temporal_bias  # [T]
alpha             = transformed_basis @ coefficient + transformed_bias
reconstructed     = Q @ alpha            # [T]
```

`reconstructed == reference`验证A6可无dense bypass地morph到RGNB coordinates。由于`Q` square orthonormal，
此检查同时证明M0只是bijective reparameterization，不能证明新function。

## Control Audits

### Overcomplete frame

`frame_control`构造`S=[G,L]: [T,T+r_g]`。它验证`SS.T=I+GG.T`、canonical reconstruction，并显式构造

```text
kernel = [I; -L.T @ G]  # [T+r_g,r_g]
```

使`S @ kernel = 0`。因此frame稳定不等于coefficients identifiable。

### Independent support groups

`function_class_budget`按`global_root/detail_depth_*`统计atom counts。对每组计算exact包含rank-256 A6 block
所需的`min(group_size,256)`。若每组size不超过256，independent group maps即为full-row-rank blocks，整体等价
full affine。

## Artifact Definitions

| Artifact | Columns / fields | Source and meaning |
| --- | --- | --- |
| `basis_checks.csv` | `length, global_rank, atoms, global_atoms, detail_atoms, depth_groups` | construction shape/counts |
|  | `orthogonality_max_abs` | `max(abs(Q.T@Q-I))` |
|  | `nested_inclusion_max_abs` | parent scaling投影回children union的max error |
|  | `global_projector_max_abs` | RGNB root与DCT root projectors的max difference |
|  | `detail_prototype_moment_max_abs` | `max(abs(Q_detail.T@G))` |
|  | `support_max_abs` | detail declared interval之外的max absolute value |
|  | `a6_morphism_max_abs, prefix_max_abs` | full/prefix reconstruction errors |
| `prefix_checks.csv` | `horizon, active_atoms, inactive_atoms, active_to_horizon_ratio` | support-intersection counts |
|  | `conservative_active_bound, bound_pass` | $H+r_g(\lceil\log_2T\rceil+1)$ bound and result |
|  | `prefix_max_abs` | active-only prefix vs full reference error |
| `active_bound_checks.csv` | `horizon_cases, bound_pass, largest_bound_excess` | each case over all $H=1,\ldots,T$ |
| `conditioning_checks.csv` | `raw/stable_*_condition, *_nodes_above_1e12` | restricted DCT vs stable local-chart conditioning |
| `frame_control_checks.csv` | frame bounds、redundancy、kernel、coherence、reconstruction gaps | naive union stability/identifiability |
| `candidate_matrix.csv` | four candidate/control gate fields | Step 5 candidate boundary |
| `function_class_budget.json` | group sizes/caps、parameter counts/ratios、class booleans | T720 no-go evidence |
| `theory_gate.json` | all gates、decision、next question、rollback | machine-readable Step 5 conclusion |

## Code-Theory Consistency

- intended theory：一个square nested basis同时保留global smooth subspace与local interval complements；
- code realization：RGNB严格按$V_I$与$(V_L\oplus V_R)\ominus V_I$构造，并审计support、moments与projectors；
- proxy boundary：DCT只代表一种global smooth subspace；本轮不证明其forecast最优；
- unproven boundary：shared atom-conditioned generator尚不存在，selective coefficient computation与实际
  inductive bias均未验证；
- falsification：任一algebra gap超过`1e-10`、frame/control解释candidate、或Step 6无法隔离dense-equivalence，
  都阻止method进入Step 7。

## Reproduction

```bash
/opt/anaconda3/envs/r2026-fsa/bin/python -m py_compile \
  scripts/check_stage_c_plgo_step5_theory.py
/opt/anaconda3/envs/r2026-fsa/bin/python \
  scripts/check_stage_c_plgo_step5_theory.py
```
