# SC1-PLGO Step 6 Design Audit Code Explanation

## Scope

`scripts/check_stage_c_plgo_step6_design.py`是CPU-only float64 design harness。它没有forecast model class、
optimizer、dataset或checkpoint I/O，只验证PAF tensor contract、prefix invariance、parameter budget与candidate
control边界。

## Computation Flow

1. 从Step 5 proof module复用`restricted_global_nested_basis`，得到
   `Q: [T,T]`与`atoms: list[Atom]`；不复制RGNB implementation。
2. `canonical_atom_descriptors`把每个atom转换为`d_j: [8]`：
   global/detail one-hot、normalized start/end/length、depth、within-node order和group-size fraction。
3. `projective_atom_functional`执行：

```text
hidden [B,C,R] @ branch_weight [R,K] -> latent [B,C,K]
descriptor [N,8] -> tanh MLP -> trunk [N,K]
einsum("bck,nk->bcn") -> coefficients [B,C,N]
```

4. full path用`Q [T,T] @ alpha [B,C,T]`；prefix path仅对`active_indices(atoms,H)`重新调用同一PAF，
   再用`Q[:H,active]`合成。
5. 同时打乱active coefficient与Q columns的顺序，验证set ordering不会改变output。

## Artifact Definitions

| Artifact | Fields | Meaning |
| --- | --- | --- |
| `projectivity_checks.csv` | `length,global_rank,horizon,active_atoms` | case identity与active support count |
|  | `coefficient_subset_max_abs` | active-only coefficients vs full coefficients indexed subset |
|  | `prefix_reconstruction_max_abs` | active synthesis vs full-output crop |
|  | `active_order_permutation_max_abs` | paired coefficient/Q-column permutation invariance |
| `parameter_budget.csv` | `dataset,history_width,design,trunk_width,descriptor_dim` | frozen profile与candidate size |
|  | `a6/paf_readout_parameters,paf_to_a6_ratio` | exact algebraic parameter counts |
|  | `output_rank_upper_bound,full_affine_output_class,exact_all_a6_table_containment` | function boundary |
| `candidate_control_matrix.csv` | arm、role、history path、geometry、function boundary、status | Step 6 control decision |
| `theory_gate.json` | algebra/narrative/internal gates、decision、rollback、next diagnostic | machine-readable cursor |

## Code-Theory Consistency

- intended theory：atomwise $F(h,d_j)$应使active subset不改变保留coefficients；
- code realization：每次只对传入descriptor rows逐行执行shared MLP，完全没有atom-axis reduction；
- verified proxy：synthetic nonlinear branch/trunk可以证明tensor invariance，不能证明forecast effectiveness；
- function boundary：linear-in-history effective rank不超过256；free atom table才exact包含A6；
- falsification：任何subset/prefix gap超过`1e-10`会阻断该tensor contract；本轮max为`4.547e-13`；
- narrative boundary：external primitive overlap只收紧claim，不自动否决task-specific contribution；algebra pass
  仍不能覆盖B11/B14 internal negative evidence，因此D7 attribution在Step 7前保持mandatory。

## Reproduction

```bash
/opt/anaconda3/envs/r2026-fsa/bin/python -m py_compile \
  scripts/check_stage_c_plgo_step6_design.py
/opt/anaconda3/envs/r2026-fsa/bin/python \
  scripts/check_stage_c_plgo_step6_design.py
```
