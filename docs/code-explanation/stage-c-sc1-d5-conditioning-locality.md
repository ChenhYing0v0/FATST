# SC1-D5 Conditioning-Locality Diagnostic Code Explanation

## Computation Flow

forecast model未修改。D5继续读取frozen A6的`memory: [B,C,P,D]`，展平为
`features: [BC,PD]`；target为`[BC,720]`。与D4相同的`GroupedNonlinearHead`输出
`coefficients: [BC,720]`，再通过`prediction = coefficients @ basis`得到`[BC,720]`。

新增的`block_diagonal_basis()`把720个future positions切为连续blocks，并在每个square block内填入
orthonormal DCT-II或fit-only PCA transform。不同blocks没有非零交叉项，因此整体仍满足
`basis @ basis.T = I`；任意prefix只激活与其相交的blocks。

## Artifact And Selection Logic

worker复用D4 artifact schema（`d4_probe_metrics.csv`、`d4_basis_geometry.csv`、`d4_metadata.json`），因为tensor、
training与invariant contract完全相同；D5 analyzer将其聚合为独立的`d5_*`结果。fit-only selector先限制
`active_atoms_h48 <= 96`，再按off-diagonal ratio、top-16 capture与family name排序。validation metrics不进入
选择。

`d5_local_family_comparisons.csv`与`d5_local_family_summary.csv`额外报告全部预注册local families相对三controls
的效果，用于审计fit-only selector是否掩盖明显反例；它们不改变预注册gate，也不能用validation事后挑选method。

## Code-Theory Consistency

- intended theory：local orthogonal transform可能在prefix selective synthesis与coefficient conditioning之间
  形成Pareto improvement；
- code realization：block support严格local，DCT/PCA分别提供fixed与fit-adaptive conditioning controls；
- proxy：当前head仍计算全部720 coefficients，所以active-atom count只是algebraic support proxy，不是实际速度；
- falsification：fit-only selected local若不能稳定改善balanced并接近DCT/PCA，则SC1-CLG不通过problem gate；
- non-claim：block PCA/DCT不是paper method，也不主张其construction novelty。
