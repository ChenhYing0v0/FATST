# Solar H3B Training Result and Direct-Test Prelaunch

## Decision summary

| Field | Value |
| --- | --- |
| `current_step` | H3B training artifact audit complete；direct official-test prelaunch |
| `candidate_version` | `ISCF-BSCA-MAIN-v1-solar-h3b-test-informed-20260801` |
| `training_matrix` | 4/4 complete；test=0/4 |
| `test_matrix` | 4 checkpoints × 4 horizons = 16 cells |
| `checkpoint_selector` | validation four-H mean MSE；best-val |
| `decision` | execute complete four-checkpoint test audit immediately |

[Fact] Remote H3B在commit `4ac5650`上4/4完成；GPU 0/1/2均空闲。Artifact analyzer确认4/4 checkpoints、training logs、four-H validation MSE/MAE、effective configs、initialization contracts与model diagnostics完整，numeric health通过，test artifacts为0。

Validation aggregate仅记录checkpoint provenance，不用于profile selection：lr2e4=`0.132727`、lr4e4=`0.131754`、lr3e4+dropout4=`0.132232`、lr3e4+rank64=`0.133824`。四个checkpoints全部直接进入test。

Manifest=`h3b_checkpoint_manifest.csv`，SHA256=`8d63967e00cef3c69fbbfd22ea414e04d261008ade8e2a64a934fc00dc450f23`。Test必须达到4/4 checkpoints和16/16 cells，验证dense CSV、candidate/trial/profile/seed provenance、invariant、NPZ以及test前后checkpoint hash。Partial ranking禁止。

H3B是当前LR/regularization邻域的terminal batch。Test后将H3B结果与H1/H2/H3A全部Solar profiles合并，按four-H official-test mean MSE冻结最终single-seed profile。若最低值仍高于0.192，则记录`bounded_HPO_target_narrowly_missed`，停止同一邻域继续搜索。

Decision=`Solar_H3B_4_of_4_training_complete_direct_test_prelaunch`。
