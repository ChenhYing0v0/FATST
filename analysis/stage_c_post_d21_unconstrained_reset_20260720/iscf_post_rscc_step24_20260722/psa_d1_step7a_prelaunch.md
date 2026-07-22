# SC-ISCF-PSA-D1 Step7A and Prelaunch Gate

## 1. Decision

Decision=`psa_d1_step7a_pass_proceed_commit_remote_preflight`。

用户已明确授权PSA-D1 Step7A与five-run validation training。D1仍是control-only：不改model/loss，不访问formal test，
不增加seed，也不允许method promotion。

本地Step7A contracts与machine-decision smokes已通过。下一步先commit/push，remote fast-forward后执行GPU preflight与
Weather one-job resource smoke；只有smoke finite、route=0、five scope gradients nonzero、initialization hash匹配时，才
启动完整5 runs。

## 2. Frozen implementation

| Artifact | Role |
| --- | --- |
| `configs/stage_c_iscf_psa_d1.json` | one new EQUAL arm、five datasets、references、gates、authorization |
| `scripts/check_stage_c_iscf_psa_d1.py` | source/config/objective/gradient contracts |
| `scripts/remote/run_stage_c_iscf_psa_d1.sh` | existing validation runner wrapper |
| `scripts/analyze_stage_c_iscf_psa_d1.py` | 20-run/80-cell H2/H3 analyzer |
| `docs/code-explanation/stage-c-iscf-psa-d1.md` | tensor/dataflow/statistics说明 |

## 3. Local verification

| Check | Result |
| --- | --- |
| training/evaluator semantic diff vs `020eea3` | none |
| EQUAL loss decomposition | pass；route loss/weight=0 |
| random-tensor gradients | arms/policy finite；5/5 scope arms nonzero |
| config JSON | parse pass |
| runner syntax | `bash -n` pass |
| dry-run | 5/5 jobs；all `equal_skill`；validation-only |
| profile hash | `80912741...990a` matches |
| analyzer synthetic matrix | 4 arms/80 metrics/100 comparisons pass |
| analyzer decision branches | run-drift与co-adaptation branches both pass |

Local PyTorch checks使用repo conda environment `r2026-fsa`。系统Python缺torch的首次调用未进入D1 logic，不计为
protocol failure。

## 4. Remote preflight contracts

1. remote repo必须fast-forward到本次commit；
2. 记录GPU index/name/memory/utilization/process；
3. Weather resource smoke使用one epoch、two train/eval batches、no final evaluation；
4. smoke artifact必须finite、无OOM/Traceback/NaN/Inf；
5. `effective_config.adapter.pcc_objective_mode=equal_skill`；
6. training log route weight与weighted route loss必须为0；
7. five scope gradients全部finite/nonzero；
8. smoke initialization hash必须等于Weather historical/ARMERR/SHUFFLED hash；
9. smoke通过后才允许full 5-run launch。

## 5. Launch boundary

| Action | Status |
| --- | --- |
| Step7A implementation | complete |
| five-run validation authorization | true, conditional on preflight |
| Weather resource smoke | pending commit/pull/GPU check |
| full five-run launch | conditional |
| partial result selection | forbidden |
| official test | false |
| confirmation seeds | false |
| method promotion | false |

## 6. Failure attribution

Preflight的config/init/numeric failure只使launch blocked，不产生H2/H3 research decision。只有5/5 runs、20/20 new
validation cells与完整references通过analyzer后，才按冻结gates输出attribution。不得读取partial favorable datasets修改matrix。
