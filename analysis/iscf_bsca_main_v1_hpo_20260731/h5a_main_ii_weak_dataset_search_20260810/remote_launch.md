# H5A Remote Resource Gate and Launch

## 1. Exact launch contract

- Remote host：`529_Lab-3090` (`star3090.iai.zju.edu.cn`)
- Repo：`/home/yingch/projects/FATST`
- Commit：`7544f76da2fdbb39f32e40e3cdd3e02dcde2f97c`
- Config SHA256：`fb1fb57927033b50e04cdc85dc05f500cf2c6a9d11baf9cfff59c83dd09c1034`
- Search-space SHA256：`e7164c22853435308103e8f3d5212ff1e9a678b09802861fae09a79d3392066d`
- Output：`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h5a`
- GPUs：0/1/2；launch前均为18 MiB used、0% utilization
- Quota：launch前`182G / 200G soft / 220G hard`；H5A budget=16 GiB

远程working tree原有三个unrelated modified CSV；它们与H5A pull范围不重叠，已原样保留，
没有stash、reset、删除或提交。

## 2. Resource smoke

2026-08-10 15:08:13--15:12:35执行48/48完整resource smoke：

- checkpoints=`48/48`，metrics=`48/48`，logs=`48/48`；
- unique checkpoint hashes=`48`；
- official-test artifacts=`0`；
- OOM/NaN/Inf/Traceback/RuntimeError=`0`；
- ECL patch12/8/4/2与large-capacity profiles全部通过；
- smoke root约1.3 GiB，完成后quota约183G。

Resource gate=`pass`。

## 3. Full training launch

Full train/validation queue于2026-08-10 15:13:32启动：

```text
MODE=train GPU_IDS="0 1 2" \
  bash scripts/remote/run_iscf_bsca_main_v1_hpo_main_ii_h5a.sh
```

Background PID=`2375625`。首批为ECL patch12/8/4，三项均进入epoch1；observed GPU
memory约1.5--1.9 GiB，utilization约47--56%。Training阶段固定`test_jobs=0`。按冻结
estimate预计14--28 wall-hours；实际结束由early stopping和Solar/ECL workload决定。

Current status=`48_job_training_active`。下一步等待queue结束后运行完整artifact analyzer，
冻结48-checkpoint manifest；在此之前formal test保持blocked，且不依据中途validation结果
删除或提前测试任何trial。
