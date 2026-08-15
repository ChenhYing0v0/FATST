# iTransformer-style Decoder-Transfer v1 Formal-Test Prelaunch

Decision：`itransformer_transfer_v1_formal_15_checkpoint_60_cell_authorized_prelaunch`

用户于2026-08-15明确要求继续formal test。冻结输入为SHA256=`062588a140ecd4fae385aa9d194c039355bef3c7d9f49f685d796779626eecc9`的15-row immutable training manifest；三臂、五datasets、seed2021与four standard horizons均不得改变。

执行面为15个validation-selected checkpoints × `{96,192,336,720}`=`60` cells。每个checkpoint只加载一次test loader并生成dense prefix audit；paper-facing result只抽取四个standard horizons。Runner在test loader access前检查formal authorization、manifest hash、15 unique checkpoint hashes和当前checkpoint bytes；每个job结束后再次检查checkpoint non-mutation。

成功gate保持预注册不变：+ISCF-BSCA相对Original Decoder必须同时改善macro MSE和macro MAE，并赢至少3/5 dataset-mean MSE comparisons。相对+ISCF的完整结果单独报告。Validation risk、dataset、horizon、metric或cell不得用于缩减formal matrix。

本轮授权不包含checkpoint retraining、extra HPO、extra seeds或canonical Decoder-Transfer table mutation。60/60 artifacts闭合前不读取partial方向；完成后先形成独立iTransformer result block与Step 9--10 decision。
