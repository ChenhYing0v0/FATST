# D20-D1 Summary Contribution Direction/Scale Diagnostic

## What we plan to test

D20表明SPEC相对A6在validation正、test负，而SPEC相对RANDOM保留弱正向specificity。D1进一步检查：训练后的
summary contribution在同一个SPEC/RANDOM模型内部是否真实降低residual，还是方向正确但幅度过强。

## Why it matters

如果summary contribution本身有益但完整模型仍差于A6，失败更可能来自joint co-adaptation或冗余shortcut；如果
贡献在多数future regions有正向oracle headroom但$\alpha=1$有害，则readout calibration是直接问题；如果最优
$\alpha\leq0$，才更支持当前statistic-to-coefficient方向错误。

## Artifact construction and metrics

使用每个run保存的`probe_fused`、`probe_targets`和`probe_history_prediction_contribution`，构造
`base=fused-contribution`。在8个冻结future bins与full H720上报告actual/oracle/clipped MSE gain、optimal alpha、
contribution-residual cosine和重构误差。

## Authorization and gate

- new training：false；
- checkpoint mutation：false；
- official-test label：只用于oracle diagnostic；
- confirmation/paper method：false；
- frozen replacement boundary：within-model removal是conditional attribution，不是architecture comparison。

D1没有performance pass gate。无论结果如何，exact D20保持关闭；结果只决定Contribution 1回Step2/4时优先审计
`scale/calibration`、`co-adaptation/intervention point`还是`statistic direction`。
