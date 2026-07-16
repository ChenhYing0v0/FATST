# Active Code Explanations

- `phase5-clean-timealign-a6-lbf.md`: active natural A6 forward path；
- `stage-c-natural-baseline-test.md`: baseline evaluator/analyzer/runner。
- `stage-c-active-mode-gate.md`: 默认 training CLI 对归档 encoder/readout/loss 的入口保护。
- `stage-c-d1-offline-diagnostic.md`: D1-A/B/C structure、probe、frozen counterfactual、basis与gradient flow。
- `stage-c-step46-theory-audit.md`: PMFO mixed-radix invariants、prefix-local count与MIPR measure geometry。
- `stage-c-five-profile-extension-pipeline.md`: ETTh1/ETTm2 validation-only profile calibration的
  training/artifact flow、统计量与freeze boundary。
- `stage-c-sc1-d2-diagnostic-pipeline.md`: frozen-memory probes、dynamic memory width、split与artifact统计。
- `stage-c-sc1-d3-crossed-diagnostic.md`: missing factorial cell worker、15-unit aggregation、gate与remote flow。
- `stage-c-sc1-d4-structured-basis.md`: seven-basis worker、dense-horizon evaluation、geometry与gate flow。
- `stage-c-d14a1-dual-carrier-grouped-mlp.md`: neutral/A6双carrier、GroupedMLP sharing topology、
  affine containment、validation artifacts与串行causal gate。
- `stage-c-pcsd-cf-step7a.md`: PCSD shared mode field、scope pooling、shared target synthesis、direct/control
  policies、chunked execution、A6 containment与Step7A artifact definitions。

下一次 model或diagnostic code 更新必须在本目录新增对应说明，并按 tensor/artifact flow定义 shape、统计列与
code-theory consistency。历史说明已移入 `docs/archive/pre-stage-c-reset-20260713/code-explanation/`。
