# ISCF-RSCC Step7A Implementation and Remote Gate

## 1. Decision record

| Field | Record |
| --- | --- |
| `current_step` | RSCC-v1 Step7A passed；resource smoke pending |
| `candidate` | `SC-ISCF-RSCC-v1` |
| `implementation` | EQUAL reliability + exact coalition route KL；matched ARMERR/SHUFFLED |
| `architecture_change` | none |
| `inference_change` | none |
| `matrix` | 3 new arms × 5 datasets × seed2021=15 runs；5 parent reused |
| `decision` | `rscc_step7a_pass_resource_smoke_authorized` |

## 2. Code contract

`equal_scope_coalition_credit`只在existing PCC composition map中同时启用`skill_kind=equal`与
`route_kind=coalition`。credit公式、detach boundary、epsilon、uniform fallback、route schedule与SCC-v0完全相同。
shuffled mode只用dedicated RNG重排scope binding。

closest control `pointwise_prior_composed`是已有EQUAL + standalone-error route KL，无需新实现。所有new arms保持
`siff-independent-scope-control` readout和same initialization。

## 3. Local verification

- RSCC skill loss与EQUAL逐值相等；
- RSCC total=`fused + weighted skill + weighted route`；
- shuffled control保持skill loss与credit marginals；
- SCC/RSCC exact、gradient boundary、uniform fallback与RNG isolation checker通过；
- existing PCC regression 36/36通过；
- 15-job config parse、shell syntax、runner dry-run与diff check通过；
- config SHA256=`fba748ff0a6abe087f58677c8aa6e277e66c65a23f74f82c5e3e70837de52fc7`。

## 4. Remote boundary

只先运行Weather RSCC与RSCC-SHUFFLED的2-batch resource smoke。必须确认skill/route同时active、five gradients nonzero、
finite且无OOM，才允许15-run validation launch。formal test与confirmation seeds false。

```text
active_method = SC-ISCF-RSCC-v1_step7b_candidate
resource_smoke_authorized = true
15_run_validation_authorized = conditional_on_smoke
formal_test_authorized = false
```
