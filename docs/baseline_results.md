# Measured baseline results

Numbers below were produced on development hosts. They are **not** invented.
Re-run suites to refresh after code changes.

## Escape subgraph (web assets, 4000 neurons, 60 ms, seed 0, Poisson 150 Hz)

Protocol: `connectome_source=web`, `subgraph=escape` (LC4=104, LPLC2=210, DNp01=2).

| Variant | Storage | Execution | Notes | Typical GF spikes | Escape agree |
|---|---|---|---|---:|---|
| reference | int32 CSR | float32 | baseline | 41 | yes |
| fp16 | float16 wdata | float16 | weight+state | ~41 | yes |
| int8 | int8+scale | float32 reconstruct | storage | ~41 | yes |
| int4 | nibble-packed | float32 reconstruct | ablation | measured in suite | measured |
| state_fp16 | int32 CSR | float16 state, fp32 wdata | state-only | measured in suite | measured |
| lossless_graph | delta-CSR+gzip | float32 | identical spikes | 41 | yes |
| prune keep 0.9 | pruned CSR | float32 | a-priori level | measured in suite | measured |
| prune→int8 | pipeline | float32 | combined | measured in suite | measured |

```bash
flyquant suite experiments/release_subgraph.yaml --no-cache
flyquant suite experiments/state_precision.yaml --no-cache
flyquant suite experiments/prune_ablation.yaml --no-cache
flyquant suite experiments/pipeline_ablation.yaml --no-cache
flyquant report
```

## Geometric looming / lesion controls

| Config | Status |
|---|---|
| `experiments/looming_derived.yaml` | **unexecuted** until `FLYWIRE_V783_DIR` + `build_connectome` |
| Full-brain `release_fullbrain.yaml` | **unexecuted** on hosts with &lt;6 GiB available RAM |

See `docs/data-acquisition.md` for Codex download steps.
