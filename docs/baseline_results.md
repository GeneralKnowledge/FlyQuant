# Measured baseline results (development machine)

These numbers were produced on the FlyQuant cloud agent host after the escape
subgraph seed-preservation fix. They are **not** invented.

**Protocol:** `connectome_source=web`, `subgraph=escape` (4000 neurons,
LC4=104, LPLC2=210, DNp01=2), Poisson 150 Hz, duration 60 ms, seed 0.

| Variant | Storage | Execution | Weight/CSR ratio | GF spikes | Active-rate corr | Escape agree | Wall (s) |
|---|---|---|---:|---:|---:|---|---:|
| reference | int32 CSR counts | float32 | — | 41 | 1.0 | yes | ~0.031 |
| fp16 | float16 wdata | float16 | ~10× vs fp32 wdata file | 41 | 0.9993 | yes | ~0.115 |
| int8 | int8+scale | float32 reconstruct | ~18× vs fp32 wdata file | 41 | 0.9979 | yes | ~0.024 |
| lossless_graph | delta-CSR int16 + gzip | float32 | ~2.9× vs in-memory CSR | 41 | 1.0 | yes | ~0.024 |

Re-run:

```bash
flyquant suite experiments/quickstart_subgraph.yaml --no-cache
flyquant report
```

Full-brain geometric looming (`experiments.01_looming_escape`) was **not**
executed here: it needs Codex-derived retinotopy tables. Status: unexecuted
pending `FLYWIRE_V783_DIR` + `build_connectome`.
