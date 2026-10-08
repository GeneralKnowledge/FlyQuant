# Architecture

```
FlyQuant
├── reference/fruit-fly-lab/   # immutable upstream submodule
├── src/flyquant/
│   ├── reference/             # adapter, web loader, synthetic graphs
│   ├── compression/           # weights, lossless graph, state, prune
│   ├── metrics/               # neural / behavioural / resources
│   ├── benchmarks/            # harness + cache
│   ├── reports/               # comparison JSON/Markdown/plots
│   └── cli.py
├── experiments/               # YAML suites
├── tests/                     # unit + optional integration
└── docs/
```

## Separation of concerns

1. **Reference** — upstream LIF engine and connectome remain unmodified.
   Adapters may wrap or reconstruct inputs; they must not edit submodule files
   during an experiment.
2. **Compression** — pure transforms of CSR / weight arrays / artefacts.
3. **Benchmarks** — execute reference and hypothesis under identical seeds and
   stimuli; persist machine-readable records.
4. **Metrics / reports** — consume records only.

## Comparison rule

Every compressed variant is scored against the **same** reference configuration
(same connectome source, subgraph, seed, duration, rate). Variants are never
compared only to each other for fidelity claims.

## Storage vs memory vs execution

| Layer | Example |
|---|---|
| Storage artefact | gzipped delta-CSR, INT8 payload file |
| In-memory model | scipy CSR of synapse counts; engine `wdata` |
| Runtime state | `v`, `g`, delay ring (`dtype`) |

A smaller file does not imply lower peak RSS or faster steps unless measured.

## Experiment progression

Phases follow the project brief: audit → harness → lossless graph → FP16/INT8 →
state precision → pruning → reports. Pruning is available but optional; it is
kept independent of weight quantisation.
