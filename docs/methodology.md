# Methodology

## Research question

How much can we compress an executable fruit-fly connectome before its neural
activity and observable behaviours diverge significantly from the original
computational model?

## Baseline protocol

1. Load connectome (`web` or `derived`).
2. Optionally restrict to the escape-circuit subgraph (~≤4000 neurons).
3. Drive LC4 + LPLC2 with Poisson rate 150 Hz (Shiu default), fixed seed.
4. Simulate duration `T` ms at `dt = 0.1` ms.
5. Record spike counts, Giant Fibre (DNp01) spikes/latency, wall time, RSS,
   and optionally spike times / membrane samples.
6. Persist JSON with versions, fingerprint, and configuration.

Warm-up: none by default (single-shot CPU trials). Repeat runs by changing
`seed` or disabling cache.

## Escape probe vs upstream looming experiment

| | FlyQuant `escape` probe | `looming` / `escape_controls` |
|---|---|---|
| Stimulus | Constant Poisson on LC4/LPLC2 | Geometric looming + tuning / lesions |
| Needs retinotopy / columns | No | Yes (Codex-derived) |
| Default CI path | Yes | Gated; `unexecuted` without data |

## Storage vs memory vs execution

| Layer | What we report |
|---|---|
| Storage artefact | Quant NPZ, nibble-packed INT4, gzipped delta-CSR |
| In-memory model | CSR synapse counts; engine `wdata` |
| Runtime state | `v`, `g`, delay ring (`state_arrays_bytes`) |

`state_fp16` changes only runtime state (weights stay float32 mV).
Weight-`fp16` changes stored/executed synaptic weights (and optionally state).

## Weight quantisation

Applied to effective mV weights `count * w_syn`, never to neuron IDs.

| Method | Storage | Execution |
|---|---|---|
| fp32 | float32 | float32 |
| fp16 | float16 | float16 (native dtype) |
| int8 | int8 + scale | reconstruct → float32 |
| int4 | nibble-packed + scale | unpack + reconstruct → float32 |

## State precision

| Method | Weights | State dtype |
|---|---|---|
| reference | float32 mV from counts | float32 |
| state_fp16 | float32 mV (override) | float16 |
| state_fp64 | float32 mV (override) | float64 |

## Pruning

Fixed a-priori levels only (`keep_fraction` ∈ {0.95,0.90,0.75,0.50},
`threshold` synapse floors, static-graph importance). **Not** tuned on
evaluation escape success. `depends_on_stimulus: false`.

## Combined pipelines

Config `method: pipeline` with ordered `steps` (`prune`, `quantise`,
`lossless_graph`). Ablations compare prune-only, quant-only, prune→quant,
quant→prune, and lossless-alone.

## Fidelity metrics

- Neural: spike-count match, rate MAE/RMSE/percentiles, correlations with
  explicit silent/zero-variance handling; optional spike-time coincidence and
  membrane MAE when recording is enabled.
- Behavioural: GF success, latency delta, false ± escape; lesion abolishment
  agreement when `escape_controls` runs.
- Resources: artefact bytes, CSR bytes, state array bytes, peak RSS, wall time.

## Locked suites

| Suite | Purpose |
|---|---|
| `experiments/release_subgraph.yaml` | Canonical CPU release |
| `experiments/state_precision.yaml` | State vs weight FP16 ablation |
| `experiments/prune_ablation.yaml` | Prune curve |
| `experiments/pipeline_ablation.yaml` | Combined ablations |
| `experiments/looming_derived.yaml` | Geometric / lesions (derived) |
| `experiments/release_fullbrain.yaml` | Full brain, RAM-gated |
| `experiments/synthetic_smoke.yaml` | Offline CI smoke |

```bash
flyquant suite experiments/release_subgraph.yaml --no-cache
flyquant report
# artefacts: reports/comparison.{md,json,csv}, reports/plots/
```
