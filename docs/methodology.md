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
5. Record spike counts, Giant Fibre (DNp01) spikes/latency, wall time, RSS.
6. Persist JSON with versions, fingerprint, and configuration.

Warm-up: none by default (single-shot CPU trials). Repeat runs by changing
`seed` or disabling cache.

## Escape probe vs upstream looming experiment

| | FlyQuant `escape` probe | Upstream `01_looming_escape` |
|---|---|---|
| Stimulus | Constant Poisson on LC4/LPLC2 | Geometric looming + tuning curves |
| Needs retinotopy / columns | No | Yes (Codex `column_assignment`) |
| Good for compression A/B | Yes | Yes, when derived data exist |

The probe is sufficient to stress synaptic weights into DNp01. It is **not**
identical to the published looming time course. Document which experiment was
used in every result file.

## Weight quantisation

Applied to effective mV weights `count * w_syn`, never to neuron IDs.

| Method | Storage | Execution |
|---|---|---|
| fp32 | float32 | float32 |
| fp16 | float16 | float16 (native dtype) |
| int8 | int8 + scale | reconstruct → float32 |
| int4 | int8 holding [-7,7] + scale | reconstruct → float32 |

## Lossless graph

CSR + within-row delta indices + narrow integer weights + optional gzip.
Round-trip must be bit-identical on indptr/indices/data.

## Fidelity metrics

- Neural: spike-count match, rate MAE/RMSE/percentiles, correlations with
  explicit silent/zero-variance handling.
- Behavioural: GF success, latency delta, false ± escape.
- Resources: artefact bytes, CSR bytes, peak RSS, wall time, simulated time.

## Calibration / evaluation

Static connectome weights: scale estimation may use the full weight vector.
Do not tune pruning thresholds on a held-out behavioural criterion and then
report that same criterion as unbiased performance. Prefer fixed a-priori
levels (`keep_fraction=0.9`, etc.) for published tables.
