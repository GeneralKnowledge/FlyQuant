# FlyQuant

**LLM-style quantisation experiments for a connectome-based fruit fly brain.**

FlyQuant asks a measurable question:

> How much can we compress an executable fruit-fly connectome before its neural
> activity and observable behaviours diverge significantly from the original
> computational model?

It does **not** build a new simulator. It reuses the open-source
[Fruit Fly Laboratory](https://github.com/vaibhavkedarisetti/fruit-fly-lab)
(LIF model of Shiu et al. 2024 on FlyWire FAFB v783) as an immutable reference,
then adds compression methods, fidelity metrics, and a reproducible benchmark
harness.

This evaluates fidelity to a **computational model**, not biological equivalence
to a living animal.

## Quick start (CPU, no Codex login)

```bash
git clone --recurse-submodules https://github.com/GeneralKnowledge/FlyQuant.git
cd FlyQuant
python -m pip install -e ".[dev]"

flyquant inspect
flyquant suite experiments/synthetic_smoke.yaml
flyquant suite experiments/quickstart_subgraph.yaml
flyquant suite experiments/release_subgraph.yaml   # Phase 5–7 release surface
flyquant report
```

## CLI

| Command | Purpose |
|---|---|
| `flyquant inspect` | Upstream commit, assets, derived-data status |
| `flyquant baseline --experiment escape_subgraph` | Reference measurements |
| `flyquant compress --method state_fp16 --experiment escape_subgraph` | State / weight / prune / pipeline |
| `flyquant verify --experiment escape_subgraph` | Lossless graph round-trip |
| `flyquant benchmark --compare reference,fp16,int8,int4,state_fp16` | Paired comparisons |
| `flyquant suite experiments/release_subgraph.yaml` | Locked suite |
| `flyquant report` | Markdown + JSON + CSV + plots under `reports/` |

Common flags: `--execution-dtype {fp32,fp16,fp64}`, `--record-spikes`, `--record-v`.

## What is implemented (v0.2)

Phases 1–7 of the project brief, adapted to upstream:

1. Upstream audit + pinned submodule
2. Benchmark harness with cache / RAM gates / `unexecuted` labels
3. Lossless CSR + delta + gzip
4. Weight quantisation FP16 / INT8 / INT4 (nibble-packed storage)
5. State-precision isolation (`state_fp16` / `state_fp64`) + recording metrics
6. Pruning (magnitude + static importance) and combined pipelines
7. Release suites, richer reports (CSV + family plots), GitHub Actions CI
8. Looming / lesion adapters (run or cleanly `unexecuted` without Codex data)

## Locked suites

| File | Role |
|---|---|
| `experiments/release_subgraph.yaml` | Canonical CPU release |
| `experiments/state_precision.yaml` | State vs weight FP16 |
| `experiments/prune_ablation.yaml` | Fixed prune levels |
| `experiments/pipeline_ablation.yaml` | Combined ablations |
| `experiments/looming_derived.yaml` | Needs Codex-derived connectome |
| `experiments/release_fullbrain.yaml` | Full brain, high RAM gate |

## Documentation

| Doc | Contents |
|---|---|
| `docs/audit.md` | Upstream inspection |
| `docs/architecture.md` | Component boundaries |
| `docs/methodology.md` | Metrics and protocols |
| `docs/data-acquisition.md` | Legal data access |
| `docs/limitations.md` | Assumptions and gaps |
| `docs/baseline_results.md` | Measured / unexecuted status |

## Data and licences

- **FlyQuant code:** Apache-2.0
- **FlyWire data:** CC BY-NC-SA 4.0 — see `docs/data-acquisition.md`
- **fruit-fly-lab:** submodule; no LICENSE file upstream

## Tests

```bash
pytest tests/ -q -m "not integration and not slow"   # CI default
pytest tests/ -q -m "integration and not slow"       # needs web/data submodule
```
