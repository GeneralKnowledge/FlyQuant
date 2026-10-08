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
flyquant suite experiments/synthetic_smoke.yaml    # no FlyWire assets needed
flyquant suite experiments/quickstart_subgraph.yaml  # uses submodule web/data
flyquant report
```

The synthetic suite always runs. The subgraph suite loads the real v783 wiring
diagram from `reference/fruit-fly-lab/web/data/` (~26 MB) and benchmarks an
escape-circuit neighbourhood (~4k neurons).

## CLI

| Command | Purpose |
|---|---|
| `flyquant inspect` | Upstream commit, assets, available methods |
| `flyquant baseline --experiment escape_subgraph` | Reference measurements |
| `flyquant compress --method fp16 --experiment escape_subgraph` | Lossy / lossless variant |
| `flyquant verify --experiment escape_subgraph` | Lossless graph round-trip |
| `flyquant benchmark --compare reference,fp16,int8,lossless_graph` | Paired comparisons |
| `flyquant suite experiments/quickstart_subgraph.yaml` | Config-driven suite |
| `flyquant report` | Markdown + JSON + plots under `reports/` |

## What is implemented (v0.1)

1. Upstream audit and pinned submodule (`docs/audit.md`)
2. Immutable reference adapter + web-binary connectome loader
3. Reproducible baseline harness (JSON results, cache fingerprints)
4. Lossy weight quantisation: FP16, INT8, INT4 (ablation)
5. Lossless graph encoding: CSR + delta indices + narrow ints + gzip
6. Neural + behavioural fidelity metrics (silent / zero-variance safe)
7. Optional magnitude pruning (independent of quantisation)
8. Automated unit tests (synthetic) + optional `@pytest.mark.integration` tests
9. Comparison report with Pareto listing

## Project layout

See `docs/architecture.md`. Compression never edits `reference/fruit-fly-lab`.

## Data and licences

- **FlyQuant code:** Apache-2.0
- **FlyWire data:** CC BY-NC-SA 4.0 — obtain via Codex; see `docs/data-acquisition.md`
- **fruit-fly-lab:** submodule; no LICENSE file upstream — do not relicense it

Do not commit large FlyWire CSV dumps into git.

## Full-brain upstream experiments

After downloading Codex files and building the derived connectome:

```bash
export FLYWIRE_V783_DIR=/path/to/FlyWire\ Brain\ Dataset\ \(FAFB\ v783\)
cd reference/fruit-fly-lab
python -m brain.connectivity.build_connectome
python -m experiments.02_escape_controls
```

Then point FlyQuant at `connectome_source: derived` (or `auto`).

## Documentation

| Doc | Contents |
|---|---|
| `docs/audit.md` | Upstream inspection findings |
| `docs/architecture.md` | Component boundaries |
| `docs/methodology.md` | Metrics and protocols |
| `docs/data-acquisition.md` | Legal data access |
| `docs/limitations.md` | Assumptions and gaps |

## Citing the science (not this repo)

- Dorkenwald et al. (2024) doi:10.1038/s41586-024-07558-y
- Schlegel et al. (2024) doi:10.1038/s41586-024-07686-5
- Shiu et al. (2024) doi:10.1038/s41586-024-07763-9

## Tests

```bash
pytest tests/ -q                 # unit tests (synthetic only)
pytest tests/ -q -m integration  # needs submodule web/data
```
