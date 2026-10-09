# Upstream audit: fruit-fly-lab

**Audit date:** 2026-10-08  
**Upstream repository:** https://github.com/vaibhavkedarisetti/fruit-fly-lab  
**Pinned commit:** `26672e06427c12c61536ce1bd93dae7442944681` (2026-08-28)  
**Integration path:** git submodule at `reference/fruit-fly-lab`

This document records what the upstream project actually implements, as verified
from source, before any FlyQuant compression work.

---

## 1. What the project is

An interactive whole-brain simulation of adult female *Drosophila melanogaster*
built on the FlyWire FAFB **v783** connectome and the published LIF model of
Shiu et al. (2024). Pipeline:

```
stimulus → sensory encoders (Poisson rates) → LIF over real wiring
         → descending-neuron readout → kinematic body
```

No LLM, no hand-written escape rule. Escape emerges from LC4/LPLC2 → Giant Fibre
(DNp01) connectivity.

## 2. Neural model (verified in source)

| Item | Value | Source file |
|---|---|---|
| Model class | Leaky integrate-and-fire (LIF) | `brain/neuron_models/lif.py` |
| Equations | `dv/dt=(v₀−v+g)/t_mbr`, `dg/dt=−g/τ` | same; matches Shiu `model.py` |
| Integration | Exact linear (Brian2 `method='linear'`) | `simulation/engine/lif_engine.py` |
| `v_0` / `v_rst` | −52 mV | `LIFParams` |
| `v_th` | −45 mV | |
| `t_mbr` | 20 ms | |
| `tau` | 5 ms | |
| `t_rfc` | 2.2 ms (22 steps) | |
| `t_dly` | 1.8 ms (18 steps) | |
| `w_syn` | 0.275 mV per synapse | |
| `dt` | 0.1 ms | |
| Default runtime dtype | `numpy.float32` | `LIFEngine.__init__` |
| Weight storage in graph | signed synapse **counts** (`int32` CSR) | `Connectome.w` |
| Effective synaptic weight | `sign × syn_count × w_syn` (mV) | applied when engine loads |

**Not present:** multi-compartment dendrites, gap junctions, plasticity,
neuromodulation, spontaneous activity, ventral nerve cord.

## 3. Connectivity representation

- Sparse CSR matrix, shape `(139255, 139255)`, `nnz = 3_732_460`
- Data values = signed synapse counts (excitatory +ve / inhibitory −ve)
- Neurotransmitter → sign convention from Shiu et al. (ACH/DA/OCT/SER → +1;
  GABA/GLUT → −1)
- Neuron indices are dense `0..n-1` sorted by ascending FlyWire `root_id`
- Derived artefact: `data/derived/connectome_v783.npz` (+ neuron index CSV)
- Browser export already uses CSR + **delta-coded** column indices + `int16`
  weights (`web/data/connectome.bin`)

## 4. Dynamic state (per neuron)

| Array | Role | Default dtype |
|---|---|---|
| `v` | membrane potential (mV) | float32 |
| `g` | synaptic drive (mV-like) | float32 |
| `rfc_left` | refractory countdown | int32 |
| `_ring` | delayed synaptic input buffer | float32, shape `(delay+1, n)` |
| `spike_counts` | cumulative spikes | int32 |

## 5. Data requirements

| Artefact | Origin | In git? | Size (approx.) |
|---|---|---|---|
| FlyWire CSV.gz sources | Codex download (login) | No | ~1.5 GB required set |
| `data/derived/*` | Built locally | No (gitignored) | tens of MB |
| `web/data/connectome.bin` | Export of full graph | **Yes** | ~22 MB |
| `web/data/neurons.bin` | Neuron metadata | **Yes** | ~3.4 MB |
| `web/data/meta.json` | Types, RFs, modalities | **Yes** | ~0.7 MB |

Full-brain experiments need either:

1. FlyWire sources + `python -m brain.connectivity.build_connectome`, or
2. FlyQuant’s web-binary loader (reads committed `web/data/*` without Codex login).

FlyWire data licence: **CC BY-NC-SA 4.0**. Users must accept FlyWire terms at
https://flywire.ai. See `docs/data-acquisition.md`.

## 6. Experiments

| Module | Purpose |
|---|---|
| `experiments.01_looming_escape` | Escape time course (behavioural benchmark) |
| `experiments.02_escape_controls` | Lesion controls (LC4/LPLC2 silence → GF = 0) |
| `experiments.03_touch_and_feeding` | Feeding / touch / honest refusals |

**Smallest meaningful behavioural baseline:** looming escape with seed fixed,
duration ~280 ms, watching DNp01 (Giant Fibre) spikes and `escape_takeoff`
channel. Upstream also supports a ~4000-neuron escape-circuit **subgraph** used
in unit tests (`tests/test_lif_engine.py`).

## 7. Dependencies

From `requirements/requirements.txt` (unpinned lower bounds):

- Python ≥ 3.11 (upstream developed on 3.13.14; FlyQuant CI uses 3.12)
- numpy, scipy, pandas
- fastapi, uvicorn, websockets (interactive UI only; not required for FlyQuant CLI)
- pytest

**GPU:** not implemented. CPU event-driven CSR propagation. RAM guidance from
upstream: 4 GB minimum, 8 GB comfortable for full brain.

## 8. Licence compatibility

| Component | Licence | Notes |
|---|---|---|
| fruit-fly-lab source | **No LICENSE file in repository** | Contact upstream author before commercial redistribution of their code. FlyQuant treats it as an immutable reference submodule and does not relicense it. |
| FlyWire / Codex data | CC BY-NC-SA 4.0 | Non-commercial share-alike; attribution required |
| Shiu et al. model constants | Published paper + reference code | Cite doi:10.1038/s41586-024-07763-9 |
| FlyQuant (this repo) | Apache-2.0 | Applies to FlyQuant code only, not upstream or FlyWire data |

## 9. Tests upstream

74 tests covering provenance checksums, circuit facts, LIF equivalence to a
literal Brian2 transcription, sensory/body behaviour. Most integration tests
require a built connectome.

## 10. Integration strategy (chosen)

1. **Submodule** pinned reference — never modify upstream files in place during
   experiments.
2. **Adapter** (`flyquant.reference`) adds sys.path / imports upstream modules;
   optional web-binary loader reconstructs a `Connectome` for machines without
   Codex downloads.
3. **Synthetic graphs** for unit tests (no FlyWire required).
4. Compression operates on weight arrays / CSR artefacts, then feeds reconstructed
   or lower-precision arrays into the **same** `LIFEngine` class.
5. If upstream becomes unavailable: keep the pinned submodule commit; document
   fallback to philshiu/Drosophila_brain_model for equations only (not interactive
   loop). Do not invent a new biology stack.

## 11. Implications for quantisation

- Graph weights are **integers** (synapse counts). Quantising them as if they
  were FP32 LLM weights would be incorrect for INT8-of-counts that already fit
  in int16 (max |w| = 2633 in the v783 export).
- Meaningful lossy quantisation targets:
  1. **Effective mV weights** `wdata = counts * w_syn` as used in the engine
  2. **Runtime state** `v`, `g`, delay ring (`dtype`)
  3. **Lossless storage** of the CSR (delta indices, int16 counts, gzip) —
     already partially present in `web/data/connectome.bin`
- Upstream engine can take `dtype=np.float16` for state and `wdata`; there is no
  native INT8 kernel. INT8 is therefore **storage + reconstruct**, or
  custom cast path, not a guaranteed speed-up.
- Default baseline is already float32 execution, not float64.

## 12. Blockers for this development environment

| Item | Status |
|---|---|
| Full Codex CSV download | Requires FlyWire login — not automated here |
| Built `data/derived/` | Absent until build or web load |
| Web binaries in submodule | Present — preferred path for FlyQuant baseline |
| Full-brain RAM | ~5 GB available on this host; short trials may work; record failures honestly |
