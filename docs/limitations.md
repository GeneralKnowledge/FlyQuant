# Known limitations and untested assumptions

## Scientific

- Fidelity is to the Shiu et al. / fruit-fly-lab **computational model**, not to
  a living fly.
- Point neurons; no gap junctions; no VNC; no spontaneous activity (see upstream
  `BIOLOGICAL_ASSUMPTIONS.md`).
- FlyQuant’s default escape **probe** uses constant Poisson drive, not the full
  geometric looming encoder (unless you run upstream experiments directly).
- Small weight errors can eliminate or create Giant Fibre spikes because of
  thresholding — mean rate error can look tiny while behaviour flips.

## Engineering

- Upstream fruit-fly-lab has **no LICENSE file**; treat redistribution of that
  code cautiously and keep it as a submodule.
- No GPU path; float16 may be slower on some CPUs.
- INT8/INT4 are storage + reconstruct in v0.1, not integer kernels.
- INT4 payload is not nibble-packed (values in int8).
- Full-brain (139k) trials need more RAM than many laptops provide; use
  `escape_subgraph` first. Harness can mark runs `unexecuted` when RAM gated.
- Web loader rebuilds neuron tables without every annotation column present in
  the derived CSV; Session features that need missing columns may differ.
- Cache keys include FlyQuant version + upstream commit + config; always pass
  `--no-cache` after intentional logic changes if unsure.

## Unimplemented / future

- Bit-packed INT4 and integer arithmetic engines.
- Membrane-potential trajectory metrics (needs trace recording hooks).
- Importance-based pruning with separate calibration/eval splits.
- Automatic Pareto plotting beyond the basic matplotlib suite.
- Docker images (intentionally avoided unless needed).

## Honest status labels

| Status | Meaning |
|---|---|
| `ok` | Measured end-to-end |
| `unexecuted` | Skipped (resources/data); reason recorded |
| `error` | Failed; traceback stored |
