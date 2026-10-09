# Known limitations and untested assumptions

## Scientific

- Fidelity is to the Shiu et al. / fruit-fly-lab **computational model**, not to
  a living fly.
- Point neurons; no gap junctions; no VNC; no spontaneous activity (see upstream
  `BIOLOGICAL_ASSUMPTIONS.md`).
- FlyQuant’s default escape **probe** uses constant Poisson drive. Geometric
  looming (`experiment: looming`) and lesion controls require Codex-derived
  data; without them runs are labelled `unexecuted`.
- Small weight errors can eliminate or create Giant Fibre spikes because of
  thresholding — mean rate error can look tiny while behaviour flips.

## Engineering

- Upstream fruit-fly-lab has **no LICENSE file**; treat redistribution of that
  code cautiously and keep it as a submodule.
- No GPU path; float16 may be slower on some CPUs.
- INT8/INT4 are storage + reconstruct in v0.2, not integer kernels.
- INT4 nibble packing is storage-only; simulation always dequantises to float32.
- `state_fp16` keeps float32 mV weights while lowering `v`/`g`/ring dtype;
  weight-`fp16` also lowers `wdata` when `native_fp16_execution` is true.
- Full-brain (139k) trials need more RAM than many laptops provide; use
  `escape_subgraph` first. Harness can mark runs `unexecuted` when RAM gated.
- Web loader rebuilds neuron tables without every annotation column present in
  the derived CSV; Session features that need missing columns may differ.
- Geometric looming with weight overrides uses a Poisson proxy for compressed
  variants (Session has no `weight_data` hook); documented in harness metadata.
- Cache keys include FlyQuant version + upstream commit + config; always pass
  `--no-cache` after intentional logic changes if unsure.
- Importance pruning uses **static graph** scores only (`depends_on_stimulus: false`).
  Prune levels are fixed a priori — not tuned on evaluation escape success.
- Activity-based / learned importance pruning is deferred.

## Honest status labels

| Status | Meaning |
|---|---|
| `ok` | Measured end-to-end |
| `unexecuted` | Skipped (resources/data); reason recorded |
| `error` | Failed; traceback stored |
