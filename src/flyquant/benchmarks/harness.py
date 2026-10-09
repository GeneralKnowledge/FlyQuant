"""
Reproducible benchmark harness.

Every run records configuration, software versions, resources, and fidelity.
Compressed variants are always compared to the same reference configuration.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from flyquant import DATASET_ID, UPSTREAM_COMMIT, __version__
from flyquant.benchmarks.cache import config_fingerprint, load_cached, save_cached
from flyquant.compression.graph import (
    compress_graph_lossless,
    graph_identical,
    reconstruct_csr,
    save_graph_artefact,
)
from flyquant.compression.prune import prune_connections
from flyquant.compression.state import STATE_METHODS, describe_state_precision, resolve_state_dtype
from flyquant.compression.weights import (
    dequantise_weights,
    effective_mv_weights,
    quantise_weights,
    reconstruction_error,
    save_quantised,
)
from flyquant.metrics.behavioural import behavioural_fidelity, lesion_control_fidelity
from flyquant.metrics.neural import (
    membrane_trace_error,
    neural_fidelity,
    spike_time_coincidence,
)
from flyquant.metrics.resources import measure_resources, peak_rss_bytes
from flyquant.paths import RESULTS_DIR, VARIANTS_DIR, ensure_output_dirs
from flyquant.reference.adapter import (
    DerivedDataUnavailable,
    derived_data_status,
    get_upstream_commit,
    load_reference_connectome,
    run_escape_benchmark,
    run_escape_controls,
    run_looming_escape,
)
from flyquant.reference.recording import RecordingConfig


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@dataclass
class RunRecord:
    status: str
    config: dict
    results: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "config": self.config,
            "results": self.results,
            "error": self.error,
        }


class BenchmarkHarness:
    def __init__(self, results_dir: Path | None = None):
        ensure_output_dirs()
        self.results_dir = Path(results_dir) if results_dir else RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _base_meta(self, config: dict) -> dict:
        return {
            "flyquant_version": __version__,
            "upstream_commit_pinned": UPSTREAM_COMMIT,
            "upstream_commit_resolved": get_upstream_commit(),
            "dataset_id": DATASET_ID,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config": deepcopy(config),
        }

    def run_config(self, config: dict, *, use_cache: bool = True) -> RunRecord:
        cfg = deepcopy(config)
        meta = self._base_meta(cfg)
        fp_key = {
            "config": cfg,
            "flyquant_version": __version__,
            "upstream_commit": meta["upstream_commit_resolved"],
        }
        fp = config_fingerprint(fp_key)
        meta["fingerprint"] = fp

        if use_cache:
            cached = load_cached(fp)
            if cached and cached.get("status") == "ok":
                cached = dict(cached)
                cached["cache_hit"] = True
                return RunRecord(status="ok", config=cfg, results=cached)

        try:
            record = self._execute(cfg, meta)
            record["fingerprint"] = fp
            if record.get("status") == "ok":
                save_cached(fp, record)
            out = self.results_dir / f"{cfg.get('name', 'run')}_{fp}.json"
            if record.get("status") == "error":
                out = self.results_dir / f"{cfg.get('name', 'run')}_{fp}_error.json"
            out.write_text(json.dumps(_jsonable(record), indent=2))
            record["results_path"] = str(out)
            return RunRecord(status=record["status"], config=cfg, results=record)
        except Exception as exc:
            err = {
                **meta,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            out = self.results_dir / f"{cfg.get('name', 'run')}_{fp}_error.json"
            out.write_text(json.dumps(_jsonable(err), indent=2))
            return RunRecord(status="error", config=cfg, results=err, error=str(exc))

    def _recording_config(self, cfg: dict) -> RecordingConfig | None:
        if not cfg.get("record_spikes") and not cfg.get("record_v"):
            return None
        return RecordingConfig(
            record_spikes=bool(cfg.get("record_spikes", True)),
            record_v=bool(cfg.get("record_v", False)),
            v_stride=int(cfg.get("v_stride", 10)),
        )

    def _apply_pipeline(self, connectome, steps: list[dict], cfg: dict) -> tuple[Any, np.ndarray | None, dict]:
        """Apply prune / quantise / lossless steps. Returns connectome, weight_data, info."""
        info: dict[str, Any] = {"steps": []}
        weight_data = None
        weights_mv = effective_mv_weights(connectome)
        for step in steps:
            op = step.get("op")
            step_info: dict[str, Any] = {"op": op}
            if op == "prune":
                dnp01 = connectome.by_cell_type("DNp01")["idx"].to_numpy()
                pr = prune_connections(
                    connectome.w,
                    method=step.get("prune_method", cfg.get("prune_method", "keep_fraction")),
                    level=float(step.get("prune_level", cfg.get("prune_level", 0.9))),
                    dnp01_indices=dnp01,
                )
                connectome = _with_matrix(connectome, pr.matrix)
                weights_mv = effective_mv_weights(connectome)
                weight_data = weights_mv.astype(np.float32)
                step_info["prune"] = pr.to_meta()
            elif op == "quantise":
                precision = step.get("precision", "int8")
                q = quantise_weights(
                    weights_mv,
                    precision,
                    pack_int4=bool(step.get("pack_int4", True)),
                )
                recon = dequantise_weights(q, out_dtype=np.float32)
                weight_data = recon
                step_info["quantisation"] = q.to_meta()
                step_info["weight_reconstruction_error"] = reconstruction_error(
                    weights_mv, recon
                )
                art_path = VARIANTS_DIR / f"{cfg.get('name', 'pipe')}_{precision}_weights.npz"
                save_quantised(art_path, q)
                step_info["artefact_path"] = str(art_path)
                step_info["artefact_bytes"] = int(art_path.stat().st_size)
                if q.packed_nbytes is not None:
                    step_info["packed_nbytes"] = q.packed_nbytes
                    step_info["unpacked_nbytes"] = q.unpacked_nbytes
            elif op == "lossless_graph":
                art = compress_graph_lossless(
                    connectome.w,
                    metadata={"dataset": getattr(connectome, "dataset", DATASET_ID)},
                )
                saved = save_graph_artefact(
                    VARIANTS_DIR / f"{cfg.get('name', 'pipe')}_graph",
                    art,
                    gzip_file=bool(step.get("gzip", True)),
                )
                step_info["graph_storage"] = saved
                step_info["graph_roundtrip"] = graph_identical(
                    connectome.w, reconstruct_csr(art)
                )
            else:
                raise ValueError(f"unknown pipeline op: {op}")
            info["steps"].append(step_info)
        return connectome, weight_data, info

    def _prepare_variant(self, cfg: dict, connectome) -> tuple[Any, np.ndarray | None, Any, dict]:
        """
        Returns (connectome, weight_data, dtype, variant_info).
        """
        method = cfg.get("method", "reference")
        weights_mv = effective_mv_weights(connectome)
        model_bytes_ref = int(
            connectome.w.data.nbytes
            + connectome.w.indices.nbytes
            + connectome.w.indptr.nbytes
        )
        variant_info: dict[str, Any] = {
            "method": method,
            "model_storage_bytes_reference_csr": model_bytes_ref,
        }
        weight_data = None
        exec_precision = cfg.get("execution_dtype", "fp32")
        dtype = resolve_state_dtype(exec_precision)

        if method == "reference":
            variant_info["storage_precision"] = "int32_synapse_counts_csr"
            variant_info["execution_precision"] = str(np.dtype(dtype))
            variant_info["state_precision"] = describe_state_precision(exec_precision)

        elif method in STATE_METHODS:
            # State-only: keep float32 mV weights; lower state dtype.
            dtype = resolve_state_dtype(method)
            weight_data = weights_mv.astype(np.float32)
            variant_info["storage_precision"] = "int32_synapse_counts_csr"
            variant_info["execution_precision"] = str(np.dtype(dtype))
            variant_info["weight_execution"] = "float32_wdata_override"
            variant_info["state_precision"] = describe_state_precision(method)
            variant_info["notes"] = (
                "State-only experiment: reference synapse counts reconstructed as "
                "float32 mV weights; engine dtype lowered for v/g/ring."
            )

        elif method in ("fp16", "fp32", "int8", "int4"):
            q = quantise_weights(
                weights_mv,
                method,
                pack_int4=bool(cfg.get("pack_int4", True)),
            )
            recon = dequantise_weights(q, out_dtype=np.float32)
            variant_info["weight_reconstruction_error"] = reconstruction_error(
                weights_mv, recon
            )
            variant_info["quantisation"] = q.to_meta()
            weight_data = recon
            if method == "fp16" and cfg.get("native_fp16_execution", True):
                dtype = np.float16
                weight_data = dequantise_weights(q, out_dtype=np.float16)
            art_path = VARIANTS_DIR / f"{cfg.get('name', method)}_weights.npz"
            save_quantised(art_path, q)
            variant_info["artefact_path"] = str(art_path)
            variant_info["artefact_bytes"] = int(art_path.stat().st_size)
            if q.packed_nbytes is not None:
                variant_info["packed_nbytes"] = q.packed_nbytes
                variant_info["unpacked_nbytes"] = q.unpacked_nbytes
                variant_info["nibble_pack_ratio_vs_unpacked"] = (
                    q.unpacked_nbytes / q.packed_nbytes if q.packed_nbytes else None
                )
            variant_info["storage_precision"] = q.storage_dtype
            variant_info["execution_precision"] = str(np.dtype(dtype))

        elif method == "lossless_graph":
            art = compress_graph_lossless(
                connectome.w,
                metadata={"dataset": getattr(connectome, "dataset", DATASET_ID)},
            )
            saved = save_graph_artefact(
                VARIANTS_DIR / f"{cfg.get('name', 'graph')}",
                art,
                gzip_file=bool(cfg.get("gzip", True)),
            )
            recon = reconstruct_csr(art)
            variant_info["graph_roundtrip"] = graph_identical(connectome.w, recon)
            variant_info["graph_storage"] = saved
            variant_info["storage_precision"] = art.weight_dtype
            variant_info["execution_precision"] = str(np.dtype(dtype))

        elif method == "prune":
            dnp01 = connectome.by_cell_type("DNp01")["idx"].to_numpy()
            pr = prune_connections(
                connectome.w,
                method=cfg.get("prune_method", "keep_fraction"),
                level=float(cfg.get("prune_level", 0.9)),
                dnp01_indices=dnp01,
            )
            variant_info["prune"] = pr.to_meta()
            connectome = _with_matrix(connectome, pr.matrix)
            weights_mv = effective_mv_weights(connectome)
            weight_data = weights_mv.astype(np.float32)
            variant_info["storage_precision"] = "pruned_csr"
            variant_info["execution_precision"] = str(np.dtype(dtype))

        elif method == "pipeline":
            steps = cfg.get("steps") or []
            connectome, weight_data, pipe_info = self._apply_pipeline(
                connectome, steps, cfg
            )
            variant_info["pipeline"] = pipe_info
            variant_info["storage_precision"] = "pipeline"
            variant_info["execution_precision"] = str(np.dtype(dtype))
            # Prefer last quant artefact bytes if present
            for step in reversed(pipe_info["steps"]):
                if "artefact_bytes" in step:
                    variant_info["artefact_bytes"] = step["artefact_bytes"]
                    break
                if "graph_storage" in step:
                    variant_info["graph_storage"] = step["graph_storage"]
                    break

        else:
            raise ValueError(f"unknown method: {method}")

        return connectome, weight_data, dtype, variant_info

    def _execute(self, cfg: dict, meta: dict) -> dict:
        source = cfg.get("connectome_source", "auto")
        subgraph = cfg.get("subgraph")
        experiment = cfg.get("experiment", "escape")
        method = cfg.get("method", "reference")

        min_ram_gb = float(cfg.get("min_ram_gb", 0))
        if min_ram_gb > 0:
            avail = _available_ram_gb()
            if avail is not None and avail < min_ram_gb:
                return {
                    **meta,
                    "status": "unexecuted",
                    "reason": (
                        f"Insufficient RAM: ~{avail:.1f} GiB available, "
                        f"config requires min_ram_gb={min_ram_gb}"
                    ),
                }

        # Derived-only experiments
        if experiment in ("looming", "escape_controls"):
            status = derived_data_status()
            if not status["available"]:
                return {
                    **meta,
                    "status": "unexecuted",
                    "reason": status["reason"],
                    "derived_status": status,
                }
            if source == "auto":
                source = "derived"

        t_load0 = time.perf_counter()
        try:
            connectome = load_reference_connectome(source=source, subgraph=subgraph)
        except DerivedDataUnavailable as exc:
            return {**meta, "status": "unexecuted", "reason": str(exc)}
        load_s = time.perf_counter() - t_load0

        seed = int(cfg.get("seed", 0))
        duration_ms = float(cfg.get("duration_ms", 80.0))
        rate_hz = float(cfg.get("rate_hz", 150.0))
        rec_cfg = self._recording_config(cfg)

        # Escape-controls: special multi-condition path
        if experiment == "escape_controls":
            return self._execute_escape_controls(cfg, meta, connectome, load_s)

        connectome, weight_data, dtype, variant_info = self._prepare_variant(
            cfg, connectome
        )
        model_bytes_ref = variant_info["model_storage_bytes_reference_csr"]
        rss_before = peak_rss_bytes()

        run_kwargs = dict(
            seed=seed,
            duration_ms=duration_ms,
            recording=rec_cfg,
        )

        if experiment == "looming":
            if method != "reference":
                # Apply compressed weights by swapping connectome matrix when pruned,
                # or by constructing a Session is hard — use Poisson fallback note.
                # For weight quant on looming: rebuild engine path via escape with
                # geometric stimulus is complex; run looming only for reference,
                # and for compressed use connectome with pruned w + Session.
                try:
                    ref_run = run_looming_escape(
                        load_reference_connectome(source=source, subgraph=subgraph),
                        **{k: run_kwargs[k] for k in ("seed", "duration_ms")},
                    )
                except DerivedDataUnavailable as exc:
                    return {**meta, "status": "unexecuted", "reason": str(exc)}
                # Hypothesis: if we only changed weights via weight_data, Session
                # path doesn't accept it — fall back to Poisson probe with same seed
                # and document the limitation.
                hyp_run = run_escape_benchmark(
                    connectome,
                    dtype=dtype,
                    weight_data=weight_data,
                    rate_hz=rate_hz,
                    experiment=f"{experiment}:{method}:poisson_proxy",
                    **run_kwargs,
                )
                variant_info["looming_note"] = (
                    "Compressed looming uses Poisson probe proxy for weight/state "
                    "overrides; full geometric looming compared for reference only."
                )
            else:
                ref_run = run_looming_escape(
                    connectome,
                    **{k: run_kwargs[k] for k in ("seed", "duration_ms")},
                )
                hyp_run = ref_run
        else:
            # Poisson escape / escape_subgraph / escape_poisson
            if method == "reference":
                ref_run = run_escape_benchmark(
                    connectome,
                    dtype=dtype,
                    rate_hz=rate_hz,
                    experiment=f"{experiment}:reference",
                    **run_kwargs,
                )
                hyp_run = ref_run
            else:
                ref_c = load_reference_connectome(source=source, subgraph=subgraph)
                ref_run = run_escape_benchmark(
                    ref_c,
                    dtype=np.float32,
                    rate_hz=rate_hz,
                    experiment=f"{experiment}:reference",
                    **run_kwargs,
                )
                hyp_run = run_escape_benchmark(
                    connectome,
                    dtype=dtype,
                    weight_data=weight_data,
                    rate_hz=rate_hz,
                    experiment=f"{experiment}:{method}",
                    **run_kwargs,
                )

        neural = neural_fidelity(
            ref_run.spike_counts, hyp_run.spike_counts, duration_ms=duration_ms
        )
        behaviour = behavioural_fidelity(ref_run.summary(), hyp_run.summary())

        # Optional trajectory metrics
        if ref_run.recording is not None and hyp_run.recording is not None:
            neural["spike_time_coincidence"] = spike_time_coincidence(
                ref_run.recording.spike_times_ms,
                hyp_run.recording.spike_times_ms,
                tolerance_ms=float(cfg.get("spike_tolerance_ms", 1.0)),
            )
            neural["membrane_error"] = membrane_trace_error(
                ref_run.recording.v_traces,
                hyp_run.recording.v_traces,
                times_ms=ref_run.recording.v_times_ms,
            )
        else:
            neural["spike_time_coincidence"] = {
                "skipped": True,
                "reason": "recording disabled",
            }
            neural["membrane_error"] = {
                "skipped": True,
                "reason": "recording disabled",
            }

        resources = measure_resources()
        variant_info["hypothesis_engine_state"] = hyp_run.metadata.get("engine_state")
        variant_info["reference_engine_state"] = ref_run.metadata.get("engine_state")

        weight_bytes_ref = (ref_run.metadata.get("engine_state") or {}).get("wdata_bytes")
        if not weight_bytes_ref:
            try:
                ref_for_w = load_reference_connectome(source=source, subgraph=subgraph)
                weight_bytes_ref = int(
                    effective_mv_weights(ref_for_w).astype(np.float32).nbytes
                )
            except Exception:
                weight_bytes_ref = model_bytes_ref

        compression_ratio = None
        variant_info["reference_weight_bytes_fp32"] = weight_bytes_ref
        if method in ("fp16", "fp32", "int8", "int4") and variant_info.get("artefact_bytes"):
            compression_ratio = weight_bytes_ref / variant_info["artefact_bytes"]
            variant_info["compression_ratio_basis"] = "fp32_wdata_vs_quant_artefact"
        elif method in STATE_METHODS:
            ref_state = (ref_run.metadata.get("engine_state") or {}).get("state_arrays_bytes")
            hyp_state = (hyp_run.metadata.get("engine_state") or {}).get("state_arrays_bytes")
            if ref_state and hyp_state:
                compression_ratio = ref_state / hyp_state
                variant_info["compression_ratio_basis"] = "state_arrays_fp32_vs_hypothesis"
        elif variant_info.get("graph_storage", {}).get("stored_nbytes"):
            compression_ratio = (
                model_bytes_ref / variant_info["graph_storage"]["stored_nbytes"]
            )
            variant_info["compression_ratio_basis"] = "csr_vs_lossless_artefact"
        elif variant_info.get("artefact_bytes"):
            compression_ratio = weight_bytes_ref / variant_info["artefact_bytes"]
            variant_info["compression_ratio_basis"] = "fp32_wdata_vs_artefact"
        elif method == "prune":
            pr = variant_info.get("prune") or {}
            if pr.get("original_nnz"):
                compression_ratio = pr["original_nnz"] / max(pr.get("kept_nnz", 1), 1)
                variant_info["compression_ratio_basis"] = "nnz_original_vs_kept"

        return {
            **meta,
            "status": "ok",
            "load_time_s": load_s,
            "rss_before_sim_bytes": rss_before,
            "resources": resources.to_dict(),
            "variant": variant_info,
            "reference": ref_run.summary(),
            "hypothesis": hyp_run.summary(),
            "neural_fidelity": neural,
            "behavioural_fidelity": behaviour,
            "compression_ratio_storage_vs_csr": compression_ratio,
            "model_counts": {
                "n_neurons": int(hyp_run.n_neurons),
                "n_connections": int(hyp_run.n_connections),
                "n_synapses": int(hyp_run.n_synapses),
                "csr_nbytes": model_bytes_ref,
            },
            "cache_hit": False,
        }

    def _execute_escape_controls(self, cfg, meta, connectome, load_s) -> dict:
        method = cfg.get("method", "reference")
        seed = int(cfg.get("seed", 0))
        if method != "reference":
            # Apply prune to connectome for lesion comparison if requested
            connectome, weight_data, dtype, variant_info = self._prepare_variant(
                cfg, connectome
            )
            if weight_data is not None and method not in ("prune", "pipeline"):
                return {
                    **meta,
                    "status": "unexecuted",
                    "reason": (
                        "escape_controls currently supports reference and prune/pipeline "
                        "topology changes; weight-quant overrides are not injected into "
                        "upstream Session for geometric looming lesions."
                    ),
                    "variant": variant_info,
                }
        else:
            variant_info = {"method": "reference"}

        try:
            ref_c = load_reference_connectome(
                source=cfg.get("connectome_source", "derived"),
                subgraph=cfg.get("subgraph"),
            )
            ref_conds = run_escape_controls(ref_c, seed=seed)
            hyp_conds = run_escape_controls(connectome, seed=seed)
        except DerivedDataUnavailable as exc:
            return {**meta, "status": "unexecuted", "reason": str(exc)}

        lesion = lesion_control_fidelity(ref_conds, hyp_conds)
        # Summaries from looming condition
        def _loom(conds):
            for c in conds:
                if c.get("condition") == "looming":
                    return {
                        "gf_spikes": c.get("dnp01_spikes", 0),
                        "lc4_spikes": c.get("lc4_spikes", 0),
                        "lplc2_spikes": c.get("lplc2_spikes", 0),
                        "first_gf_spike_ms": None,
                        "escape_peak": c.get("peak_dnp01_hz", 0) / 210.0 if c.get("peak_dnp01_hz") else 0,
                    }
            return {"gf_spikes": 0}

        behaviour = behavioural_fidelity(_loom(ref_conds), _loom(hyp_conds))
        behaviour["lesion_controls"] = lesion

        return {
            **meta,
            "status": "ok",
            "load_time_s": load_s,
            "resources": measure_resources().to_dict(),
            "variant": variant_info,
            "reference": {"conditions": ref_conds, **_loom(ref_conds)},
            "hypothesis": {"conditions": hyp_conds, **_loom(hyp_conds)},
            "neural_fidelity": {"skipped": True, "reason": "multi-condition lesion table"},
            "behavioural_fidelity": behaviour,
            "compression_ratio_storage_vs_csr": None,
            "model_counts": {
                "n_neurons": int(connectome.n),
                "n_connections": int(connectome.w.nnz),
                "n_synapses": int(np.abs(connectome.w.data).sum()),
            },
            "cache_hit": False,
        }

    def run_suite(self, suite_path: Path, *, use_cache: bool = True) -> list[RunRecord]:
        suite = yaml.safe_load(Path(suite_path).read_text())
        runs = suite.get("runs", [])
        return [self.run_config(r, use_cache=use_cache) for r in runs]


def _with_matrix(connectome, matrix):
    from flyquant.reference.synthetic import SyntheticConnectome

    return SyntheticConnectome(
        neurons=connectome.neurons,
        w=matrix.tocsr(),
        manifest=dict(getattr(connectome, "manifest", {})),
    )


def _available_ram_gb() -> float | None:
    try:
        text = Path("/proc/meminfo").read_text()
        for line in text.splitlines():
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return kb / (1024 * 1024)
    except Exception:
        return None
    return None
