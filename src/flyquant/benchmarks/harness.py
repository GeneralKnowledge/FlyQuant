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
from flyquant.compression.state import resolve_state_dtype
from flyquant.compression.weights import (
    dequantise_weights,
    effective_mv_weights,
    quantise_weights,
    reconstruction_error,
    save_quantised,
)
from flyquant.metrics.behavioural import behavioural_fidelity
from flyquant.metrics.neural import neural_fidelity
from flyquant.metrics.resources import measure_resources, peak_rss_bytes
from flyquant.paths import RESULTS_DIR, VARIANTS_DIR, ensure_output_dirs
from flyquant.reference.adapter import (
    get_upstream_commit,
    load_reference_connectome,
    run_escape_benchmark,
)


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
            save_cached(fp, record)
            out = self.results_dir / f"{cfg.get('name', 'run')}_{fp}.json"
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

    def _execute(self, cfg: dict, meta: dict) -> dict:
        source = cfg.get("connectome_source", "auto")
        subgraph = cfg.get("subgraph")
        # Resource gate
        min_ram_gb = float(cfg.get("min_ram_gb", 0))
        if min_ram_gb > 0:
            # rough available estimate from /proc/meminfo if present
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

        t_load0 = time.perf_counter()
        connectome = load_reference_connectome(source=source, subgraph=subgraph)
        load_s = time.perf_counter() - t_load0

        method = cfg.get("method", "reference")
        seed = int(cfg.get("seed", 0))
        duration_ms = float(cfg.get("duration_ms", 80.0))
        rate_hz = float(cfg.get("rate_hz", 150.0))
        experiment = cfg.get("experiment", "escape")

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
        dtype = np.float32
        exec_precision = cfg.get("execution_dtype", "fp32")
        dtype = resolve_state_dtype(exec_precision) if exec_precision in (
            "fp16",
            "fp32",
            "fp64",
        ) else np.float32

        if method == "reference":
            variant_info["storage_precision"] = "int32_synapse_counts_csr"
            variant_info["execution_precision"] = str(np.dtype(dtype))
        elif method in ("fp16", "fp32", "int8", "int4"):
            q = quantise_weights(weights_mv, method)  # type: ignore[arg-type]
            recon = dequantise_weights(q, out_dtype=np.float32)
            variant_info["weight_reconstruction_error"] = reconstruction_error(
                weights_mv, recon
            )
            variant_info["quantisation"] = q.to_meta()
            weight_data = recon
            if method == "fp16" and cfg.get("native_fp16_execution", True):
                dtype = np.float16
                weight_data = dequantise_weights(q, out_dtype=np.float16)
            # persist artefact
            art_path = VARIANTS_DIR / f"{cfg.get('name', method)}_weights.npz"
            save_quantised(art_path, q)
            variant_info["artefact_path"] = str(art_path)
            variant_info["artefact_bytes"] = int(art_path.stat().st_size)
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
            # Live simulation still uses original connectome (lossless storage demo)
        elif method == "prune":
            pr = prune_connections(
                connectome.w,
                method=cfg.get("prune_method", "keep_fraction"),
                level=float(cfg.get("prune_level", 0.9)),
            )
            variant_info["prune"] = pr.to_meta()
            # Build a shallow wrapper with pruned matrix
            connectome = _with_matrix(connectome, pr.matrix)
            weights_mv = effective_mv_weights(connectome)
            weight_data = weights_mv.astype(np.float32)
        else:
            raise ValueError(f"unknown method: {method}")

        # Reference spike train: always run a reference pass for non-reference methods
        # using the same connectome topology before mutation... For prune, topology
        # changed — compare against unpruned reference loaded separately.
        rss_before = peak_rss_bytes()

        if method == "reference":
            ref_run = run_escape_benchmark(
                connectome,
                seed=seed,
                duration_ms=duration_ms,
                dtype=dtype,
                rate_hz=rate_hz,
                experiment=f"{experiment}:reference",
            )
            hyp_run = ref_run
            neural = neural_fidelity(
                ref_run.spike_counts, hyp_run.spike_counts, duration_ms=duration_ms
            )
            behaviour = behavioural_fidelity(ref_run.summary(), hyp_run.summary())
        else:
            # Load a clean reference connectome for comparison when topology may change
            ref_c = load_reference_connectome(source=source, subgraph=subgraph)
            ref_run = run_escape_benchmark(
                ref_c,
                seed=seed,
                duration_ms=duration_ms,
                dtype=np.float32,
                rate_hz=rate_hz,
                experiment=f"{experiment}:reference",
            )
            hyp_run = run_escape_benchmark(
                connectome,
                seed=seed,
                duration_ms=duration_ms,
                dtype=dtype,
                weight_data=weight_data,
                rate_hz=rate_hz,
                experiment=f"{experiment}:{method}",
            )
            neural = neural_fidelity(
                ref_run.spike_counts, hyp_run.spike_counts, duration_ms=duration_ms
            )
            behaviour = behavioural_fidelity(ref_run.summary(), hyp_run.summary())

        resources = measure_resources()
        # Ratio definitions (reported explicitly; never conflate file vs RAM):
        # - graph methods: reference CSR bytes / stored artefact bytes
        # - weight methods: reference wdata bytes / quantised payload (+meta) bytes
        compression_ratio = None
        weight_bytes_ref = int(weights_mv.astype(np.float32).nbytes)
        variant_info["reference_weight_bytes_fp32"] = weight_bytes_ref
        if method in ("fp16", "fp32", "int8", "int4") and variant_info.get("artefact_bytes"):
            compression_ratio = weight_bytes_ref / variant_info["artefact_bytes"]
            variant_info["compression_ratio_basis"] = "fp32_wdata_vs_quant_artefact"
        elif variant_info.get("graph_storage", {}).get("stored_nbytes"):
            compression_ratio = (
                model_bytes_ref / variant_info["graph_storage"]["stored_nbytes"]
            )
            variant_info["compression_ratio_basis"] = "csr_vs_lossless_artefact"
        elif variant_info.get("artefact_bytes"):
            compression_ratio = model_bytes_ref / variant_info["artefact_bytes"]
            variant_info["compression_ratio_basis"] = "csr_vs_artefact"

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
                "n_neurons": int(connectome.n),
                "n_connections": int(connectome.w.nnz),
                "n_synapses": int(np.abs(connectome.w.data).sum()),
                "weight_dtype": str(connectome.w.data.dtype),
                "csr_nbytes": model_bytes_ref,
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
