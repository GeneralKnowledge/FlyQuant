"""
Lossless connectivity-graph compression.

Methods
-------
* CSR layout (already the live representation)
* int32 / int16 / int8 index/weight width selection where values fit
* Delta encoding of sorted CSR column indices (as in upstream web export)
* Optional gzip of the serialised artefact

Lossless file compression is reported separately from live simulation memory.
Round-trip tests require exact equality of indptr, indices, and data.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp


def delta_encode_indices(indices: np.ndarray, indptr: np.ndarray) -> np.ndarray:
    """Delta-code CSR column indices within each row (sorted rows)."""
    d = indices.astype(np.int64).copy()
    if d.size == 0:
        return d.astype(np.int32)
    prev = np.empty_like(d)
    prev[0] = 0
    prev[1:] = d[:-1]
    delta = d - prev
    row_starts = indptr[:-1]
    row_starts = row_starts[row_starts < len(d)]
    delta[row_starts] = d[row_starts]
    return delta.astype(np.int32)


def delta_decode_indices(delta: np.ndarray, indptr: np.ndarray) -> np.ndarray:
    out = delta.astype(np.int64).copy()
    for a, b in zip(indptr[:-1], indptr[1:]):
        if b > a:
            out[a:b] = np.cumsum(out[a:b])
    return out.astype(np.int32)


@dataclass
class GraphArtefact:
    """Lossless serialisation of a signed synapse-count CSR."""

    n: int
    nnz: int
    indptr: np.ndarray
    indices_delta: np.ndarray
    weights: np.ndarray
    weight_dtype: str
    format: str = "csr_delta_v1"
    metadata: dict | None = None

    def nbytes_raw(self) -> int:
        return int(self.indptr.nbytes + self.indices_delta.nbytes + self.weights.nbytes)

    def sha256(self) -> str:
        h = hashlib.sha256()
        h.update(self.indptr.tobytes())
        h.update(self.indices_delta.tobytes())
        h.update(self.weights.tobytes())
        return h.hexdigest()


def _choose_weight_dtype(data: np.ndarray) -> tuple[np.dtype, np.ndarray]:
    max_abs = int(np.max(np.abs(data))) if data.size else 0
    if max_abs <= 127:
        dt = np.dtype(np.int8)
    elif max_abs <= 32767:
        dt = np.dtype(np.int16)
    else:
        dt = np.dtype(np.int32)
    return dt, data.astype(dt)


def compress_graph_lossless(csr: sp.spmatrix, metadata: dict | None = None) -> GraphArtefact:
    w = csr.tocsr()
    w.sort_indices()
    if not np.issubdtype(w.data.dtype, np.integer):
        # Allow float arrays that are integral-valued (e.g. after load).
        if not np.allclose(w.data, np.rint(w.data)):
            raise ValueError("lossless graph compressor requires integer synapse counts")
        data = np.rint(w.data).astype(np.int32)
    else:
        data = w.data.astype(np.int32)

    indptr = w.indptr.astype(np.int32)
    delta = delta_encode_indices(w.indices.astype(np.int32), indptr)
    # verify
    if not np.array_equal(delta_decode_indices(delta, indptr), w.indices.astype(np.int32)):
        raise RuntimeError("delta encoding round-trip failed before save")

    wdt, weights = _choose_weight_dtype(data)
    return GraphArtefact(
        n=int(w.shape[0]),
        nnz=int(w.nnz),
        indptr=indptr,
        indices_delta=delta,
        weights=weights,
        weight_dtype=str(wdt),
        metadata=metadata or {},
    )


def reconstruct_csr(art: GraphArtefact) -> sp.csr_matrix:
    indices = delta_decode_indices(art.indices_delta, art.indptr)
    data = art.weights.astype(np.int32)
    mat = sp.csr_matrix((data, indices, art.indptr.copy()), shape=(art.n, art.n))
    mat.sort_indices()
    return mat


def graph_identical(a: sp.spmatrix, b: sp.spmatrix) -> dict[str, Any]:
    aa, bb = a.tocsr(), b.tocsr()
    aa.sort_indices()
    bb.sort_indices()
    same_shape = aa.shape == bb.shape
    same_nnz = aa.nnz == bb.nnz
    same_indptr = same_shape and np.array_equal(aa.indptr, bb.indptr)
    same_indices = same_shape and same_nnz and np.array_equal(aa.indices, bb.indices)
    same_data = same_shape and same_nnz and np.array_equal(aa.data, bb.data)
    return {
        "identical": bool(same_shape and same_indptr and same_indices and same_data),
        "same_shape": bool(same_shape),
        "same_nnz": bool(same_nnz),
        "same_indptr": bool(same_indptr),
        "same_indices": bool(same_indices),
        "same_data": bool(same_data),
        "a_nnz": int(aa.nnz),
        "b_nnz": int(bb.nnz),
        "a_abs_sum": int(np.abs(aa.data).sum()) if aa.nnz else 0,
        "b_abs_sum": int(np.abs(bb.data).sum()) if bb.nnz else 0,
    }


def save_graph_artefact(path: Path, art: GraphArtefact, *, gzip_file: bool = True) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "format": art.format,
        "n": art.n,
        "nnz": art.nnz,
        "weight_dtype": art.weight_dtype,
        "sha256": art.sha256(),
        "raw_nbytes": art.nbytes_raw(),
        "gzip": gzip_file,
        "metadata": art.metadata or {},
    }
    raw = art.indptr.tobytes() + art.indices_delta.tobytes() + art.weights.tobytes()
    bin_path = path.with_suffix(".bin")
    meta_path = path.with_suffix(".json")
    if gzip_file:
        payload = gzip.compress(raw, compresslevel=6)
        bin_path = path.with_suffix(".bin.gz")
        bin_path.write_bytes(payload)
        meta["stored_nbytes"] = len(payload)
        meta["compression_ratio_vs_raw"] = (
            meta["raw_nbytes"] / len(payload) if payload else None
        )
    else:
        bin_path.write_bytes(raw)
        meta["stored_nbytes"] = len(raw)
        meta["compression_ratio_vs_raw"] = 1.0
    meta_path.write_text(json.dumps(meta, indent=2))
    meta["bin_path"] = str(bin_path)
    meta["meta_path"] = str(meta_path)
    return meta


def load_graph_artefact(meta_path: Path) -> GraphArtefact:
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text())
    bin_path = Path(meta.get("bin_path", meta_path.with_suffix(".bin.gz")))
    if not bin_path.exists():
        # try beside meta
        cand = meta_path.with_suffix(".bin.gz")
        bin_path = cand if cand.exists() else meta_path.with_suffix(".bin")
    blob = bin_path.read_bytes()
    if meta.get("gzip") or str(bin_path).endswith(".gz"):
        blob = gzip.decompress(blob)
    n = int(meta["n"])
    nnz = int(meta["nnz"])
    wdt = np.dtype(meta["weight_dtype"])
    indptr = np.frombuffer(blob, dtype=np.int32, count=n + 1)
    delta = np.frombuffer(blob, dtype=np.int32, count=nnz, offset=4 * (n + 1))
    w_off = 4 * (n + 1) + 4 * nnz
    weights = np.frombuffer(blob, dtype=wdt, count=nnz, offset=w_off)
    return GraphArtefact(
        n=n,
        nnz=nnz,
        indptr=np.array(indptr, copy=True),
        indices_delta=np.array(delta, copy=True),
        weights=np.array(weights, copy=True),
        weight_dtype=str(wdt),
        metadata=meta.get("metadata") or {},
    )
