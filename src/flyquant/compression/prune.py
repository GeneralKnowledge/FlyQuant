"""
Connection pruning (Phase 6 — kept independent of weight quantisation).

Magnitude / threshold / percentage / static-graph importance methods.
Do not tune on the final evaluation set; use fixed a-priori levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

PruneMethod = Literal[
    "threshold",
    "percentile",
    "keep_fraction",
    "importance_magnitude_outdegree",
    "importance_dnp01_path",
]


@dataclass
class PruneResult:
    method: PruneMethod
    level: float
    original_nnz: int
    kept_nnz: int
    removed_nnz: int
    matrix: sp.csr_matrix
    depends_on_stimulus: bool = False
    abs_hist_before: dict[str, Any] = field(default_factory=dict)
    abs_hist_after: dict[str, Any] = field(default_factory=dict)
    importance_notes: str | None = None

    def to_meta(self) -> dict:
        return {
            "method": self.method,
            "level": self.level,
            "original_nnz": self.original_nnz,
            "kept_nnz": self.kept_nnz,
            "removed_nnz": self.removed_nnz,
            "depends_on_stimulus": self.depends_on_stimulus,
            "fraction_removed": (
                self.removed_nnz / self.original_nnz if self.original_nnz else 0.0
            ),
            "abs_hist_before": self.abs_hist_before,
            "abs_hist_after": self.abs_hist_after,
            "importance_notes": self.importance_notes,
        }


def _abs_histogram(data: np.ndarray, bins: int = 10) -> dict[str, Any]:
    if data.size == 0:
        return {"bins": [], "counts": [], "n": 0}
    abs_w = np.abs(data.astype(np.float64))
    counts, edges = np.histogram(abs_w, bins=bins)
    return {
        "n": int(abs_w.size),
        "min": float(abs_w.min()),
        "max": float(abs_w.max()),
        "mean": float(abs_w.mean()),
        "edges": [float(x) for x in edges],
        "counts": [int(x) for x in counts],
    }


def _importance_scores(
    w: sp.csr_matrix,
    method: str,
    dnp01_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Per-edge importance on the static graph (COO order of w.tocoo())."""
    coo = w.tocoo()
    abs_w = np.abs(coo.data.astype(np.float64))
    if method == "importance_magnitude_outdegree":
        # score = |w| * out_degree(pre)
        out_deg = np.asarray(w.getnnz(axis=1), dtype=np.float64)
        return abs_w * out_deg[coo.row]
    if method == "importance_dnp01_path":
        # Prefer edges into DNp01; others scored by |w| only (still static).
        targets = set(int(i) for i in (dnp01_indices if dnp01_indices is not None else []))
        boost = np.array([4.0 if int(j) in targets else 1.0 for j in coo.col])
        return abs_w * boost
    raise ValueError(f"unknown importance method: {method}")


def prune_connections(
    csr: sp.spmatrix,
    method: PruneMethod,
    level: float,
    *,
    dnp01_indices: np.ndarray | None = None,
) -> PruneResult:
    """
    Return a new CSR with low-importance / low-magnitude connections removed.

    * threshold: keep edges with |w| >= level
    * percentile: drop edges with |w| below the given percentile of |w|
    * keep_fraction: keep the top fraction of edges by |w| (level in (0,1])
    * importance_*: keep top fraction by static-graph score (level in (0,1])
    """
    w = csr.tocsr(copy=True)
    w.sort_indices()
    data = w.data.astype(np.float64)
    abs_w = np.abs(data)
    original = int(w.nnz)
    hist_before = _abs_histogram(data)
    importance_notes = None

    if original == 0:
        return PruneResult(method, level, 0, 0, 0, w, abs_hist_before=hist_before)

    if method == "threshold":
        mask = abs_w >= float(level)
    elif method == "percentile":
        thr = float(np.percentile(abs_w, float(level)))
        mask = abs_w >= thr
    elif method == "keep_fraction":
        frac = float(level)
        if not 0.0 < frac <= 1.0:
            raise ValueError("keep_fraction level must be in (0, 1]")
        k = max(1, int(round(frac * original)))
        order = np.argsort(-abs_w)
        mask = np.zeros(original, dtype=bool)
        mask[order[:k]] = True
    elif method in ("importance_magnitude_outdegree", "importance_dnp01_path"):
        frac = float(level)
        if not 0.0 < frac <= 1.0:
            raise ValueError("importance keep fraction must be in (0, 1]")
        scores = _importance_scores(w, method, dnp01_indices=dnp01_indices)
        k = max(1, int(round(frac * original)))
        order = np.argsort(-scores)
        mask = np.zeros(original, dtype=bool)
        # scores align with COO order — rebuild via COO mask
        coo = w.tocoo()
        mask_coo = np.zeros(original, dtype=bool)
        mask_coo[order[:k]] = True
        new = sp.csr_matrix(
            (coo.data[mask_coo], (coo.row[mask_coo], coo.col[mask_coo])),
            shape=w.shape,
        )
        new.sum_duplicates()
        new.sort_indices()
        importance_notes = (
            "Static-graph importance only; depends_on_stimulus=false. "
            "Not tuned on behavioural evaluation outcomes."
        )
        return PruneResult(
            method=method,
            level=float(level),
            original_nnz=original,
            kept_nnz=int(new.nnz),
            removed_nnz=original - int(new.nnz),
            matrix=new,
            depends_on_stimulus=False,
            abs_hist_before=hist_before,
            abs_hist_after=_abs_histogram(new.data),
            importance_notes=importance_notes,
        )
    else:
        raise ValueError(f"unknown prune method: {method}")

    coo = w.tocoo()
    # For threshold/percentile/keep_fraction, mask is in CSR data order which
    # matches sorted CSR; convert via COO carefully: use CSR data mask on CSR.
    # Rebuild from CSR by zeroing then eliminate zeros.
    data_kept = data.copy()
    data_kept[~mask] = 0
    new = sp.csr_matrix((data_kept, w.indices.copy(), w.indptr.copy()), shape=w.shape)
    new.eliminate_zeros()
    new.sort_indices()
    kept_nnz = int(new.nnz)
    return PruneResult(
        method=method,
        level=float(level),
        original_nnz=original,
        kept_nnz=kept_nnz,
        removed_nnz=original - kept_nnz,
        matrix=new,
        depends_on_stimulus=False,
        abs_hist_before=hist_before,
        abs_hist_after=_abs_histogram(new.data),
        importance_notes=importance_notes,
    )
