"""
Connection pruning (Phase 6 — kept independent of weight quantisation).

Only magnitude / threshold / percentage methods are implemented. Do not tune on
the final evaluation set; use calibration masks separately when exploring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp

PruneMethod = Literal["threshold", "percentile", "keep_fraction"]


@dataclass
class PruneResult:
    method: PruneMethod
    level: float
    original_nnz: int
    kept_nnz: int
    removed_nnz: int
    matrix: sp.csr_matrix
    depends_on_stimulus: bool = False

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
        }


def prune_connections(
    csr: sp.spmatrix,
    method: PruneMethod,
    level: float,
) -> PruneResult:
    """
    Return a new CSR with low-magnitude connections removed.

    * threshold: keep edges with |w| >= level
    * percentile: drop edges with |w| below the given percentile of |w|
    * keep_fraction: keep the top fraction of edges by |w| (level in (0,1])
    """
    w = csr.tocsr(copy=True)
    w.sort_indices()
    data = w.data.astype(np.float64)
    abs_w = np.abs(data)
    original = int(w.nnz)

    if original == 0:
        return PruneResult(method, level, 0, 0, 0, w)

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
    else:
        raise ValueError(f"unknown prune method: {method}")

    # Rebuild CSR from COO masked entries
    coo = w.tocoo()
    kept = mask
    new = sp.csr_matrix(
        (coo.data[kept], (coo.row[kept], coo.col[kept])),
        shape=w.shape,
    )
    new.sum_duplicates()
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
    )
