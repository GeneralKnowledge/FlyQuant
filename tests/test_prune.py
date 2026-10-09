from __future__ import annotations

from flyquant.compression.prune import prune_connections
from flyquant.reference.synthetic import make_synthetic_connectome


def test_keep_fraction_reduces_nnz():
    c = make_synthetic_connectome(n=40, seed=4)
    pr = prune_connections(c.w, "keep_fraction", 0.5)
    assert pr.kept_nnz <= c.w.nnz
    assert pr.removed_nnz == c.w.nnz - pr.kept_nnz
    assert pr.depends_on_stimulus is False


def test_threshold_pruning():
    c = make_synthetic_connectome(n=30, seed=5)
    pr = prune_connections(c.w, "threshold", 5)
    assert pr.kept_nnz <= c.w.nnz
    if pr.kept_nnz:
        assert abs(pr.matrix.data).min() >= 5
