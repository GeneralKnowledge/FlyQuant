"""Lossless graph compression tests on synthetic CSR graphs."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from flyquant.compression.graph import (
    compress_graph_lossless,
    graph_identical,
    load_graph_artefact,
    reconstruct_csr,
    save_graph_artefact,
)
from flyquant.reference.synthetic import make_synthetic_connectome


def test_delta_roundtrip_identical():
    c = make_synthetic_connectome(n=40, seed=2)
    art = compress_graph_lossless(c.w)
    recon = reconstruct_csr(art)
    result = graph_identical(c.w, recon)
    assert result["identical"]


def test_save_load_gzip_roundtrip(tmp_path):
    c = make_synthetic_connectome(n=24, seed=3)
    art = compress_graph_lossless(c.w, metadata={"test": True})
    meta = save_graph_artefact(tmp_path / "g", art, gzip_file=True)
    assert meta["stored_nbytes"] <= meta["raw_nbytes"]
    loaded = load_graph_artefact(tmp_path / "g.json")
    recon = reconstruct_csr(loaded)
    assert graph_identical(c.w, recon)["identical"]


def test_int16_weight_dtype_when_needed():
    rows = [0, 1]
    cols = [1, 0]
    data = [400, -400]  # needs int16
    w = sp.csr_matrix((data, (rows, cols)), shape=(3, 3))
    art = compress_graph_lossless(w)
    assert art.weight_dtype == "int16"
    assert graph_identical(w, reconstruct_csr(art))["identical"]


def test_rejects_non_integral_floats():
    w = sp.csr_matrix(([0.5], ([0], [1])), shape=(2, 2))
    try:
        compress_graph_lossless(w)
        assert False, "expected ValueError"
    except ValueError:
        pass
