"""Unit tests for weight quantisation (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pytest

from flyquant.compression.weights import (
    dequantise_weights,
    quantise_weights,
    reconstruction_error,
)
from flyquant.reference.synthetic import make_synthetic_connectome
from flyquant.compression.weights import effective_mv_weights


def test_fp32_is_identity():
    w = np.array([0.275, -1.1, 3.575], dtype=np.float64)
    q = quantise_weights(w, "fp32")
    recon = dequantise_weights(q)
    assert recon.dtype == np.float32
    np.testing.assert_allclose(recon, w, rtol=0, atol=1e-6)


def test_fp16_roundtrip_bound():
    w = np.linspace(-5, 5, 1001, dtype=np.float64)
    q = quantise_weights(w, "fp16")
    recon = dequantise_weights(q, out_dtype=np.float32)
    err = reconstruction_error(w, recon)
    assert err["max_abs"] < 0.01
    assert err["corr"] > 0.999


def test_int8_scale_clip_and_bounds():
    w = np.array([-10.0, -0.1, 0.0, 0.2, 10.0], dtype=np.float64)
    q = quantise_weights(w, "int8")
    assert q.payload.dtype == np.int8
    assert q.payload.min() >= -127
    assert q.payload.max() <= 127
    assert q.zero_point == 0
    recon = dequantise_weights(q)
    # extremes should reconstruct near original with symmetric scale
    assert abs(float(recon[0]) - (-10.0)) < 0.1
    assert abs(float(recon[-1]) - 10.0) < 0.1


def test_int4_clips_to_nibble_range():
    w = np.array([-100.0, 100.0], dtype=np.float64)
    q = quantise_weights(w, "int4")
    assert q.payload.min() >= -7
    assert q.payload.max() <= 7
    recon = dequantise_weights(q)
    err = reconstruction_error(w, recon)
    assert err["mae"] >= 0.0


def test_does_not_treat_indices_as_weights():
    # Sanity: quantiser operates on provided vector only; indices never passed in.
    c = make_synthetic_connectome(n=16, seed=1)
    w = effective_mv_weights(c)
    q = quantise_weights(w, "int8")
    assert q.n_weights == c.w.nnz
    assert q.n_weights != c.n


def test_empty_weights():
    w = np.array([], dtype=np.float64)
    q = quantise_weights(w, "int8")
    recon = dequantise_weights(q)
    assert recon.size == 0
    err = reconstruction_error(w, recon)
    assert err["n"] == 0
