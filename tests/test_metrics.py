"""Fidelity metric edge cases."""

from __future__ import annotations

import numpy as np

from flyquant.metrics.behavioural import behavioural_fidelity
from flyquant.metrics.neural import neural_fidelity


def test_all_silent_neurons():
    a = np.zeros(10)
    b = np.zeros(10)
    m = neural_fidelity(a, b, duration_ms=100.0)
    assert m["n_silent_reference"] == 10
    assert m["rate_corr_all"] == 1.0
    assert m["spike_count_exact_match_fraction"] == 1.0


def test_zero_variance_mismatch_returns_none_corr():
    a = np.zeros(5)
    b = np.array([0, 0, 0, 0, 3], dtype=float)
    m = neural_fidelity(a, b, duration_ms=50.0)
    assert m["rate_corr_all"] is None
    assert m["false_active"] == 1


def test_behaviour_false_negative():
    ref = {"gf_spikes": 10, "first_gf_spike_ms": 40.0, "escape_peak": 0.8}
    hyp = {"gf_spikes": 0, "first_gf_spike_ms": None, "escape_peak": 0.0}
    b = behavioural_fidelity(ref, hyp)
    assert b["false_negative"] is True
    assert b["escape_agreement"] is False


def test_behaviour_latency_delta():
    ref = {"gf_spikes": 5, "first_gf_spike_ms": 30.0}
    hyp = {"gf_spikes": 4, "first_gf_spike_ms": 35.0}
    b = behavioural_fidelity(ref, hyp)
    assert b["latency_delta_ms"] == 5.0
    assert b["escape_agreement"] is True
