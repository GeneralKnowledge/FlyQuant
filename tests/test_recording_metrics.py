"""Recording wrapper and spike-time / membrane metrics."""

from __future__ import annotations

import numpy as np

from flyquant.metrics.neural import membrane_trace_error, spike_time_coincidence
from flyquant.reference.adapter import run_escape_benchmark
from flyquant.reference.recording import RecordingConfig, RecordingEngine
from flyquant.reference.synthetic import make_synthetic_connectome
from flyquant.reference.adapter import make_engine


def test_spike_time_coincidence_perfect():
    times = {0: [1.0, 2.0, 3.0], 1: []}
    m = spike_time_coincidence(times, times, tolerance_ms=0.5)
    assert m["skipped"] is False
    assert m["mean_f1"] == 1.0


def test_spike_time_empty_skipped():
    m = spike_time_coincidence({}, {}, tolerance_ms=1.0)
    assert m["skipped"] is True


def test_membrane_skipped_without_traces():
    m = membrane_trace_error(None, None)
    assert m["skipped"] is True


def test_membrane_error_identical():
    t = np.zeros((5, 3), dtype=np.float32)
    m = membrane_trace_error(t, t, times_ms=[0, 1, 2, 3, 4])
    assert m["skipped"] is False
    assert m["mae_mV"] == 0.0


def test_recording_engine_watches():
    c = make_synthetic_connectome(n=16, seed=1)
    eng = make_engine(c, seed=0)
    watch = c.by_cell_type("DNp01")["idx"].to_numpy()
    rec = RecordingEngine(
        eng,
        RecordingConfig(record_spikes=True, record_v=True, watch_indices=watch, v_stride=5),
    )
    drive = np.concatenate(
        [
            c.by_cell_type("LC4")["idx"].to_numpy(),
            c.by_cell_type("LPLC2")["idx"].to_numpy(),
        ]
    )
    eng.set_poisson(drive, 200.0)
    for _ in range(50):
        rec.step()
    result = rec.finalise()
    assert result.enabled
    assert result.v_traces is not None
    assert result.v_traces.shape[0] >= 1


def test_escape_with_recording():
    c = make_synthetic_connectome(n=24, seed=2)
    r = run_escape_benchmark(
        c,
        seed=0,
        duration_ms=10.0,
        rate_hz=200.0,
        recording=RecordingConfig(record_spikes=True, record_v=True, v_stride=5),
    )
    assert r.recording is not None
    assert r.recording.to_meta()["enabled"] is True
