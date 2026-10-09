"""State-precision helpers and harness isolation."""

from __future__ import annotations

import numpy as np

from flyquant.benchmarks.harness import BenchmarkHarness
from flyquant.compression.state import (
    describe_state_precision,
    estimate_state_bytes,
    resolve_state_dtype,
)


def test_resolve_state_dtype():
    assert resolve_state_dtype("fp16") == np.float16
    assert resolve_state_dtype("state_fp16") == np.float16
    assert resolve_state_dtype("fp32") == np.float32
    assert resolve_state_dtype("fp64") == np.float64
    meta = describe_state_precision("state_fp16")
    assert meta["itemsize"] == 2


def test_estimate_state_bytes_scales_with_dtype():
    a = estimate_state_bytes(1000, 19, np.float32)
    b = estimate_state_bytes(1000, 19, np.float16)
    assert a["state_arrays_bytes"] == 2 * b["state_arrays_bytes"]


def test_state_fp16_harness(tmp_path):
    harness = BenchmarkHarness(results_dir=tmp_path / "results")
    ref = harness.run_config(
        {
            "name": "s_ref",
            "method": "reference",
            "experiment": "escape",
            "connectome_source": "synthetic",
            "subgraph": None,
            "seed": 0,
            "duration_ms": 15.0,
            "rate_hz": 200.0,
            "min_ram_gb": 0,
        },
        use_cache=False,
    )
    hyp = harness.run_config(
        {
            "name": "s_fp16",
            "method": "state_fp16",
            "experiment": "escape",
            "connectome_source": "synthetic",
            "subgraph": None,
            "seed": 0,
            "duration_ms": 15.0,
            "rate_hz": 200.0,
            "min_ram_gb": 0,
        },
        use_cache=False,
    )
    assert ref.status == "ok"
    assert hyp.status == "ok"
    assert hyp.results["variant"]["weight_execution"] == "float32_wdata_override"
    assert "float16" in hyp.results["variant"]["execution_precision"]
