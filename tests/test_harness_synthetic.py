"""End-to-end harness on synthetic connectome (no FlyWire download)."""

from __future__ import annotations

from flyquant.benchmarks.harness import BenchmarkHarness
from flyquant.reference.adapter import run_escape_benchmark
from flyquant.reference.synthetic import make_synthetic_connectome


def test_synthetic_escape_runs():
    c = make_synthetic_connectome(n=32, seed=0)
    result = run_escape_benchmark(c, seed=0, duration_ms=20.0, rate_hz=200.0)
    assert result.n_neurons == 32
    assert result.sim_steps == 200
    assert result.lc4_spikes >= 0


def test_harness_reference_and_fp16(tmp_path):
    harness = BenchmarkHarness(results_dir=tmp_path / "results")
    cfg = {
        "name": "unit_ref",
        "method": "reference",
        "experiment": "escape",
        "connectome_source": "synthetic",
        "subgraph": None,
        "seed": 1,
        "duration_ms": 15.0,
        "rate_hz": 200.0,
        "min_ram_gb": 0,
    }
    ref = harness.run_config(cfg, use_cache=False)
    assert ref.status == "ok"

    cfg2 = dict(cfg)
    cfg2["name"] = "unit_fp16"
    cfg2["method"] = "fp16"
    hyp = harness.run_config(cfg2, use_cache=False)
    assert hyp.status == "ok"
    assert "neural_fidelity" in hyp.results
    assert hyp.results["behavioural_fidelity"]["escape_agreement"] in (True, False)


def test_harness_int8_and_lossless(tmp_path):
    harness = BenchmarkHarness(results_dir=tmp_path / "results")
    base = {
        "experiment": "escape",
        "connectome_source": "synthetic",
        "subgraph": None,
        "seed": 2,
        "duration_ms": 15.0,
        "rate_hz": 200.0,
        "min_ram_gb": 0,
    }
    r1 = harness.run_config({**base, "name": "u_int8", "method": "int8"}, use_cache=False)
    r2 = harness.run_config(
        {**base, "name": "u_lossy", "method": "lossless_graph", "gzip": True},
        use_cache=False,
    )
    assert r1.status == "ok"
    assert r2.status == "ok"
    assert r2.results["variant"]["graph_roundtrip"]["identical"] is True


def test_cache_hit(tmp_path):
    harness = BenchmarkHarness(results_dir=tmp_path / "results")
    cfg = {
        "name": "cache_me",
        "method": "reference",
        "experiment": "escape",
        "connectome_source": "synthetic",
        "subgraph": None,
        "seed": 3,
        "duration_ms": 10.0,
        "rate_hz": 200.0,
        "min_ram_gb": 0,
    }
    a = harness.run_config(cfg, use_cache=True)
    b = harness.run_config(cfg, use_cache=True)
    assert a.status == "ok"
    assert b.status == "ok"
    assert b.results.get("cache_hit") is True
