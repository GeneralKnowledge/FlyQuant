"""Prune histograms, importance methods, and pipeline harness."""

from __future__ import annotations

from flyquant.benchmarks.harness import BenchmarkHarness
from flyquant.compression.prune import prune_connections
from flyquant.reference.synthetic import make_synthetic_connectome


def test_prune_histogram_present():
    c = make_synthetic_connectome(n=40, seed=4)
    pr = prune_connections(c.w, "keep_fraction", 0.5)
    assert pr.abs_hist_before["n"] == c.w.nnz
    assert "counts" in pr.abs_hist_after
    assert pr.depends_on_stimulus is False


def test_importance_outdegree():
    c = make_synthetic_connectome(n=40, seed=5)
    pr = prune_connections(c.w, "importance_magnitude_outdegree", 0.8)
    assert pr.kept_nnz <= c.w.nnz
    assert pr.importance_notes is not None


def test_importance_dnp01():
    c = make_synthetic_connectome(n=40, seed=6)
    dnp = c.by_cell_type("DNp01")["idx"].to_numpy()
    pr = prune_connections(c.w, "importance_dnp01_path", 0.8, dnp01_indices=dnp)
    assert pr.kept_nnz > 0


def test_harness_prune_and_pipeline(tmp_path):
    harness = BenchmarkHarness(results_dir=tmp_path / "results")
    base = {
        "experiment": "escape",
        "connectome_source": "synthetic",
        "subgraph": None,
        "seed": 0,
        "duration_ms": 12.0,
        "rate_hz": 200.0,
        "min_ram_gb": 0,
    }
    pr = harness.run_config(
        {**base, "name": "p", "method": "prune", "prune_method": "keep_fraction", "prune_level": 0.8},
        use_cache=False,
    )
    pipe = harness.run_config(
        {
            **base,
            "name": "pipe",
            "method": "pipeline",
            "steps": [
                {"op": "prune", "prune_method": "keep_fraction", "prune_level": 0.9},
                {"op": "quantise", "precision": "int8"},
            ],
        },
        use_cache=False,
    )
    assert pr.status == "ok"
    assert "abs_hist_before" in pr.results["variant"]["prune"]
    assert pipe.status == "ok"
    assert len(pipe.results["variant"]["pipeline"]["steps"]) == 2
