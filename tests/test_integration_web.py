"""Optional integration tests requiring fruit-fly-lab web connectome assets."""

from __future__ import annotations

import pytest

from flyquant.paths import REFERENCE_DIR
from flyquant.reference.web_loader import web_assets_available


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def web_connectome():
    if not REFERENCE_DIR.exists() or not web_assets_available():
        pytest.skip("fruit-fly-lab web assets not available")
    from flyquant.reference.adapter import load_reference_connectome

    return load_reference_connectome(source="web", subgraph="escape")


def test_web_escape_subgraph_loads(web_connectome):
    assert web_connectome.n > 100
    assert web_connectome.n <= 4000
    assert web_connectome.w.nnz > 0
    assert not web_connectome.by_cell_type("DNp01").empty


def test_web_lossless_roundtrip(web_connectome):
    from flyquant.compression.graph import (
        compress_graph_lossless,
        graph_identical,
        reconstruct_csr,
    )

    art = compress_graph_lossless(web_connectome.w)
    assert graph_identical(web_connectome.w, reconstruct_csr(art))["identical"]


@pytest.mark.slow
def test_web_fp16_benchmark_smoke(web_connectome, tmp_path):
    from flyquant.benchmarks.harness import BenchmarkHarness

    harness = BenchmarkHarness(results_dir=tmp_path / "results")
    rec = harness.run_config(
        {
            "name": "web_fp16_smoke",
            "method": "fp16",
            "experiment": "escape_subgraph",
            "connectome_source": "web",
            "subgraph": "escape",
            "seed": 0,
            "duration_ms": 30.0,
            "rate_hz": 150.0,
            "min_ram_gb": 0,
        },
        use_cache=False,
    )
    assert rec.status == "ok"
    assert "neural_fidelity" in rec.results
