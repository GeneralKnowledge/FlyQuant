from __future__ import annotations

import json

from flyquant.benchmarks.harness import BenchmarkHarness
from flyquant.reports.generate import generate_report


def test_report_generation(tmp_path):
    results = tmp_path / "results"
    reports = tmp_path / "reports"
    harness = BenchmarkHarness(results_dir=results)
    for method, name in [("reference", "r"), ("fp16", "f"), ("int8", "i")]:
        harness.run_config(
            {
                "name": name,
                "method": method,
                "experiment": "escape",
                "connectome_source": "synthetic",
                "subgraph": None,
                "seed": 0,
                "duration_ms": 10.0,
                "rate_hz": 200.0,
                "min_ram_gb": 0,
            },
            use_cache=False,
        )
    paths = generate_report(results_dir=results, out_dir=reports)
    assert paths["markdown"].exists()
    agg = json.loads(paths["json"].read_text())
    assert agg["n_ok"] >= 3
