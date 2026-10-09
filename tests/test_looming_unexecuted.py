"""Looming / escape_controls must unexecuted cleanly without derived data."""

from __future__ import annotations

from flyquant.benchmarks.harness import BenchmarkHarness
from flyquant.reference.adapter import derived_data_status


def test_derived_status_reports_missing():
    status = derived_data_status()
    # In this environment derived NPZ is typically absent.
    assert "available" in status
    assert "missing" in status or status["available"]


def test_looming_unexecuted_without_derived(tmp_path):
    harness = BenchmarkHarness(results_dir=tmp_path / "results")
    rec = harness.run_config(
        {
            "name": "loom_test",
            "method": "reference",
            "experiment": "looming",
            "connectome_source": "derived",
            "subgraph": None,
            "seed": 0,
            "duration_ms": 50.0,
            "min_ram_gb": 0,
        },
        use_cache=False,
    )
    status = derived_data_status()
    if status["available"]:
        assert rec.status in ("ok", "error")
    else:
        assert rec.status == "unexecuted"
        assert "Derived" in (rec.results.get("reason") or "") or "missing" in (
            rec.results.get("reason") or ""
        ).lower()


def test_escape_controls_unexecuted_without_derived(tmp_path):
    harness = BenchmarkHarness(results_dir=tmp_path / "results")
    rec = harness.run_config(
        {
            "name": "ctrl_test",
            "method": "reference",
            "experiment": "escape_controls",
            "connectome_source": "derived",
            "subgraph": None,
            "seed": 0,
            "duration_ms": 50.0,
            "min_ram_gb": 0,
        },
        use_cache=False,
    )
    if not derived_data_status()["available"]:
        assert rec.status == "unexecuted"
