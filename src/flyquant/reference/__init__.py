"""Immutable upstream adapter. Never mutates fruit-fly-lab source during runs."""

from flyquant.reference.adapter import (
    DerivedDataUnavailable,
    derived_data_status,
    ensure_upstream_on_path,
    get_upstream_commit,
    load_reference_connectome,
    make_engine,
    run_escape_benchmark,
    run_escape_controls,
    run_looming_escape,
)
from flyquant.reference.synthetic import make_synthetic_connectome

__all__ = [
    "DerivedDataUnavailable",
    "derived_data_status",
    "ensure_upstream_on_path",
    "get_upstream_commit",
    "load_reference_connectome",
    "make_engine",
    "run_escape_benchmark",
    "run_escape_controls",
    "run_looming_escape",
    "make_synthetic_connectome",
]
