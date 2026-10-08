"""Immutable upstream adapter. Never mutates fruit-fly-lab source during runs."""

from flyquant.reference.adapter import (
    ensure_upstream_on_path,
    get_upstream_commit,
    load_reference_connectome,
    make_engine,
    run_escape_benchmark,
)
from flyquant.reference.synthetic import make_synthetic_connectome

__all__ = [
    "ensure_upstream_on_path",
    "get_upstream_commit",
    "load_reference_connectome",
    "make_engine",
    "run_escape_benchmark",
    "make_synthetic_connectome",
]
