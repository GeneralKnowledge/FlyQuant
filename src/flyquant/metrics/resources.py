"""Process memory / timing helpers."""

from __future__ import annotations

import os
import platform
import resource
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import numpy as np

from flyquant.compression.state import estimate_state_bytes


@dataclass
class ResourceSnapshot:
    peak_rss_bytes: int | None
    platform: str
    python_version: str
    cpu_count: int | None
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def peak_rss_bytes() -> int | None:
    """Peak resident set size in bytes (best-effort, Unix)."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: kilobytes; macOS: bytes
        if sys.platform == "darwin":
            return int(usage)
        return int(usage) * 1024
    except Exception:
        return None


def measure_resources() -> ResourceSnapshot:
    return ResourceSnapshot(
        peak_rss_bytes=peak_rss_bytes(),
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        cpu_count=os.cpu_count(),
        notes="ru_maxrss is process peak RSS; device/GPU memory N/A (CPU-only engine).",
    )


def engine_state_footprint(engine) -> dict[str, Any]:
    """
    Measure in-memory footprint of dynamic state arrays on a live engine.

    Separates CSR / weight storage from v, g, and the synaptic delay ring.
    """
    v_b = int(getattr(engine, "v").nbytes)
    g_b = int(getattr(engine, "g").nbytes)
    ring_b = int(getattr(engine, "_ring").nbytes)
    wdata_b = int(getattr(engine, "wdata").nbytes)
    estimated = estimate_state_bytes(
        n_neurons=int(engine.n),
        delay_slots=int(engine._ring.shape[0]),
        dtype=engine.dtype,
    )
    return {
        "dtype": str(np.dtype(engine.dtype)),
        "v_bytes": v_b,
        "g_bytes": g_b,
        "ring_bytes": ring_b,
        "wdata_bytes": wdata_b,
        "state_arrays_bytes": v_b + g_b + ring_b,
        "estimated": estimated,
    }


@contextmanager
def timed_section() -> Iterator[dict]:
    box: dict = {}
    t0 = time.perf_counter()
    yield box
    box["wall_time_s"] = time.perf_counter() - t0
    box["peak_rss_bytes"] = peak_rss_bytes()
