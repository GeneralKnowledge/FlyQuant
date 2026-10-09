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


@contextmanager
def timed_section() -> Iterator[dict]:
    box: dict = {}
    t0 = time.perf_counter()
    yield box
    box["wall_time_s"] = time.perf_counter() - t0
    box["peak_rss_bytes"] = peak_rss_bytes()
