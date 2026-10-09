"""
Neuron parameter / runtime-state precision experiments (Phase 5).

Treats state precision separately from weight quantisation. Upstream LIFEngine
accepts a numpy dtype for v, g, ring buffer and wdata. There is no INT8 state
kernel; integer state would require a custom integrator (out of scope for v0.1).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

StatePrecision = Literal["fp32", "fp16", "fp64"]


def resolve_state_dtype(precision: StatePrecision):
    return {
        "fp32": np.float32,
        "fp16": np.float16,
        "fp64": np.float64,
    }[precision]


def describe_state_precision(precision: StatePrecision) -> dict:
    dt = np.dtype(resolve_state_dtype(precision))
    return {
        "precision": precision,
        "numpy_dtype": str(dt),
        "itemsize": int(dt.itemsize),
        "notes": (
            "Applies to engine arrays v, g, synaptic delay ring, and wdata when "
            "constructed via LIFEngine(..., dtype=...). Integer state arithmetic "
            "is not supported by the upstream engine."
        ),
    }
