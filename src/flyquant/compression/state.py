"""
Neuron parameter / runtime-state precision experiments (Phase 5).

Treats state precision separately from weight quantisation. Upstream LIFEngine
accepts a numpy dtype for v, g, ring buffer and wdata. There is no INT8 state
kernel; integer state would require a custom integrator.

Methods ``state_fp16`` / ``state_fp64`` keep reference synapse-count weights
(reconstructed as float32 mV) while lowering only the dynamic-state dtype.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

StatePrecision = Literal["fp32", "fp16", "fp64"]
STATE_METHODS = frozenset({"state_fp16", "state_fp64"})


def resolve_state_dtype(precision: StatePrecision | str):
    key = str(precision).replace("state_", "")
    mapping = {
        "fp32": np.float32,
        "fp16": np.float16,
        "fp64": np.float64,
    }
    if key not in mapping:
        raise ValueError(f"unsupported state precision: {precision}")
    return mapping[key]


def describe_state_precision(precision: StatePrecision | str) -> dict[str, Any]:
    dt = np.dtype(resolve_state_dtype(precision))
    return {
        "precision": str(precision),
        "numpy_dtype": str(dt),
        "itemsize": int(dt.itemsize),
        "notes": (
            "Applies to engine arrays v, g, synaptic delay ring, and (when the "
            "engine is constructed with this dtype) wdata. For state-only "
            "experiments, FlyQuant overrides wdata back to float32 mV weights "
            "so weight and state precision are not confounded. Integer state "
            "arithmetic is not supported by the upstream engine."
        ),
    }


def estimate_state_bytes(n_neurons: int, delay_slots: int, dtype) -> dict[str, int]:
    """Bytes for v, g, and delay ring alone (excludes CSR / spike counts)."""
    dt = np.dtype(dtype)
    item = int(dt.itemsize)
    v = n_neurons * item
    g = n_neurons * item
    ring = delay_slots * n_neurons * item
    return {
        "v_bytes": v,
        "g_bytes": g,
        "ring_bytes": ring,
        "state_arrays_bytes": v + g + ring,
        "itemsize": item,
        "n_neurons": int(n_neurons),
        "delay_slots": int(delay_slots),
    }
