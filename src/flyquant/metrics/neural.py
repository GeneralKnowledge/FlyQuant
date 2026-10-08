"""Neural activity fidelity metrics (reference vs compressed)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2:
        return None
    if np.std(a) == 0 or np.std(b) == 0:
        # Zero-variance: perfect if identical, else undefined
        if np.array_equal(a, b):
            return 1.0
        return None
    return float(np.corrcoef(a, b)[0, 1])


def neural_fidelity(
    ref_spikes: np.ndarray,
    hyp_spikes: np.ndarray,
    *,
    duration_ms: float,
) -> dict[str, Any]:
    """
    Compare per-neuron spike counts / rates.

    Handles silent neurons explicitly. Reports distributions, not only means.
    """
    a = np.asarray(ref_spikes, dtype=np.float64).ravel()
    b = np.asarray(hyp_spikes, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"spike count shape mismatch: {a.shape} vs {b.shape}")

    elapsed_s = max(duration_ms, 0.0) * 1e-3
    rate_a = a / elapsed_s if elapsed_s > 0 else a * 0.0
    rate_b = b / elapsed_s if elapsed_s > 0 else b * 0.0
    err = rate_b - rate_a
    abs_err = np.abs(err)

    silent_ref = a == 0
    silent_hyp = b == 0
    active_ref = ~silent_ref

    # Spike-count agreement among neurons that spiked in either run
    either = (a + b) > 0
    if either.any():
        rel = np.abs(a[either] - b[either]) / np.maximum(a[either], 1.0)
        agreement = float((a[either] == b[either]).mean())
    else:
        rel = np.array([])
        agreement = 1.0

    return {
        "n_neurons": int(a.size),
        "n_silent_reference": int(silent_ref.sum()),
        "n_silent_hypothesis": int(silent_hyp.sum()),
        "n_active_reference": int(active_ref.sum()),
        "total_spikes_ref": int(a.sum()),
        "total_spikes_hyp": int(b.sum()),
        "spike_count_l1": float(np.abs(a - b).sum()),
        "spike_count_exact_match_fraction": float(np.mean(a == b)) if a.size else 1.0,
        "active_either_exact_match_fraction": agreement,
        "rate_mae_hz": float(abs_err.mean()) if a.size else 0.0,
        "rate_rmse_hz": float(np.sqrt((err ** 2).mean())) if a.size else 0.0,
        "rate_max_abs_hz": float(abs_err.max()) if a.size else 0.0,
        "rate_p50_abs_hz": float(np.percentile(abs_err, 50)) if a.size else 0.0,
        "rate_p95_abs_hz": float(np.percentile(abs_err, 95)) if a.size else 0.0,
        "rate_p99_abs_hz": float(np.percentile(abs_err, 99)) if a.size else 0.0,
        "rate_corr_all": _safe_corr(rate_a, rate_b),
        "rate_corr_active_ref": (
            _safe_corr(rate_a[active_ref], rate_b[active_ref]) if active_ref.any() else None
        ),
        "mean_rel_spike_error_active_either": float(rel.mean()) if rel.size else 0.0,
        "false_silent": int((active_ref & silent_hyp).sum()),
        "false_active": int((silent_ref & (~silent_hyp)).sum()),
        "notes": (
            "Correlations are None when variance is zero and vectors differ. "
            "Membrane-potential error omitted unless engines expose full traces."
        ),
    }
