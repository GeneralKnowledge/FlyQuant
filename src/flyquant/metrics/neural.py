"""Neural activity fidelity metrics (reference vs compressed)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2:
        return None
    if np.std(a) == 0 or np.std(b) == 0:
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
            "Membrane-potential / spike-time metrics added separately when traces exist."
        ),
    }


def spike_time_coincidence(
    ref_times: dict[int, list[float]],
    hyp_times: dict[int, list[float]],
    *,
    tolerance_ms: float = 1.0,
) -> dict[str, Any]:
    """
    Per-neuron spike-time coincidence within ±tolerance_ms.

    For each reference spike, a hypothesis spike within tolerance counts as a hit
    (greedy matching). Silent neurons on both sides contribute perfect scores.
    """
    neurons = sorted(set(ref_times) | set(hyp_times))
    if not neurons:
        return {
            "skipped": True,
            "reason": "no watched spike times recorded",
            "tolerance_ms": tolerance_ms,
        }

    precisions, recalls, f1s = [], [], []
    n_ref_total = 0
    n_hyp_total = 0
    n_matched = 0
    for nid in neurons:
        r = np.asarray(ref_times.get(nid, []), dtype=np.float64)
        h = np.asarray(hyp_times.get(nid, []), dtype=np.float64)
        n_ref_total += int(r.size)
        n_hyp_total += int(h.size)
        if r.size == 0 and h.size == 0:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
            continue
        if r.size == 0:
            precisions.append(0.0 if h.size else 1.0)
            recalls.append(1.0)
            f1s.append(0.0)
            continue
        if h.size == 0:
            precisions.append(1.0)
            recalls.append(0.0)
            f1s.append(0.0)
            continue
        used = np.zeros(h.size, dtype=bool)
        hits = 0
        for t in r:
            d = np.abs(h - t)
            d[used] = np.inf
            j = int(np.argmin(d))
            if d[j] <= tolerance_ms:
                used[j] = True
                hits += 1
        n_matched += hits
        recall = hits / r.size
        precision = hits / h.size if h.size else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "skipped": False,
        "tolerance_ms": tolerance_ms,
        "n_neurons": len(neurons),
        "n_ref_spikes": n_ref_total,
        "n_hyp_spikes": n_hyp_total,
        "n_matched": n_matched,
        "mean_precision": float(np.mean(precisions)),
        "mean_recall": float(np.mean(recalls)),
        "mean_f1": float(np.mean(f1s)),
        "p50_f1": float(np.percentile(f1s, 50)),
        "p05_f1": float(np.percentile(f1s, 5)),
    }


def membrane_trace_error(
    ref_traces: np.ndarray | None,
    hyp_traces: np.ndarray | None,
    *,
    times_ms: list[float] | None = None,
) -> dict[str, Any]:
    """MAE/RMSE on sampled membrane traces when both sides recorded."""
    if ref_traces is None or hyp_traces is None:
        return {
            "skipped": True,
            "reason": "membrane traces not recorded (record_v=false or missing)",
        }
    a = np.asarray(ref_traces, dtype=np.float64)
    b = np.asarray(hyp_traces, dtype=np.float64)
    n = min(a.shape[0], b.shape[0])
    if n == 0 or a.ndim != 2 or b.ndim != 2:
        return {"skipped": True, "reason": "empty or malformed traces"}
    a, b = a[:n], b[:n]
    cols = min(a.shape[1], b.shape[1])
    a, b = a[:, :cols], b[:, :cols]
    err = b - a
    abs_err = np.abs(err)
    return {
        "skipped": False,
        "n_samples": int(n),
        "n_neurons_tracked": int(cols),
        "mae_mV": float(abs_err.mean()),
        "rmse_mV": float(np.sqrt((err ** 2).mean())),
        "max_abs_mV": float(abs_err.max()),
        "p95_abs_mV": float(np.percentile(abs_err, 95)),
        "duration_span_ms": (
            None
            if not times_ms
            else float(times_ms[min(n - 1, len(times_ms) - 1)] - times_ms[0])
        ),
    }
