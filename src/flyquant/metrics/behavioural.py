"""Behavioural / circuit-level fidelity for escape-style probes."""

from __future__ import annotations

from typing import Any


def behavioural_fidelity(ref: dict, hyp: dict) -> dict[str, Any]:
    """
    Compare escape-probe summaries.

    Success is defined as Giant Fibre (DNp01) producing ≥1 spike during the trial,
    matching the qualitative upstream escape readout. This is fidelity to the
    computational model, not proof of biological equivalence.
    """
    ref_gf = int(ref.get("gf_spikes", 0))
    hyp_gf = int(hyp.get("gf_spikes", 0))
    ref_success = ref_gf > 0
    hyp_success = hyp_gf > 0

    ref_lat = ref.get("first_gf_spike_ms")
    hyp_lat = hyp.get("first_gf_spike_ms")
    if ref_lat is not None and hyp_lat is not None:
        latency_delta_ms = float(hyp_lat) - float(ref_lat)
    else:
        latency_delta_ms = None

    return {
        "ref_gf_spikes": ref_gf,
        "hyp_gf_spikes": hyp_gf,
        "ref_escape_success": ref_success,
        "hyp_escape_success": hyp_success,
        "escape_agreement": ref_success == hyp_success,
        "false_negative": bool(ref_success and not hyp_success),
        "false_positive": bool((not ref_success) and hyp_success),
        "gf_spike_abs_error": abs(hyp_gf - ref_gf),
        "gf_spike_rel_error": (
            abs(hyp_gf - ref_gf) / ref_gf if ref_gf > 0 else (0.0 if hyp_gf == 0 else None)
        ),
        "ref_first_gf_spike_ms": ref_lat,
        "hyp_first_gf_spike_ms": hyp_lat,
        "latency_delta_ms": latency_delta_ms,
        "ref_escape_peak": float(ref.get("escape_peak", 0.0)),
        "hyp_escape_peak": float(hyp.get("escape_peak", 0.0)),
        "ref_lc4_spikes": int(ref.get("lc4_spikes", 0)),
        "hyp_lc4_spikes": int(hyp.get("lc4_spikes", 0)),
        "ref_lplc2_spikes": int(ref.get("lplc2_spikes", 0)),
        "hyp_lplc2_spikes": int(hyp.get("lplc2_spikes", 0)),
        "caveat": (
            "Matching the simulator does not prove equivalence to a living fly; "
            "scores measure fidelity to the selected computational model."
        ),
    }
