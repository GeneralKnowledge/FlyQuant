"""Behavioural / circuit-level fidelity for escape-style probes and lesions."""

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


def lesion_control_fidelity(ref_conditions: list[dict], hyp_conditions: list[dict]) -> dict[str, Any]:
    """
    Compare escape-control condition tables.

    Expects each condition dict to include at least:
      condition, dnp01_spikes (or gf_spikes), silenced (optional list)
    """
    def _key(c: dict) -> str:
        return str(c.get("condition", ""))

    def _gf(c: dict) -> int:
        if "dnp01_spikes" in c:
            return int(c["dnp01_spikes"])
        return int(c.get("gf_spikes", 0))

    ref_map = {_key(c): c for c in ref_conditions}
    hyp_map = {_key(c): c for c in hyp_conditions}
    keys = sorted(set(ref_map) | set(hyp_map))
    rows = []
    abolishment_ok = None
    for k in keys:
        r = ref_map.get(k)
        h = hyp_map.get(k)
        if r is None or h is None:
            rows.append({"condition": k, "missing": True})
            continue
        rg, hg = _gf(r), _gf(h)
        row = {
            "condition": k,
            "ref_gf": rg,
            "hyp_gf": hg,
            "gf_abs_error": abs(hg - rg),
            "both_silent": rg == 0 and hg == 0,
            "both_active": rg > 0 and hg > 0,
            "agreement_success": (rg > 0) == (hg > 0),
        }
        rows.append(row)
        if "LC4" in k and "LPLC2" in k and ("-" in k or "silence" in k.lower()):
            # Lesion abolishment: both should be ~0
            abolishment_ok = (rg == 0) and (hg == 0)
        elif k.endswith("-LC4/-LPLC2") or "-both" in k or k.endswith(", -LC4/-LPLC2"):
            abolishment_ok = (rg == 0) and (hg == 0)

    # Also detect silenced both via silenced field
    if abolishment_ok is None:
        for k, r in ref_map.items():
            sil = r.get("silenced") or []
            if set(sil) >= {"LC4", "LPLC2"}:
                h = hyp_map.get(k)
                if h is not None:
                    abolishment_ok = (_gf(r) == 0) and (_gf(h) == 0)

    n_agree = sum(1 for r in rows if r.get("agreement_success") is True)
    n_cmp = sum(1 for r in rows if "agreement_success" in r)
    return {
        "n_conditions": len(keys),
        "n_compared": n_cmp,
        "success_agreement_fraction": (n_agree / n_cmp) if n_cmp else None,
        "lesion_abolishment_agreement": abolishment_ok,
        "rows": rows,
        "caveat": (
            "Lesion abolishment agreement requires GF≈0 under LC4+LPLC2 silence "
            "on both reference and compressed models."
        ),
    }
