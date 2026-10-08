"""
Weight quantisation for effective synaptic weights (mV).

Important distinctions
----------------------
* Connectome CSR stores **integer synapse counts** (not LLM-style FP32 weights).
* The simulator multiplies by `w_syn` to form `wdata` in mV (`LIFEngine`).
* This module quantises **eligible floating mV weights** (`wdata`), never neuron
  IDs, CSR indices, or categorical metadata.
* INT8/INT4 paths define scale (and optional zero-point), clip, round, and
  reconstruct. Upstream has no native INT8 kernel: simulation uses reconstructed
  float32/float16 unless a dtype override is requested for FP16.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

Precision = Literal["fp32", "fp16", "int8", "int4"]


@dataclass
class QuantisedWeights:
    method: Precision
    storage_dtype: str
    execution_dtype: str
    n_weights: int
    scale: float | None
    zero_point: int | None
    clip_min: float | None
    clip_max: float | None
    payload: np.ndarray
    w_syn: float
    notes: str

    def to_meta(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("payload")
        d["payload_nbytes"] = int(self.payload.nbytes)
        return d


def effective_mv_weights(connectome, w_syn: float = 0.275) -> np.ndarray:
    """CSR data as float64 mV weights (signed synapse count × w_syn)."""
    return connectome.w.data.astype(np.float64) * float(w_syn)


def quantise_weights(
    weights_mv: np.ndarray,
    method: Precision,
    *,
    w_syn: float = 0.275,
    calibration: np.ndarray | None = None,
) -> QuantisedWeights:
    """
    Quantise effective mV weights.

    Calibration (optional) supplies a representative sample for scale estimation;
    by default the full weight vector is used (allowed for this static connectome).
    """
    w = np.asarray(weights_mv, dtype=np.float64).ravel()
    cal = w if calibration is None else np.asarray(calibration, dtype=np.float64).ravel()

    if method == "fp32":
        payload = w.astype(np.float32)
        return QuantisedWeights(
            method="fp32",
            storage_dtype="float32",
            execution_dtype="float32",
            n_weights=int(w.size),
            scale=None,
            zero_point=None,
            clip_min=None,
            clip_max=None,
            payload=payload,
            w_syn=w_syn,
            notes="Identity cast to float32 (upstream engine default).",
        )

    if method == "fp16":
        payload = w.astype(np.float16)
        return QuantisedWeights(
            method="fp16",
            storage_dtype="float16",
            execution_dtype="float16",
            n_weights=int(w.size),
            scale=None,
            zero_point=None,
            clip_min=None,
            clip_max=None,
            payload=payload,
            w_syn=w_syn,
            notes="IEEE float16 storage; engine can run dtype=float16.",
        )

    if method == "int8":
        # Symmetric max-abs scale, zero-point 0 (signed weights).
        max_abs = float(np.max(np.abs(cal))) if cal.size else 1.0
        scale = max_abs / 127.0 if max_abs > 0 else 1.0
        q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
        return QuantisedWeights(
            method="int8",
            storage_dtype="int8",
            execution_dtype="float32",  # reconstruct then simulate
            n_weights=int(w.size),
            scale=scale,
            zero_point=0,
            clip_min=-127,
            clip_max=127,
            payload=q,
            w_syn=w_syn,
            notes="Symmetric INT8; reconstruct to float32 for simulation.",
        )

    if method == "int4":
        # Packed as int8 values in [-7, 7] (signed 4-bit range); not bit-packed
        # in v0.1 — packing would be a storage-only further step.
        max_abs = float(np.max(np.abs(cal))) if cal.size else 1.0
        scale = max_abs / 7.0 if max_abs > 0 else 1.0
        q = np.clip(np.rint(w / scale), -7, 7).astype(np.int8)
        return QuantisedWeights(
            method="int4",
            storage_dtype="int8_values_int4_range",
            execution_dtype="float32",
            n_weights=int(w.size),
            scale=scale,
            zero_point=0,
            clip_min=-7,
            clip_max=7,
            payload=q,
            w_syn=w_syn,
            notes=(
                "Signed 4-bit range stored in int8 elements (not nibble-packed). "
                "Technically justified only as an extreme ablation; expect large error."
            ),
        )

    raise ValueError(f"unsupported quantisation method: {method}")


def dequantise_weights(q: QuantisedWeights, out_dtype=np.float32) -> np.ndarray:
    """Reconstruct floating mV weights for simulation or error measurement."""
    if q.method in ("fp32", "fp16"):
        return q.payload.astype(out_dtype, copy=False)
    if q.method in ("int8", "int4"):
        if q.scale is None:
            raise ValueError("missing scale for integer quantisation")
        return (q.payload.astype(np.float64) * q.scale).astype(out_dtype)
    raise ValueError(f"unsupported method: {q.method}")


def reconstruction_error(original_mv: np.ndarray, reconstructed_mv: np.ndarray) -> dict:
    a = np.asarray(original_mv, dtype=np.float64).ravel()
    b = np.asarray(reconstructed_mv, dtype=np.float64).ravel()
    err = b - a
    abs_err = np.abs(err)
    return {
        "mae": float(abs_err.mean()) if a.size else 0.0,
        "rmse": float(np.sqrt((err ** 2).mean())) if a.size else 0.0,
        "max_abs": float(abs_err.max()) if a.size else 0.0,
        "p50_abs": float(np.percentile(abs_err, 50)) if a.size else 0.0,
        "p95_abs": float(np.percentile(abs_err, 95)) if a.size else 0.0,
        "p99_abs": float(np.percentile(abs_err, 99)) if a.size else 0.0,
        "corr": float(np.corrcoef(a, b)[0, 1]) if a.size > 1 and a.std() > 0 and b.std() > 0 else None,
        "n": int(a.size),
    }


def save_quantised(path: Path, q: QuantisedWeights) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        payload=q.payload,
        method=np.array(q.method),
        storage_dtype=np.array(q.storage_dtype),
        execution_dtype=np.array(q.execution_dtype),
        scale=np.array(np.nan if q.scale is None else q.scale),
        zero_point=np.array(-999 if q.zero_point is None else q.zero_point),
        w_syn=np.array(q.w_syn),
        notes=np.array(q.notes),
    )


def load_quantised(path: Path) -> QuantisedWeights:
    z = np.load(Path(path), allow_pickle=False)
    scale_v = float(z["scale"])
    zp_v = int(z["zero_point"])
    return QuantisedWeights(
        method=str(z["method"]),
        storage_dtype=str(z["storage_dtype"]),
        execution_dtype=str(z["execution_dtype"]),
        n_weights=int(z["payload"].size),
        scale=None if np.isnan(scale_v) else scale_v,
        zero_point=None if zp_v == -999 else zp_v,
        clip_min=None,
        clip_max=None,
        payload=z["payload"],
        w_syn=float(z["w_syn"]),
        notes=str(z["notes"]),
    )
