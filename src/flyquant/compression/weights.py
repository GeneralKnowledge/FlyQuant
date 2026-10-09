"""
Weight quantisation for effective synaptic weights (mV).

Important distinctions
----------------------
* Connectome CSR stores **integer synapse counts** (not LLM-style FP32 weights).
* The simulator multiplies by `w_syn` to form `wdata` in mV (`LIFEngine`).
* This module quantises **eligible floating mV weights** (`wdata`), never neuron
  IDs, CSR indices, or categorical metadata.
* INT8/INT4 paths define scale (and optional zero-point), clip, round, and
  reconstruct. Upstream has no native INT8/INT4 kernel: simulation uses
  reconstructed float32/float16 unless a dtype override is requested for FP16.
* INT4 nibble packing is a **storage-only** optimisation; values are unpacked
  before dequantisation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    packed_payload: np.ndarray | None = None
    packed_nbytes: int | None = None
    unpacked_nbytes: int | None = None

    def to_meta(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("payload")
        d.pop("packed_payload")
        d["payload_nbytes"] = int(self.payload.nbytes)
        if self.packed_payload is not None:
            d["packed_payload_nbytes"] = int(self.packed_payload.nbytes)
        return d


def effective_mv_weights(connectome, w_syn: float = 0.275) -> np.ndarray:
    """CSR data as float64 mV weights (signed synapse count × w_syn)."""
    return connectome.w.data.astype(np.float64) * float(w_syn)


def pack_int4_nibbles(values: np.ndarray) -> np.ndarray:
    """
    Pack signed int4-range values in [-7, 7] into nibbles (two per byte).

    Encoding: store (v + 8) as an unsigned nibble in 1..15 (0 unused for zero
    offset). Odd trailing element is padded with 0 nibble.
    """
    v = np.asarray(values, dtype=np.int16).ravel()
    if v.size and (v.min() < -7 or v.max() > 7):
        raise ValueError("INT4 pack requires values in [-7, 7]")
    u = (v.astype(np.int16) + 8).astype(np.uint8)
    if u.size % 2 == 1:
        u = np.concatenate([u, np.array([8], dtype=np.uint8)])  # pad = 0
    hi = u[0::2]
    lo = u[1::2]
    return ((hi << 4) | lo).astype(np.uint8)


def unpack_int4_nibbles(packed: np.ndarray, n: int) -> np.ndarray:
    """Inverse of pack_int4_nibbles; return exactly n signed values."""
    p = np.asarray(packed, dtype=np.uint8).ravel()
    hi = ((p >> 4) & 0x0F).astype(np.int16) - 8
    lo = (p & 0x0F).astype(np.int16) - 8
    out = np.empty(p.size * 2, dtype=np.int8)
    out[0::2] = hi.astype(np.int8)
    out[1::2] = lo.astype(np.int8)
    return out[:n]


def quantise_weights(
    weights_mv: np.ndarray,
    method: Precision,
    *,
    w_syn: float = 0.275,
    calibration: np.ndarray | None = None,
    pack_int4: bool = True,
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
            unpacked_nbytes=int(payload.nbytes),
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
            unpacked_nbytes=int(payload.nbytes),
        )

    if method == "int8":
        max_abs = float(np.max(np.abs(cal))) if cal.size else 1.0
        scale = max_abs / 127.0 if max_abs > 0 else 1.0
        q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
        return QuantisedWeights(
            method="int8",
            storage_dtype="int8",
            execution_dtype="float32",
            n_weights=int(w.size),
            scale=scale,
            zero_point=0,
            clip_min=-127,
            clip_max=127,
            payload=q,
            w_syn=w_syn,
            notes="Symmetric INT8; reconstruct to float32 for simulation.",
            unpacked_nbytes=int(q.nbytes),
        )

    if method == "int4":
        max_abs = float(np.max(np.abs(cal))) if cal.size else 1.0
        scale = max_abs / 7.0 if max_abs > 0 else 1.0
        q = np.clip(np.rint(w / scale), -7, 7).astype(np.int8)
        packed = pack_int4_nibbles(q) if pack_int4 else None
        return QuantisedWeights(
            method="int4",
            storage_dtype="int4_nibbles" if pack_int4 else "int8_values_int4_range",
            execution_dtype="float32",
            n_weights=int(w.size),
            scale=scale,
            zero_point=0,
            clip_min=-7,
            clip_max=7,
            payload=q,
            w_syn=w_syn,
            notes=(
                "Signed 4-bit range; nibble-packed for storage when pack_int4=True. "
                "Simulation always dequantises to float32 — not a native INT4 kernel."
            ),
            packed_payload=packed,
            packed_nbytes=int(packed.nbytes) if packed is not None else None,
            unpacked_nbytes=int(q.nbytes),
        )

    raise ValueError(f"unsupported quantisation method: {method}")


def dequantise_weights(q: QuantisedWeights, out_dtype=np.float32) -> np.ndarray:
    """Reconstruct floating mV weights for simulation or error measurement."""
    if q.method in ("fp32", "fp16"):
        return q.payload.astype(out_dtype, copy=False)
    if q.method in ("int8", "int4"):
        if q.scale is None:
            raise ValueError("missing scale for integer quantisation")
        payload = q.payload
        if (
            q.method == "int4"
            and q.packed_payload is not None
            and (payload is None or payload.size == 0)
        ):
            payload = unpack_int4_nibbles(q.packed_payload, q.n_weights)
        return (payload.astype(np.float64) * q.scale).astype(out_dtype)
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
    arrays: dict[str, Any] = {
        "payload": q.payload,
        "method": np.array(q.method),
        "storage_dtype": np.array(q.storage_dtype),
        "execution_dtype": np.array(q.execution_dtype),
        "scale": np.array(np.nan if q.scale is None else q.scale),
        "zero_point": np.array(-999 if q.zero_point is None else q.zero_point),
        "w_syn": np.array(q.w_syn),
        "notes": np.array(q.notes),
        "n_weights": np.array(q.n_weights),
    }
    if q.packed_payload is not None:
        arrays["packed_payload"] = q.packed_payload
    np.savez_compressed(path, **arrays)


def load_quantised(path: Path) -> QuantisedWeights:
    z = np.load(Path(path), allow_pickle=False)
    scale_v = float(z["scale"])
    zp_v = int(z["zero_point"])
    packed = z["packed_payload"] if "packed_payload" in z.files else None
    n_weights = int(z["n_weights"]) if "n_weights" in z.files else int(z["payload"].size)
    payload = z["payload"]
    if packed is not None and str(z["method"]) == "int4":
        # Prefer unpacked payload for simulation; keep packed for size stats.
        if payload.size != n_weights:
            payload = unpack_int4_nibbles(packed, n_weights)
    return QuantisedWeights(
        method=str(z["method"]),  # type: ignore[arg-type]
        storage_dtype=str(z["storage_dtype"]),
        execution_dtype=str(z["execution_dtype"]),
        n_weights=n_weights,
        scale=None if np.isnan(scale_v) else scale_v,
        zero_point=None if zp_v == -999 else zp_v,
        clip_min=None,
        clip_max=None,
        payload=payload,
        w_syn=float(z["w_syn"]),
        notes=str(z["notes"]),
        packed_payload=packed,
        packed_nbytes=int(packed.nbytes) if packed is not None else None,
        unpacked_nbytes=int(payload.nbytes),
    )
