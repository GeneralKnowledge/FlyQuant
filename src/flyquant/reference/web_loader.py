"""
Load a Connectome from fruit-fly-lab's committed web binary assets.

This reconstructs the same CSR signed-synapse-count graph used by the Python
engine without requiring a Codex login or `build_connectome` step. The binaries
are produced by upstream `tools/export_web_connectome.py` and already ship in
the submodule under `web/data/`.

Licence note: FlyWire data are CC BY-NC-SA 4.0. Loading these assets does not
grant rights beyond that licence.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flyquant.paths import CONNECTOME_BIN, META_JSON, NEURONS_BIN, WEB_DATA_DIR
from flyquant.reference.synthetic import SyntheticConnectome


def _delta_decode(delta: np.ndarray, indptr: np.ndarray) -> np.ndarray:
    """Inverse of upstream `tools.export_web_connectome.delta_encode`."""
    out = delta.astype(np.int64).copy()
    for a, b in zip(indptr[:-1], indptr[1:]):
        if b > a:
            out[a:b] = np.cumsum(out[a:b])
    return out.astype(np.int32)


def load_web_connectome(
    data_dir: Path | None = None,
) -> SyntheticConnectome:
    """
    Reconstruct a Connectome-compatible object from web/data binaries.

    Returns a SyntheticConnectome wrapper (same interface as upstream Connectome
    for fields FlyQuant uses). The graph is the real v783 wiring diagram.
    """
    root = Path(data_dir) if data_dir is not None else WEB_DATA_DIR
    conn_path = root / "connectome.bin" if data_dir else CONNECTOME_BIN
    neu_path = root / "neurons.bin" if data_dir else NEURONS_BIN
    meta_path = root / "meta.json" if data_dir else META_JSON

    for p in (conn_path, neu_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing web connectome asset: {p}. "
                "Initialise the fruit-fly-lab submodule and ensure web/data is present."
            )

    meta = json.loads(meta_path.read_text())
    n = int(meta["n"])
    nnz = int(meta["nnz"])

    blob = conn_path.read_bytes()
    # Layout: int32 indptr[n+1] | int32 indices_delta[nnz] | int16 weights[nnz]
    indptr_nbytes = 4 * (n + 1)
    delta_nbytes = 4 * nnz
    weights_nbytes = 2 * nnz
    expected = indptr_nbytes + delta_nbytes + weights_nbytes
    if len(blob) != expected:
        raise ValueError(
            f"connectome.bin size mismatch: got {len(blob)} bytes, expected {expected}"
        )

    indptr = np.frombuffer(blob, dtype=np.int32, count=n + 1)
    delta = np.frombuffer(blob, dtype=np.int32, count=nnz, offset=indptr_nbytes)
    weights = np.frombuffer(
        blob, dtype=np.int16, count=nnz, offset=indptr_nbytes + delta_nbytes
    ).astype(np.int32)

    indices = _delta_decode(delta, indptr)
    w = sp.csr_matrix((weights, indices, indptr.copy()), shape=(n, n))
    w.sort_indices()

    # neurons.bin layout from export_neurons
    neu = neu_path.read_bytes()
    root_nbytes = 8 * n
    pos_nbytes = 4 * n * 3
    tcode_nbytes = 2 * n
    ccode_nbytes = 1 * n
    scode_nbytes = 1 * n
    sign_nbytes = 1 * n
    neu_expected = (
        root_nbytes + pos_nbytes + tcode_nbytes + ccode_nbytes + scode_nbytes + sign_nbytes
    )
    if len(neu) != neu_expected:
        raise ValueError(
            f"neurons.bin size mismatch: got {len(neu)} bytes, expected {neu_expected}"
        )

    off = 0
    root_ids = np.frombuffer(neu, dtype=np.int64, count=n, offset=off)
    off += root_nbytes
    pos = np.frombuffer(neu, dtype=np.float32, count=n * 3, offset=off).reshape(n, 3)
    off += pos_nbytes
    tcode = np.frombuffer(neu, dtype=np.uint16, count=n, offset=off)
    off += tcode_nbytes
    ccode = np.frombuffer(neu, dtype=np.uint8, count=n, offset=off)
    off += ccode_nbytes
    scode = np.frombuffer(neu, dtype=np.uint8, count=n, offset=off)
    off += scode_nbytes
    sign = np.frombuffer(neu, dtype=np.int8, count=n, offset=off)

    cell_types = list(meta["cell_types"])
    super_classes = list(meta["super_classes"])
    sides = list(meta["sides"])
    neuropils = list(meta.get("neuropils", [""]))
    np_code = meta.get("neuropil_code", [0] * n)

    primary_type = np.array([cell_types[int(i)] for i in tcode], dtype=object)
    super_class = np.array([super_classes[int(i)] for i in ccode], dtype=object)
    side = np.array([sides[int(i)] for i in scode], dtype=object)
    primary_neuropil = np.array(
        [neuropils[int(i)] if neuropils else "" for i in np_code], dtype=object
    )

    # Positions in neurons.bin are centred micrometres; store nm-scale placeholders
    # compatible with upstream column names used by Session/SpikeRecorder.
    neurons = pd.DataFrame({
        "idx": np.arange(n, dtype=np.int32),
        "root_id": np.asarray(root_ids),
        "primary_type": primary_type,
        "super_class": super_class,
        "class": primary_type,
        "side": side,
        "nt_resolved": np.where(sign > 0, "ACH", np.where(sign < 0, "GABA", "")),
        "sign": np.asarray(sign),
        "primary_neuropil": primary_neuropil,
        "pos_x_nm": pos[:, 0].astype(np.float64) * 1000.0,
        "pos_y_nm": pos[:, 1].astype(np.float64) * 1000.0,
        "pos_z_nm": pos[:, 2].astype(np.float64) * 1000.0,
    })

    manifest = dict(meta.get("manifest", {}))
    manifest.setdefault("dataset", "FlyWire FAFB")
    manifest.setdefault("version", "783")
    manifest["n_neurons"] = n
    manifest["n_neuron_pairs"] = int(w.nnz)
    manifest["n_synapses"] = int(np.abs(w.data).sum())
    manifest["loaded_from"] = "web/data"
    manifest["connectome_bin_sha256"] = meta.get("files", {}).get("connectome.bin", {}).get(
        "sha256"
    )

    # Sanity against published counts when meta claims full brain
    if n != 139_255:
        raise ValueError(f"unexpected neuron count in web assets: {n}")
    if w.nnz != 3_732_460:
        raise ValueError(f"unexpected nnz in web assets: {w.nnz}")

    return SyntheticConnectome(neurons, w, manifest)


def web_assets_available() -> bool:
    return CONNECTOME_BIN.exists() and NEURONS_BIN.exists() and META_JSON.exists()


def estimate_web_load_bytes() -> int:
    """Rough lower bound on bytes needed to hold CSR + neuron table in RAM."""
    if not CONNECTOME_BIN.exists():
        return 0
    # CSR int32 data/indices/indptr + float state later dwarfs this; report file sizes.
    total = CONNECTOME_BIN.stat().st_size + NEURONS_BIN.stat().st_size
    return int(total)
