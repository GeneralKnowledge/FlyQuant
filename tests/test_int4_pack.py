"""INT4 nibble pack/unpack and quantisation."""

from __future__ import annotations

import numpy as np

from flyquant.compression.weights import (
    dequantise_weights,
    pack_int4_nibbles,
    quantise_weights,
    reconstruction_error,
    unpack_int4_nibbles,
)


def test_nibble_roundtrip():
    vals = np.array([-7, -3, 0, 1, 7, 2], dtype=np.int8)
    packed = pack_int4_nibbles(vals)
    assert packed.dtype == np.uint8
    assert packed.nbytes == 3  # 6 values -> 3 bytes
    out = unpack_int4_nibbles(packed, len(vals))
    np.testing.assert_array_equal(out, vals)


def test_int4_quant_with_packing():
    w = np.linspace(-5, 5, 101, dtype=np.float64)
    q = quantise_weights(w, "int4", pack_int4=True)
    assert q.packed_payload is not None
    assert q.packed_nbytes is not None
    assert q.packed_nbytes < q.unpacked_nbytes
    recon = dequantise_weights(q)
    err = reconstruction_error(w, recon)
    assert err["n"] == 101
    assert q.storage_dtype == "int4_nibbles"


def test_int4_rejects_out_of_range_pack():
    try:
        pack_int4_nibbles(np.array([8], dtype=np.int8))
        assert False, "expected ValueError"
    except ValueError:
        pass
