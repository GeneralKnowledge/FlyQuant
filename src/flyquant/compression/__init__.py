"""Compression methods: weight quantisation, lossless graph encoding, state precision."""

from flyquant.compression.graph import (
    GraphArtefact,
    compress_graph_lossless,
    graph_identical,
    load_graph_artefact,
    reconstruct_csr,
    save_graph_artefact,
)
from flyquant.compression.prune import prune_connections
from flyquant.compression.state import describe_state_precision, resolve_state_dtype
from flyquant.compression.weights import (
    QuantisedWeights,
    dequantise_weights,
    effective_mv_weights,
    pack_int4_nibbles,
    quantise_weights,
    unpack_int4_nibbles,
)

__all__ = [
    "GraphArtefact",
    "compress_graph_lossless",
    "graph_identical",
    "load_graph_artefact",
    "reconstruct_csr",
    "save_graph_artefact",
    "QuantisedWeights",
    "dequantise_weights",
    "quantise_weights",
    "effective_mv_weights",
    "pack_int4_nibbles",
    "unpack_int4_nibbles",
    "prune_connections",
    "resolve_state_dtype",
    "describe_state_precision",
]
