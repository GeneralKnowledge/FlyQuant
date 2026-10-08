"""Tiny synthetic connectomes for unit tests (no FlyWire download required)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class SyntheticConnectome:
    """Minimal Connectome-compatible object for FlyQuant unit tests."""

    neurons: pd.DataFrame
    w: sp.csr_matrix
    manifest: dict

    @property
    def dataset(self) -> str:
        return "%s v%s" % (self.manifest["dataset"], self.manifest["version"])

    @property
    def n(self) -> int:
        return int(self.w.shape[0])

    @property
    def root_ids(self) -> np.ndarray:
        return self.neurons["root_id"].to_numpy()

    def by_cell_type(self, cell_type, side=None) -> pd.DataFrame:
        m = self.neurons["primary_type"].astype(str) == str(cell_type)
        if side is not None:
            m &= self.neurons["side"].astype(str) == str(side)
        return self.neurons.loc[m]

    def by_cell_types(self, cell_types, side=None) -> pd.DataFrame:
        wanted = set(str(c) for c in cell_types)
        m = self.neurons["primary_type"].astype(str).isin(wanted)
        if side is not None:
            m &= self.neurons["side"].astype(str) == str(side)
        return self.neurons.loc[m]

    def subgraph(self, indices) -> "SyntheticConnectome":
        idx = np.sort(np.asarray(indices, dtype=np.int64))
        neurons = self.neurons.iloc[idx].copy().reset_index(drop=True)
        neurons["idx"] = np.arange(len(idx), dtype=np.int32)
        sub = self.w[idx, :][:, idx].tocsr()
        sub.sort_indices()
        m = dict(self.manifest)
        m["n_neurons"] = int(len(idx))
        return SyntheticConnectome(neurons, sub, m)


def make_synthetic_connectome(
    n: int = 32,
    seed: int = 0,
    density: float = 0.15,
) -> SyntheticConnectome:
    """
    Build a small signed sparse graph with escape-like cell labels.

    Weights are integer synapse counts in [-20, 20], excluding zero.
    Topology is random but reproducible; used only for unit tests.
    """
    if n < 8:
        raise ValueError("synthetic connectome needs at least 8 neurons")
    rng = np.random.default_rng(seed)

    types = np.array(["other"] * n, dtype=object)
    types[0:2] = "LC4"
    types[2:5] = "LPLC2"
    types[5:7] = "DNp01"
    sides = np.array(["left" if i % 2 == 0 else "right" for i in range(n)], dtype=object)
    signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)

    rows, cols, data = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if rng.random() < density:
                syn = int(rng.integers(1, 21)) * int(signs[i])
                rows.append(i)
                cols.append(j)
                data.append(syn)

    # Ensure LC4/LPLC2 → DNp01 edges exist so escape-style probes are meaningful.
    for pre in (0, 1, 2, 3, 4):
        for post in (5, 6):
            rows.append(pre)
            cols.append(post)
            data.append(int(10 * signs[pre]))

    w = sp.csr_matrix(
        (np.asarray(data, dtype=np.int32), (np.asarray(rows), np.asarray(cols))),
        shape=(n, n),
    )
    w.sum_duplicates()
    w.sort_indices()

    neurons = pd.DataFrame({
        "idx": np.arange(n, dtype=np.int32),
        # Distinct fake root IDs (not FlyWire); length must be n, not arange(huge).
        "root_id": (np.arange(n, dtype=np.int64) + 7_205_759_400_000_000),
        "primary_type": types,
        "super_class": np.where(
            types == "DNp01", "descending",
            np.where(np.isin(types, ["LC4", "LPLC2"]), "visual_projection", "other"),
        ),
        "class": types,
        "side": sides,
        "nt_resolved": np.where(signs > 0, "ACH", "GABA"),
        "sign": signs.astype(np.int8),
        "primary_neuropil": np.array(["ME"] * n, dtype=object),
        "pos_x_nm": rng.normal(0, 1e5, n),
        "pos_y_nm": rng.normal(0, 1e5, n),
        "pos_z_nm": rng.normal(0, 1e5, n),
    })

    manifest = {
        "dataset": "FlyQuantSynthetic",
        "version": "test",
        "n_neurons": n,
        "n_neuron_pairs": int(w.nnz),
        "n_synapses": int(np.abs(w.data).sum()),
        "synthetic": True,
        "seed": seed,
    }
    return SyntheticConnectome(neurons, w, manifest)
