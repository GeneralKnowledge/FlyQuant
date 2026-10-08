"""
Adapter to the immutable fruit-fly-lab reference.

Import path setup is the only coupling. Compression code never edits upstream
files; it constructs engines with alternate weight arrays / dtypes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flyquant import DATASET_ID, UPSTREAM_COMMIT, __version__
from flyquant.paths import REFERENCE_DIR
from flyquant.reference.web_loader import load_web_connectome, web_assets_available


def ensure_upstream_on_path() -> None:
    root = str(REFERENCE_DIR.resolve())
    if not REFERENCE_DIR.exists():
        raise FileNotFoundError(
            f"Upstream submodule missing at {REFERENCE_DIR}. "
            "Run: git submodule update --init --recursive"
        )
    if root not in sys.path:
        sys.path.insert(0, root)


def get_upstream_commit() -> str:
    if not (REFERENCE_DIR / ".git").exists() and not (
        REFERENCE_DIR.parent.parent / ".git"
    ).exists():
        return UPSTREAM_COMMIT
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REFERENCE_DIR,
            text=True,
        ).strip()
        return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return UPSTREAM_COMMIT


def load_reference_connectome(source: str = "auto", subgraph: str | None = None):
    """
    Load the reference connectome.

    source:
      - "auto": prefer built derived NPZ, else web binaries
      - "derived": upstream `load_connectome()` only
      - "web": web/data binaries only
      - "synthetic": tiny test graph (not a biological reference)
    subgraph:
      - None: full graph
      - "escape": ~escape-circuit neighbourhood (for smaller CPU experiments)
    """
    ensure_upstream_on_path()

    if source == "synthetic":
        from flyquant.reference.synthetic import make_synthetic_connectome

        c = make_synthetic_connectome()
    elif source == "web":
        c = load_web_connectome()
    elif source == "derived":
        from brain.neurons.registry import load_connectome

        c = load_connectome()
    else:
        # auto
        try:
            from brain.neurons.registry import load_connectome
            import config as lab_config

            if lab_config.CONNECTOME_NPZ.exists():
                c = load_connectome()
            elif web_assets_available():
                c = load_web_connectome()
            else:
                raise FileNotFoundError(
                    "No connectome available. Build derived artefacts or ensure "
                    "reference/fruit-fly-lab/web/data exists."
                )
        except FileNotFoundError:
            if web_assets_available():
                c = load_web_connectome()
            else:
                raise

    if subgraph == "escape":
        seeds = c.by_cell_types(["LPLC2", "LC4", "DNp01"])["idx"].to_numpy()
        seed_set = {int(i) for i in seeds}
        keep: set[int] = set(seed_set)
        for i in seeds:
            row = c.w.getrow(int(i))
            keep.update(int(j) for j in row.indices)
            # one hop upstream into LC4/LPLC2 also helps density for short trials
            col = c.w.getcol(int(i)).tocoo()
            keep.update(int(j) for j in col.row[:200])
        # Cap for CPU-only machines, but never drop seed circuit neurons.
        # Taking sorted(keep)[:max_n] alone can exclude high-index DNp01 cells.
        max_n = int(os.environ.get("FLYQUANT_ESCAPE_SUBGRAPH_MAX", "4000"))
        if len(keep) > max_n:
            extras = sorted(keep - seed_set)
            budget = max(0, max_n - len(seed_set))
            keep = set(seed_set) | set(extras[:budget])
        idx = np.array(sorted(keep), dtype=np.int64)
        c = c.subgraph(idx)
    elif subgraph is not None:
        raise ValueError(f"unknown subgraph: {subgraph}")

    return c


def make_engine(
    connectome,
    seed: int = 0,
    dtype=np.float32,
    weight_data: np.ndarray | None = None,
):
    """
    Construct upstream LIFEngine.

    If weight_data is provided, it replaces engine.wdata after init (mV units).
    """
    ensure_upstream_on_path()
    from simulation.engine.lif_engine import LIFEngine

    engine = LIFEngine(connectome, seed=seed, dtype=dtype)
    if weight_data is not None:
        w = np.asarray(weight_data)
        if w.shape != engine.wdata.shape:
            raise ValueError(
                f"weight_data shape {w.shape} != engine.wdata shape {engine.wdata.shape}"
            )
        engine.wdata = w.astype(engine.dtype, copy=False)
    return engine


@dataclass
class EscapeRunResult:
    experiment: str
    seed: int
    duration_ms: float
    n_neurons: int
    n_connections: int
    n_synapses: int
    dtype: str
    total_spikes: int
    active_neurons: int
    gf_spikes: int
    lc4_spikes: int
    lplc2_spikes: int
    first_gf_spike_ms: float | None
    escape_peak: float
    spike_counts: np.ndarray
    wall_time_s: float
    sim_steps: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "seed": self.seed,
            "duration_ms": self.duration_ms,
            "n_neurons": self.n_neurons,
            "n_connections": self.n_connections,
            "n_synapses": self.n_synapses,
            "dtype": self.dtype,
            "total_spikes": self.total_spikes,
            "active_neurons": self.active_neurons,
            "gf_spikes": self.gf_spikes,
            "lc4_spikes": self.lc4_spikes,
            "lplc2_spikes": self.lplc2_spikes,
            "first_gf_spike_ms": self.first_gf_spike_ms,
            "escape_peak": self.escape_peak,
            "wall_time_s": self.wall_time_s,
            "sim_steps": self.sim_steps,
            "metadata": self.metadata,
        }


def _indices_of(c, cell_type: str) -> np.ndarray:
    df = c.by_cell_type(cell_type)
    if df.empty:
        return np.empty(0, dtype=np.int64)
    return df["idx"].to_numpy(dtype=np.int64)


def run_escape_benchmark(
    connectome,
    *,
    seed: int = 0,
    duration_ms: float = 280.0,
    dtype=np.float32,
    weight_data: np.ndarray | None = None,
    rate_hz: float = 150.0,
    experiment: str = "escape_poisson",
) -> EscapeRunResult:
    """
    Minimal reproducible escape-style probe.

    Drives all LC4 + LPLC2 neurons with constant Poisson rate (reference default
    150 Hz) and records Giant Fibre (DNp01) activity. This mirrors the core of
    upstream looming escape without requiring retinotopy tables when using the
    web-loaded connectome (retinotopy needs column_assignment from Codex).

    For the full geometric looming pipeline, use upstream
    `python -m experiments.01_looming_escape` after building derived data.
    """
    engine = make_engine(connectome, seed=seed, dtype=dtype, weight_data=weight_data)

    lc4 = _indices_of(connectome, "LC4")
    lplc2 = _indices_of(connectome, "LPLC2")
    gf = _indices_of(connectome, "DNp01")
    drive = np.concatenate([lc4, lplc2])
    if drive.size == 0:
        raise RuntimeError("No LC4/LPLC2 neurons in connectome; cannot run escape probe")

    engine.set_poisson(drive, rate_hz)

    # Optional descending readout if columns allow
    escape_peak = 0.0
    try:
        ensure_upstream_on_path()
        from brain.motor.descending import DescendingReadout

        readout = DescendingReadout(connectome)
        window = np.zeros(connectome.n, dtype=np.int32)
        window_ms = 50.0
    except Exception:
        readout = None
        window = None
        window_ms = 50.0

    t0 = time.perf_counter()
    n_steps = int(round(duration_ms / engine.p.dt))
    first_gf = None
    gf_before = 0
    for _ in range(n_steps):
        spk = engine.step()
        if readout is not None and window is not None:
            if spk.size:
                window += np.bincount(spk, minlength=connectome.n).astype(np.int32)
            # decay-ish: keep a simple cumulative for peak channel only every 10 steps
            if engine.step_count % 10 == 0:
                ch = readout.channels(window, max(engine.t_ms, window_ms))
                escape_peak = max(escape_peak, float(ch.get("escape_takeoff", 0.0)))
                window[:] = 0
        if first_gf is None and gf.size:
            cur = int(engine.spike_counts[gf].sum())
            if cur > gf_before:
                first_gf = float(engine.t_ms)
            gf_before = cur
    wall = time.perf_counter() - t0

    return EscapeRunResult(
        experiment=experiment,
        seed=seed,
        duration_ms=duration_ms,
        n_neurons=int(connectome.n),
        n_connections=int(connectome.w.nnz),
        n_synapses=int(np.abs(connectome.w.data).sum()),
        dtype=str(np.dtype(dtype)),
        total_spikes=int(engine.spike_counts.sum()),
        active_neurons=int((engine.spike_counts > 0).sum()),
        gf_spikes=int(engine.spike_counts[gf].sum()) if gf.size else 0,
        lc4_spikes=int(engine.spike_counts[lc4].sum()) if lc4.size else 0,
        lplc2_spikes=int(engine.spike_counts[lplc2].sum()) if lplc2.size else 0,
        first_gf_spike_ms=first_gf,
        escape_peak=escape_peak,
        spike_counts=engine.spike_counts.copy(),
        wall_time_s=wall,
        sim_steps=n_steps,
        metadata={
            "flyquant_version": __version__,
            "upstream_commit": get_upstream_commit(),
            "dataset": getattr(connectome, "dataset", DATASET_ID),
            "rate_hz": rate_hz,
            "n_drive": int(drive.size),
            "storage_weight_dtype": str(connectome.w.data.dtype),
            "execution_dtype": str(np.dtype(dtype)),
            "weight_override": weight_data is not None,
        },
    )
