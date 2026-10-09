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
from flyquant.reference.recording import RecordingConfig, RecordingEngine
from flyquant.reference.web_loader import load_web_connectome, web_assets_available


class DerivedDataUnavailable(RuntimeError):
    """Raised when Codex-derived connectome / retinotopy assets are missing."""


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
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REFERENCE_DIR,
            text=True,
        ).strip()
        return out
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return UPSTREAM_COMMIT


def derived_data_status() -> dict[str, Any]:
    """Inspect whether upstream derived connectome + FlyWire sources exist."""
    ensure_upstream_on_path()
    import config as lab_config

    npz = lab_config.CONNECTOME_NPZ
    index = lab_config.NEURON_INDEX
    col = lab_config.SRC.get("column_assignment")
    missing = []
    if not npz.exists():
        missing.append(str(npz))
    if not index.exists():
        missing.append(str(index))
    if col is not None and not col.exists():
        missing.append(str(col))
    return {
        "available": len(missing) == 0,
        "missing": missing,
        "flywire_dir": str(lab_config.FLYWIRE_DIR),
        "connectome_npz": str(npz),
        "reason": (
            None
            if not missing
            else (
                "Derived connectome / retinotopy assets missing. "
                "Set FLYWIRE_V783_DIR, download Codex files, and run "
                "`python -m brain.connectivity.build_connectome` in the submodule. "
                f"Missing: {missing}"
            )
        ),
    }


def load_reference_connectome(source: str = "auto", subgraph: str | None = None):
    """
    Load the reference connectome.

    source: auto | derived | web | synthetic
    subgraph: None | escape
    """
    ensure_upstream_on_path()

    if source == "synthetic":
        from flyquant.reference.synthetic import make_synthetic_connectome

        c = make_synthetic_connectome()
    elif source == "web":
        c = load_web_connectome()
    elif source == "derived":
        status = derived_data_status()
        if not status["available"]:
            raise DerivedDataUnavailable(status["reason"])
        from brain.neurons.registry import load_connectome

        c = load_connectome()
    else:
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
            col = c.w.getcol(int(i)).tocoo()
            keep.update(int(j) for j in col.row[:200])
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
    recording: Any | None = None

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
            "recording_meta": (
                None if self.recording is None else self.recording.to_meta()
            ),
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
    recording: RecordingConfig | None = None,
    silence_types: tuple[str, ...] = (),
) -> EscapeRunResult:
    """Poisson LC4/LPLC2 → DNp01 probe (default FlyQuant behavioural surface)."""
    raw = make_engine(connectome, seed=seed, dtype=dtype, weight_data=weight_data)
    for t in silence_types:
        idx = _indices_of(connectome, t)
        if idx.size:
            raw.silence(idx)

    lc4 = _indices_of(connectome, "LC4")
    lplc2 = _indices_of(connectome, "LPLC2")
    gf = _indices_of(connectome, "DNp01")
    drive = np.concatenate([lc4, lplc2])
    if drive.size == 0:
        raise RuntimeError("No LC4/LPLC2 neurons in connectome; cannot run escape probe")

    raw.set_poisson(drive, rate_hz)

    rec_cfg = recording
    if rec_cfg is None:
        engine = raw
        recorder = None
    else:
        if rec_cfg.watch_indices is None:
            watch = np.concatenate([gf, lc4[:8], lplc2[:8]]) if gf.size else drive[:32]
            rec_cfg = RecordingConfig(
                record_spikes=rec_cfg.record_spikes,
                record_v=rec_cfg.record_v,
                v_indices=rec_cfg.v_indices,
                v_stride=rec_cfg.v_stride,
                watch_indices=np.unique(watch.astype(np.int64)),
            )
        recorder = RecordingEngine(raw, rec_cfg)
        engine = recorder

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
    n_steps = int(round(duration_ms / raw.p.dt))
    first_gf = None
    gf_before = 0
    for _ in range(n_steps):
        spk = engine.step()
        if readout is not None and window is not None:
            if spk.size:
                window += np.bincount(spk, minlength=connectome.n).astype(np.int32)
            if raw.step_count % 10 == 0:
                ch = readout.channels(window, max(raw.t_ms, window_ms))
                escape_peak = max(escape_peak, float(ch.get("escape_takeoff", 0.0)))
                window[:] = 0
        if first_gf is None and gf.size:
            cur = int(raw.spike_counts[gf].sum())
            if cur > gf_before:
                first_gf = float(raw.t_ms)
            gf_before = cur
    wall = time.perf_counter() - t0

    rec_result = recorder.finalise() if recorder is not None else None

    return EscapeRunResult(
        experiment=experiment,
        seed=seed,
        duration_ms=duration_ms,
        n_neurons=int(connectome.n),
        n_connections=int(connectome.w.nnz),
        n_synapses=int(np.abs(connectome.w.data).sum()),
        dtype=str(np.dtype(dtype)),
        total_spikes=int(raw.spike_counts.sum()),
        active_neurons=int((raw.spike_counts > 0).sum()),
        gf_spikes=int(raw.spike_counts[gf].sum()) if gf.size else 0,
        lc4_spikes=int(raw.spike_counts[lc4].sum()) if lc4.size else 0,
        lplc2_spikes=int(raw.spike_counts[lplc2].sum()) if lplc2.size else 0,
        first_gf_spike_ms=first_gf,
        escape_peak=escape_peak,
        spike_counts=raw.spike_counts.copy(),
        wall_time_s=wall,
        sim_steps=n_steps,
        recording=rec_result,
        metadata={
            "flyquant_version": __version__,
            "upstream_commit": get_upstream_commit(),
            "dataset": getattr(connectome, "dataset", DATASET_ID),
            "rate_hz": rate_hz,
            "n_drive": int(drive.size),
            "storage_weight_dtype": str(connectome.w.data.dtype),
            "execution_dtype": str(np.dtype(dtype)),
            "weight_override": weight_data is not None,
            "silence_types": list(silence_types),
            "engine_state": {
                "v_bytes": int(raw.v.nbytes),
                "g_bytes": int(raw.g.nbytes),
                "ring_bytes": int(raw._ring.nbytes),
                "wdata_bytes": int(raw.wdata.nbytes),
                "state_arrays_bytes": int(raw.v.nbytes + raw.g.nbytes + raw._ring.nbytes),
                "dtype": str(np.dtype(raw.dtype)),
            },
        },
    )


def run_looming_escape(
    connectome,
    *,
    seed: int = 0,
    duration_ms: float = 280.0,
    azimuth_deg: float = 45.0,
    l_over_v_ms: float = 20.0,
    quiet: bool = True,
) -> EscapeRunResult:
    """
    Upstream geometric looming escape (needs derived connectome + retinotopy).

    Applies weight overrides indirectly by mutating engine after Session creation
    is not supported here — use reference weights. For compressed comparisons,
    prefer applying compression to connectome.w before calling, or use the
    Poisson probe with weight_data.
    """
    status = derived_data_status()
    if not status["available"]:
        raise DerivedDataUnavailable(status["reason"])

    ensure_upstream_on_path()
    from brain.sensory.encoders import LoomingEncoder
    from brain.sensory.retinotopy import load_retinotopy
    from simulation.engine.session import Session
    from simulation.stimuli.looming import LoomingStimulus

    retino = load_retinotopy(connectome)
    sess = Session(connectome, seed=seed)
    enc = LoomingEncoder(connectome, retino)
    stim = LoomingStimulus(
        azimuth_deg=azimuth_deg,
        elevation_deg=0.0,
        half_size_mm=5.0,
        speed_mm_s=5.0 / (l_over_v_ms / 1000.0),
        start_distance_mm=50.0,
        t_start_ms=20.0,
    )
    sess.add_stimulus(enc, stim)

    gf = _indices_of(connectome, "DNp01")
    lc4 = _indices_of(connectome, "LC4")
    lplc2 = _indices_of(connectome, "LPLC2")
    prev_gf = 0
    first_gf = None
    escape_peak = 0.0

    t0 = time.perf_counter()
    steps_5 = int(duration_ms / 5.0)
    for _ in range(steps_5):
        sess.advance(5.0)
        ch = sess.readout.channels(sess.recorder.window_sum, sess.window_ms)
        escape_peak = max(escape_peak, float(ch.get("escape_takeoff", 0.0)))
        if first_gf is None and gf.size:
            cur = int(sess.engine.spike_counts[gf].sum())
            if cur > prev_gf:
                first_gf = float(sess.engine.t_ms)
            prev_gf = cur
    wall = time.perf_counter() - t0
    if not quiet:
        print("looming escape done: GF", int(sess.engine.spike_counts[gf].sum()))

    return EscapeRunResult(
        experiment="looming",
        seed=seed,
        duration_ms=duration_ms,
        n_neurons=int(connectome.n),
        n_connections=int(connectome.w.nnz),
        n_synapses=int(np.abs(connectome.w.data).sum()),
        dtype=str(np.dtype(sess.engine.dtype)),
        total_spikes=int(sess.engine.spike_counts.sum()),
        active_neurons=int((sess.engine.spike_counts > 0).sum()),
        gf_spikes=int(sess.engine.spike_counts[gf].sum()) if gf.size else 0,
        lc4_spikes=int(sess.engine.spike_counts[lc4].sum()) if lc4.size else 0,
        lplc2_spikes=int(sess.engine.spike_counts[lplc2].sum()) if lplc2.size else 0,
        first_gf_spike_ms=first_gf,
        escape_peak=escape_peak,
        spike_counts=sess.engine.spike_counts.copy(),
        wall_time_s=wall,
        sim_steps=int(round(duration_ms / sess.engine.p.dt)),
        metadata={
            "flyquant_version": __version__,
            "upstream_commit": get_upstream_commit(),
            "dataset": getattr(connectome, "dataset", DATASET_ID),
            "stimulus": "geometric_looming",
            "azimuth_deg": azimuth_deg,
            "l_over_v_ms": l_over_v_ms,
            "engine_state": {
                "v_bytes": int(sess.engine.v.nbytes),
                "g_bytes": int(sess.engine.g.nbytes),
                "ring_bytes": int(sess.engine._ring.nbytes),
                "wdata_bytes": int(sess.engine.wdata.nbytes),
                "state_arrays_bytes": int(
                    sess.engine.v.nbytes + sess.engine.g.nbytes + sess.engine._ring.nbytes
                ),
                "dtype": str(np.dtype(sess.engine.dtype)),
            },
        },
    )


def run_escape_controls(
    connectome,
    *,
    seed: int = 0,
    duration_ms: float = 240.0,
    azimuth_deg: float = 45.0,
) -> list[dict[str, Any]]:
    """Upstream-style lesion controls; requires derived retinotopy assets."""
    status = derived_data_status()
    if not status["available"]:
        raise DerivedDataUnavailable(status["reason"])

    ensure_upstream_on_path()
    import importlib

    from brain.sensory.retinotopy import load_retinotopy

    # Upstream module name starts with digits — cannot use a normal import.
    controls = importlib.import_module("experiments.02_escape_controls")
    run_condition = controls.run_condition

    retino = load_retinotopy(connectome)
    conditions = [
        ("looming", ()),
        ("receding", ()),
        ("static", ()),
        ("looming", ("LC4",)),
        ("looming", ("LPLC2",)),
        ("looming", ("LC4", "LPLC2")),
    ]
    # Upstream run_condition uses fixed DURATION_MS / AZIMUTH_DEG.
    _ = duration_ms
    _ = azimuth_deg
    results = []
    for kind, sil in conditions:
        r = run_condition(connectome, retino, kind, sil, seed=seed)
        r["gf_spikes"] = r.get("dnp01_spikes", 0)
        results.append(r)
    return results
