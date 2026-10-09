"""
Thin recording wrapper around upstream LIFEngine (no upstream edits).

Optionally records spike times for watched neurons and membrane snapshots
for a configurable index set every K steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RecordingConfig:
    record_spikes: bool = True
    record_v: bool = False
    v_indices: np.ndarray | None = None
    v_stride: int = 10  # sample membrane every K steps
    watch_indices: np.ndarray | None = None

    def normalised(self, n: int) -> "RecordingConfig":
        watch = self.watch_indices
        if watch is not None:
            watch = np.asarray(watch, dtype=np.int64)
        vind = self.v_indices
        if vind is not None:
            vind = np.asarray(vind, dtype=np.int64)
        elif self.record_v:
            # Default: first min(64, n) neurons — never full population by accident.
            vind = np.arange(min(64, n), dtype=np.int64)
        return RecordingConfig(
            record_spikes=self.record_spikes,
            record_v=bool(self.record_v),
            v_indices=vind,
            v_stride=max(1, int(self.v_stride)),
            watch_indices=watch,
        )


@dataclass
class RecordingResult:
    spike_times_ms: dict[int, list[float]] = field(default_factory=dict)
    v_times_ms: list[float] = field(default_factory=list)
    v_traces: np.ndarray | None = None  # shape (n_samples, n_tracked)
    v_indices: np.ndarray | None = None
    enabled: bool = False

    def to_meta(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "n_watched_with_spikes": len(self.spike_times_ms),
            "n_v_samples": len(self.v_times_ms),
            "v_indices": None if self.v_indices is None else self.v_indices.tolist(),
            "record_v": self.v_traces is not None,
        }


class RecordingEngine:
    """Wraps an upstream LIFEngine, forwarding attributes and recording steps."""

    def __init__(self, engine, config: RecordingConfig | None = None):
        self.engine = engine
        self.config = (config or RecordingConfig()).normalised(engine.n)
        self.result = RecordingResult(enabled=True)
        if self.config.record_v and self.config.v_indices is not None:
            self.result.v_indices = np.asarray(self.config.v_indices, dtype=np.int64)
            self._v_buf: list[np.ndarray] = []
        else:
            self._v_buf = []
        self._spike_map: dict[int, list[float]] = {}
        if self.config.watch_indices is not None:
            for i in self.config.watch_indices:
                self._spike_map[int(i)] = []
        self._watch_set = set(self._spike_map.keys())

    def __getattr__(self, name: str):
        return getattr(self.engine, name)

    def step(self) -> np.ndarray:
        spk = self.engine.step()
        cfg = self.config
        t = float(self.engine.t_ms)
        if cfg.record_spikes and spk.size:
            if self._watch_set:
                for i in spk:
                    ii = int(i)
                    if ii in self._watch_set:
                        self._spike_map[ii].append(t)
            else:
                # Record all spikes into map (can be large — prefer watch_indices)
                for i in spk:
                    self._spike_map.setdefault(int(i), []).append(t)
        if cfg.record_v and self.result.v_indices is not None:
            if self.engine.step_count % cfg.v_stride == 0:
                self.result.v_times_ms.append(t)
                self._v_buf.append(self.engine.v[self.result.v_indices].astype(np.float32).copy())
        return spk

    def run(self, duration_ms: float):
        n_steps = int(round(duration_ms / self.engine.p.dt))
        return [self.step() for _ in range(n_steps)]

    def finalise(self) -> RecordingResult:
        self.result.spike_times_ms = {k: list(v) for k, v in self._spike_map.items()}
        if self._v_buf:
            self.result.v_traces = np.stack(self._v_buf, axis=0)
        return self.result
