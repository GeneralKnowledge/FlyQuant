"""Filesystem layout for FlyQuant and the pinned upstream submodule."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
REFERENCE_DIR = REPO_ROOT / "reference" / "fruit-fly-lab"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
REPORTS_DIR = REPO_ROOT / "reports"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
CACHE_DIR = REPO_ROOT / "benchmarks" / "cache"
VARIANTS_DIR = REPO_ROOT / "benchmarks" / "variants"

WEB_DATA_DIR = REFERENCE_DIR / "web" / "data"
CONNECTOME_BIN = WEB_DATA_DIR / "connectome.bin"
NEURONS_BIN = WEB_DATA_DIR / "neurons.bin"
META_JSON = WEB_DATA_DIR / "meta.json"


def ensure_output_dirs() -> None:
    for d in (RESULTS_DIR, CACHE_DIR, VARIANTS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
