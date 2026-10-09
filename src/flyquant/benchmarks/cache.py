"""Result caching with invalidation on code/config/dataset identity changes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from flyquant.paths import CACHE_DIR, ensure_output_dirs


def config_fingerprint(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def cache_path(fingerprint: str) -> Path:
    ensure_output_dirs()
    return CACHE_DIR / f"{fingerprint}.json"


def load_cached(fingerprint: str) -> dict | None:
    path = cache_path(fingerprint)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_cached(fingerprint: str, record: dict) -> Path:
    path = cache_path(fingerprint)
    path.write_text(json.dumps(record, indent=2, default=str))
    return path
