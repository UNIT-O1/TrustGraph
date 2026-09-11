"""Content-addressed disk cache for provider calls.

§6.2 asks for paraphrase caching per category so re-runs are cheap. The same
mechanism is applied to measurement and re-check calls, which makes an
interrupted demo resumable and keeps iteration on the frontend from re-billing
the whole grid.

Caching measurement calls is a real tradeoff: a cached run is a replay, not a
fresh measurement. So every cached cell is flagged ``cached: true`` and the
dashboard shows the count, and ``fresh=true`` on a run bypasses reads entirely.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class DiskCache:
    def __init__(self, directory: str | Path, enabled: bool = True) -> None:
        self._dir = Path(directory)
        self._enabled = enabled
        if self._enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def key(namespace: str, payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:40]
        return f"{namespace}-{digest}"

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if not self._enabled:
            return None
        path = self._path(key)
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        if not self._enabled:
            return
        path = self._path(key)
        # Atomic replace, so a killed process can never leave a half-written
        # entry that later reads as corrupt.
        try:
            fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle)
            os.replace(tmp, path)
        except OSError:
            return

    def clear(self) -> int:
        if not self._dir.exists():
            return 0
        removed = 0
        for path in self._dir.glob("*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        return removed
