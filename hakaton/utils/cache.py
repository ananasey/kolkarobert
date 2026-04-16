"""Simple disk cache helpers used by API clients."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def make_cache_key(*parts: object) -> str:
    """Build a stable SHA-256 key from arbitrary JSON-serializable values."""

    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json_cache(cache_dir: Path, key: str) -> dict[str, Any] | None:
    """Read a JSON cache object or return None when it does not exist."""

    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json_cache(cache_dir: Path, key: str, data: dict[str, Any]) -> None:
    """Write JSON data to cache, ignoring non-critical filesystem errors."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    try:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        return


def read_bytes_cache(cache_dir: Path, key: str, suffix: str = ".bin") -> bytes | None:
    """Read bytes from cache or return None."""

    path = cache_dir / f"{key}{suffix}"
    if not path.exists():
        return None
    try:
        return path.read_bytes()
    except OSError:
        return None


def write_bytes_cache(
    cache_dir: Path,
    key: str,
    content: bytes,
    suffix: str = ".bin",
) -> None:
    """Write bytes to cache, ignoring non-critical filesystem errors."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}{suffix}"
    try:
        path.write_bytes(content)
    except OSError:
        return
