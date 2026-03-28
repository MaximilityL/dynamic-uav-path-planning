"""I/O helpers for saving scaffold artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _to_serializable(value: Any) -> Any:
    """Convert numpy-heavy payloads into JSON-friendly objects."""
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    """Write a JSON artifact with consistent formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_to_serializable(payload), handle, indent=2)
    return path


def append_jsonl(path: str | Path, payload: Dict[str, Any]) -> Path:
    """Append one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_serializable(payload)))
        handle.write("\n")
    return path


def save_npz(path: str | Path, payload: Dict[str, Any]) -> Path:
    """Save a compressed NumPy payload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path
