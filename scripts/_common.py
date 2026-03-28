"""Common script bootstrap helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_local_venv() -> None:
    """Re-exec the script with the repository-local virtualenv when available."""
    if os.environ.get("DUPP_SKIP_VENV") == "1":
        return

    venv_root = PROJECT_ROOT / ".venv"
    venv_python = venv_root / "bin" / "python"
    if not venv_python.exists():
        return

    current_python = Path(sys.executable)
    if current_python == venv_python or Path(sys.prefix) == venv_root:
        return

    os.execv(str(venv_python), [str(venv_python), *sys.argv])


def bootstrap_project() -> Path:
    """Ensure the correct interpreter and import path are active."""
    ensure_local_venv()
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT
