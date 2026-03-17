"""CLI helpers shared by ReactionOptimisation scripts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


def load_module(path: Path, module_name: str):
    """Load a Python module directly from a file path."""
    resolved = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(module_name, str(resolved))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: str | None, payload: dict[str, Any]) -> None:
    """Write a JSON payload when an output path is provided."""
    if not path:
        return
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
