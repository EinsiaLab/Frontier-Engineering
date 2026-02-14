from __future__ import annotations

import os
from pathlib import Path


def find_dotenv(start: Path) -> Path | None:
    start = start.expanduser().resolve()
    for p in [start, *start.parents]:
        candidate = p / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_dotenv(path: Path, override: bool = False) -> None:
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if sep != "=":
            continue
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]

        if (not override) and (key in os.environ):
            continue
        os.environ[key] = value

