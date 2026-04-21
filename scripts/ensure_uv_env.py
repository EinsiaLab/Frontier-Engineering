#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", shlex.join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or update a uv-managed virtual environment.")
    parser.add_argument("manifest", type=Path, help="Path to env manifest JSON.")
    parser.add_argument("--root", type=Path, required=True, help="Repository root.")
    parser.add_argument("--envs-dir", type=Path, required=True, help="Directory containing managed venvs.")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    name = str(manifest["name"]).strip()
    python_version = str(manifest.get("python", "3.12")).strip() or "3.12"
    env_dir = (args.envs_dir / name).resolve()

    if manifest.get("manual"):
        print(f"[skip] {name}: manual setup required")
        for note in manifest.get("notes", []):
            print(f"  - {note}")
        return 0

    env_dir.parent.mkdir(parents=True, exist_ok=True)

    run(["uv", "python", "install", python_version])
    if env_dir.exists():
        print(f"[reuse] {name}: {env_dir}")
    else:
        run(["uv", "venv", "--python", python_version, str(env_dir)])

    python_bin = env_dir / "bin" / "python"
    run(["uv", "pip", "install", "--python", str(python_bin), "-U", "pip", "setuptools", "wheel"])

    install_cmd = ["uv", "pip", "install", "--python", str(python_bin)]
    for requirement in manifest.get("requirements", []):
      install_cmd.extend(["-r", str((args.root / requirement).resolve())])
    install_cmd.extend(str(pkg) for pkg in manifest.get("packages", []))
    if len(install_cmd) > 5:
        run(install_cmd)

    print(f"[ready] {name}: {python_bin}")
    if manifest.get("system_requirements"):
        print("  System requirements:")
        for dep in manifest["system_requirements"]:
            print(f"    - {dep}")
    for note in manifest.get("notes", []):
        print(f"  Note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
