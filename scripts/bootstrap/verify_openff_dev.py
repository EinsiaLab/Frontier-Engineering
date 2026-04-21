#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the openff-dev benchmark runtime.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root. Defaults to the parent repo of this script.",
    )
    return parser.parse_args()


def safe_version(name: str) -> str:
    try:
        return metadata.version(name)
    except Exception:
        return "unknown"


def check_imports() -> None:
    modules = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("openmm", "openmm"),
        ("rdkit", "rdkit"),
        ("openff.toolkit", "openff-toolkit"),
        ("openff.units", "openff-units"),
        ("openff.utilities", "openff-utilities"),
    ]
    print("== Import check ==")
    for module_name, package_name in modules:
        importlib.import_module(module_name)
        print(f"{module_name}: ok ({safe_version(package_name)})")


def run_prepare_smoke(repo_root: Path, task_rel: str) -> None:
    task_dir = repo_root / "benchmarks" / "MolecularMechanics" / task_rel
    script = task_dir / "verification" / "evaluate.py"
    raw_task = task_dir / "data" / "raw_task.json"
    with tempfile.TemporaryDirectory(prefix=f"openff_{task_rel}_") as tmpdir:
        out_path = Path(tmpdir) / "prepared.json"
        cmd = [
            sys.executable,
            str(script),
            "prepare",
            "--raw-task",
            str(raw_task),
            "--prepared-output",
            str(out_path),
        ]
        print(f"== Smoke prepare: {task_rel} ==")
        subprocess.run(cmd, cwd=str(repo_root), check=True)
        if not out_path.is_file():
            raise RuntimeError(f"Expected output not produced: {out_path}")
        print(f"{task_rel}: ok")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()

    check_imports()
    run_prepare_smoke(repo_root, "weighted_parameter_coverage")
    run_prepare_smoke(repo_root, "diverse_conformer_portfolio")
    run_prepare_smoke(repo_root, "torsion_profile_fitting")

    print("openff-dev verification complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
