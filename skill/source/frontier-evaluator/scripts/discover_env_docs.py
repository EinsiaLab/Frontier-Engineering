#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
BENCHMARKS_ROOT = REPO_ROOT / "benchmarks"
FRAMEWORK_DOCS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "README_zh-CN.md",
    REPO_ROOT / "frontier_eval" / "README.md",
    REPO_ROOT / "frontier_eval" / "README_zh-CN.md",
]

LINE_PATTERNS = [
    re.compile(r"\bconda\b", re.IGNORECASE),
    re.compile(r"\bmamba\b", re.IGNORECASE),
    re.compile(r"\bvenv\b", re.IGNORECASE),
    re.compile(r"pip install", re.IGNORECASE),
    re.compile(r"requirements?\.txt", re.IGNORECASE),
    re.compile(r"docker", re.IGNORECASE),
    re.compile(r"environment|环境", re.IGNORECASE),
    re.compile(r"task\.runtime\.(conda_env|python_path|use_conda_run)"),
    re.compile(r"python_path"),
    re.compile(r"ENGDESIGN_EVAL_MODE"),
]

ENV_PATTERNS = [
    re.compile(r"conda create -n ([A-Za-z0-9_.-]+)"),
    re.compile(r"conda run -n ([A-Za-z0-9_.-]+)"),
    re.compile(r"conda activate ([A-Za-z0-9_.-]+)"),
    re.compile(r"task\.runtime\.conda_env=([A-Za-z0-9_.-]+)"),
]


def is_env_line(line: str) -> bool:
    return any(pattern.search(line) for pattern in LINE_PATTERNS)


def extract_env_names(lines: list[str]) -> list[str]:
    envs: set[str] = set()
    for line in lines:
        for pattern in ENV_PATTERNS:
            for match in pattern.findall(line):
                if match in {"...", ".", ".."}:
                    continue
                envs.add(match)
    return sorted(envs)


def read_env_matches(path: Path) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    matched = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if is_env_line(line):
            matched.append({"line": lineno, "text": line.rstrip()})
    if not matched:
        return None
    env_names = extract_env_names([item["text"] for item in matched])
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "env_names": env_names,
        "matches": matched,
    }


def normalize_target(raw: str) -> str:
    raw = raw.strip().strip("/")
    if raw.startswith("benchmarks/"):
        raw = raw[len("benchmarks/") :]
    return raw


def gather_docs_for_target(target: str) -> list[Path]:
    target = normalize_target(target)
    parts = [part for part in target.split("/") if part]
    if not parts:
        return []

    domain_dir = BENCHMARKS_ROOT / parts[0]
    docs: list[Path] = []
    docs.extend(sorted(domain_dir.glob("README*")))

    if len(parts) >= 2:
        task_dir = domain_dir / parts[1]
        docs.extend(sorted(task_dir.glob("README*")))
        return docs

    for child in sorted(domain_dir.iterdir()) if domain_dir.exists() else []:
        if child.is_dir():
            docs.extend(sorted(child.glob("README*")))
    return docs


def parse_matrix_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    benchmarks = sorted(set(re.findall(r"task\.benchmark=([A-Za-z0-9_.\\/-]+)", text)))
    conda_envs = sorted(set(re.findall(r"task\.runtime\.conda_env=([A-Za-z0-9_.-]+)", text)))
    python_paths = sorted(set(re.findall(r"task\.runtime\.python_path=([^\s\"']+)", text)))
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "benchmarks": benchmarks,
        "conda_envs": conda_envs,
        "python_paths": python_paths,
    }


def discover_from_all() -> list[str]:
    targets: list[str] = []
    for path in sorted(BENCHMARKS_ROOT.glob("*")):
        if path.is_dir():
            targets.append(path.name)
    return targets


def build_report(targets: list[str], matrices: list[Path]) -> dict:
    matrix_reports = [parse_matrix_file(path) for path in matrices]

    derived_targets = list(targets)
    for report in matrix_reports:
        derived_targets.extend(report["benchmarks"])

    unique_targets = []
    seen = set()
    for target in derived_targets:
        normalized = normalize_target(target)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_targets.append(normalized)

    framework_docs = []
    for path in FRAMEWORK_DOCS:
        match = read_env_matches(path)
        if match:
            framework_docs.append(match)

    target_reports = []
    for target in unique_targets:
        docs = []
        seen_docs = set()
        for path in gather_docs_for_target(target):
            if path in seen_docs:
                continue
            seen_docs.add(path)
            match = read_env_matches(path)
            if match:
                docs.append(match)
        target_reports.append({"target": target, "docs": docs})

    return {
        "repo_root": str(REPO_ROOT),
        "framework_docs": framework_docs,
        "matrices": matrix_reports,
        "targets": target_reports,
    }


def print_report(report: dict) -> None:
    print(f"repo_root: {report['repo_root']}")
    print()

    if report["framework_docs"]:
        print("framework_docs:")
        for doc in report["framework_docs"]:
            envs = ", ".join(doc["env_names"]) if doc["env_names"] else "-"
            print(f"  - {doc['path']} | envs: {envs}")
        print()

    if report["matrices"]:
        print("matrices:")
        for matrix in report["matrices"]:
            benchmarks = ", ".join(matrix["benchmarks"]) or "-"
            conda_envs = ", ".join(matrix["conda_envs"]) or "-"
            python_paths = ", ".join(matrix["python_paths"]) or "-"
            print(f"  - {matrix['path']}")
            print(f"    benchmarks: {benchmarks}")
            print(f"    conda_envs: {conda_envs}")
            print(f"    python_paths: {python_paths}")
        print()

    for target in report["targets"]:
        print(f"[{target['target']}]")
        if not target["docs"]:
            print("  no matching environment hints found")
            print()
            continue
        for doc in target["docs"]:
            envs = ", ".join(doc["env_names"]) if doc["env_names"] else "-"
            print(f"  doc: {doc['path']}")
            print(f"  envs: {envs}")
            for match in doc["matches"][:12]:
                print(f"    {match['line']}: {match['text']}")
            if len(doc["matches"]) > 12:
                remaining = len(doc["matches"]) - 12
                print(f"    ... ({remaining} more matching lines)")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover Frontier-Engineering benchmark environment docs."
    )
    parser.add_argument("targets", nargs="*", help="Benchmark or domain names to inspect.")
    parser.add_argument(
        "--matrix",
        action="append",
        default=[],
        help="Batch matrix YAML file to scan for benchmark ids and runtime overrides.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan all benchmark domains for environment-related docs.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of human-readable text.",
    )
    args = parser.parse_args()

    targets = list(args.targets)
    if args.all:
        targets.extend(discover_from_all())

    matrix_paths = [Path(path).resolve() for path in args.matrix]
    for path in matrix_paths:
        if not path.exists():
            print(f"missing matrix file: {path}", file=sys.stderr)
            return 1

    report = build_report(targets=targets, matrices=matrix_paths)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

