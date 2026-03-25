from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = REPO_ROOT / "skill"
COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


@dataclass(frozen=True)
class SkillSpec:
    key: str
    title: str
    summary: str


@dataclass(frozen=True)
class TargetSpec:
    key: str
    title: str
    summary: str
    relative_destination: Path


SKILLS: tuple[SkillSpec, ...] = (
    SkillSpec(
        key="contributor",
        title="Contributor",
        summary="Help add or update benchmarks and prepare clean pull requests.",
    ),
    SkillSpec(
        key="evaluator",
        title="Evaluator",
        summary="Help set up environments and run or debug benchmark evaluations.",
    ),
)

TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec(
        key="codex",
        title="Codex",
        summary="Install into .codex/skills for OpenAI Codex.",
        relative_destination=Path(".codex") / "skills",
    ),
    TargetSpec(
        key="claude",
        title="Claude Code",
        summary="Install into .claude/skills for Claude Code.",
        relative_destination=Path(".claude") / "skills",
    ),
    TargetSpec(
        key="cursor",
        title="Cursor",
        summary="Install into .cursor/skills for Cursor Agent Skills.",
        relative_destination=Path(".cursor") / "skills",
    ),
)

SKILL_BY_KEY = {item.key: item for item in SKILLS}
TARGET_BY_KEY = {item.key: item for item in TARGETS}


def _canonical_skill_name(role: str) -> str:
    return f"frontier-{role}"


def _source_directory(role: str, target: str) -> Path:
    return SKILL_ROOT / f".{target}" / "skills" / _canonical_skill_name(role)


def _destination_directory(role: str, target: str, dest_root: Path) -> Path:
    target_spec = TARGET_BY_KEY[target]
    return dest_root / target_spec.relative_destination / _canonical_skill_name(role)


def _validate_destination_root(dest_root: Path) -> None:
    required = (
        dest_root / "README.md",
        dest_root / "frontier_eval" / "README.md",
        dest_root / "benchmarks",
    )
    missing = [str(path.relative_to(dest_root)) for path in required if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise NotADirectoryError(
            f"{dest_root} does not look like a Frontier-Engineering repo root. Missing: {joined}"
        )


def _print_catalog() -> None:
    print("Available skills:")
    for item in SKILLS:
        print(f"  - {item.key}: {item.summary}")
    print()
    print("Available targets:")
    for item in TARGETS:
        print(f"  - {item.key}: {item.summary}")


def _prompt_selection(kind: str, entries: tuple[SkillSpec, ...] | tuple[TargetSpec, ...]) -> str:
    print(f"Select {kind}:")
    for idx, item in enumerate(entries, start=1):
        print(f"  {idx}. {item.key} - {item.summary}")

    while True:
        raw = input("> ").strip().lower()
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(entries):
                return entries[index].key

        for item in entries:
            if raw == item.key:
                return item.key

        print(f"Enter a number from 1 to {len(entries)} or a valid {kind} name.")


def _normalize_argv(argv: list[str]) -> list[str]:
    if argv and argv[0] == "install":
        return argv[1:]
    if argv == ["list"]:
        return ["--list"]
    return argv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m frontier_eval skill",
        description="Install bundled project skills for Codex, Claude Code, or Cursor.",
    )
    parser.add_argument("items", nargs="*")
    parser.add_argument(
        "--dest",
        default=".",
        help="Project root where the tool-specific skill directory should be installed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing installed skill directory.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List bundled skills and targets, then exit.",
    )
    parser.epilog = (
        "Usage: python -m frontier_eval skill [role] [target]\n"
        "Order does not matter. Missing values are prompted interactively."
    )
    return parser


def _resolve_items(items: list[str], parser: argparse.ArgumentParser) -> tuple[str | None, str | None]:
    if len(items) > 2:
        parser.error("accepts at most two positional values: one role and one target")

    role: str | None = None
    target: str | None = None
    unknown: list[str] = []

    for raw in items:
        token = raw.strip().lower()
        is_role = token in SKILL_BY_KEY
        is_target = token in TARGET_BY_KEY

        if is_role and is_target:
            parser.error(f"ambiguous token: {raw}")
        if is_role:
            if role is not None:
                parser.error("received more than one role")
            role = token
            continue
        if is_target:
            if target is not None:
                parser.error("received more than one target")
            target = token
            continue
        unknown.append(raw)

    if unknown:
        parser.error(f"unknown value(s): {', '.join(unknown)}")

    return role, target


def _install(role: str, target: str, dest_root: Path, force: bool) -> Path:
    source_dir = _source_directory(role=role, target=target)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"missing bundled skill directory: {source_dir}")

    destination = _destination_directory(role=role, target=target, dest_root=dest_root)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if not force:
            raise FileExistsError(
                f"{destination} already exists. Re-run with --force to overwrite it."
            )
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()

    shutil.copytree(source_dir, destination, ignore=COPY_IGNORE)
    return destination


def main(argv: list[str] | None = None) -> int:
    argv = _normalize_argv(list(sys.argv[1:] if argv is None else argv))
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list:
        _print_catalog()
        return 0

    role, target = _resolve_items(args.items, parser)
    role = role or _prompt_selection("skill", SKILLS)
    target = target or _prompt_selection("target", TARGETS)
    dest_root = Path(args.dest).expanduser().resolve()
    _validate_destination_root(dest_root)

    installed_path = _install(role=role, target=target, dest_root=dest_root, force=args.force)

    print(f"Installed {role} for {target} at {installed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
