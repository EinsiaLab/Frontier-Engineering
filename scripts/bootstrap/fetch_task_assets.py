#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "scripts" / "bootstrap" / "assets_manifest.json"
PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")
GDRIVE_ENTRY_RE = re.compile(
    r'<div class="flip-entry" id="entry-([^"]+)".*?<a href="([^"]+)"[^>]*>.*?<div class="flip-entry-title">(.*?)</div>',
    re.S,
)
GDRIVE_ID_PATH_RE = re.compile(r"/(?:folders|d)/([A-Za-z0-9_-]+)")
HTTP_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) Frontier-Eng asset bootstrap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch benchmark and algorithm assets for Frontier-Eng.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the asset manifest JSON.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Bundle id or tag to execute. Repeatable. Use --target all to run every bundle.",
    )
    parser.add_argument("--list", action="store_true", help="List available bundles and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing them.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run commands even if outputs already exist. Existing downloads/repos are still updated conservatively.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def venv_python(env_name: str) -> Path:
    primary = REPO_ROOT / ".venvs" / env_name / "bin" / "python"
    if primary.is_file():
        return primary
    legacy_aliases = {
        "frontier-eval-driver": "frontier-eval-2",
    }
    legacy_name = legacy_aliases.get(env_name)
    if legacy_name:
        legacy = REPO_ROOT / ".venvs" / legacy_name / "bin" / "python"
        if legacy.is_file():
            return legacy
    return primary


def resolve_placeholders(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(1)
        if token == "repo_root":
            return str(REPO_ROOT)
        if token.startswith("env_python:"):
            env_name = token.split(":", 1)[1]
            return str(venv_python(env_name))
        raise ValueError(f"Unsupported placeholder: {{{token}}}")

    return PLACEHOLDER_RE.sub(repl, value)


def resolve_relpath(path_str: str) -> Path:
    return (REPO_ROOT / resolve_placeholders(path_str)).resolve()


def run_cmd(argv: list[str], *, cwd: Path | None = None, dry_run: bool) -> None:
    pretty = " ".join(shlex_quote(arg) for arg in argv)
    if cwd is not None:
        print(f"[run] (cd {cwd} && {pretty})")
    else:
        print(f"[run] {pretty}")
    if dry_run:
        return
    subprocess.run(argv, cwd=str(cwd) if cwd else None, check=True)


def shlex_quote(value: str) -> str:
    return subprocess.list2cmdline([value]) if os.name == "nt" else "'" + value.replace("'", "'\"'\"'") + "'"


def bundle_matches(bundle: dict[str, Any], targets: list[str]) -> bool:
    if not targets:
        return False
    bundle_id = str(bundle["id"])
    tags = {bundle_id, *[str(tag) for tag in bundle.get("tags", [])]}
    return "all" in targets or any(target in tags for target in targets)


def list_bundles(manifest: dict[str, Any]) -> None:
    for bundle in manifest.get("bundles", []):
        tags = ", ".join(str(tag) for tag in bundle.get("tags", []))
        print(f"- {bundle['id']}: {bundle.get('description', '')}")
        if tags:
            print(f"  tags: {tags}")


def ensure_expected(dest: Path, expected_paths: list[str], expected_any: list[str]) -> bool:
    all_ok = all((dest / rel).exists() for rel in expected_paths)
    any_ok = True if not expected_any else any((dest / rel).exists() for rel in expected_any)
    return all_ok and any_ok


def normalize_single_child(dest: Path, expected_paths: list[str], expected_any: list[str]) -> None:
    if ensure_expected(dest, expected_paths, expected_any):
        return
    children = [path for path in dest.iterdir()]
    if len(children) != 1 or not children[0].is_dir():
        return
    child = children[0]
    if not ensure_expected(child, expected_paths, expected_any):
        return
    for item in child.iterdir():
        target = dest / item.name
        if target.exists():
            raise RuntimeError(f"Cannot normalize download into {dest}: {target} already exists.")
        shutil.move(str(item), str(target))
    child.rmdir()


def gdrive_request(url: str, *, method: str = "GET"):
    request = urllib.request.Request(url, headers={"User-Agent": HTTP_USER_AGENT}, method=method)
    return urllib.request.urlopen(request, timeout=120)


def extract_gdrive_id(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    if query.get("id"):
        return query["id"][0]
    match = GDRIVE_ID_PATH_RE.search(parsed.path)
    if match:
        return match.group(1)
    raise RuntimeError(f"Could not extract a Google Drive id from: {url}")


def list_public_gdrive_folder(folder_id: str) -> list[dict[str, str]]:
    with gdrive_request(f"https://drive.google.com/embeddedfolderview?id={folder_id}#list") as response:
        text = response.read().decode("utf-8", "replace")

    entries: list[dict[str, str]] = []
    for _, href, raw_title in GDRIVE_ENTRY_RE.findall(text):
        href = html.unescape(href)
        title = html.unescape(raw_title).strip()
        if "/drive/folders/" in href:
            entry_type = "folder"
        elif "/file/d/" in href or "uc?id=" in href or "open?id=" in href:
            entry_type = "file"
        else:
            continue
        entries.append({"type": entry_type, "id": extract_gdrive_id(href), "title": title})

    if not entries:
        raise RuntimeError(f"Could not enumerate public Google Drive folder contents for id={folder_id}")

    entries.sort(key=lambda item: (0 if item["type"] == "folder" else 1, item["title"]))
    return entries


def expected_remote_size(file_id: str) -> int | None:
    try:
        with gdrive_request(f"https://drive.google.com/uc?export=download&id={file_id}", method="HEAD") as response:
            size = response.headers.get("Content-Length")
    except Exception:
        return None
    if not size:
        return None
    try:
        return int(size)
    except ValueError:
        return None


def download_public_gdrive_file(file_id: str, dest: Path, *, dry_run: bool) -> None:
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    print(f"[run] public-gdrive-file {file_id} -> {dest}")
    if dry_run:
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    remote_size = expected_remote_size(file_id)
    if dest.exists():
        if dest.is_dir():
            raise RuntimeError(f"Cannot download file into directory path: {dest}")
        local_size = dest.stat().st_size
        if remote_size is not None and local_size == remote_size:
            print(f"[skip] File already present: {dest}")
            return
        if remote_size is None and local_size > 0:
            print(f"[skip] File already present (size unchecked): {dest}")
            return

    temp_path = dest.with_name(dest.name + ".tmp")
    with gdrive_request(url) as response, temp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)

    if remote_size is not None and temp_path.stat().st_size != remote_size:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Incomplete download for {dest}: expected {remote_size} bytes")

    temp_path.replace(dest)


def download_public_gdrive_folder(url: str, dest: Path, *, dry_run: bool) -> None:
    def recurse(folder_id: str, current_dest: Path) -> None:
        if not dry_run:
            current_dest.mkdir(parents=True, exist_ok=True)
        for entry in list_public_gdrive_folder(folder_id):
            target = current_dest / entry["title"]
            if entry["type"] == "folder":
                print(f"[run] public-gdrive-folder {entry['id']} -> {target}")
                recurse(entry["id"], target)
            else:
                download_public_gdrive_file(entry["id"], target, dry_run=dry_run)

    recurse(extract_gdrive_id(url), dest)


def apply_patch(repo_dir: Path, patch_path: Path, *, dry_run: bool) -> None:
    check_cmd = ["git", "-C", str(repo_dir), "apply", "--check", str(patch_path)]
    reverse_cmd = ["git", "-C", str(repo_dir), "apply", "--reverse", "--check", str(patch_path)]
    apply_cmd = ["git", "-C", str(repo_dir), "apply", str(patch_path)]

    check_ok = subprocess.run(check_cmd, capture_output=True).returncode == 0
    if check_ok:
        run_cmd(apply_cmd, dry_run=dry_run)
        return

    reverse_ok = subprocess.run(reverse_cmd, capture_output=True).returncode == 0
    if reverse_ok:
        print(f"[skip] Patch already applied: {patch_path}")
        return

    raise RuntimeError(f"Patch does not apply cleanly: {patch_path}")


def handle_git_clone(step: dict[str, Any], *, dry_run: bool) -> None:
    repo_url = str(step["repo"])
    dest = resolve_relpath(str(step["dest"]))
    checkout = str(step.get("checkout", "") or "").strip()
    patch_rel = str(step.get("patch", "") or "").strip()

    if dest.exists() and not (dest / ".git").is_dir():
        print(f"[skip] {dest} exists and is not a git checkout; leaving it unchanged.")
        return

    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(["git", "clone", repo_url, str(dest)], dry_run=dry_run)
    else:
        print(f"[skip] Repo already present: {dest}")

    if checkout and (dest / ".git").is_dir():
        run_cmd(["git", "-C", str(dest), "fetch", "--all", "--tags"], dry_run=dry_run)
        run_cmd(["git", "-C", str(dest), "checkout", checkout], dry_run=dry_run)

    if patch_rel:
        patch_path = resolve_relpath(patch_rel)
        if dry_run:
            print(f"[run] git -C {dest} apply {patch_path}")
        elif (dest / ".git").is_dir():
            apply_patch(dest, patch_path, dry_run=dry_run)


def handle_editable_install(step: dict[str, Any], *, dry_run: bool) -> None:
    env_name = str(step["env_name"])
    env_python = venv_python(env_name)
    if not env_python.is_file() and not dry_run:
        raise RuntimeError(
            f"Missing env python for {env_name}: {env_python}. Run bash init.sh or bash scripts/env/setup_v1_task_envs.sh first."
        )
    project = resolve_relpath(str(step["project"]))
    run_cmd([str(env_python), "-m", "pip", "install", "-e", str(project)], dry_run=dry_run)


def handle_gdrive_folder(step: dict[str, Any], *, dry_run: bool) -> None:
    url = str(step["url"])
    dest = resolve_relpath(str(step["dest"]))
    expected_paths = [str(item) for item in step.get("expected_paths", [])]
    expected_any = [str(item) for item in step.get("expected_any", [])]

    if dest.exists() and ensure_expected(dest, expected_paths, expected_any):
        print(f"[skip] Download already present: {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.mkdir(parents=True, exist_ok=True)
    try:
        run_cmd(["uvx", "--from", "gdown", "gdown", "--folder", url, "-O", str(dest)], dry_run=dry_run)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        if dry_run:
            raise
        print(f"[warn] gdown folder download failed, falling back to direct public downloads: {exc}")
        download_public_gdrive_folder(url, dest, dry_run=dry_run)
    if not dry_run:
        normalize_single_child(dest, expected_paths, expected_any)
        if not ensure_expected(dest, expected_paths, expected_any):
            raise RuntimeError(f"Downloaded folder does not contain the expected files: {dest}")


def handle_command(step: dict[str, Any], *, dry_run: bool, force: bool) -> None:
    cwd = resolve_relpath(str(step.get("cwd", ".")))
    argv = [resolve_placeholders(str(arg)) for arg in step["argv"]]
    outputs = [cwd / resolve_placeholders(str(item)) for item in step.get("outputs", [])]
    if outputs and not force and all(path.exists() for path in outputs):
        print(f"[skip] Outputs already present for command in {cwd}")
        return
    run_cmd(argv, cwd=cwd, dry_run=dry_run)


def handle_shinkaevolve_patch(step: dict[str, Any], *, dry_run: bool) -> None:
    repo_dir = resolve_relpath(str(step["repo"]))
    display_path = repo_dir / "shinka" / "database" / "display.py"
    pricing_path = repo_dir / "shinka" / "llm" / "providers" / "pricing.csv"

    display_old = """        table.add_row(
            f\"Best: {best_score_str}\",
            island_display,
            status_display,
            score_display,
            program.metadata.get(\"patch_name\", \"[dim]N/A[/dim]\")[:30],
            program.metadata.get(\"patch_type\", \"[dim]N/A[/dim]\"),
            f\"{program.complexity:.1f}\",
            cost_display,
            time_display,
        )
"""
    display_new = """        patch_name = None
        patch_type = None
        if program.metadata:
            patch_name = program.metadata.get(\"patch_name\")
            patch_type = program.metadata.get(\"patch_type\")
        patch_name_display = (patch_name or \"[dim]N/A[/dim]\")[:30]
        patch_type_display = patch_type or \"[dim]N/A[/dim]\"
        table.add_row(
            f\"Best: {best_score_str}\",
            island_display,
            status_display,
            score_display,
            patch_name_display,
            patch_type_display,
            f\"{program.complexity:.1f}\",
            cost_display,
            time_display,
        )
"""
    display_text = display_path.read_text(encoding="utf-8")
    if "patch_name_display" not in display_text:
        if display_old not in display_text:
            raise RuntimeError(f"Could not locate expected ShinkaEvolve display block in {display_path}")
        if dry_run:
            print(f"[run] patch {display_path}")
        else:
            display_path.write_text(display_text.replace(display_old, display_new, 1), encoding="utf-8")
    else:
        print(f"[skip] ShinkaEvolve display patch already present: {display_path}")

    pricing_lines = pricing_path.read_text(encoding="utf-8").splitlines()
    extra_rows = [
        "qwen/qwen3-coder-next,openrouter,N/A,N/A,,,,\" False\",\" 0\",\" 0\"",
        "qwen3-coder-next,openai,N/A,N/A,,,,\" False\",\" 0\",\" 0\"",
    ]
    missing_rows = [row for row in extra_rows if row not in pricing_lines]
    if not missing_rows:
        print(f"[skip] ShinkaEvolve pricing patch already present: {pricing_path}")
        return
    if dry_run:
        print(f"[run] patch {pricing_path}")
        return
    anchor = "qwen/qwen3-coder,openrouter,N/A,N/A,,,,\" False\",\" 0\",\" 0\""
    try:
        idx = pricing_lines.index(anchor) + 1
    except ValueError as exc:
        raise RuntimeError(f"Could not locate pricing anchor row in {pricing_path}") from exc
    pricing_lines[idx:idx] = missing_rows
    pricing_path.write_text("\n".join(pricing_lines) + "\n", encoding="utf-8")


def execute_bundle(bundle: dict[str, Any], *, dry_run: bool, force: bool) -> None:
    print(f"== {bundle['id']} ==")
    if bundle.get("description"):
        print(bundle["description"])
    for step in bundle.get("steps", []):
        step_type = str(step["type"])
        if step_type == "git_clone":
          handle_git_clone(step, dry_run=dry_run)
        elif step_type == "editable_install":
          handle_editable_install(step, dry_run=dry_run)
        elif step_type == "gdrive_folder":
          handle_gdrive_folder(step, dry_run=dry_run)
        elif step_type == "command":
          handle_command(step, dry_run=dry_run, force=force)
        elif step_type == "shinkaevolve_patch":
          handle_shinkaevolve_patch(step, dry_run=dry_run)
        else:
          raise ValueError(f"Unsupported step type: {step_type}")
    print("")


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)

    if args.list:
        list_bundles(manifest)
        return 0

    targets = [target.strip() for target in args.target if target.strip()]
    if not targets:
        print("No --target specified. Use --list to inspect bundles or pass --target <bundle-or-tag>.", file=sys.stderr)
        return 2

    selected = [bundle for bundle in manifest.get("bundles", []) if bundle_matches(bundle, targets)]
    if not selected:
        print(f"No bundles matched: {targets}", file=sys.stderr)
        return 2

    for bundle in selected:
        execute_bundle(bundle, dry_run=args.dry_run, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
