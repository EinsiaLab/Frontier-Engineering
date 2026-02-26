from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _tail(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def parse_result(
    *,
    stdout_text: str,
    stderr_text: str,
    mdriver_returncode: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    combined = (stdout_text or "") + "\n" + (stderr_text or "")

    metrics: dict[str, float] = {
        "combined_score": 0.0,
        "valid": 0.0,
    }
    artifacts: dict[str, Any] = {
        "mdriver_stdout_tail": _tail(stdout_text),
        "mdriver_stderr_tail": _tail(stderr_text),
    }

    score_line = ""
    for raw in combined.splitlines():
        line = raw.strip()
        if line.startswith("Score =") or line.startswith("Perf index ="):
            score_line = line
    if score_line:
        artifacts["score_line"] = score_line

    score_match = re.search(r"=\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*100\b", score_line or combined)
    if score_match:
        score = float(score_match.group(1))
        metrics["score_100"] = score
        metrics["score_ratio"] = score / 100.0
        metrics["combined_score"] = score

    testcase_match = re.search(r"\*\s*([0-9]+)\s*/\s*([0-9]+)\s*\(testcase\)", score_line or combined)
    if testcase_match:
        passed = float(testcase_match.group(1))
        total = float(testcase_match.group(2))
        metrics["testcases_passed"] = passed
        metrics["testcases_total"] = total
        if total > 0:
            metrics["testcase_pass_rate"] = passed / total

    metrics["mdriver_returncode"] = float(mdriver_returncode)
    if mdriver_returncode == 0 and "score_100" in metrics:
        metrics["valid"] = 1.0
    else:
        metrics["valid"] = 0.0
        metrics["combined_score"] = 0.0
        if mdriver_returncode != 0:
            artifacts["error_message"] = f"mdriver failed with return code {mdriver_returncode}"
        else:
            artifacts["error_message"] = "failed to parse score from mdriver output"

    return metrics, artifacts


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse MallocLab mdriver output into metrics/artifacts.")
    p.add_argument("--stdout-file", type=str, required=True)
    p.add_argument("--stderr-file", type=str, required=True)
    p.add_argument("--mdriver-returncode", type=int, required=True)
    p.add_argument("--metrics-out", type=str, required=True)
    p.add_argument("--artifacts-out", type=str, required=True)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    stdout_file = Path(args.stdout_file).expanduser().resolve()
    stderr_file = Path(args.stderr_file).expanduser().resolve()
    metrics_out = Path(args.metrics_out).expanduser().resolve()
    artifacts_out = Path(args.artifacts_out).expanduser().resolve()

    stdout_text = _read_text(stdout_file)
    stderr_text = _read_text(stderr_file)
    metrics, artifacts = parse_result(
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        mdriver_returncode=int(args.mdriver_returncode),
    )
    artifacts["mdriver_stdout_file"] = str(stdout_file)
    artifacts["mdriver_stderr_file"] = str(stderr_file)

    _write_json(metrics_out, metrics)
    _write_json(artifacts_out, artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
