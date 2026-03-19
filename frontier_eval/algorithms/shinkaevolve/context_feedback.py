from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


@dataclass(frozen=True)
class ContextBundle:
    metrics: dict[str, Any]
    artifacts: dict[str, Any]
    correct: bool
    primary_error: str
    text_feedback: str
    stdout_bridge: str
    stderr_bridge: str
    selected_keys: list[str]
    omitted_keys: list[str]


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(value)
    return str(value)


def truncate_middle(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    if limit <= 64:
        return text[:limit]
    keep = max(1, (limit - 32) // 2)
    omitted = len(text) - (2 * keep)
    return text[:keep] + f"\n[... truncated {omitted} chars ...]\n" + text[-keep:]


def primary_error_message(artifacts: dict[str, Any]) -> str:
    for key in (
        "error_message",
        "user_artifact::error_message",
        "failure_summary",
        "user_artifact::failure_summary",
    ):
        text = _clean_text(stringify(artifacts.get(key))).strip()
        if text:
            return text
    return ""


def build_context_bundle(
    metrics: dict[str, Any],
    artifacts: dict[str, Any],
    *,
    text_feedback_max_chars: int = 8_000,
    stdout_bridge_max_chars: int = 6_000,
    stderr_bridge_max_chars: int = 12_000,
) -> ContextBundle:
    metrics_copy = dict(metrics or {})
    artifacts_copy = dict(artifacts or {})

    primary_error = primary_error_message(artifacts_copy)
    correct = _derive_correct(metrics_copy, primary_error=primary_error)

    selected_keys: list[str] = []
    text_feedback = _build_text_feedback(
        metrics_copy,
        artifacts_copy,
        selected_keys=selected_keys,
        max_chars=text_feedback_max_chars,
    )
    stdout_bridge = _build_stdout_bridge(
        metrics_copy,
        artifacts_copy,
        max_chars=stdout_bridge_max_chars,
    )
    stderr_bridge = _build_stderr_bridge(
        metrics_copy,
        artifacts_copy,
        primary_error=primary_error,
        max_chars=stderr_bridge_max_chars,
    )

    omitted_keys = [
        key
        for key in artifacts_copy
        if _clean_text(stringify(artifacts_copy.get(key))).strip() and key not in set(selected_keys)
    ]

    return ContextBundle(
        metrics=metrics_copy,
        artifacts=artifacts_copy,
        correct=correct,
        primary_error=primary_error,
        text_feedback=text_feedback,
        stdout_bridge=stdout_bridge,
        stderr_bridge=stderr_bridge,
        selected_keys=selected_keys,
        omitted_keys=omitted_keys,
    )


def build_context_manifest(bundle: ContextBundle) -> dict[str, Any]:
    return {
        "correct": bool(bundle.correct),
        "primary_error": bundle.primary_error,
        "selected_keys": list(bundle.selected_keys),
        "omitted_keys": list(bundle.omitted_keys),
        "lengths": {
            "text_feedback": len(bundle.text_feedback),
            "stdout_bridge": len(bundle.stdout_bridge),
            "stderr_bridge": len(bundle.stderr_bridge),
        },
    }


def _derive_correct(metrics: dict[str, Any], *, primary_error: str) -> bool:
    valid = metrics.get("valid", None)
    if isinstance(valid, (int, float)) and not isinstance(valid, bool):
        correct = float(valid) > 0.0
    else:
        correct = True
    if primary_error:
        return False
    return correct


def _build_text_feedback(
    metrics: dict[str, Any],
    artifacts: dict[str, Any],
    *,
    selected_keys: list[str],
    max_chars: int,
) -> str:
    sections: list[str] = []

    existing = _clean_text(stringify(metrics.get("text_feedback"))).strip()
    if existing:
        sections.append(_render_text_block("Task-Provided Feedback", existing, limit=2200))

    outcome_lines: list[str] = []
    for key in (
        "combined_score",
        "valid",
        "runtime_s",
        "timeout",
        "benchmark_returncode",
        "make_returncode",
        "mdriver_returncode",
        "errors_count",
        "testcases_passed",
        "testcases_total",
        "score_100",
        "geom_mean_ns",
        "program_returncode",
    ):
        if key in metrics:
            value = _clean_text(stringify(metrics.get(key))).strip()
            if value:
                outcome_lines.append(f"{key}: {value}")
    if outcome_lines:
        sections.append(_render_text_block("Outcome", "\n".join(outcome_lines), limit=1200))

    selected_set: set[str] = set()
    used_texts: set[str] = set()

    for key in _select_diagnostic_keys(artifacts, limit=6):
        block = _render_artifact_block(artifacts, key, limit=2200, used_texts=used_texts)
        if block:
            sections.append(block)
            selected_keys.append(key)
            selected_set.add(key)

    evidence_keys = _select_evidence_keys(artifacts, limit=4, exclude=selected_set)
    if evidence_keys:
        evidence_blocks: list[str] = []
        for key in evidence_keys:
            block = _render_artifact_block(artifacts, key, limit=2000, used_texts=used_texts)
            if block:
                evidence_blocks.append(block)
                selected_keys.append(key)
                selected_set.add(key)
        if evidence_blocks:
            sections.append("\n\n".join(evidence_blocks))

    for key in ("constraints", "interface_contract", "task_spec", "task_spec_zh_cn", "agent_files"):
        if key in artifacts and key not in selected_set:
            block = _render_artifact_block(artifacts, key, limit=1800, used_texts=used_texts)
            if block:
                sections.append(block)
                selected_keys.append(key)
                selected_set.add(key)

    context_file_keys = _select_file_context_keys(artifacts, limit=4, exclude=selected_set)
    if context_file_keys:
        context_blocks: list[str] = []
        for key in context_file_keys:
            block = _render_artifact_block(
                artifacts,
                key,
                limit=4000 if _artifact_relpath(key).lower() == "runtime/problem.py" else 1800,
                used_texts=used_texts,
            )
            if block:
                context_blocks.append(block)
                selected_keys.append(key)
                selected_set.add(key)
        if context_blocks:
            sections.append("\n\n".join(context_blocks))

    omitted = [
        key
        for key in artifacts
        if _clean_text(stringify(artifacts.get(key))).strip() and key not in selected_set
    ]
    if omitted:
        extra = "\n".join(omitted[:12])
        if len(omitted) > 12:
            extra += f"\n... and {len(omitted) - 12} more"
        sections.append(_render_text_block("Omitted Context", extra, limit=1200))

    feedback = "\n\n".join(section for section in sections if section).strip()
    return truncate_middle(feedback, max_chars) if feedback else ""


def _build_stderr_bridge(
    metrics: dict[str, Any],
    artifacts: dict[str, Any],
    *,
    primary_error: str,
    max_chars: int,
) -> str:
    sections: list[str] = []
    used_texts: set[str] = set()

    if primary_error:
        cleaned = _clean_text(primary_error).strip()
        if cleaned:
            used_texts.add(cleaned)
            sections.append(_render_text_block("Primary Error", cleaned, limit=1800))

    for key in _select_diagnostic_keys(artifacts, limit=4):
        block = _render_artifact_block(artifacts, key, limit=2500, used_texts=used_texts)
        if block:
            sections.append(block)

    stderr_keys = _select_stderr_keys(artifacts, limit=4)
    for key in stderr_keys:
        block = _render_artifact_block(artifacts, key, limit=3500, used_texts=used_texts)
        if block:
            sections.append(block)

    if not sections:
        fallback = _build_minimal_outcome(metrics)
        if fallback:
            sections.append(_render_text_block("Outcome", fallback, limit=1200))

    bridge = "\n\n".join(section for section in sections if section).strip()
    return truncate_middle(bridge, max_chars) if bridge else ""


def _build_stdout_bridge(
    metrics: dict[str, Any],
    artifacts: dict[str, Any],
    *,
    max_chars: int,
) -> str:
    sections: list[str] = []
    used_texts: set[str] = set()

    stdout_keys = _select_stdout_keys(artifacts, limit=4)
    for key in stdout_keys:
        block = _render_artifact_block(artifacts, key, limit=3000, used_texts=used_texts)
        if block:
            sections.append(block)

    for key in ("check", "score_line"):
        if key in artifacts:
            block = _render_artifact_block(artifacts, key, limit=800, used_texts=used_texts)
            if block:
                sections.append(block)

    if not sections:
        fallback = _build_minimal_outcome(metrics)
        if fallback:
            sections.append(_render_text_block("Outcome", fallback, limit=1200))

    bridge = "\n\n".join(section for section in sections if section).strip()
    return truncate_middle(bridge, max_chars) if bridge else ""


def _build_minimal_outcome(metrics: dict[str, Any]) -> str:
    lines: list[str] = []
    for key in ("combined_score", "valid", "runtime_s", "timeout"):
        if key in metrics:
            value = _clean_text(stringify(metrics.get(key))).strip()
            if value:
                lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _render_artifact_block(
    artifacts: dict[str, Any],
    key: str,
    *,
    limit: int,
    used_texts: set[str],
) -> str:
    text = _clean_text(stringify(artifacts.get(key))).strip()
    if not text or text in used_texts:
        return ""
    used_texts.add(text)
    return _render_text_block(_pretty_artifact_title(key), text, limit=limit)


def _render_text_block(title: str, body: str, *, limit: int) -> str:
    cleaned = _clean_text(body).strip()
    if not cleaned:
        return ""
    return f"## {title}\n{truncate_middle(cleaned, limit)}"


def _clean_text(text: str) -> str:
    if not text:
        return ""
    cleaned = _ANSI_ESCAPE_RE.sub("", text)
    cleaned = cleaned.replace("\x00", "")
    return cleaned


def _normalized_key(key: str) -> str:
    lowered = key.lower()
    if lowered.startswith("user_artifact::"):
        lowered = lowered[len("user_artifact::") :]
    return lowered


def _variant_base(key: str) -> str:
    normalized = _normalized_key(key)
    if normalized.endswith("_full"):
        return normalized[:-5]
    return normalized


def _pretty_artifact_title(key: str) -> str:
    key_wo_prefix = key.replace("user_artifact::", "")
    if key_wo_prefix.startswith("agent_file::"):
        return f"Agent File: {_artifact_relpath(key_wo_prefix)}"
    if key_wo_prefix.startswith("collected_artifact::"):
        return f"Collected Artifact: {_artifact_relpath(key_wo_prefix)}"
    return key_wo_prefix.replace("::", " / ").replace("_", " ").title()


def _artifact_relpath(key: str) -> str:
    return key.split("::", 1)[-1]


def _select_diagnostic_keys(artifacts: dict[str, Any], *, limit: int) -> list[str]:
    return _select_keys(
        artifacts,
        predicate=_is_diagnostic_key,
        sort_key=_diagnostic_sort_key,
        limit=limit,
    )


def _select_stderr_keys(artifacts: dict[str, Any], *, limit: int) -> list[str]:
    return _select_keys(
        artifacts,
        predicate=_is_stderr_like_key,
        sort_key=_stderr_sort_key,
        limit=limit,
    )


def _select_stdout_keys(artifacts: dict[str, Any], *, limit: int) -> list[str]:
    return _select_keys(
        artifacts,
        predicate=_is_stdout_like_key,
        sort_key=_stdout_sort_key,
        limit=limit,
    )


def _select_evidence_keys(
    artifacts: dict[str, Any],
    *,
    limit: int,
    exclude: set[str],
) -> list[str]:
    return _select_keys(
        artifacts,
        predicate=lambda key, _value: (_is_stderr_like_key(key, _value) or _is_stdout_like_key(key, _value))
        and key not in exclude,
        sort_key=_evidence_sort_key,
        limit=limit,
    )


def _select_file_context_keys(
    artifacts: dict[str, Any],
    *,
    limit: int,
    exclude: set[str],
) -> list[str]:
    return _select_keys(
        artifacts,
        predicate=lambda key, _value: (
            isinstance(key, str)
            and (key.startswith("agent_file::") or key.startswith("collected_artifact::"))
            and "::error" not in key
            and key not in exclude
        ),
        sort_key=_file_context_sort_key,
        limit=limit,
    )


def _select_keys(
    artifacts: dict[str, Any],
    *,
    predicate,
    sort_key,
    limit: int,
) -> list[str]:
    candidates: list[str] = []
    seen_bases: set[str] = set()
    for key in sorted(artifacts.keys(), key=sort_key):
        value = artifacts.get(key)
        if not isinstance(key, str) or not predicate(key, value):
            continue
        text = _clean_text(stringify(value)).strip()
        if not text:
            continue
        base = _variant_base(key)
        if base in seen_bases:
            continue
        seen_bases.add(base)
        candidates.append(key)
        if len(candidates) >= limit:
            break
    return candidates


def _is_diagnostic_key(key: str, value: Any) -> bool:
    normalized = _normalized_key(key)
    if normalized in {
        "error_message",
        "failure_summary",
        "readonly_violations",
        "metrics_json_error",
        "artifacts_json_error",
    }:
        return True
    if "traceback" in normalized:
        return True
    if normalized.endswith("::error"):
        return True
    return False


def _is_stderr_like_key(key: str, value: Any) -> bool:
    normalized = _normalized_key(key)
    return "stderr" in normalized or "traceback" in normalized


def _is_stdout_like_key(key: str, value: Any) -> bool:
    normalized = _normalized_key(key)
    return "stdout" in normalized or "log" in normalized


def _diagnostic_sort_key(key: str) -> tuple[int, int, str]:
    normalized = _normalized_key(key)
    order = {
        "error_message": 0,
        "failure_summary": 1,
        "traceback": 2,
        "readonly_violations": 3,
        "metrics_json_error": 4,
        "artifacts_json_error": 5,
    }
    score = order.get(normalized, 10 if normalized.endswith("::error") else 20)
    return score, 0 if normalized.endswith("_full") else 1, key


def _stderr_sort_key(key: str) -> tuple[int, int, str]:
    normalized = _normalized_key(key)
    score = 0 if "traceback" in normalized else 1
    return score, 0 if normalized.endswith("_full") else 1, key


def _stdout_sort_key(key: str) -> tuple[int, int, str]:
    normalized = _normalized_key(key)
    score = 0 if "stdout" in normalized else 1
    return score, 0 if normalized.endswith("_full") else 1, key


def _evidence_sort_key(key: str) -> tuple[int, int, str]:
    normalized = _normalized_key(key)
    if "stderr" in normalized:
        category = 0
    elif "traceback" in normalized:
        category = 1
    elif "stdout" in normalized:
        category = 2
    else:
        category = 3
    return category, 0 if normalized.endswith("_full") else 1, key


def _file_context_sort_key(key: str) -> tuple[int, int, int, str]:
    relpath = _artifact_relpath(key).lower()
    basename = Path(relpath).name

    if relpath.startswith("runtime/problem."):
        priority = 0
    elif relpath.startswith("baseline/solution."):
        priority = 1
    elif basename == "task.md":
        priority = 2
    elif basename == "task_zh-cn.md":
        priority = 3
    elif basename == "readme.md":
        priority = 4
    elif basename == "readme_zh-cn.md":
        priority = 5
    elif basename.endswith(".h"):
        priority = 6
    elif "interface" in basename or "config" in basename:
        priority = 7
    else:
        priority = 8

    return priority, len(relpath.split("/")), len(relpath), key
