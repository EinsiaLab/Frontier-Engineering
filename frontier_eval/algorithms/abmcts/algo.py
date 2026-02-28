from __future__ import annotations

"""TreeQuest AB-MCTS algorithm adapter for Frontier Eval."""

import asyncio
import hashlib
import json
import math
import os
import random
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from frontier_eval.algorithms.base import Algorithm
from frontier_eval.tasks.base import Task


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _extract_metrics_and_artifacts(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if result is None:
        return {}, {}
    if isinstance(result, dict):
        return result, {}
    metrics = getattr(result, "metrics", None)
    artifacts = getattr(result, "artifacts", None)
    if isinstance(metrics, dict):
        return metrics, artifacts if isinstance(artifacts, dict) else {}
    raise TypeError(f"Unsupported evaluation result type: {type(result)}")


def _tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _strip_code_fences(text: str) -> str:
    raw = str(text or "").strip()
    if "```" not in raw:
        return raw

    best: str | None = None
    cursor = 0
    while True:
        start = raw.find("```", cursor)
        if start < 0:
            break
        end = raw.find("```", start + 3)
        if end < 0:
            break

        inner = raw[start + 3 : end]
        inner_lines = inner.splitlines()
        if inner_lines and inner_lines[0].strip().lower() in {"python", "py"}:
            inner = "\n".join(inner_lines[1:])
        inner = inner.strip()
        if inner and (best is None or len(inner) > len(best)):
            best = inner

        cursor = end + 3

    if best is not None:
        return best.strip()

    # Fallback: handle unbalanced fences like "```python\n...\n" (missing closing ```).
    cleaned_lines: list[str] = []
    for line in raw.splitlines():
        if line.strip().startswith("```"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _split_evolve_block(code: str) -> tuple[str, str, str] | None:
    start_marker = "# EVOLVE-BLOCK-START"
    end_marker = "# EVOLVE-BLOCK-END"
    start = code.find(start_marker)
    end = code.find(end_marker)
    if start < 0 or end < 0 or start >= end:
        return None

    start_line_end = code.find("\n", start)
    if start_line_end < 0:
        start_line_end = len(code)
    content_start = min(len(code), start_line_end + 1)
    content_end = end
    return code[:content_start], code[content_start:content_end], code[content_end:]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _url_join(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def _chat_completions_sync(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    retries: int,
    retry_delay_s: float,
) -> str:
    url = _url_join(api_base, "chat/completions")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_err: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=max(1.0, float(timeout_s))) as resp:
                raw = resp.read()
            parsed = json.loads(raw.decode("utf-8", errors="replace"))
            if not isinstance(parsed, dict):
                raise TypeError(f"Unexpected response type: {type(parsed)}")
            choices = parsed.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    msg = first.get("message")
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        return str(msg["content"])
                    if isinstance(first.get("text"), str):
                        return str(first["text"])
            raise ValueError("No completion choices in response")
        except Exception as e:
            last_err = e
            if attempt + 1 >= max(1, int(retries)):
                break
            time.sleep(max(0.0, float(retry_delay_s)))
            continue
    raise RuntimeError(f"LLM request failed: {last_err}") from last_err


def _signed_log1p(x: float) -> float:
    if x == 0.0:
        return 0.0
    return math.copysign(math.log1p(abs(float(x))), float(x))


def _sigmoid(x: float) -> float:
    z = float(x)
    if z >= 60.0:
        return 1.0
    if z <= -60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def _as_plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, DictConfig):
        plain = OmegaConf.to_container(value, resolve=True)
        if plain is None:
            return {}
        if isinstance(plain, dict):
            return dict(plain)
        raise TypeError(f"Expected a mapping, got {type(plain)}")
    raise TypeError(f"Expected a mapping, got {type(value)}")


@dataclass(frozen=True)
class _ActionSpec:
    name: str
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


def _parse_actions(value: Any) -> list[_ActionSpec]:
    if value is None:
        return [_ActionSpec(name="default")]

    if isinstance(value, str):
        text = value.strip()
        return [_ActionSpec(name=text or "default")]

    if isinstance(value, (DictConfig, ListConfig)):
        value = OmegaConf.to_container(value, resolve=True)

    if isinstance(value, list):
        out: list[_ActionSpec] = []
        for item in value:
            if isinstance(item, str):
                name = item.strip()
                if not name:
                    continue
                out.append(_ActionSpec(name=name))
                continue
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip()
                if not name:
                    continue
                out.append(
                    _ActionSpec(
                        name=name,
                        model=str(item["model"]).strip() if item.get("model") else None,
                        temperature=float(item["temperature"]) if item.get("temperature") is not None else None,
                        max_tokens=int(item["max_tokens"]) if item.get("max_tokens") is not None else None,
                    )
                )
                continue
        return out or [_ActionSpec(name="default")]

    if isinstance(value, dict):
        out = []
        for name, cfg in value.items():
            name = str(name).strip()
            if not name:
                continue
            if isinstance(cfg, dict):
                out.append(
                    _ActionSpec(
                        name=name,
                        model=str(cfg["model"]).strip() if cfg.get("model") else None,
                        temperature=float(cfg["temperature"]) if cfg.get("temperature") is not None else None,
                        max_tokens=int(cfg["max_tokens"]) if cfg.get("max_tokens") is not None else None,
                    )
                )
            else:
                out.append(_ActionSpec(name=name))
        return out or [_ActionSpec(name="default")]

    raise TypeError(f"`algorithm.actions` must be list/dict/str, got {type(value)}")


@dataclass(frozen=True)
class ProgramState:
    code: str
    metrics: dict[str, Any]
    artifacts: dict[str, Any]
    combined_score: float
    reward: float


class ABMCTSAlgorithm(Algorithm):
    """
    Frontier Eval adapter that drives TreeQuest's AB-MCTS implementations (ABMCTSA / ABMCTSM).

    TreeQuest requires node rewards in the [0, 1] range. This adapter transforms the
    task's `combined_score` into a bounded reward for AB-MCTS, while still selecting
    the best program by raw `combined_score`.
    """

    NAME = "abmcts"

    def __init__(self, cfg: DictConfig, repo_root: Path):
        super().__init__(cfg=cfg, repo_root=repo_root)

        treequest_src = (self.repo_root / "third_party" / "treequest" / "src").resolve()
        if treequest_src.is_dir():
            treequest_src_str = str(treequest_src)
            if treequest_src_str not in sys.path:
                sys.path.insert(0, treequest_src_str)

        try:
            import treequest as tq
            from treequest.algos.ab_mcts_a.prob_state import PriorConfig
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "TreeQuest is not importable.\n"
                "Install it from the local repo (recommended):\n"
                "  pip install -e third_party/treequest\n"
                "Or install from PyPI:\n"
                "  pip install treequest\n"
            ) from e

        self._tq = tq
        self._tq_PriorConfig = PriorConfig

    async def run(self, task: Task) -> None:
        algo_cfg = self.cfg.algorithm
        llm_cfg = self.cfg.llm

        output_dir = Path(str(self.cfg.run.output_dir)).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        ab_dir = (output_dir / "abmcts").resolve()
        tree_dir = ab_dir / "tree"
        best_dir = ab_dir / "best"
        baseline_dir = ab_dir / "baseline"
        ab_dir.mkdir(parents=True, exist_ok=True)
        tree_dir.mkdir(parents=True, exist_ok=True)
        best_dir.mkdir(parents=True, exist_ok=True)
        baseline_dir.mkdir(parents=True, exist_ok=True)

        iterations = int(getattr(algo_cfg, "iterations", 0) or 0)
        batch_size = int(getattr(algo_cfg, "batch_size", 1) or 1)
        if batch_size <= 0:
            raise ValueError("`algorithm.batch_size` must be >= 1")

        variant = str(getattr(algo_cfg, "variant", "a") or "a").strip().lower()
        if variant not in {"a", "m"}:
            raise ValueError("`algorithm.variant` must be one of: a, m")

        # Must be an int because some evaluators parse this env var via `int(...)`
        # (e.g. `predict_modality`). Using float here would stringify as "300.0"
        # and crash downstream.
        evaluator_timeout_s = int(getattr(algo_cfg, "evaluator_timeout_s", 300) or 300)
        max_code_length = int(getattr(algo_cfg, "max_code_length", 20000) or 20000)
        artifact_char_limit = int(getattr(algo_cfg, "artifact_char_limit", 12000) or 12000)
        seed = int(getattr(algo_cfg, "seed", 0) or 0)
        max_llm_attempts = int(getattr(algo_cfg, "max_llm_attempts", 6) or 6)

        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except Exception:
            pass

        task_cfg_view = OmegaConf.to_container(getattr(self.cfg, "task", None), resolve=True)
        task_cfg_payload: dict[str, Any] = task_cfg_view if isinstance(task_cfg_view, dict) else {}
        task_cfg_payload = dict(task_cfg_payload)
        task_cfg_payload.setdefault("name", task.NAME)
        os.environ["FRONTIER_EVAL_TASK_NAME"] = task.NAME
        os.environ["FRONTIER_EVAL_TASK_CFG_JSON"] = json.dumps(task_cfg_payload, ensure_ascii=False)
        os.environ["FRONTIER_EVAL_EVALUATOR_TIMEOUT_S"] = str(evaluator_timeout_s)
        os.environ.setdefault("FRONTIER_ENGINEERING_ROOT", str(self.repo_root))
        os.environ.setdefault("PYTHONUNBUFFERED", "1")

        constraints_text = ""
        if task.NAME == "unified":
            try:
                from frontier_eval.tasks.unified.spec import load_unified_task_spec

                spec = load_unified_task_spec(task_cfg=getattr(self.cfg, "task", None), repo_root=self.repo_root)
                if spec.constraints_text:
                    constraints_text = str(spec.constraints_text)
            except Exception:
                constraints_text = ""

        api_base = str(getattr(llm_cfg, "api_base", "") or "") or os.environ.get(
            "OPENAI_API_BASE", "https://api.openai.com/v1"
        )
        model_default = str(getattr(llm_cfg, "model", "") or "") or os.environ.get(
            "OPENAI_MODEL", "gpt-4o-mini"
        )
        api_key = str(getattr(llm_cfg, "api_key", "") or "") or os.environ.get("OPENAI_API_KEY", "") or ""

        if iterations > 0 and not api_key:
            raise RuntimeError(
                "Missing API key for AB-MCTS. Set `OPENAI_API_KEY` or `llm.api_key` "
                "when `algorithm.iterations > 0`."
            )

        llm_temperature_default = float(getattr(llm_cfg, "temperature", 0.7) or 0.7)
        llm_max_tokens_default = int(getattr(llm_cfg, "max_tokens", 4096) or 4096)
        llm_timeout_s = float(getattr(llm_cfg, "timeout", 60) or 60)
        llm_retries = int(getattr(llm_cfg, "retries", 3) or 3)
        llm_retry_delay_s = float(getattr(llm_cfg, "retry_delay", 5) or 5)

        prompt_cfg = getattr(algo_cfg, "prompt", None)
        sys_prompt_override = str(getattr(prompt_cfg, "system", "") or "").strip() if prompt_cfg is not None else ""
        root_prompt_override = str(getattr(prompt_cfg, "root", "") or "").strip() if prompt_cfg is not None else ""
        mutate_prompt_override = str(getattr(prompt_cfg, "mutate", "") or "").strip() if prompt_cfg is not None else ""

        sys_prompt_default = (
            "You are optimizing a Python program for an automated benchmark.\n"
            "Goal: maximize metrics['combined_score'] (higher is better) while keeping the "
            "program correct, fast, and self-contained.\n"
            "Return only plain Python code. Do not include explanations, markdown, or ``` code fences.\n"
        )
        sys_prompt = sys_prompt_override or sys_prompt_default

        reward_cfg = getattr(algo_cfg, "reward", None)
        reward_transform = str(getattr(reward_cfg, "transform", "signed_log1p_sigmoid") or "signed_log1p_sigmoid")
        reward_center_mode = str(getattr(reward_cfg, "center", "baseline") or "baseline").strip().lower()
        reward_scale = float(getattr(reward_cfg, "scale", 1.0) or 1.0)
        invalid_reward = float(getattr(reward_cfg, "invalid_reward", 0.0) or 0.0)
        if reward_scale <= 0:
            raise ValueError("`algorithm.reward.scale` must be > 0")

        actions = _parse_actions(getattr(algo_cfg, "actions", None))
        action_names = [a.name for a in actions]
        action_by_name = {a.name: a for a in actions}

        def _reward_from_metrics(metrics: dict[str, Any], *, baseline_score: float, baseline_valid: bool) -> tuple[float, float]:
            raw = _as_float(metrics.get("combined_score", metrics.get("score", 0.0))) or 0.0
            valid_raw = metrics.get("valid", None)
            valid = True
            if isinstance(valid_raw, (int, float)) and not isinstance(valid_raw, bool):
                valid = float(valid_raw) > 0.0
            if not valid:
                return float(raw), float(invalid_reward)

            if reward_transform == "clip_01":
                r = max(0.0, min(1.0, float(raw)))
                return float(raw), float(r)

            x = _signed_log1p(float(raw)) if reward_transform == "signed_log1p_sigmoid" else float(raw)
            if reward_center_mode == "baseline":
                center_raw = float(baseline_score) if baseline_valid else 0.0
                center_x = _signed_log1p(center_raw) if reward_transform == "signed_log1p_sigmoid" else center_raw
            elif reward_center_mode == "zero":
                center_x = 0.0
            else:
                try:
                    center_x = float(reward_center_mode)
                except Exception:
                    center_x = 0.0

            r = _sigmoid((x - center_x) / float(reward_scale))
            r = max(0.0, min(1.0, float(r)))
            return float(raw), float(r)

        async def _evaluate_program(code: str, *, out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
            out_dir.mkdir(parents=True, exist_ok=True)
            program_path = out_dir / "program.py"
            program_path.write_text(code, encoding="utf-8", errors="replace")
            try:
                raw = await asyncio.to_thread(task.evaluate_program, program_path)
                metrics, artifacts = _extract_metrics_and_artifacts(raw)
            except Exception as e:
                metrics = {"combined_score": 0.0, "valid": 0.0, "error": str(e)}
                artifacts = {"error_message": str(e)}
            (out_dir / "metrics.json").write_text(_safe_json(metrics), encoding="utf-8")
            if artifacts:
                (out_dir / "artifacts.json").write_text(_safe_json(artifacts), encoding="utf-8")
            return metrics, artifacts

        initial_path = task.initial_program_path()
        baseline_code = initial_path.read_text(encoding="utf-8", errors="replace")
        if len(baseline_code) > max_code_length:
            raise ValueError(f"Initial program exceeds algorithm.max_code_length ({max_code_length})")

        baseline_metrics, baseline_artifacts = await _evaluate_program(baseline_code, out_dir=baseline_dir)
        baseline_valid_raw = baseline_metrics.get("valid", None)
        baseline_valid = True
        if isinstance(baseline_valid_raw, (int, float)) and not isinstance(baseline_valid_raw, bool):
            baseline_valid = float(baseline_valid_raw) > 0.0
        baseline_score_for_center = (
            _as_float(baseline_metrics.get("combined_score", baseline_metrics.get("score", 0.0)))
            or 0.0
        )
        baseline_raw_score, baseline_reward = _reward_from_metrics(
            baseline_metrics,
            baseline_score=float(baseline_score_for_center),
            baseline_valid=baseline_valid,
        )

        best_state: ProgramState = ProgramState(
            code=baseline_code,
            metrics=baseline_metrics,
            artifacts=baseline_artifacts,
            combined_score=float(baseline_raw_score),
            reward=float(baseline_reward),
        )

        def _write_best(state: ProgramState) -> None:
            best_program_path = best_dir / "program.py"
            best_program_path.write_text(state.code, encoding="utf-8", errors="replace")
            info = {
                "metrics": state.metrics,
                "combined_score": float(state.combined_score),
                "reward": float(state.reward),
                "program_path": str(best_program_path),
            }
            (best_dir / "best_program_info.json").write_text(_safe_json(info), encoding="utf-8")

        _write_best(best_state)

        if iterations <= 0:
            print(f"Best score: {best_state.combined_score}")
            print(f"Saved: {ab_dir}")
            print(f"Saved: {best_dir / 'best_program_info.json'}")
            return

        tq_cfg = _as_plain_mapping(getattr(algo_cfg, "tq", None))
        tq_a_cfg = _as_plain_mapping(tq_cfg.get("a")) if isinstance(tq_cfg, dict) else {}
        tq_m_cfg = _as_plain_mapping(tq_cfg.get("m")) if isinstance(tq_cfg, dict) else {}

        if variant == "a":
            dist_type = str(tq_a_cfg.get("dist_type", "gaussian") or "gaussian")
            model_selection_strategy = str(
                tq_a_cfg.get("model_selection_strategy", "multiarm_bandit_thompson") or "multiarm_bandit_thompson"
            )
            reward_average_priors = tq_a_cfg.get("reward_average_priors", None)
            prior_cfg_raw = tq_a_cfg.get("prior_config", None)
            prior_config = None
            if isinstance(prior_cfg_raw, dict):
                prior_config = self._tq_PriorConfig(
                    dist_type=str(prior_cfg_raw.get("dist_type", dist_type) or dist_type),
                    prior=prior_cfg_raw.get("prior", None),
                )
            algo = self._tq.ABMCTSA(
                dist_type=dist_type,
                reward_average_priors=reward_average_priors,
                prior_config=prior_config,
                model_selection_strategy=model_selection_strategy,
            )
        else:
            try:
                from treequest.algos.ab_mcts_m.algo import ABMCTSM, ABMCTSMAdvancedConfig
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "ABMCTSM is not importable (TreeQuest optional deps missing).\n"
                    "Install with extras (local repo):\n"
                    "  pip install -e 'third_party/treequest[abmcts-m]'\n"
                    "Or disable it by using `algorithm.variant=a`."
                ) from e

            advanced_cfg_raw = _as_plain_mapping(tq_m_cfg.get("advanced"))
            advanced_cfg = None
            if advanced_cfg_raw:
                advanced_cfg = ABMCTSMAdvancedConfig(
                    validate_reward_range=bool(advanced_cfg_raw.get("validate_reward_range", True)),
                    prior_mu_alpha_sigma=float(advanced_cfg_raw.get("prior_mu_alpha_sigma", 0.2)),
                    prior_sigma_alpha_sigma=float(advanced_cfg_raw.get("prior_sigma_alpha_sigma", 0.2)),
                    prior_sigma_y_sigma=float(advanced_cfg_raw.get("prior_sigma_y_sigma", 0.3)),
                )

            algo = ABMCTSM(
                enable_pruning=bool(tq_m_cfg.get("enable_pruning", True)),
                reward_average_priors=tq_m_cfg.get("reward_average_priors", None),
                model_selection_strategy=str(
                    tq_m_cfg.get("model_selection_strategy", "multiarm_bandit_thompson") or "multiarm_bandit_thompson"
                ),
                min_subtree_size_for_pruning=int(tq_m_cfg.get("min_subtree_size_for_pruning", 4) or 4),
                same_score_proportion_threshold=float(tq_m_cfg.get("same_score_proportion_threshold", 0.75) or 0.75),
                max_process_workers=int(tq_m_cfg.get("max_process_workers", os.cpu_count() or 1) or (os.cpu_count() or 1)),
                _advanced_config=advanced_cfg,
            )

        search_state = algo.init_tree()

        trace_cfg = getattr(algo_cfg, "trace", None)
        trace_enabled = bool(getattr(trace_cfg, "enabled", True)) if trace_cfg is not None else True
        trace_path = ab_dir / "trace.jsonl"
        if trace_enabled:
            trace_path.write_text("", encoding="utf-8")
        trace_f = trace_path.open("a", encoding="utf-8") if trace_enabled else None

        code_hashes: set[str] = {_sha256_text(baseline_code)}

        async def _propose_code(
            *,
            parent: ProgramState | None,
            action: _ActionSpec,
            attempt: int,
            feedback: str | None = None,
        ) -> tuple[str, str, str]:
            base_code = baseline_code if parent is None else parent.code
            parent_metrics = baseline_metrics if parent is None else parent.metrics
            parent_artifacts = baseline_artifacts if parent is None else parent.artifacts

            artifacts_preview: dict[str, Any] = {}
            for k, v in (parent_artifacts or {}).items():
                if len(artifacts_preview) >= 12:
                    break
                if isinstance(v, str):
                    artifacts_preview[k] = _tail(v, max(2000, artifact_char_limit // 6))
                else:
                    artifacts_preview[k] = v

            block_parts = _split_evolve_block(base_code)
            is_root = parent is None
            user_prefix = f"Task: {task.NAME}\n"
            if constraints_text:
                user_prefix += f"Constraints:\n{constraints_text}\n\n"

            if is_root and root_prompt_override:
                user_prefix += root_prompt_override.strip() + "\n\n"
            if (not is_root) and mutate_prompt_override:
                user_prefix += mutate_prompt_override.strip() + "\n\n"

            if block_parts is not None:
                prefix, block, suffix = block_parts
                user_prompt = (
                    user_prefix
                    + "Parent evaluation metrics (JSON):\n"
                    + _safe_json(parent_metrics)
                    + "\n\nParent artifacts (truncated, JSON):\n"
                    + _safe_json(artifacts_preview)
                    + "\n\nUpdate ONLY the code between EVOLVE-BLOCK markers.\n"
                    + "Return the replacement block content ONLY (no markers).\n\n"
                    + "Current block:\n"
                    + block
                    + "\n"
                )
                expect_mode = "block"
            else:
                user_prompt = (
                    user_prefix
                    + "Parent evaluation metrics (JSON):\n"
                    + _safe_json(parent_metrics)
                    + "\n\nParent artifacts (truncated, JSON):\n"
                    + _safe_json(artifacts_preview)
                    + "\n\nReturn the full updated Python file content.\n\n"
                    + "Current program:\n"
                    + base_code
                    + "\n"
                )
                expect_mode = "full"

            if attempt > 1:
                user_prompt += "\nAvoid producing identical code.\n"
            if feedback:
                user_prompt += (
                    "\nPrevious attempt feedback (fix this):\n"
                    + _tail(str(feedback).strip(), 1200)
                    + "\n"
                )

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
            model = action.model or model_default
            temperature = action.temperature if action.temperature is not None else llm_temperature_default
            max_tokens = action.max_tokens if action.max_tokens is not None else llm_max_tokens_default

            raw = await asyncio.to_thread(
                _chat_completions_sync,
                api_base=api_base,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=llm_timeout_s,
                retries=llm_retries,
                retry_delay_s=llm_retry_delay_s,
            )
            cleaned = _strip_code_fences(raw)
            if expect_mode == "block" and block_parts is not None:
                prefix, _block, suffix = block_parts
                if "# EVOLVE-BLOCK-START" in cleaned and "# EVOLVE-BLOCK-END" in cleaned:
                    inner_parts = _split_evolve_block(cleaned)
                    if inner_parts is not None:
                        _, cleaned, _ = inner_parts
                code = prefix + cleaned.rstrip() + "\n" + suffix.lstrip("\n")
            else:
                code = cleaned
            return user_prompt, raw, code

        def _write_node(
            *,
            node_id: int,
            parent_id: int,
            action: str,
            trial_id: str,
            program: ProgramState,
            user_prompt: str,
            llm_raw: str,
        ) -> None:
            node_dir = tree_dir / f"node_{node_id:06d}"
            node_dir.mkdir(parents=True, exist_ok=True)
            (node_dir / "program.py").write_text(program.code, encoding="utf-8", errors="replace")
            (node_dir / "metrics.json").write_text(_safe_json(program.metrics), encoding="utf-8")
            if program.artifacts:
                (node_dir / "artifacts.json").write_text(_safe_json(program.artifacts), encoding="utf-8")
            (node_dir / "meta.json").write_text(
                _safe_json(
                    {
                        "node_id": int(node_id),
                        "parent_id": int(parent_id),
                        "action": str(action),
                        "trial_id": str(trial_id),
                        "combined_score": float(program.combined_score),
                        "reward": float(program.reward),
                    }
                ),
                encoding="utf-8",
            )
            (node_dir / "llm_system.txt").write_text(sys_prompt, encoding="utf-8", errors="replace")
            (node_dir / "llm_user.txt").write_text(user_prompt, encoding="utf-8", errors="replace")
            (node_dir / "llm_raw.txt").write_text(llm_raw, encoding="utf-8", errors="replace")

        total_reflected = 0
        generation_idx = 0
        try:
            while generation_idx < iterations:
                this_batch = min(batch_size, iterations - generation_idx)
                search_state, trials = algo.ask_batch(
                    search_state, batch_size=this_batch, actions=action_names
                )
                for trial in trials:
                    generation_idx += 1
                    parent_node = search_state.tree.get_node(trial.node_to_expand)
                    parent_state = parent_node.state
                    action_spec = action_by_name.get(trial.action)
                    if action_spec is None:
                        action_spec = actions[0]

                    node_id = int(search_state.tree.size - 1)
                    parent_id = int(parent_node.expand_idx)

                    user_prompt = ""
                    llm_raw = ""
                    code = ""
                    feedback: str | None = None
                    for attempt in range(1, max(1, max_llm_attempts) + 1):
                        user_prompt, llm_raw, code = await _propose_code(
                            parent=parent_state,
                            action=action_spec,
                            attempt=attempt,
                            feedback=feedback,
                        )
                        feedback = None
                        if not code.strip():
                            feedback = "Empty output. Return valid Python code only."
                            continue
                        if len(code) > max_code_length:
                            feedback = (
                                f"Output too long ({len(code)} chars). Keep it under {max_code_length} chars."
                            )
                            continue
                        try:
                            compile(code, "<abmcts_candidate>", "exec")
                        except SyntaxError as e:
                            loc = f"line {e.lineno}" if e.lineno is not None else "unknown line"
                            msg = str(e.msg) if getattr(e, "msg", None) else str(e)
                            feedback = (
                                f"Python syntax error ({loc}): {msg}. "
                                "Return only valid Python code (no markdown fences)."
                            )
                            continue
                        h = _sha256_text(code)
                        if h in code_hashes and attempt < max(1, max_llm_attempts):
                            feedback = "Output was identical to a previous program. Produce a different variant."
                            continue
                        break

                    if not code.strip():
                        code = baseline_code

                    metrics, artifacts = await _evaluate_program(code, out_dir=tree_dir / f"node_{node_id:06d}")
                    raw_score, reward = _reward_from_metrics(
                        metrics, baseline_score=float(baseline_raw_score), baseline_valid=baseline_valid
                    )
                    program_state = ProgramState(
                        code=code,
                        metrics=metrics,
                        artifacts=artifacts,
                        combined_score=float(raw_score),
                        reward=float(reward),
                    )

                    before_size = int(search_state.tree.size)
                    search_state = algo.tell(search_state, trial.trial_id, (program_state, float(reward)))
                    if int(search_state.tree.size) == before_size:
                        continue

                    total_reflected += 1
                    code_hashes.add(_sha256_text(code))

                    _write_node(
                        node_id=node_id,
                        parent_id=parent_id,
                        action=trial.action,
                        trial_id=trial.trial_id,
                        program=program_state,
                        user_prompt=user_prompt,
                        llm_raw=llm_raw,
                    )

                    if program_state.combined_score > best_state.combined_score:
                        best_state = program_state
                        _write_best(best_state)

                    if trace_f is not None:
                        trace_f.write(
                            json.dumps(
                                {
                                    "step": int(generation_idx),
                                    "reflected": int(total_reflected),
                                    "node_id": int(node_id),
                                    "parent_id": int(parent_id),
                                    "action": str(trial.action),
                                    "combined_score": float(program_state.combined_score),
                                    "reward": float(program_state.reward),
                                    "best_combined_score": float(best_state.combined_score),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        trace_f.flush()
        finally:
            if trace_f is not None:
                trace_f.close()

        render_cfg = getattr(algo_cfg, "render", None)
        render_enabled = bool(getattr(render_cfg, "enabled", False)) if render_cfg is not None else False
        render_format = str(getattr(render_cfg, "format", "json")) if render_cfg is not None else "json"
        if render_enabled:
            try:
                def _fmt(state: ProgramState) -> str:
                    return json.dumps(
                        {
                            "combined_score": state.combined_score,
                            "reward": state.reward,
                            "code_sha256": _sha256_text(state.code)[:12],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )

                self._tq.render(
                    search_state,
                    ab_dir / "treequest_tree",
                    format=render_format,
                    state_formatter=_fmt,
                )
            except Exception:
                pass

        print(f"Best score: {best_state.combined_score}")
        print(f"Saved: {ab_dir}")
        info_path = best_dir / "best_program_info.json"
        if info_path.is_file():
            print(f"Saved: {info_path}")
