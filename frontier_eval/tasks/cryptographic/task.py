from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from frontier_eval.tasks.base import Task

from .spec import (
    CRYPTO_AES128_SPEC,
    CRYPTO_SHA3_256_SPEC,
    CRYPTO_SHA256_SPEC,
    CryptographicSpec,
)


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


class _BaseCryptographicTask(Task):
    SPEC: CryptographicSpec

    def _include_pdf_reference(self) -> bool:
        task_cfg = getattr(self.cfg, "task", None)
        raw: Any = None
        if task_cfg is not None:
            try:
                raw = getattr(task_cfg, "include_pdf_reference", None)
            except Exception:
                raw = None
        if raw is None:
            raw = os.environ.get("FRONTIER_EVAL_CRYPTO_INCLUDE_PDF_REFERENCE", None)
        return _as_bool(raw, default=False)

    def initial_program_path(self) -> Path:
        benchmark_dir = self.SPEC.benchmark_dir(self.repo_root)
        return (benchmark_dir / "baseline" / self.SPEC.baseline_source).resolve()

    def evaluate_program(self, program_path: Path) -> Any:
        from .evaluator.python import evaluate
        return evaluate(
            str(program_path),
            repo_root=self.repo_root,
            spec=self.SPEC,
            include_pdf_reference=self._include_pdf_reference(),
        )


class CryptoAES128Task(_BaseCryptographicTask):
    NAME = "crypto_aes128"
    SPEC = CRYPTO_AES128_SPEC


class CryptoSHA256Task(_BaseCryptographicTask):
    NAME = "crypto_sha256"
    SPEC = CRYPTO_SHA256_SPEC


class CryptoSHA3_256Task(_BaseCryptographicTask):
    NAME = "crypto_sha3_256"
    SPEC = CRYPTO_SHA3_256_SPEC

