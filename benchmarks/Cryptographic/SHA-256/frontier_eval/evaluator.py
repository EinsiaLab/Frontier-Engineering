from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from evaluator_impl import evaluate as _evaluate
from spec import (
    CRYPTO_AES128_SPEC,
    CRYPTO_SHA256_SPEC,
    CRYPTO_SHA3_256_SPEC,
)

_SPEC_BY_BENCHMARK = {
    "AES-128": CRYPTO_AES128_SPEC,
    "SHA-256": CRYPTO_SHA256_SPEC,
    "SHA3-256": CRYPTO_SHA3_256_SPEC,
}


def _flag_from_env(name: str) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def evaluate(program_path: str, *, repo_root: Path | None = None) -> Any:
    benchmark_name = Path(__file__).resolve().parents[1].name
    spec = _SPEC_BY_BENCHMARK[benchmark_name]
    return _evaluate(
        program_path,
        repo_root=repo_root,
        spec=spec,
        include_pdf_reference=_flag_from_env(
            "FRONTIER_EVAL_UNIFIED_CRYPTO_INCLUDE_PDF_REFERENCE"
        ),
    )
