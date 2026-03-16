"""Compatibility helpers for importing Summit in a modern Python environment."""

from __future__ import annotations

import warnings


def apply_summit_compat(silence_warnings: bool = True) -> None:
    """Patch a few sklearn compatibility breaks needed by Summit 0.8.x."""
    if silence_warnings:
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="skorch")

    import sklearn.utils.fixes as fixes
    import sklearn.utils.validation as validation

    if not hasattr(validation, "_check_fit_params"):

        def _check_fit_params(X, fit_params, indices=None):  # noqa: ANN001, ARG001
            return {} if fit_params is None else fit_params

        validation._check_fit_params = _check_fit_params

    if not hasattr(fixes, "delayed"):
        from joblib import delayed

        fixes.delayed = delayed
