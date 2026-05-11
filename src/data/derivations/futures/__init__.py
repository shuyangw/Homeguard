"""Futures-specific derivation pipelines."""

from src.data.derivations.futures.sofr import derive_sofr
from src.data.derivations.futures.yields import get_treasury_yield


def compute_es_realized_vol(*args, **kwargs):
    """Deferred per docs/progress/20260509_VALIDATION_FRAMEWORK_DEFERMENTS.md."""
    raise NotImplementedError(
        "ES realized volatility derivation deferred. "
        "See docs/progress/20260509_VALIDATION_FRAMEWORK_DEFERMENTS.md."
    )


def compute_carry_glbx(*args, **kwargs):
    """Deferred per docs/progress/20260509_VALIDATION_FRAMEWORK_DEFERMENTS.md."""
    raise NotImplementedError(
        "Per-asset-class carry computation deferred. "
        "See docs/progress/20260509_VALIDATION_FRAMEWORK_DEFERMENTS.md."
    )


__all__ = [
    "derive_sofr",
    "get_treasury_yield",
    "compute_es_realized_vol",
    "compute_carry_glbx",
]
