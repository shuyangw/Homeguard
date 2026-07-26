"""Tripwire for signals that silently evaluate to a constant.

The failure this exists to stop: the EM carry "seatbelt" trial ran under its
pre-registered name with a crash filter that was exactly 0.0 on every date. Its
four terms are JPY/CHF/AUDJPY/XAUUSD constructs, none of which exist in the EM7
universe, and each term falls back to a zero series when its input is absent.
The strategy silently degraded to plain carry and still produced a verdict.

A signal that never varies cannot carry information. Raising is correct: a
constant filter is a code defect, and a verdict computed over one is not a
verdict about the mechanism the pre-registration named.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import logger

_ATOL = 1e-12


class DegenerateSignalError(ValueError):
    """A declared signal is constant (or entirely missing) over the whole run."""


def _is_constant(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return True
    return bool(np.ptp(finite) <= _ATOL)


def constant_columns(frame: pd.DataFrame) -> list[str]:
    """Columns that never vary. Sorted, so callers get a stable message."""
    return sorted(c for c in frame.columns
                  if _is_constant(frame[c].to_numpy(dtype=float, na_value=np.nan)))


def assert_not_degenerate(signal: pd.Series | pd.DataFrame, name: str) -> None:
    """Raise if `signal` is constant everywhere; log the dead columns otherwise.

    A DataFrame is degenerate only when EVERY column is constant. One flat pair
    is a legitimate strategy state and must not halt a run, but it is worth
    surfacing, so it is logged.
    """
    if signal.empty:
        raise DegenerateSignalError(
            f"{name!r} is empty; a signal with no observations cannot be evaluated")

    if isinstance(signal, pd.Series):
        if _is_constant(signal.to_numpy(dtype=float, na_value=np.nan)):
            raise DegenerateSignalError(
                f"{name!r} is CONSTANT over all {len(signal)} observations "
                f"(value {signal.iloc[0]!r}). A signal that never varies carries no "
                "information -- check that its inputs exist in this universe.")
        return

    dead = constant_columns(signal)
    if len(dead) == len(signal.columns):
        raise DegenerateSignalError(
            f"{name!r} is CONSTANT in every one of its {len(dead)} columns. "
            "A signal that never varies carries no information -- check that its "
            "inputs exist in this universe.")
    if dead:
        logger.warning(f"[degenerate_signal] {name}: {len(dead)} of "
                       f"{len(signal.columns)} columns never vary: {', '.join(dead)}")
