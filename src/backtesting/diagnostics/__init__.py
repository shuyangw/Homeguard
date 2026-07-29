"""Data-property diagnostics.

This package measures PROPERTIES OF PRICE/STATE DATA (conditional means,
variances, frequencies, episode counts). It contains NO strategy simulation:
no positions, no fills, no equity curves, no performance metrics. Keeping the
boundary explicit is what keeps these measurements out of the project's
multiple-testing trial count.
"""

from src.backtesting.diagnostics.session_bars import (
    build_session_marks,
    infer_bar_label_convention,
    load_minute_bars,
    session_close_schedule,
)

__all__ = [
    "build_session_marks",
    "infer_bar_label_convention",
    "load_minute_bars",
    "session_close_schedule",
]
