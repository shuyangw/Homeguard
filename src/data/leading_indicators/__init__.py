"""Leading-indicator data acquirers for the WS-3d detector replacement.

Each submodule exposes `load_<indicator>(start, end) -> pd.DataFrame`.
The unified loader `load_leading_indicators(start, end)` returns a single
DataFrame with all indicators joined on date.

See docs/superpowers/specs/2026-05-25-ws3d-detector-replacement-design.md
for the canonical input set.
"""

from src.data.leading_indicators.vix_term import load_vix_term
from src.data.leading_indicators.fred_hy_oas import load_hy_oas
from src.data.leading_indicators.breadth import load_breadth
from src.data.leading_indicators.skew import load_skew
from src.data.leading_indicators.loader import load_leading_indicators

__all__ = [
    'load_vix_term',
    'load_hy_oas',
    'load_breadth',
    'load_skew',
    'load_leading_indicators',
]
