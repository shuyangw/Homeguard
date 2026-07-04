"""Asness-style futures value (5yr-to-1yr reversal) strategy.

Signal: value_raw = -(log(P).shift(252) - log(P).shift(1260)), i.e. the negative
of the return earned from t-5yr to t-1yr (the most recent 1yr is skipped to avoid
momentum contamination). Assets that rose over that long window are treated as
"expensive" -> shorted; assets that fell are "cheap" -> bought. Both shifts (252,
1260) reference only past prices relative to t, so the signal is causal.

The raw signal is normalized by the expected vol of a 1008-trading-day return
(1260 - 252 = 1008, the span of the skip-return window) so forecasts are
comparable in scale across instruments, then scaled by a FIXED doctrine constant
and clipped to +/-cap, mirroring the Carver forecast-scalar convention used in
futures_carry_strategy.py / carver_momentum_strategy.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.volatility import close_to_close_rv

_SKIP_DAYS = 252     # ~1yr: length of the recent window excluded from the signal
_LOOKBACK_DAYS = 1260  # ~5yr: total history window used
_WINDOW_DAYS = _LOOKBACK_DAYS - _SKIP_DAYS  # 1008 trading days spanned by skip_return
_VOL_WINDOW = 25

# Fixed doctrine constant (NOT tuned/swept): chosen so the average absolute
# forecast lands near Carver's ~10 convention, calibrated once on synthetic
# futures-like price paths (mean abs normalized signal ~1-2.3 depending on
# drift/vol assumptions -> implied scalar ~4.4-9.6). 7.0 is a fixed midpoint;
# since Sharpe/correlation (the metrics this experiment is judged on) are
# scale-invariant, precision here is not critical.
_VALUE_SCALAR = 7.0


class FuturesValueStrategy:
    def __init__(self, universe, cap: float = 20.0, **params):
        self.universe = list(universe)
        self.cap = float(cap)

    def _forecast_root(self, close: pd.Series) -> pd.Series:
        lp = np.log(close.astype(float))
        skip_return = lp.shift(_SKIP_DAYS) - lp.shift(_LOOKBACK_DAYS)
        value_raw = -skip_return

        rets = close.pct_change(fill_method=None)
        daily_vol = close_to_close_rv(rets, _VOL_WINDOW, annualization_factor=1)
        window_vol = daily_vol * np.sqrt(_WINDOW_DAYS)

        normalized = value_raw / window_vol.replace(0, np.nan)
        forecast = (normalized * _VALUE_SCALAR).clip(-self.cap, self.cap)
        return forecast

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, pd.Series] = {}
        for root in self.universe:
            if root not in close_panel.columns:
                out[root] = pd.Series(np.nan, index=close_panel.index)
                continue
            out[root] = self._forecast_root(close_panel[root])
        return pd.DataFrame(out).reindex(columns=self.universe)
