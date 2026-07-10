"""CME equity-index session boundaries (ET) -> UTC, DST-aware.

Boundaries are fixed conventions for the SP-B session engine: RTH cash open/close,
the FOMC announcement time, and the NY-Fed overnight hour-slice. ET<->UTC uses
fx_clock.local_to_utc (America/New_York, DST-correct). 1-min futures bars are
UTC-timestamped, so a strategy maps a (date, ET time) to the UTC instant to look
up the bar."""
from __future__ import annotations

import datetime as dt
from datetime import date, time

import pandas as pd

from src.backtesting.sessions.fx_clock import local_to_utc

_ET = "America/New_York"

RTH_OPEN = time(9, 30)     # ET cash open
RTH_CLOSE = time(16, 0)    # ET cash close
FOMC_TIME = time(14, 0)    # ET FOMC statement release
SLICE_START = time(2, 0)   # ET, NY-Fed overnight drift window start (approx SR-917)
SLICE_END = time(5, 0)     # ET, hour-slice end


def et_to_utc(d: date, t: time) -> pd.Timestamp:
    return local_to_utc(_ET, dt.datetime.combine(d, t))
