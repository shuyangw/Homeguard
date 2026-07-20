import datetime as dt
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "backtest_scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from run_fx_spread_backtest import run_spread_backtest


def test_audnzd_backtest_produces_daily_return_series():
    s = run_spread_backtest("AudNzdPairs", ["AUDUSD", "NZDUSD"],
                            dt.date(2015, 1, 1), dt.date(2018, 1, 1))
    assert s is not None and len(s) > 200
    assert s.index.is_monotonic_increasing
