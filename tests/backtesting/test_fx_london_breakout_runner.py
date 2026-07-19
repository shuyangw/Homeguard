import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "backtest_scripts"))

from run_fx_london_breakout_walkforward import build_daily_returns


def test_build_daily_returns_short_window_produces_series():
    s = build_daily_returns(["GBPUSD"], dt.date(2020, 6, 1), dt.date(2020, 6, 30))
    assert s is not None
    assert len(s) > 5
    assert s.index.is_monotonic_increasing
