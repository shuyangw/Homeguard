"""Regression tests for the load_daily_panel silent-basket-shrink bug.

An over-broad `except Exception` used to swallow a genuine bug (a roll-date
KeyError in aggregate_to_daily) and silently drop the whole root from the
window, quietly shrinking the basket. The except is now narrowed to
FileNotFoundError (the legitimate no-data-file case); everything else must
propagate.
"""
from datetime import date, datetime, timezone

import polars as pl
import pytest

from src.backtesting.data import futures_backtest_loader as fbl
from src.data.continuous_contract_loader import continuous_1min_dir


def _valid_daily(_root):
    return pl.DataFrame({
        "timestamp": [datetime(2020, 1, 2, tzinfo=timezone.utc),
                      datetime(2020, 1, 3, tzinfo=timezone.utc)],
        "close": [100.0, 101.0],
    })


def test_unexpected_error_propagates(monkeypatch):
    # A bug (e.g. the roll-date KeyError) must FAIL LOUD, not silently drop a root.
    def boom(self, root, method, start, end):
        raise KeyError("simulated roll-date bug")
    monkeypatch.setattr(fbl.ContinuousContractDataLoader, "aggregate_to_daily", boom)
    with pytest.raises(KeyError):
        fbl.load_daily_panel(["ES", "NQ"], date(2020, 1, 1), date(2020, 6, 30))


def test_missing_file_skipped_others_kept(monkeypatch):
    # Only FileNotFoundError (genuine no-data) is skipped; other roots survive.
    def maybe(self, root, method, start, end):
        if root == "GONE":
            raise FileNotFoundError("no parquet for GONE")
        return _valid_daily(root)
    monkeypatch.setattr(fbl.ContinuousContractDataLoader, "aggregate_to_daily", maybe)
    panel = fbl.load_daily_panel(["ES", "GONE"], date(2020, 1, 1), date(2020, 1, 31))
    assert sorted({r for r, _ in panel.columns}) == ["ES"]


@pytest.mark.skipif(not (continuous_1min_dir() / "symbol=TN").exists(),
                    reason="futures store not present")
def test_roll_date_gap_basket_preserved():
    # The real regression: TN/BZ used to KeyError on a roll-date gap over
    # 2010-2014 and vanish, collapsing the panel to just ES. All 3 must survive.
    panel = fbl.load_daily_panel(["TN", "BZ", "ES"], date(2010, 6, 7), date(2014, 6, 7))
    assert sorted({r for r, _ in panel.columns}) == ["BZ", "ES", "TN"]
