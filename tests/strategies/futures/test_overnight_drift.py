from datetime import date
import pandas as pd
from src.strategies.advanced.overnight_drift_strategy import overnight_trades


def test_one_close_to_next_open_trade_per_trading_day():
    idx = pd.Index([date(2015, 1, 5), date(2015, 1, 6), date(2015, 1, 7)], name="date")
    bars = pd.DataFrame({"et_1600": [2000.0, 2010.0, 2020.0], "et_0930": [2005, 2015, 2025]}, index=idx)
    trades = overnight_trades({"ES": bars})
    # 3 dates -> 2 overnight trades (the last date has no next trading day)
    assert len(trades) == 2
    t0 = trades[0]
    assert (t0.root, t0.entry_date, t0.entry_col, t0.exit_date, t0.exit_col, t0.sign) == \
           ("ES", date(2015, 1, 5), "et_1600", date(2015, 1, 6), "et_0930", 1.0)


def test_skips_last_day_with_no_next():
    idx = pd.Index([date(2015, 1, 5)], name="date")
    bars = pd.DataFrame({"et_1600": [2000.0], "et_0930": [2005.0]}, index=idx)
    assert overnight_trades({"ES": bars}) == []
