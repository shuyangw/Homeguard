from datetime import date
import pandas as pd
from src.strategies.advanced.prefomc_strategy import prefomc_trades


def test_prefomc_prev_trading_day_to_fomc_day():
    # cache dates include the FOMC day and the prior trading day
    idx = pd.Index([date(2015, 3, 17), date(2015, 3, 18)], name="date")  # FOMC 2015-03-18
    bars = pd.DataFrame({"et_1400": [2000.0, 2010.0]}, index=idx)
    trades = prefomc_trades({"ES": bars}, [date(2015, 3, 18)])
    assert len(trades) == 1
    t = trades[0]
    assert (t.entry_date, t.entry_col, t.exit_date, t.exit_col, t.sign) == \
           (date(2015, 3, 17), "et_1400", date(2015, 3, 18), "et_1400", 1.0)


def test_prefomc_skips_dates_not_in_cache():
    idx = pd.Index([date(2015, 3, 18)], name="date")  # no prior trading day in cache
    bars = pd.DataFrame({"et_1400": [2010.0]}, index=idx)
    assert prefomc_trades({"ES": bars}, [date(2015, 3, 18)]) == []
