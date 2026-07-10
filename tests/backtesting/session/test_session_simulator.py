from datetime import date
import numpy as np
import pandas as pd
from src.backtesting.session.session_simulator import SessionTrade, simulate_session_returns


def _bars(dates, et_1600, et_0930):
    return pd.DataFrame({"et_1600": et_1600, "et_0930": et_0930},
                        index=pd.Index(dates, name="date"))


def test_long_overnight_return_net_of_cost():
    bars = {"ES": _bars([date(2015, 1, 5), date(2015, 1, 6)],
                        et_1600=[2000.0, 2010.0], et_0930=[2005.0, 2015.0])}
    # long overnight: entry 16:00 on 1/5 (2000), exit 09:30 on 1/6 (2015) -> raw = +0.75%
    tr = SessionTrade("ES", date(2015, 1, 5), "et_1600", date(2015, 1, 6), "et_0930", 1.0)
    r = simulate_session_returns([tr], bars, cost_mult=1.0)
    raw = (2015.0 - 2000.0) / 2000.0
    # cost_ret = round_trip_usd(ES) / (2000 * 50). Assert net < raw and within a small band.
    assert r.loc[date(2015, 1, 6)] < raw
    assert abs(r.loc[date(2015, 1, 6)] - raw) < 0.001  # cost is small (sub-2bp)


def test_1_5x_cost_is_more_expensive():
    bars = {"ES": _bars([date(2015, 1, 5), date(2015, 1, 6)],
                        et_1600=[2000.0, 2010.0], et_0930=[2005.0, 2015.0])}
    tr = SessionTrade("ES", date(2015, 1, 5), "et_1600", date(2015, 1, 6), "et_0930", 1.0)
    r1 = simulate_session_returns([tr], bars, cost_mult=1.0).iloc[0]
    r15 = simulate_session_returns([tr], bars, cost_mult=1.5).iloc[0]
    assert r15 < r1  # 1.5x cost eats more


def test_nan_close_skips_trade():
    bars = {"ES": _bars([date(2015, 1, 5), date(2015, 1, 6)],
                        et_1600=[2000.0, 2010.0], et_0930=[2005.0, np.nan])}
    tr = SessionTrade("ES", date(2015, 1, 5), "et_1600", date(2015, 1, 6), "et_0930", 1.0)
    r = simulate_session_returns([tr], bars, cost_mult=1.0)
    assert len(r) == 0  # exit close NaN -> skipped
