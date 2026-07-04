import collections
import pandas as pd
from src.backtesting.engine.futures_portfolio_simulator import FuturesPortfolioSimulator
from src.backtesting.margin.futures_margin import MarginModel
from src.backtesting.costs.futures import futures_round_trip_usd


def _panels():
    idx = pd.date_range("2022-01-03", periods=8, freq="B")
    close = pd.DataFrame({"GC": 1800.0, "CL": 80.0}, index=idx)
    fc = pd.DataFrame({"GC": 10.0, "CL": 10.0}, index=idx)    # equal forecast
    vol = pd.DataFrame({"GC": 0.01, "CL": 0.01}, index=idx)   # equal daily vol
    return close, fc, vol


def _sim():
    return FuturesPortfolioSimulator(initial_capital=1_000_000, cost_fn=futures_round_trip_usd,
                                     margin_model=MarginModel(), rebalance="weekly", cost_mult=1.0)


def _contracts_by_root(res):
    d = collections.defaultdict(int)
    for _, row in res.trades.iterrows():
        d[row["root"]] += abs(int(row["contracts"]))
    return d


def test_dict_divmult_scales_per_root():
    close, fc, vol = _panels()
    base = _contracts_by_root(_sim().run_sized(close, fc, vol, 0.20, div_mult=1.0))
    scaled = _contracts_by_root(_sim().run_sized(close, fc, vol, 0.20,
                                                 div_mult={"GC": 2.0, "CL": 0.5}))
    assert scaled["GC"] > base["GC"]   # GC up-weighted 2x -> more contracts
    assert scaled["CL"] < base["CL"]   # CL down-weighted 0.5x -> fewer


def test_scalar_divmult_still_works():
    close, fc, vol = _panels()
    res = _sim().run_sized(close, fc, vol, 0.20, div_mult=1.0)   # float path back-compat
    assert len(res.equity_curve) == 8
