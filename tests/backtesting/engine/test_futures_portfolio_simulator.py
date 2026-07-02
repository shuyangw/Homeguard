import pandas as pd

from src.backtesting.engine.futures_portfolio_simulator import FuturesPortfolioSimulator
from src.backtesting.margin.futures_margin import MarginModel
from src.backtesting.utils.position_sizer_futures import size_from_forecast


def _zero_cost(root, regular_hours=True, n_contracts=1):
    return 0.0


def test_mtm_pnl_known_scenario():
    # 1 MES (multiplier 5), price 5000 -> 5100 over one day = +$500 MTM
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    close = pd.DataFrame({"MES": [5000.0, 5100.0, 5100.0]}, index=dates)
    targets = pd.DataFrame({"MES": [1, 1, 1]}, index=dates)
    sim = FuturesPortfolioSimulator(initial_capital=25000, cost_fn=_zero_cost,
                                    margin_model=MarginModel(), rebalance="daily")
    res = sim.run(close, targets)
    # day2 MTM = 1 * 5 * (5100-5000) = 500; day3 = 0
    assert res.equity_curve.iloc[0] == 25000                 # day1: position opened, no prior close
    assert res.equity_curve.iloc[1] == 25000 + 500
    assert res.equity_curve.iloc[2] == 25000 + 500


def test_cost_charged_only_on_rebalance():
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    close = pd.DataFrame({"MES": [5000.0, 5000.0, 5000.0]}, index=dates)
    targets = pd.DataFrame({"MES": [1, 1, 2]}, index=dates)  # trade on day1 (0->1) and day3 (1->2)

    def cost(root, regular_hours=True, n_contracts=1):
        return 3.0 * n_contracts  # cost_fn returns TOTAL cost for n_contracts

    sim = FuturesPortfolioSimulator(25000, cost_fn=cost, margin_model=MarginModel(), rebalance="daily")
    res = sim.run(close, targets)
    # total cost = 3.0*1 (day1 open) + 3.0*1 (day3 add) = 6; no MTM (flat price)
    assert res.equity_curve.iloc[-1] == 25000 - 6


def test_bankruptcy_floor_flattens_and_stays_at_zero():
    # 100 MES (multiplier 5) held from day1; day2 price craters 5000 -> 4000
    # MTM = 100 * 5 * (4000-5000) = -500,000, far exceeding the 25,000 account.
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    close = pd.DataFrame({"MES": [5000.0, 4000.0, 3000.0, 3000.0]}, index=dates)
    targets = pd.DataFrame({"MES": [100, 100, 100, 100]}, index=dates)
    sim = FuturesPortfolioSimulator(initial_capital=25000, cost_fn=_zero_cost,
                                    margin_model=MarginModel(), rebalance="daily")
    res = sim.run(close, targets)

    assert (res.equity_curve >= 0).all()
    assert res.equity_curve.iloc[1] == 0.0  # blown up on day2's MTM
    assert res.equity_curve.iloc[2] == 0.0
    assert res.equity_curve.iloc[3] == 0.0
    assert res.trades.loc[res.trades["date"] == dates[1], "contracts"].empty  # no rebalance trade on blow-up day


def test_bankruptcy_floor_catches_cost_driven_negative_equity_same_day():
    # Tiny account, flat price (no MTM movement), but the rebalance cost on
    # day1 exceeds remaining cash -- equity must floor at 0.0 the SAME day,
    # not go negative.
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    close = pd.DataFrame({"MES": [5000.0, 5000.0, 5000.0]}, index=dates)
    targets = pd.DataFrame({"MES": [1, 1, 1]}, index=dates)

    def cost(root, regular_hours=True, n_contracts=1):
        return 50.0 * n_contracts  # exceeds the tiny initial_capital below

    sim = FuturesPortfolioSimulator(initial_capital=10.0, cost_fn=cost,
                                    margin_model=MarginModel(), rebalance="daily")
    res = sim.run(close, targets)

    assert (res.equity_curve >= 0).all()
    assert res.equity_curve.iloc[0] == 0.0  # cost debit crossed zero on day1
    assert res.equity_curve.iloc[1] == 0.0
    assert res.equity_curve.iloc[2] == 0.0
    # only day1's rebalance trade is recorded -- no further trading once blown
    assert res.trades.loc[res.trades["date"] == dates[1], "contracts"].empty


def test_run_sized_scales_contracts_with_live_equity():
    # Day1 sizes against initial_capital (25,000) at a high price (1,000).
    # Day2's price crater to 600 costs the day1 position ~12,000 in MTM, so the
    # day2 rebalance sizes against LOWER live equity (~13,000) -- fewer
    # contracts than an equivalent sizing at the (higher) initial equity for
    # the same forecast/price/vol.
    dates = pd.date_range("2024-01-02", periods=4, freq="D")
    close = pd.DataFrame({"MES": [1000.0, 600.0, 600.0, 600.0]}, index=dates)
    forecasts = pd.DataFrame({"MES": [10.0, 10.0, 10.0, 10.0]}, index=dates)
    daily_vol = pd.DataFrame({"MES": [0.01, 0.01, 0.01, 0.01]}, index=dates)
    vol_target = 0.20
    initial_capital = 25000

    sim = FuturesPortfolioSimulator(initial_capital=initial_capital, cost_fn=_zero_cost,
                                    margin_model=MarginModel(), rebalance="daily")
    res = sim.run_sized(close, forecasts, daily_vol, vol_target)

    day1_contracts = res.trades.loc[res.trades["date"] == dates[0], "contracts"].iloc[0]
    day2_trade = res.trades[res.trades["date"] == dates[1]]
    live_equity_day2 = res.equity_curve.iloc[1]

    assert live_equity_day2 < initial_capital  # drawdown reduced equity

    margin = MarginModel()
    raw_at_initial = size_from_forecast(10.0, initial_capital, vol_target, "MES",
                                         price=600.0, daily_vol=0.01)
    raw_at_live = size_from_forecast(10.0, live_equity_day2, vol_target, "MES",
                                      price=600.0, daily_vol=0.01)
    reference_at_initial = margin.check_and_scale({"MES": raw_at_initial}, equity=initial_capital)["MES"]
    reference_at_live = margin.check_and_scale({"MES": raw_at_live}, equity=live_equity_day2)["MES"]
    assert reference_at_live < reference_at_initial

    assert day1_contracts == size_from_forecast(10.0, initial_capital, vol_target, "MES",
                                                  price=1000.0, daily_vol=0.01)

    final_day2_position = day1_contracts + (day2_trade["contracts"].iloc[0] if not day2_trade.empty else 0)
    assert final_day2_position == reference_at_live
    assert final_day2_position < day1_contracts  # sized down as equity fell
