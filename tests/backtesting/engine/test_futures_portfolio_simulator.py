import pandas as pd

from src.backtesting.engine.futures_portfolio_simulator import FuturesPortfolioSimulator
from src.backtesting.margin.futures_margin import MarginModel


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
