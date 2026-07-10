import numpy as np
import pandas as pd
from src.backtesting.spreads.convergence import rolling_z, simulate_convergence, SpreadTrade


def test_enters_and_converges():
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    base = np.zeros(200)
    base[100:110] = 3.0  # a stretch that then reverts
    signal = pd.Series(base, index=idx)
    unit_return = signal.diff().fillna(0.0)  # spread level moves with signal
    trades, daily = simulate_convergence(signal, unit_return, cost_return=0.0,
                                         window=50, entry_z=2.0, max_hold=60)
    assert len(trades) >= 1
    t = trades[0]
    assert t.direction == -1  # positive stretch -> short the spread
    assert isinstance(t, SpreadTrade)


def test_structural_stop_is_tighter_on_short_side():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    # a stretch that keeps trending up (never converges) -> structural stop fires
    signal = pd.Series(np.concatenate([np.zeros(100), np.linspace(0, 10, 200)]), index=idx)
    unit_return = signal.diff().fillna(0.0)
    trades, _ = simulate_convergence(signal, unit_return, cost_return=0.0, window=50,
                                     entry_z=2.0, structural_z=4.0, structural_z_short=3.0,
                                     max_hold=999)
    # short-side trade must exit at the tighter |z|>3, not |z|>4
    short_trades = [t for t in trades if t.direction == -1]
    assert short_trades
    assert abs(short_trades[0].exit_z) < 4.0


def test_cost_reduces_trade_return():
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    base = np.zeros(200); base[100:108] = 3.0
    signal = pd.Series(base, index=idx)
    unit_return = signal.diff().fillna(0.0)
    t_free, _ = simulate_convergence(signal, unit_return, 0.0, window=50)
    t_cost, _ = simulate_convergence(signal, unit_return, 0.02, window=50)
    assert sum(t.ret for t in t_cost) < sum(t.ret for t in t_free)
