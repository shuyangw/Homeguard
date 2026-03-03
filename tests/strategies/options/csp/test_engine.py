"""Tests for CSPBacktestEngine - event-driven backtest with callback architecture."""

from datetime import date, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import pytest

from src.strategies.options.csp.engine import CSPBacktestEngine, CSPBacktestResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(**kwargs) -> CSPBacktestEngine:
    defaults = dict(
        initial_capital=100_000,
        max_csp_allocation=0.30,
        max_positions=5,
        profit_target_pct=0.50,
        loss_limit_multiple=2.0,
        min_dte_exit=5,
    )
    defaults.update(kwargs)
    return CSPBacktestEngine(**defaults)


def _make_chain_df(
    target_date: date,
    strike: float = 150.0,
    delta: float = -0.30,
    bid: float = 2.50,
    ask: float = 2.70,
    dte: int = 28,
    oi: int = 500,
    underlying: float = 160.0,
    iv: float = 0.30,
) -> pd.DataFrame:
    expiry = target_date + timedelta(days=dte)
    return pd.DataFrame([{
        "option_type": "P",
        "strike": strike,
        "expiry": expiry,
        "days_to_expiry": dte,
        "delta": delta,
        "bid": bid,
        "ask": ask,
        "mid_price": (bid + ask) / 2,
        "open_interest": oi,
        "underlying_price": underlying,
        "implied_vol": iv,
        "gamma": 0.02,
        "theta": -0.05,
        "vega": 0.15,
        "volume": 200,
    }])


def _constant_regime(regime: str, confidence: float = 0.8):
    """Return a get_regime callback that always returns the same regime."""
    def get_regime(d: date) -> Tuple[str, float]:
        return (regime, confidence)
    return get_regime


def _constant_crash(active: bool):
    """Return a get_crash_protection callback with a fixed value."""
    def get_crash(d: date) -> bool:
        return active
    return get_crash


def _constant_top_n(symbols: List[str]):
    """Return a get_top_n_symbols callback that always returns the same list."""
    def get_top_n(d: date) -> List[str]:
        return symbols
    return get_top_n


def _chain_lookup(chains: Dict[str, pd.DataFrame]):
    """Return a get_chain callback backed by a dict of symbol -> DataFrame.

    If a chain was built for a single reference date, it is returned
    for all query dates (simulating unchanging chain data).
    """
    def get_chain(symbol: str, d: date) -> pd.DataFrame:
        if symbol in chains:
            return chains[symbol]
        return pd.DataFrame()
    return get_chain


def _price_lookup(prices: Dict[str, float]):
    """Return a get_underlying_price callback backed by a dict."""
    def get_price(symbol: str, d: date) -> float:
        return prices.get(symbol, 0.0)
    return get_price


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCSPBacktestEngineInit:
    def test_initialization(self):
        engine = _make_engine()

        assert engine.cash == 100_000
        assert engine.positions == []
        assert engine.closed_trades == []
        assert engine.daily_snapshots == []
        assert engine.max_csp_allocation == 0.30
        assert engine.max_positions == 5
        assert engine.profit_target_pct == 0.50
        assert engine.loss_limit_multiple == 2.0
        assert engine.min_dte_exit == 5


class TestNoTradesInBearRegime:
    def test_no_trades_in_bear_regime(self):
        """In BEAR regime no positions should be opened; equity stays flat."""
        engine = _make_engine()
        trading_days = [date(2024, 6, 3), date(2024, 6, 4), date(2024, 6, 5)]

        chain = _make_chain_df(trading_days[0])
        result = engine.run(
            trading_days=trading_days,
            get_regime=_constant_regime("BEAR"),
            get_crash_protection=_constant_crash(False),
            get_top_n_symbols=_constant_top_n(["AAPL", "MSFT"]),
            get_chain=_chain_lookup({"AAPL": chain, "MSFT": chain}),
            get_underlying_price=_price_lookup({"AAPL": 160.0, "MSFT": 160.0}),
            options_symbols=["AAPL", "MSFT"],
        )

        assert isinstance(result, CSPBacktestResult)
        assert len(result.closed_trades) == 0
        assert len(result.daily_snapshots) == len(trading_days)
        assert result.equity_curve is not None
        assert len(result.equity_curve) == len(trading_days)
        # Equity should be unchanged (no trades, no fees)
        for snap in result.daily_snapshots:
            assert snap["equity"] == pytest.approx(100_000, abs=1e-2)


class TestOpensPositionInStrongBull:
    def test_opens_position_in_strong_bull(self):
        """In STRONG_BULL regime, the engine should open a CSP position."""
        engine = _make_engine()
        day1 = date(2024, 6, 3)
        trading_days = [day1, day1 + timedelta(days=1)]

        chain = _make_chain_df(day1, strike=150.0, bid=2.50, ask=2.70, dte=28)
        result = engine.run(
            trading_days=trading_days,
            get_regime=_constant_regime("STRONG_BULL"),
            get_crash_protection=_constant_crash(False),
            get_top_n_symbols=_constant_top_n(["AAPL"]),
            get_chain=_chain_lookup({"AAPL": chain}),
            get_underlying_price=_price_lookup({"AAPL": 160.0}),
            options_symbols=["AAPL"],
        )

        # At least one position should have been opened; since STRONG_BULL
        # persists and there is no exit trigger, position remains open through
        # both days and is closed at "backtest_end".
        assert len(result.closed_trades) >= 1
        assert result.closed_trades[-1].exit_reason == "backtest_end"
        assert len(result.daily_snapshots) == 2

        # Verify the trade captured correct metadata
        trade = result.closed_trades[0]
        assert trade.symbol == "AAPL"
        assert trade.strike == 150.0
        assert trade.regime_at_entry == "STRONG_BULL"


class TestRegimeChangeClosesPositions:
    def test_regime_change_closes_positions(self):
        """Position opened in STRONG_BULL is closed when regime flips to BEAR."""
        engine = _make_engine()
        day1 = date(2024, 6, 3)
        day2 = date(2024, 6, 4)

        chain = _make_chain_df(day1, strike=150.0, bid=2.50, ask=2.70, dte=28)

        regime_schedule = {day1: "STRONG_BULL", day2: "BEAR"}

        def get_regime(d: date) -> Tuple[str, float]:
            return (regime_schedule.get(d, "SIDEWAYS"), 0.8)

        result = engine.run(
            trading_days=[day1, day2],
            get_regime=get_regime,
            get_crash_protection=_constant_crash(False),
            get_top_n_symbols=_constant_top_n(["AAPL"]),
            get_chain=_chain_lookup({"AAPL": chain}),
            get_underlying_price=_price_lookup({"AAPL": 160.0}),
            options_symbols=["AAPL"],
        )

        # Position should be closed on day2 due to regime change, then
        # no remaining positions at backtest_end.
        regime_trades = [t for t in result.closed_trades if t.exit_reason == "regime_change"]
        assert len(regime_trades) >= 1
        assert regime_trades[0].regime_at_exit == "BEAR"
        assert regime_trades[0].exit_date == day2


class TestMaxPositionsRespected:
    def test_max_positions_respected(self):
        """With max_positions=2 and 3 candidates, only 2 should be opened."""
        engine = _make_engine(max_positions=2)
        day1 = date(2024, 6, 3)
        day2 = date(2024, 6, 4)

        chain_aapl = _make_chain_df(day1, strike=150.0, bid=2.50, ask=2.70, dte=28,
                                     underlying=160.0)
        chain_msft = _make_chain_df(day1, strike=300.0, bid=4.00, ask=4.40, dte=28,
                                     underlying=320.0)
        chain_goog = _make_chain_df(day1, strike=140.0, bid=3.00, ask=3.30, dte=28,
                                     underlying=155.0)

        result = engine.run(
            trading_days=[day1, day2],
            get_regime=_constant_regime("STRONG_BULL"),
            get_crash_protection=_constant_crash(False),
            get_top_n_symbols=_constant_top_n(["AAPL", "MSFT", "GOOG"]),
            get_chain=_chain_lookup({
                "AAPL": chain_aapl,
                "MSFT": chain_msft,
                "GOOG": chain_goog,
            }),
            get_underlying_price=_price_lookup({
                "AAPL": 160.0,
                "MSFT": 320.0,
                "GOOG": 155.0,
            }),
            options_symbols=["AAPL", "MSFT", "GOOG"],
        )

        # At backtest_end all remaining positions are closed.
        # Total unique symbols traded should be <= 2 on day 1.
        # Check the snapshots: max num_positions on any day should be <= 2.
        max_positions_held = max(s["num_positions"] for s in result.daily_snapshots)
        assert max_positions_held <= 2

        # Also, the total number of backtest_end trades should be <= 2
        end_trades = [t for t in result.closed_trades if t.exit_reason == "backtest_end"]
        assert len(end_trades) <= 2
