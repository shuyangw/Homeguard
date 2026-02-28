"""
Tests for backtesting v2 framework.

Validates:
- Module imports
- BaseStrategyV2 interface
- PortfolioV2 state tracking
- Trade enrichment
- Numba simulation
"""

import pytest
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, Any

from src.backtesting_v2 import (
    BaseStrategyV2,
    BacktestEngineV2,
    PortfolioV2,
    from_signals_v2,
    TradeLoggerV2,
    TradeType,
    ExitReason,
)
from src.backtesting.utils.risk_config import RiskConfig


class SimpleCrossoverStrategy(BaseStrategyV2):
    """Test strategy for v2 framework."""

    def __init__(self, short_period: int = 5, long_period: int = 20):
        super().__init__(name="SimpleCrossover")
        self.short_period = short_period
        self.long_period = long_period
        self._context_cache = {}

    def generate_signals(
        self, data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        ma_short = data["close"].rolling(self.short_period).mean()
        ma_long = data["close"].rolling(self.long_period).mean()

        entries = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
        exits = (ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1))

        return entries.fillna(False), exits.fillna(False)

    def get_signals_context(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        ma_short = data["close"].rolling(self.short_period).mean()
        ma_long = data["close"].rolling(self.long_period).mean()

        for ts in data.index:
            self._context_cache[ts] = {
                "ma_short": ma_short.loc[ts] if ts in ma_short.index else None,
                "ma_long": ma_long.loc[ts] if ts in ma_long.index else None,
            }

        return self._context_cache

    def enrich_trades(
        self,
        trades_df: pd.DataFrame,
        price_data: pd.DataFrame,
        signals_context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        if trades_df.empty or not signals_context:
            return trades_df

        trades_df["ma_short"] = trades_df["timestamp"].apply(
            lambda ts: signals_context.get(ts, {}).get("ma_short")
        )
        trades_df["ma_long"] = trades_df["timestamp"].apply(
            lambda ts: signals_context.get(ts, {}).get("ma_long")
        )

        return trades_df


@pytest.fixture
def sample_data():
    """Create sample OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01 09:30", periods=1000, freq="1min")
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.1)

    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.1,
            "low": prices - 0.1,
            "close": prices,
            "volume": np.random.randint(1000, 10000, 1000),
        },
        index=dates,
    )


class TestBaseStrategyV2:
    """Tests for BaseStrategyV2 interface."""

    def test_strategy_instantiation(self):
        strategy = SimpleCrossoverStrategy()
        assert strategy.name == "SimpleCrossover"

    def test_generate_signals(self, sample_data):
        strategy = SimpleCrossoverStrategy()
        entries, exits = strategy.generate_signals(sample_data)

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(entries) == len(sample_data)
        assert len(exits) == len(sample_data)
        assert entries.dtype == bool
        assert exits.dtype == bool

    def test_get_signals_context(self, sample_data):
        strategy = SimpleCrossoverStrategy()
        context = strategy.get_signals_context(sample_data)

        assert context is not None
        assert len(context) == len(sample_data)
        assert "ma_short" in list(context.values())[0]
        assert "ma_long" in list(context.values())[0]


class TestPortfolioV2:
    """Tests for PortfolioV2 state tracking."""

    def test_portfolio_creation(self, sample_data):
        strategy = SimpleCrossoverStrategy()
        entries, exits = strategy.generate_signals(sample_data)

        portfolio = from_signals_v2(
            close=sample_data["close"],
            entries=entries,
            exits=exits,
            init_cash=100000,
            fees=0.001,
            slippage=0.0005,
            market_hours_only=False,
        )

        assert isinstance(portfolio, PortfolioV2)
        assert portfolio.equity_curve is not None
        assert len(portfolio.equity_curve) == len(sample_data)

    def test_portfolio_state_tracking(self, sample_data):
        strategy = SimpleCrossoverStrategy()
        entries, exits = strategy.generate_signals(sample_data)

        portfolio = from_signals_v2(
            close=sample_data["close"],
            entries=entries,
            exits=exits,
            init_cash=100000,
            fees=0.001,
            slippage=0.0005,
            market_hours_only=False,
            track_state=True,
        )

        state = portfolio.portfolio_state
        assert not state.empty
        assert len(state) == len(sample_data)
        assert "cash" in state.columns
        assert "position_shares" in state.columns
        assert "total_equity" in state.columns
        assert "exposure_pct" in state.columns

    def test_trade_enrichment(self, sample_data):
        strategy = SimpleCrossoverStrategy()

        # Get signals and context
        entries, exits = strategy.generate_signals(sample_data)
        context = strategy.get_signals_context(sample_data)

        portfolio = from_signals_v2(
            close=sample_data["close"],
            entries=entries,
            exits=exits,
            init_cash=100000,
            fees=0.001,
            slippage=0.0005,
            market_hours_only=False,
        )

        # Set context and finalize
        portfolio.set_signals_context(context)
        portfolio.finalize_trades(strategy)

        trades = portfolio.trades_df
        if not trades.empty:
            assert "ma_short" in trades.columns
            assert "ma_long" in trades.columns


class TestTradeSchema:
    """Tests for trade schema constants."""

    def test_trade_types(self):
        assert TradeType.ENTRY == 0
        assert TradeType.EXIT == 1
        assert TradeType.SHORT_ENTRY == 2
        assert TradeType.COVER_SHORT == 3

    def test_exit_reasons(self):
        assert ExitReason.SIGNAL == 0
        assert ExitReason.STOP_LOSS == 1
        assert ExitReason.TIME_STOP == 2
        assert ExitReason.PROFIT_TARGET == 3


class TestNumbaSimulation:
    """Tests for Numba-accelerated simulation."""

    def test_numba_parity(self, sample_data):
        """Verify Numba and Python simulations produce same results."""
        strategy = SimpleCrossoverStrategy()
        entries, exits = strategy.generate_signals(sample_data)

        # With Numba
        portfolio_numba = from_signals_v2(
            close=sample_data["close"],
            entries=entries,
            exits=exits,
            init_cash=100000,
            fees=0.001,
            slippage=0.0005,
            market_hours_only=False,
            use_numba=True,
        )

        # Without Numba
        portfolio_python = from_signals_v2(
            close=sample_data["close"],
            entries=entries,
            exits=exits,
            init_cash=100000,
            fees=0.001,
            slippage=0.0005,
            market_hours_only=False,
            use_numba=False,
        )

        # Compare final equity (allowing small floating point differences)
        numba_final = portfolio_numba.equity_curve.iloc[-1]
        python_final = portfolio_python.equity_curve.iloc[-1]

        assert abs(numba_final - python_final) < 1.0  # Within $1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
