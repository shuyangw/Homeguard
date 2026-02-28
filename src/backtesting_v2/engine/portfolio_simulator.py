"""
Portfolio simulator v2 with bar-by-bar state tracking and trade enrichment.

This module extends the V1 Portfolio class (via subclassing) with:
- Bar-by-bar portfolio state recording (via _on_bar_start hook)
- Trade enrichment via strategy adapter pattern
- Signals context caching for post-simulation analysis
- V2 Numba simulation with state tracking arrays

The simulation logic itself is inherited from V1's Portfolio class.
V2 only overrides hooks and adds state-tracking-specific methods.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from src.backtesting.engine.portfolio_simulator import Portfolio
from src.backtesting.utils.risk_config import RiskConfig
from src.backtesting_v2.engine.trade_schema import (
    BASE_TRADE_COLUMNS,
    TradeType,
    ExitReason,
    TRADE_TYPE_NAMES,
    EXIT_REASON_NAMES,
)

if TYPE_CHECKING:
    from src.backtesting_v2.base.strategy import BaseStrategyV2

# Try to import Numba simulation modules
try:
    from src.backtesting_v2.engine.numba_sim import (
        simulate_portfolio_numba_v2,
        NUMBA_V2_AVAILABLE,
    )
except ImportError:
    NUMBA_V2_AVAILABLE = False


class PortfolioV2(Portfolio):
    """
    Enhanced Portfolio with bar-by-bar state tracking and trade enrichment.

    Subclasses the V1 Portfolio class, inheriting all simulation logic.
    Adds V2-specific features via hook overrides:
    - portfolio_state: DataFrame of bar-by-bar state (cash, position, equity)
    - trades_df: DataFrame of trades with strategy-specific columns
    - signals_context: Cached strategy signals for enrichment

    Usage:
        portfolio = PortfolioV2(price, entries, exits, ...)
        portfolio.set_signals_context(context)
        portfolio.finalize_trades(strategy)

        # Access enriched data
        df = portfolio.trades_df
        state = portfolio.portfolio_state
    """

    def __init__(
        self,
        price: pd.Series,
        entries: pd.Series,
        exits: pd.Series,
        init_cash: float,
        fees: float,
        slippage: float,
        freq: str = "1min",
        market_hours_only: bool = True,
        risk_config: Optional[RiskConfig] = None,
        price_data: Optional[pd.DataFrame] = None,
        allow_shorts: bool = False,
        use_numba: bool = True,
        fractional_shares: bool = False,
        track_state: bool = False,
    ):
        """
        Initialize portfolio with optional state tracking.

        Args:
            price: Price series for backtesting
            entries: Boolean series for entry signals
            exits: Boolean series for exit signals
            init_cash: Initial cash
            fees: Trading fees as decimal (e.g., 0.001 = 0.1%)
            slippage: Slippage as decimal
            freq: Data frequency
            market_hours_only: If True, only trade during market hours
            risk_config: Risk management configuration
            price_data: Full OHLCV data for indicators
            allow_shorts: If True, enable short selling
            use_numba: If True, use Numba JIT for performance
            fractional_shares: If True, allow fractional shares (crypto)
            track_state: If True, track per-bar portfolio state (slower but
                        enables portfolio_state DataFrame). Default False.
        """
        # V2-specific state (must be set BEFORE super().__init__ runs simulation)
        self.track_state = track_state
        self._state_cash: Optional[np.ndarray] = None
        self._state_position: Optional[np.ndarray] = None
        self._state_position_price: Optional[np.ndarray] = None
        self._portfolio_state_df: Optional[pd.DataFrame] = None
        self._signals_context: Optional[Dict[str, Any]] = None
        self._trades_df: Optional[pd.DataFrame] = None

        # Pre-allocate state arrays before simulation runs (if tracking)
        if track_state:
            n_bars = len(price)
            self._state_cash = np.empty(n_bars, dtype=np.float64)
            self._state_position = np.empty(n_bars, dtype=np.float64)
            self._state_position_price = np.empty(n_bars, dtype=np.float64)

        # Call V1 constructor - this runs _simulate() or _simulate_fast()
        super().__init__(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=init_cash,
            fees=fees,
            slippage=slippage,
            freq=freq,
            market_hours_only=market_hours_only,
            risk_config=risk_config,
            price_data=price_data,
            allow_shorts=allow_shorts,
            use_numba=use_numba,
            fractional_shares=fractional_shares,
        )

    # === Hook overrides for state tracking ===

    def _on_bar_start(self, i: int, price: float, cash: float,
                     position: float, position_price: float):
        """Record per-bar state into pre-allocated arrays."""
        if self.track_state and self._state_cash is not None:
            self._state_cash[i] = cash
            self._state_position[i] = position
            self._state_position_price[i] = position_price

    # === V2-specific Numba handling ===

    def _can_use_numba(self) -> bool:
        """
        Check if Numba simulation can be used.

        Extends V1 check with V2 Numba availability for state tracking.
        """
        if not self.use_numba:
            return False

        # When tracking state, need V2 Numba
        if self.track_state:
            if not NUMBA_V2_AVAILABLE:
                return False
        else:
            # When not tracking, can use V1 Numba (via parent check)
            # or V2 Numba as fallback
            if not super()._can_use_numba() and not NUMBA_V2_AVAILABLE:
                return False
            # If V1 Numba is available, delegate to parent
            if super()._can_use_numba():
                return True

        # Shared constraints
        if self.risk_config.position_sizing_method not in ["fixed_percentage", "fixed_dollar"]:
            return False
        if self.risk_config.stop_loss_type == "atr":
            return False
        return True

    def _simulate_fast(self):
        """
        Fast simulation using Numba, optionally with state tracking.

        When track_state=True, uses V2 Numba with state arrays.
        When track_state=False, delegates to V1 Numba (faster).
        """
        if not self.track_state or not NUMBA_V2_AVAILABLE:
            # Use V1 Numba (faster, no state tracking)
            super()._simulate_fast()
            return

        # Use V2 Numba with state tracking
        prices = self.price.values.astype(np.float64)
        entries = self.entries.values.astype(np.bool_)
        exits = self.exits.values.astype(np.bool_)
        market_hours = self._compute_market_hours_mask()

        stop_loss_type = self.risk_config.stop_loss_type
        use_stop_loss = (
            self.risk_config.use_stop_loss
            and stop_loss_type in ["percentage", "profit_target"]
        )
        stop_loss_pct = self.risk_config.stop_loss_pct

        use_profit_target = (
            self.risk_config.use_stop_loss
            and stop_loss_type == "profit_target"
            and self.risk_config.take_profit_pct is not None
        )
        profit_target_pct = self.risk_config.take_profit_pct or 0.0

        use_time_stop = (
            self.risk_config.use_stop_loss
            and stop_loss_type == "time"
            and self.risk_config.max_holding_bars is not None
        )
        max_bars_in_position = self.risk_config.max_holding_bars or 99999999

        result = simulate_portfolio_numba_v2(
            prices=prices,
            entries=entries,
            exits=exits,
            market_hours=market_hours,
            init_cash=self.init_cash,
            fees=self.fees,
            slippage=self.slippage,
            position_size_pct=self.risk_config.position_size_pct,
            use_stop_loss=use_stop_loss,
            stop_loss_pct=stop_loss_pct,
            use_profit_target=use_profit_target,
            profit_target_pct=profit_target_pct,
            use_time_stop=use_time_stop,
            max_bars_in_position=max_bars_in_position,
            allow_shorts=self.allow_shorts,
            fractional_shares=self.fractional_shares,
        )

        # Unpack results (v2 includes state arrays)
        (
            equity,
            trade_bars,
            trade_types,
            trade_prices,
            trade_shares,
            trade_pnls,
            trade_pnl_pcts,
            trade_exit_reasons,
            trade_costs,
            trade_proceeds,
            trade_count,
            state_cash,
            state_position,
            state_position_price,
        ) = result

        # Store raw state arrays for lazy DataFrame construction
        self._state_cash = state_cash
        self._state_position = state_position
        self._state_position_price = state_position_price

        self.equity_curve = pd.Series(equity, index=self.price.index)

        # Convert trades using V2 trade schema (enums)
        self.trades = self._convert_numba_trades_v2(
            trade_bars, trade_types, trade_prices, trade_shares,
            trade_pnls, trade_pnl_pcts, trade_exit_reasons,
            trade_costs, trade_proceeds,
        )

    def _convert_numba_trades_v2(
        self,
        trade_bars: np.ndarray,
        trade_types: np.ndarray,
        trade_prices: np.ndarray,
        trade_shares: np.ndarray,
        trade_pnls: np.ndarray,
        trade_pnl_pcts: np.ndarray,
        trade_exit_reasons: np.ndarray,
        trade_costs: np.ndarray,
        trade_proceeds: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Convert Numba trade arrays to list of dicts using V2 trade schema."""
        trades = []
        timestamps = self.price.index

        for i in range(len(trade_bars)):
            bar_idx = trade_bars[i]
            trade_type = trade_types[i]
            timestamp = timestamps[bar_idx]

            trade = {
                "timestamp": timestamp,
                "type": TRADE_TYPE_NAMES.get(TradeType(trade_type), "unknown"),
                "price": trade_prices[i],
                "shares": trade_shares[i],
            }

            if trade_type in [TradeType.ENTRY, TradeType.SHORT_ENTRY]:
                if trade_costs[i] > 0:
                    trade["cost"] = trade_costs[i]
                if trade_type == TradeType.SHORT_ENTRY and trade_proceeds[i] > 0:
                    trade["proceeds"] = trade_proceeds[i]
            else:
                trade["pnl"] = trade_pnls[i]
                trade["pnl_pct"] = trade_pnl_pcts[i]
                if trade_proceeds[i] > 0:
                    trade["proceeds"] = trade_proceeds[i]
                if trade_costs[i] > 0:
                    trade["cost"] = trade_costs[i]

                exit_reason = trade_exit_reasons[i]
                if exit_reason >= 0:
                    trade["exit_reason"] = EXIT_REASON_NAMES.get(
                        ExitReason(exit_reason), "unknown"
                    )

            trades.append(trade)

        return trades

    # === V2-specific properties and methods ===

    @property
    def portfolio_state(self) -> pd.DataFrame:
        """
        Bar-by-bar portfolio state (lazy, vectorized construction).

        Returns DataFrame with columns:
        - timestamp: Bar timestamp
        - cash: Cash balance
        - position_shares: Current position size
        - position_entry_price: Entry price (NaN if flat)
        - position_value: Market value of position
        - total_equity: Total portfolio value
        - exposure_pct: Position value / total equity
        """
        if self._portfolio_state_df is not None:
            return self._portfolio_state_df

        if not self.track_state or self._state_cash is None:
            return pd.DataFrame()

        prices = self.price.values
        pos_val = np.abs(self._state_position) * prices
        equity = self.equity_curve.values

        self._portfolio_state_df = pd.DataFrame({
            "timestamp": self.price.index,
            "cash": self._state_cash,
            "position_shares": self._state_position,
            "position_entry_price": np.where(
                self._state_position != 0,
                self._state_position_price,
                np.nan
            ),
            "position_value": pos_val,
            "total_equity": equity,
            "exposure_pct": np.where(equity > 0, pos_val / equity, 0.0),
        })

        return self._portfolio_state_df

    @property
    def trades_df(self) -> pd.DataFrame:
        """
        Enriched trades DataFrame.

        Returns DataFrame with base columns plus any strategy-specific
        columns added by enrich_trades().
        """
        if self._trades_df is not None:
            return self._trades_df
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)

    def set_signals_context(self, context: Optional[Dict[str, Any]]):
        """
        Set signals context from strategy for trade enrichment.

        Args:
            context: Dictionary from strategy.get_signals_context()
        """
        self._signals_context = context

    def finalize_trades(self, strategy: Optional["BaseStrategyV2"] = None):
        """
        Convert trades to DataFrame and apply strategy enrichment.

        Args:
            strategy: Strategy instance with enrich_trades() method
        """
        if not self.trades:
            self._trades_df = pd.DataFrame()
            return

        self._trades_df = pd.DataFrame(self.trades)

        if strategy is not None and hasattr(strategy, "enrich_trades"):
            price_data = self.price_data if self.price_data is not None else pd.DataFrame({"close": self.price})
            self._trades_df = strategy.enrich_trades(
                self._trades_df,
                price_data,
                self._signals_context,
            )


def from_signals_v2(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    init_cash: float,
    fees: float,
    slippage: float = 0.0,
    freq: str = "1min",
    market_hours_only: bool = True,
    risk_config: Optional[RiskConfig] = None,
    price_data: Optional[pd.DataFrame] = None,
    allow_shorts: bool = False,
    use_numba: bool = True,
    fractional_shares: bool = False,
    track_state: bool = False,
    **kwargs,
) -> PortfolioV2:
    """
    Create a v2 portfolio from entry/exit signals.

    Args:
        close: Price series
        entries: Entry signals
        exits: Exit signals
        init_cash: Initial capital
        fees: Trading fees
        slippage: Slippage
        freq: Data frequency
        market_hours_only: If True, only trade during market hours
        risk_config: Risk management configuration
        price_data: Historical OHLC data
        allow_shorts: If True, enable short selling
        use_numba: If True, use Numba JIT for performance
        fractional_shares: If True, allow fractional shares
        track_state: If True, track per-bar portfolio state (slower).
                     Default False for performance.

    Returns:
        PortfolioV2 object (with state tracking if track_state=True)
    """
    return PortfolioV2(
        price=close,
        entries=entries,
        exits=exits,
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        freq=freq,
        market_hours_only=market_hours_only,
        risk_config=risk_config,
        price_data=price_data,
        allow_shorts=allow_shorts,
        use_numba=use_numba,
        fractional_shares=fractional_shares,
        track_state=track_state,
    )
