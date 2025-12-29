"""
Portfolio simulator v2 with bar-by-bar state tracking and trade enrichment.

This module extends the original Portfolio class with:
- Bar-by-bar portfolio state recording
- Trade enrichment via strategy adapter pattern
- Signals context caching for post-simulation analysis
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from src.backtesting.engine.base_portfolio import BasePortfolio
from src.backtesting.utils.risk_config import RiskConfig
from src.backtesting.utils.position_sizer import (
    FixedPercentageSizer,
    VolatilityBasedSizer,
)
from src.backtesting.utils.risk_manager import RiskManager, Position
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

# Import original V1 Numba simulation for fast mode (no state tracking)
try:
    from src.backtesting.engine.numba_sim import (
        simulate_portfolio_numba,
        NUMBA_AVAILABLE as NUMBA_V1_AVAILABLE,
    )
except ImportError:
    NUMBA_V1_AVAILABLE = False


class PortfolioV2(BasePortfolio):
    """
    Enhanced Portfolio with bar-by-bar state tracking and trade enrichment.

    Extends the original Portfolio class with:
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
        super().__init__(
            init_cash=init_cash,
            fees=fees,
            slippage=slippage,
            freq=freq,
            market_hours_only=market_hours_only,
        )

        self.price = price
        self.entries = entries.astype(bool)
        self.exits = exits.astype(bool)
        self.risk_config = risk_config or RiskConfig.moderate()
        self.price_data = price_data
        self.allow_shorts = allow_shorts
        self.use_numba = use_numba
        self.fractional_shares = fractional_shares
        self.track_state = track_state
        self.borrow_cost = 0.0030

        self.trades: List[Dict[str, Any]] = []

        # V2 additions: state tracking (only populated if track_state=True)
        # Store raw arrays for lazy DataFrame construction
        self._state_cash: Optional[np.ndarray] = None
        self._state_position: Optional[np.ndarray] = None
        self._state_position_price: Optional[np.ndarray] = None
        self._portfolio_state_df: Optional[pd.DataFrame] = None  # Lazy-built
        self._signals_context: Optional[Dict[str, Any]] = None
        self._trades_df: Optional[pd.DataFrame] = None

        # Initialize position sizer
        self._init_position_sizer()
        self._init_risk_manager()

        # Run simulation
        if self._can_use_numba():
            self._simulate_fast()
        else:
            self._simulate()

    def _init_position_sizer(self):
        """Initialize position sizer based on risk config."""
        method = self.risk_config.position_sizing_method

        if method == "fixed_percentage":
            self.position_sizer = FixedPercentageSizer(
                position_pct=self.risk_config.position_size_pct
            )
        elif method == "volatility" and self.price_data is not None:
            self.position_sizer = VolatilityBasedSizer(
                risk_pct=self.risk_config.position_size_pct / 10,
                atr_multiplier=self.risk_config.atr_multiplier,
                atr_lookback=self.risk_config.atr_lookback,
            )
        else:
            self.position_sizer = FixedPercentageSizer(
                position_pct=self.risk_config.position_size_pct
            )

    def _init_risk_manager(self):
        """Initialize risk manager if needed."""
        if self.risk_config.use_stop_loss or self.risk_config.max_positions is not None:
            self.risk_manager = RiskManager(self.risk_config)
        else:
            self.risk_manager = None

    def _can_use_numba(self) -> bool:
        """Check if Numba simulation can be used."""
        if not self.use_numba:
            return False

        # Check if appropriate Numba version is available
        if self.track_state:
            # Need V2 Numba for state tracking
            if not NUMBA_V2_AVAILABLE:
                return False
        else:
            # Can use V1 Numba (faster) when not tracking state
            if not NUMBA_V1_AVAILABLE and not NUMBA_V2_AVAILABLE:
                return False

        if self.risk_config.position_sizing_method not in ["fixed_percentage", "fixed_dollar"]:
            return False
        if self.risk_config.stop_loss_type == "atr":
            return False
        return True

    def _compute_market_hours_mask(self) -> np.ndarray:
        """Pre-compute market hours mask (vectorized)."""
        if not self.market_hours_only:
            return np.ones(len(self.price), dtype=np.bool_)

        index = self.price.index
        if index.tz is None:
            eastern = index
        else:
            eastern = index.tz_convert("US/Eastern")

        is_weekday = eastern.weekday < 5
        times = pd.Series(eastern).dt.time
        is_after_open = times >= self.market_open
        is_before_close = times <= self.market_close
        is_market_hours = is_after_open & is_before_close

        return (is_weekday & is_market_hours.values).astype(np.bool_)

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
        # Return cached DataFrame if already built
        if self._portfolio_state_df is not None:
            return self._portfolio_state_df

        # No state tracking enabled or no arrays available
        if not self.track_state or self._state_cash is None:
            return pd.DataFrame()

        # Build DataFrame using vectorized operations (fast!)
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

    def _simulate_fast(self):
        """Fast simulation using Numba, optionally with state tracking."""
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

        # Choose V1 (fast, no state) or V2 (slower, with state) based on track_state
        if self.track_state and NUMBA_V2_AVAILABLE:
            # Use V2 Numba with state tracking
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
        else:
            # Use V1 Numba (faster, no state tracking)
            result = simulate_portfolio_numba(
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

            # Unpack V1 results (no state arrays)
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
            ) = result

            # No state tracking - leave _portfolio_state empty
            self._portfolio_state = []

        self.equity_curve = pd.Series(equity, index=self.price.index)

        # Convert trades
        self.trades = self._convert_numba_trades(
            trade_bars,
            trade_types,
            trade_prices,
            trade_shares,
            trade_pnls,
            trade_pnl_pcts,
            trade_exit_reasons,
            trade_costs,
            trade_proceeds,
        )

    def _convert_numba_trades(
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
        """Convert Numba trade arrays to list of dicts."""
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

    def _simulate(self):
        """Pure Python simulation with optional state tracking."""
        n_bars = len(self.price)
        cash = self.init_cash
        position = 0.0
        position_price = 0.0
        entry_timestamp = None
        entry_bar = 0
        equity = []
        trades = []
        bars_in_position = 0
        bar_index = 0

        # Pre-allocate state arrays if tracking enabled
        if self.track_state:
            state_cash = np.empty(n_bars, dtype=np.float64)
            state_position = np.empty(n_bars, dtype=np.float64)
            state_position_price = np.empty(n_bars, dtype=np.float64)

        for i, (timestamp, price) in enumerate(self.price.items()):
            bar_index += 1
            entry_signal = self.entries.get(timestamp, False)
            exit_signal = self.exits.get(timestamp, False)

            in_market_hours = self._is_market_hours(timestamp)

            # Calculate portfolio value
            if position > 0:
                portfolio_value = cash + (position * price)
            elif position < 0:
                short_liability = abs(position) * (position_price - price)
                portfolio_value = cash + short_liability
            else:
                portfolio_value = cash

            # Record state at start of bar (only if tracking enabled)
            if self.track_state:
                state_cash[i] = cash
                state_position[i] = position
                state_position_price[i] = position_price

            # Risk management checks
            if position != 0 and self.risk_manager is not None:
                bars_in_position += 1
                current_position = Position(
                    symbol="ASSET",
                    entry_price=position_price,
                    shares=int(position),
                    entry_bar=entry_bar,
                )

                idx = self.price.index.get_loc(timestamp)
                start_idx = max(0, idx - 20)
                recent_prices = self.price.iloc[start_idx : idx + 1]

                if self.price_data is not None:
                    price_df = self.price_data.iloc[start_idx : idx + 1]
                else:
                    price_df = pd.DataFrame({
                        "close": recent_prices,
                        "high": recent_prices,
                        "low": recent_prices,
                    })

                exit_signals = self.risk_manager.check_exits(
                    current_prices={"ASSET": price},
                    current_bar=bar_index,
                    price_data_dict={"ASSET": price_df},
                )

                should_exit = "ASSET" in exit_signals
                exit_reason = exit_signals.get("ASSET") if should_exit else None

                if should_exit and in_market_hours:
                    if position > 0:
                        slippage_adj = price * (1 - self.slippage)
                        proceeds = position * slippage_adj
                        fee = proceeds * self.fees
                        net_proceeds = proceeds - fee
                        cash += net_proceeds

                        pnl = net_proceeds - (position * position_price)
                        pnl_pct = (pnl / (position * position_price)) * 100 if position_price > 0 else 0

                        trades.append({
                            "timestamp": timestamp,
                            "type": "exit",
                            "exit_reason": exit_reason,
                            "price": price,
                            "shares": position,
                            "proceeds": net_proceeds,
                            "pnl": pnl,
                            "pnl_pct": pnl_pct,
                        })
                    else:
                        slippage_adj = price * (1 + self.slippage)
                        cost_to_cover = abs(position) * slippage_adj
                        fee = cost_to_cover * self.fees
                        total_cost = cost_to_cover + fee

                        proceeds_from_short = abs(position) * position_price
                        pnl = proceeds_from_short - cost_to_cover - fee
                        pnl_pct = (pnl / proceeds_from_short) * 100 if proceeds_from_short > 0 else 0
                        cash -= total_cost

                        trades.append({
                            "timestamp": timestamp,
                            "type": "cover_short",
                            "exit_reason": exit_reason,
                            "price": price,
                            "shares": abs(position),
                            "cost": total_cost,
                            "pnl": pnl,
                            "pnl_pct": pnl_pct,
                        })

                    if self.risk_manager is not None:
                        self.risk_manager.remove_position("ASSET")

                    position = 0
                    position_price = 0
                    entry_timestamp = None
                    entry_bar = 0
                    bars_in_position = 0

            # Entry signals
            if entry_signal and in_market_hours:
                if position < 0:
                    slippage_adj = price * (1 + self.slippage)
                    cost_to_cover = abs(position) * slippage_adj
                    fee = cost_to_cover * self.fees
                    total_cost = cost_to_cover + fee

                    proceeds_from_short = abs(position) * position_price
                    pnl = proceeds_from_short - cost_to_cover - fee
                    pnl_pct = (pnl / proceeds_from_short) * 100 if proceeds_from_short > 0 else 0
                    cash -= total_cost

                    trades.append({
                        "timestamp": timestamp,
                        "type": "cover_short",
                        "price": price,
                        "shares": abs(position),
                        "cost": total_cost,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                    })

                    if self.risk_manager is not None:
                        self.risk_manager.remove_position("ASSET")

                    position = 0
                    position_price = 0
                    entry_timestamp = None
                    entry_bar = 0
                    bars_in_position = 0
                    portfolio_value = cash

                if position == 0 and cash > 0:
                    can_enter = True
                    if self.risk_manager is not None:
                        can_enter = self.risk_manager.can_open_position("ASSET", price, portfolio_value)

                    if can_enter:
                        if self.risk_config.position_sizing_method == "volatility" and self.price_data is not None:
                            shares = self.position_sizer.calculate_shares(
                                portfolio_value=portfolio_value,
                                price=price,
                                price_data=self.price_data,
                            )
                        else:
                            shares = self.position_sizer.calculate_shares(
                                portfolio_value=portfolio_value,
                                price=price,
                            )

                        # Handle fractional shares for crypto/high-value assets
                        if self.fractional_shares and shares == 0:
                            position_value = portfolio_value * self.risk_config.position_size_pct
                            shares = position_value / price  # Keep as float

                        slippage_adj = price * (1 + self.slippage)
                        cost = shares * slippage_adj
                        fee = cost * self.fees
                        total_cost = cost + fee

                        if total_cost <= cash and shares > 0:
                            position = shares
                            position_price = price
                            entry_timestamp = timestamp
                            entry_bar = bar_index
                            cash -= total_cost
                            bars_in_position = 0

                            trades.append({
                                "timestamp": timestamp,
                                "type": "entry",
                                "price": price,
                                "shares": shares,
                                "cost": total_cost,
                            })

                            if self.risk_manager is not None:
                                idx = self.price.index.get_loc(timestamp)
                                start_idx = max(0, idx - 20)
                                recent_price_series = self.price.iloc[start_idx : idx + 1]

                                if self.price_data is not None:
                                    price_df = self.price_data.iloc[start_idx : idx + 1]
                                else:
                                    price_df = pd.DataFrame({
                                        "close": recent_price_series,
                                        "high": recent_price_series,
                                        "low": recent_price_series,
                                    })

                                self.risk_manager.add_position(
                                    symbol="ASSET",
                                    entry_price=position_price,
                                    shares=int(position),
                                    entry_bar=entry_bar,
                                    price_data=price_df,
                                )

            # Exit signals
            elif exit_signal and in_market_hours:
                if position > 0:
                    slippage_adj = price * (1 - self.slippage)
                    proceeds = position * slippage_adj
                    fee = proceeds * self.fees
                    net_proceeds = proceeds - fee
                    cash += net_proceeds

                    pnl = net_proceeds - (position * position_price)
                    pnl_pct = (pnl / (position * position_price)) * 100 if position_price > 0 else 0

                    trades.append({
                        "timestamp": timestamp,
                        "type": "exit",
                        "exit_reason": "strategy_signal",
                        "price": price,
                        "shares": position,
                        "proceeds": net_proceeds,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                    })

                    if self.risk_manager is not None:
                        self.risk_manager.remove_position("ASSET")

                    position = 0
                    position_price = 0
                    entry_timestamp = None
                    entry_bar = 0
                    bars_in_position = 0
                    portfolio_value = cash

                if position == 0 and self.allow_shorts and cash > 0:
                    can_enter = True
                    if self.risk_manager is not None:
                        can_enter = self.risk_manager.can_open_position("ASSET", price, portfolio_value)

                    if can_enter:
                        if self.risk_config.position_sizing_method == "volatility" and self.price_data is not None:
                            shares = self.position_sizer.calculate_shares(
                                portfolio_value=portfolio_value,
                                price=price,
                                price_data=self.price_data,
                            )
                        else:
                            shares = self.position_sizer.calculate_shares(
                                portfolio_value=portfolio_value,
                                price=price,
                            )

                        # Handle fractional shares for crypto/high-value assets
                        if self.fractional_shares and shares == 0:
                            position_value = portfolio_value * self.risk_config.position_size_pct
                            shares = position_value / price  # Keep as float

                        slippage_adj = price * (1 - self.slippage)
                        proceeds_from_short = shares * slippage_adj
                        fee = proceeds_from_short * self.fees
                        net_proceeds = proceeds_from_short - fee

                        if shares > 0:
                            position = -shares
                            position_price = price
                            entry_timestamp = timestamp
                            entry_bar = bar_index
                            cash += net_proceeds
                            bars_in_position = 0

                            trades.append({
                                "timestamp": timestamp,
                                "type": "short_entry",
                                "price": price,
                                "shares": shares,
                                "proceeds": net_proceeds,
                            })

                            if self.risk_manager is not None:
                                idx = self.price.index.get_loc(timestamp)
                                start_idx = max(0, idx - 20)
                                recent_price_series = self.price.iloc[start_idx : idx + 1]

                                if self.price_data is not None:
                                    price_df = self.price_data.iloc[start_idx : idx + 1]
                                else:
                                    price_df = pd.DataFrame({
                                        "close": recent_price_series,
                                        "high": recent_price_series,
                                        "low": recent_price_series,
                                    })

                                self.risk_manager.add_position(
                                    symbol="ASSET",
                                    entry_price=position_price,
                                    shares=-int(shares),
                                    entry_bar=entry_bar,
                                    price_data=price_df,
                                )

            # Final equity
            if position > 0:
                portfolio_value = cash + (position * price)
            elif position < 0:
                short_liability = abs(position) * (position_price - price)
                portfolio_value = cash + short_liability
            else:
                portfolio_value = cash
            equity.append(portfolio_value)

        self.trades = trades
        self.equity_curve = pd.Series(equity, index=self.price.index)

        # Store state arrays for lazy DataFrame construction
        if self.track_state:
            self._state_cash = state_cash
            self._state_position = state_position
            self._state_position_price = state_position_price

    def stats(self) -> Optional[Dict[str, Any]]:
        """Calculate portfolio statistics."""
        if self._stats is not None:
            return self._stats

        if self.equity_curve is None or len(self.equity_curve) == 0:
            return None

        total_return_pct = ((self.equity_curve.iloc[-1] - self.init_cash) / self.init_cash) * 100
        returns = self.equity_curve.pct_change(fill_method=None).dropna()

        start_date = self.equity_curve.index[0]
        end_date = self.equity_curve.index[-1]
        total_days = (end_date - start_date).days
        years = max(total_days / 365.25, 1 / 252)

        if self.equity_curve.iloc[-1] > 0 and self.init_cash > 0 and years > 0:
            cagr = ((self.equity_curve.iloc[-1] / self.init_cash) ** (1 / years) - 1) * 100
            annual_return_pct = cagr
        else:
            annual_return_pct = total_return_pct

        if len(returns) == 0:
            sharpe_ratio = 0
        else:
            mean_return = returns.mean()
            std_return = returns.std()
            periods_per_year = self._get_periods_per_year()

            if std_return > 0:
                sharpe_ratio = (mean_return / std_return) * np.sqrt(periods_per_year)
            else:
                sharpe_ratio = 0

        cummax = self.equity_curve.cummax()
        cummax = cummax.replace(0, 1)
        drawdown = (self.equity_curve - cummax) / cummax * 100
        max_drawdown_pct = drawdown.min()

        exit_trades = [t for t in self.trades if t.get("type") in ["exit", "cover_short"]]
        winning_trades = [t for t in exit_trades if t.get("pnl", 0) > 0]
        total_trades = len(exit_trades)
        win_rate_pct = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

        short_trades = [t for t in self.trades if t.get("type") in ["short_entry", "cover_short"]]
        num_short_trades = len([t for t in short_trades if t.get("type") == "short_entry"])

        self._stats = {
            "Total Return [%]": total_return_pct,
            "Annual Return [%]": annual_return_pct,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown [%]": max_drawdown_pct,
            "Win Rate [%]": win_rate_pct,
            "Total Trades": total_trades,
            "Start Value": self.init_cash,
            "End Value": self.equity_curve.iloc[-1],
            "Short Selling": "Enabled" if self.allow_shorts else "Disabled",
            "Short Trades": num_short_trades if self.allow_shorts else 0,
        }

        return self._stats

    def returns(self, freq: Optional[str] = None) -> pd.Series:
        """Get returns series."""
        if self.equity_curve is None or len(self.equity_curve) == 0:
            return pd.Series(dtype=float)

        if freq and len(self.equity_curve) > 1000:
            resampled_equity = self.equity_curve.resample(freq).last()
            return resampled_equity.pct_change().fillna(0)
        else:
            return self.equity_curve.pct_change().fillna(0)


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
