"""
BacktestEngine v2 with trade enrichment and state tracking.

Extends the original BacktestEngine with:
- Signals context capture for trade enrichment
- Integration with PortfolioV2 for state tracking
- Optional minute-level stop simulation
"""

import pandas as pd
from typing import Union, Dict, Any, Optional, List, TYPE_CHECKING
from pathlib import Path

from src.backtesting.engine.streaming_data_loader import StreamingDataLoader
from src.backtesting.utils.risk_config import RiskConfig
from src.backtesting_v2.engine.portfolio_simulator import PortfolioV2, from_signals_v2
from src.backtesting_v2.engine.intraday_stop_simulator import IntradayStopSimulator
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.backtesting_v2.base.strategy import BaseStrategyV2

logger = get_logger(__name__)


class BacktestEngineV2:
    """
    Enhanced BacktestEngine with trade enrichment and state tracking.

    Key enhancements over original:
    - Captures signals context from strategy for trade enrichment
    - Uses PortfolioV2 with bar-by-bar state tracking
    - Supports minute-level stop simulation

    Usage:
        from src.backtesting_v2.engine import BacktestEngineV2
        from src.backtesting_v2.base import BaseStrategyV2

        engine = BacktestEngineV2(initial_capital=100000)
        portfolio = engine.run(strategy, "SPY", "2023-01-01", "2023-12-31")

        # Access enriched data
        trades = portfolio.trades_df
        state = portfolio.portfolio_state
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        fees: float = 0.001,
        slippage: float = 0.0,
        freq: str = "1min",
        market_hours_only: bool = True,
        risk_config: Optional[RiskConfig] = None,
        allow_shorts: bool = True,
        timeframe: str = "1min",
    ):
        """
        Initialize v2 backtesting engine.

        Args:
            initial_capital: Starting capital (default: $100,000)
            fees: Trading fees as decimal (default: 0.001 = 0.1%)
            slippage: Slippage as decimal (default: 0.0)
            freq: Data frequency (default: '1min')
            market_hours_only: If True, only trade during market hours
            risk_config: Risk management configuration
            allow_shorts: If True, enable short selling (default: True)
            timeframe: Data timeframe - '1min', '1hour', '1day', 'crypto_1min'
        """
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")
        if fees < 0:
            raise ValueError(f"fees must be non-negative, got {fees}")
        if slippage < 0:
            raise ValueError(f"slippage must be non-negative, got {slippage}")

        self.initial_capital = initial_capital
        self.fees = fees
        self.slippage = slippage
        self.freq = timeframe if timeframe else freq
        self.market_hours_only = market_hours_only
        self.risk_config = risk_config or RiskConfig.moderate()
        self.allow_shorts = allow_shorts
        self.timeframe = timeframe

        # Crypto detection
        is_crypto = timeframe.startswith("crypto_")
        filter_market_days = not is_crypto
        self.fractional_shares = is_crypto

        self.data_loader = StreamingDataLoader(
            timeframe=timeframe, filter_market_days=filter_market_days
        )

    def _load_symbol_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Load data for a single symbol."""
        for sym, pl_df in self.data_loader.iter_symbols([symbol], start_date, end_date):
            pdf = pl_df.to_pandas()
            pdf = pdf.set_index("timestamp")
            return pdf

        raise ValueError(f"No data found for {symbol} between {start_date} and {end_date}")

    def run(
        self,
        strategy: "BaseStrategyV2",
        symbol: str,
        start_date: str,
        end_date: str,
        price_type: str = "close",
    ) -> PortfolioV2:
        """
        Run single-symbol backtest with trade enrichment.

        Args:
            strategy: Strategy instance (must implement generate_signals)
            symbol: Symbol to backtest
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            price_type: Price column for signals ('close', 'open', etc.)

        Returns:
            PortfolioV2 with enriched trades and state tracking
        """
        logger.info(f"Running v2 backtest: {strategy} on {symbol}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Capital: ${self.initial_capital:,.2f}, Fees: {self.fees*100:.2f}%")

        # Load data
        data = self._load_symbol_data(symbol, start_date, end_date)
        logger.info(f"Loaded {len(data)} bars")

        # Capture signals context before simulation
        signals_context = None
        if hasattr(strategy, "get_signals_context"):
            signals_context = strategy.get_signals_context(data)
            if signals_context:
                logger.info(f"Captured signals context with {len(signals_context)} entries")

        # Generate signals
        entries, exits = strategy.generate_signals(data)

        # Get price series
        price = data[price_type] if price_type in data.columns else data["close"]

        # Create v2 portfolio with state tracking
        portfolio = from_signals_v2(
            close=price,
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            fees=self.fees,
            slippage=self.slippage,
            freq=self.freq,
            market_hours_only=self.market_hours_only,
            risk_config=self.risk_config,
            price_data=data,
            allow_shorts=self.allow_shorts,
            use_numba=True,
            fractional_shares=self.fractional_shares,
        )

        # Store context and finalize trades
        portfolio.set_signals_context(signals_context)
        portfolio.finalize_trades(strategy)

        # Log summary
        stats = portfolio.stats()
        if stats:
            logger.success(
                f"Backtest complete: {stats['Total Return [%]']:.2f}% return, "
                f"{stats['Total Trades']} trades, "
                f"{stats['Sharpe Ratio']:.2f} Sharpe"
            )

        return portfolio

    def run_with_intraday_stops(
        self,
        strategy: "BaseStrategyV2",
        symbol: str,
        start_date: str,
        end_date: str,
        stop_loss_pct: float = 0.02,
        trailing_stop_pct: Optional[float] = None,
        entry_time: str = "15:50",
        exit_time: str = "09:31",
        price_type: str = "close",
    ) -> PortfolioV2:
        """
        Run backtest with minute-level stop simulation.

        For strategies like OMR that hold overnight, this method:
        1. Runs standard simulation
        2. Post-processes trades with minute-level stop checks
        3. Recalculates P&L based on actual exit prices

        Args:
            strategy: Strategy instance
            symbol: Symbol to backtest
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            stop_loss_pct: Fixed stop loss percentage
            trailing_stop_pct: Optional trailing stop percentage
            entry_time: Entry time reference
            exit_time: Default exit time if no stop
            price_type: Price column for signals

        Returns:
            PortfolioV2 with updated trades reflecting stop triggers
        """
        logger.info("Running v2 backtest with intraday stop simulation")

        # Run standard simulation first
        portfolio = self.run(strategy, symbol, start_date, end_date, price_type)

        # Apply intraday stop simulation
        if not portfolio.trades_df.empty:
            stop_sim = IntradayStopSimulator(
                stop_loss_pct=stop_loss_pct,
                trailing_stop_pct=trailing_stop_pct,
                data_loader=self.data_loader,
            )

            # Add symbol column if not present
            trades_df = portfolio.trades_df.copy()
            if "symbol" not in trades_df.columns:
                trades_df["symbol"] = symbol

            # Simulate stops
            updated_trades = stop_sim.simulate_stops(
                trades_df,
                entry_time=entry_time,
                exit_time=exit_time,
                direction="long",
            )

            # Update portfolio trades
            portfolio._trades_df = updated_trades

            # Count stop triggers
            stop_count = updated_trades["stop_triggered"].sum() if "stop_triggered" in updated_trades.columns else 0
            logger.info(f"Intraday stops: {stop_count} of {len(updated_trades)} trades stopped out")

        return portfolio

    def run_sweep(
        self,
        strategy: "BaseStrategyV2",
        symbols: List[str],
        start_date: str,
        end_date: str,
        price_type: str = "close",
    ) -> Dict[str, PortfolioV2]:
        """
        Run backtest across multiple symbols.

        Args:
            strategy: Strategy instance
            symbols: List of symbols to backtest
            start_date: Start date
            end_date: End date
            price_type: Price column for signals

        Returns:
            Dict mapping symbol -> PortfolioV2
        """
        results = {}

        for symbol in symbols:
            try:
                portfolio = self.run(strategy, symbol, start_date, end_date, price_type)
                results[symbol] = portfolio
            except Exception as e:
                logger.error(f"Failed to backtest {symbol}: {e}")

        return results


def run_backtest_v2(
    strategy: "BaseStrategyV2",
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
    fees: float = 0.001,
    slippage: float = 0.0,
    risk_config: Optional[RiskConfig] = None,
    timeframe: str = "1min",
) -> PortfolioV2:
    """
    Convenience function to run v2 backtest.

    Args:
        strategy: Strategy instance
        symbol: Symbol to backtest
        start_date: Start date
        end_date: End date
        initial_capital: Starting capital
        fees: Trading fees
        slippage: Slippage
        risk_config: Risk configuration
        timeframe: Data timeframe

    Returns:
        PortfolioV2 with enriched trades and state
    """
    engine = BacktestEngineV2(
        initial_capital=initial_capital,
        fees=fees,
        slippage=slippage,
        risk_config=risk_config,
        timeframe=timeframe,
    )

    return engine.run(strategy, symbol, start_date, end_date)
