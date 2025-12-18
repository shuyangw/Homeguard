"""
Core backtesting engine with custom portfolio simulator.
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, Any, Optional, List, TYPE_CHECKING
from pathlib import Path

from src.backtesting.base.strategy import BaseStrategy, MultiSymbolStrategy, LongShortStrategy
from src.backtesting.base.pairs_strategy import PairsStrategy
from src.backtesting.engine.streaming_data_loader import StreamingDataLoader
from src.backtesting.engine.portfolio_simulator import Portfolio, from_signals
from src.backtesting.engine.rolling_results import RollingWindowResults
from src.backtesting.utils.risk_config import RiskConfig
from src.backtesting.utils.market_calendar import MarketCalendar
from src.visualization.reports.quantstats_reporter import QuantStatsReporter
from src.utils import logger

if TYPE_CHECKING:
    from src.backtesting.engine.multi_asset_portfolio import MultiAssetPortfolio

# Type alias for portfolio return types
PortfolioType = Union[Portfolio, 'MultiAssetPortfolio', RollingWindowResults]


class BacktestEngine:
    """
    Backtesting engine that executes strategies with custom portfolio simulator.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        fees: float = 0.001,
        slippage: float = 0.0,
        freq: str = '1min',
        market_hours_only: bool = True,
        benchmark: str = 'SPY',
        risk_config: Optional[RiskConfig] = None,
        enable_regime_analysis: bool = False,
        allow_shorts: bool = True,
        timeframe: str = '1min'
    ):
        """
        Initialize backtesting engine.

        Args:
            initial_capital: Starting capital for backtests (default: $100,000)
            fees: Trading fees as decimal (default: 0.001 = 0.1%)
            slippage: Slippage as decimal (default: 0.0)
            freq: Data frequency for resampling (default: '1min')
            market_hours_only: If True, only execute trades during market hours (9:35 AM - 3:55 PM EST)
            benchmark: Benchmark ticker for QuantStats reports (default: 'SPY')
            risk_config: Risk management configuration (default: RiskConfig.moderate())
            enable_regime_analysis: If True, automatically analyze performance by market regime (default: False)
            allow_shorts: If True, enable short selling for strategies (default: True)
            timeframe: Data timeframe for loading - '1min', '1hour', '1day', 'crypto_1min', etc. (default: '1min')

        Raises:
            ValueError: If initial_capital <= 0
        """
        # Validate inputs
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")
        if fees < 0:
            raise ValueError(f"fees must be non-negative, got {fees}")
        if slippage < 0:
            raise ValueError(f"slippage must be non-negative, got {slippage}")

        self.initial_capital = initial_capital
        self.fees = fees
        self.slippage = slippage
        self.freq = freq
        self.market_hours_only = market_hours_only
        self.risk_config = risk_config or RiskConfig.moderate()
        self.enable_regime_analysis = enable_regime_analysis
        self.allow_shorts = allow_shorts
        self.timeframe = timeframe
        # For crypto data, disable market day filtering (24/7 trading)
        filter_market_days = not timeframe.startswith('crypto_')
        self.data_loader = StreamingDataLoader(timeframe=timeframe, filter_market_days=filter_market_days)
        self.reporter = QuantStatsReporter(benchmark=benchmark)

        # Cache for regime analysis
        self._last_market_data = None
        self._last_symbols = None
        self._last_start_date = None
        self._last_end_date = None

    def _load_data_as_pandas(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Load data using streaming loader and convert to pandas MultiIndex DataFrame.

        Uses the StreamingDataLoader for efficient I/O, then converts to pandas
        format expected by strategies and portfolio simulator.

        Args:
            symbols: List of symbols to load
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            pandas DataFrame with MultiIndex (symbol, timestamp) and OHLCV columns
        """
        dfs = []

        for symbol, pl_df in self.data_loader.iter_symbols(symbols, start_date, end_date):
            # Convert Polars to pandas
            pdf = pl_df.to_pandas()

            # Set timestamp as index
            pdf = pdf.set_index('timestamp')

            # Add symbol column for MultiIndex
            pdf['symbol'] = symbol

            dfs.append(pdf)

        if not dfs:
            raise ValueError(f"No data found for symbols {symbols} between {start_date} and {end_date}")

        # Concatenate and create MultiIndex
        combined = pd.concat(dfs)
        combined = combined.reset_index().set_index(['symbol', 'timestamp']).sort_index()

        logger.success(f"Loaded {len(combined)} bars for {len(symbols)} symbol(s) from {start_date} to {end_date}")

        return combined

    def run(
        self,
        strategy: BaseStrategy,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        price_type: str = 'close',
        portfolio_mode: str = 'single',
        mode: str = 'batch',
        window_days: int = 60,
        step_days: Optional[int] = None
    ) -> PortfolioType:
        """
        Run backtest for a strategy.

        Args:
            strategy: Strategy instance to backtest
            symbols: Symbol or list of symbols to trade
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            price_type: Price to use for signals ('close', 'open', 'high', 'low')
            portfolio_mode: 'single' or 'multi'
                - 'single': Test each symbol independently (default, backward compatible)
                - 'multi': Hold multiple symbols in one portfolio simultaneously
            mode: Execution mode ('batch' or 'rolling')
                - 'batch': Load all data upfront, single simulation pass (default)
                - 'rolling': Rolling window simulation, more realistic for live trading
            window_days: Number of TRADING days per window (for rolling mode, default: 60)
            step_days: Number of TRADING days to step forward (default: window_days)

        Returns:
            Portfolio object (batch mode) or RollingWindowResults (rolling mode)
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        logger.blank()
        logger.separator()
        logger.header(f"Running backtest: {strategy}")
        logger.info(f"Symbols: {', '.join(symbols)}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.metric(f"Initial capital: ${self.initial_capital:,.2f}")
        logger.metric(f"Fees: {self.fees*100:.2f}%")

        # Display short selling status prominently
        if self.allow_shorts:
            logger.success(f"Short selling: ENABLED")
        else:
            logger.warning(f"Short selling: DISABLED (long-only mode)")

        if self.market_hours_only:
            logger.info(f"Market hours only: 9:35 AM - 3:55 PM EST")
        else:
            logger.info(f"Market hours filtering: Disabled")

        # Log risk management settings
        logger.info(f"Risk management: {self.risk_config}")
        if self.risk_config.enabled:
            logger.metric(f"Position size: {self.risk_config.position_size_pct*100:.1f}% per trade")
            if self.risk_config.use_stop_loss:
                logger.metric(f"Stop loss: {self.risk_config.stop_loss_pct*100:.1f}% ({self.risk_config.stop_loss_type})")

        # Log execution mode
        if mode == 'rolling':
            logger.info(f"Mode: ROLLING (window={window_days} trading days, step={step_days or window_days})")
        else:
            logger.info(f"Mode: BATCH")

        logger.separator()
        logger.blank()

        # Route to rolling window mode if requested
        if mode == 'rolling':
            return self._run_rolling(
                strategy=strategy,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                window_days=window_days,
                step_days=step_days or window_days,
                price_type=price_type
            )

        # BATCH MODE (default) - existing logic below

        # Cache for regime analysis (batch mode only)
        if self.enable_regime_analysis:
            self._last_symbols = symbols
            self._last_start_date = start_date
            self._last_end_date = end_date

        # Detect strategy type
        is_multi_symbol = self._detect_strategy_type(strategy)

        # Route based on strategy type and portfolio mode
        if is_multi_symbol:
            # MULTI-SYMBOL STRATEGY (e.g., PairsTrading)
            if len(symbols) < 2:
                raise ValueError(
                    f"{strategy.__class__.__name__} is a multi-symbol strategy that requires "
                    f"at least 2 symbols, got {len(symbols)}"
                )

            # Validate symbol requirements
            required = strategy.get_required_symbols()  # type: ignore
            if isinstance(required, int):
                if len(symbols) != required:
                    raise ValueError(
                        f"{strategy.__class__.__name__} requires exactly {required} symbols, "
                        f"got {len(symbols)}: {symbols}"
                    )
            elif isinstance(required, list):
                if symbols != required:
                    raise ValueError(
                        f"{strategy.__class__.__name__} requires specific symbols {required}, "
                        f"got {symbols}"
                    )

            portfolio = self._run_multi_symbol_strategy(strategy, symbols, start_date, end_date, price_type)

        elif portfolio_mode == 'single':
            # SINGLE-SYMBOL MODE (backward compatible)
            if len(symbols) == 1:
                portfolio = self._run_single_symbol(strategy, symbols[0], start_date, end_date, price_type)
            else:
                # Sweep mode: run each symbol separately
                portfolio = self._run_multiple_symbols(strategy, symbols, start_date, end_date, price_type)

        elif portfolio_mode == 'multi':
            # MULTI-SYMBOL PORTFOLIO MODE (new)
            if len(symbols) == 1:
                # Single symbol in multi mode = same as single mode
                portfolio = self._run_single_symbol(strategy, symbols[0], start_date, end_date, price_type)
            else:
                # True multi-asset portfolio
                portfolio = self._run_multi_asset_portfolio(strategy, symbols, start_date, end_date, price_type)

        else:
            raise ValueError(f"Invalid portfolio_mode: {portfolio_mode}. Must be 'single' or 'multi'.")

        self._print_summary(portfolio)

        # Optionally run regime analysis
        if self.enable_regime_analysis:
            self._print_regime_analysis(portfolio)

        return portfolio

    def run_and_report(
        self,
        strategy: BaseStrategy,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        output_dir: Path,
        price_type: str = 'close',
        include_pdf: bool = False,
        portfolio_mode: str = 'single'
    ) -> Portfolio:
        """
        Run backtest and generate QuantStats report.

        This method combines backtesting with automatic report generation,
        creating comprehensive tearsheets with performance metrics and visualizations.

        Args:
            strategy: Strategy instance to backtest
            symbols: Symbol or list of symbols to trade
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            output_dir: Directory to save reports (tearsheet, metrics, etc.)
            price_type: Price to use for signals ('close', 'open', 'high', 'low')
            include_pdf: If True, also generate PDF version of report

        Returns:
            Portfolio object with backtest results

        Example:
            >>> engine = BacktestEngine(initial_capital=50000, fees=0.001)
            >>> strategy = MovingAverageCrossover(fast=10, slow=50)
            >>> from src.settings import get_backtest_results_dir
            >>> portfolio = engine.run_and_report(
            ...     strategy=strategy,
            ...     symbols=['AAPL'],
            ...     start_date='2024-01-01',
            ...     end_date='2024-12-31',
            ...     output_dir=get_backtest_results_dir() / 'backtest_001'
            ... )
        """
        # Run the backtest
        portfolio = self.run(
            strategy=strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            price_type=price_type,
            portfolio_mode=portfolio_mode
        )

        # Generate report title
        if isinstance(symbols, str):
            symbols_list = [symbols]
        else:
            symbols_list = symbols

        title = f"{strategy.__class__.__name__} - {', '.join(symbols_list)}"

        # Generate QuantStats report
        logger.blank()
        logger.separator()
        logger.header("GENERATING QUANTSTATS REPORT")
        logger.separator()
        logger.blank()

        # Prepare strategy info for executive summary
        strategy_info = {
            'name': strategy.__class__.__name__,
            'symbols': symbols_list,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.initial_capital,
            'fees': f"{self.fees:.4f}" if self.fees else "0.0000",
            'allow_shorts': self.allow_shorts
        }

        try:
            report_path = self.reporter.generate_report(
                portfolio=portfolio,
                output_dir=output_dir,
                title=title,
                include_pdf=include_pdf,
                strategy_info=strategy_info
            )
            logger.blank()
            logger.success("Report generation complete!")
            logger.info(f"View report: {report_path}")
            logger.blank()
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            logger.warning("Backtest completed successfully, but report generation failed.")

        return portfolio

    def _run_single_symbol(
        self,
        strategy: BaseStrategy,
        symbol: str,
        start_date: str,
        end_date: str,
        price_type: str
    ) -> Portfolio:
        """
        Run backtest for a single symbol.
        """
        # Load data specifically for this symbol (streaming-friendly)
        pl_df = self.data_loader.load_symbol(symbol, start_date, end_date)
        if pl_df.is_empty():
             raise ValueError(f"No data found for {symbol}")
        
        symbol_data = pl_df.to_pandas().set_index('timestamp')
        # Compatibility with existing logic expecting 'symbol' context in some places? 
        # Actually strategy.generate_signals usually takes the DF directly.
        
        price = symbol_data[price_type]

        # Check if strategy is a LongShortStrategy (returns 4 signals)
        if isinstance(strategy, LongShortStrategy):
            long_entries, long_exits, short_entries, short_exits = strategy.generate_signals(symbol_data)

            long_entries = long_entries.fillna(False).astype(bool)
            long_exits = long_exits.fillna(False).astype(bool)
            short_entries = short_entries.fillna(False).astype(bool)
            short_exits = short_exits.fillna(False).astype(bool)

            # Combine long and short signals for the current Portfolio implementation
            # Long entries and short exits are "buy" signals
            # Short entries and long exits are "sell/exit" signals
            # Note: Current portfolio simulator handles position flipping
            combined_entries = long_entries | short_exits  # Enter long or cover short
            combined_exits = long_exits | short_entries    # Exit long or enter short

            portfolio = from_signals(
                close=price,
                entries=combined_entries,
                exits=combined_exits,
                init_cash=self.initial_capital,
                fees=self.fees,
                slippage=self.slippage,
                freq=self.freq,
                market_hours_only=self.market_hours_only,
                risk_config=self.risk_config,
                price_data=symbol_data,
                allow_shorts=self.allow_shorts
            )
        else:
            # Standard strategy (long-only, returns 2 signals)
            entries, exits = strategy.generate_signals(symbol_data)

            entries = entries.fillna(False).astype(bool)
            exits = exits.fillna(False).astype(bool)

            portfolio = from_signals(
                close=price,
                entries=entries,
                exits=exits,
                init_cash=self.initial_capital,
                fees=self.fees,
                slippage=self.slippage,
                freq=self.freq,
                market_hours_only=self.market_hours_only,
                risk_config=self.risk_config,
                price_data=symbol_data,
                allow_shorts=self.allow_shorts
            )

        return portfolio

    def _run_multiple_symbols(
        self,
        strategy: BaseStrategy,
        symbols: List[str],
        start_date: str,
        end_date: str,
        price_type: str
    ) -> Portfolio:
        """
        Run backtest for multiple symbols in sweep mode (each symbol separately).

        Note: This is the old behavior - each symbol tested independently.
        For true portfolio mode, use _run_multi_asset_portfolio.
        """
        """
        Run backtest for multiple symbols in sweep mode. 
        Currently simplified to first symbol only to maintain API compatibility while using streaming loader.
        """
        logger.warning(f"Multi-symbol backtesting simplified to first symbol only.")
        return self._run_single_symbol(strategy, symbols[0], start_date, end_date, price_type)

    def _run_multi_asset_portfolio(
        self,
        strategy: BaseStrategy,
        symbols: List[str],
        start_date: str,
        end_date: str,
        price_type: str
    ) -> Portfolio:
        """
        Run backtest for multiple symbols in portfolio mode (hold all simultaneously).
        """
        from src.backtesting.engine.multi_asset_portfolio import MultiAssetPortfolio

        logger.info(f"Running multi-asset portfolio with {len(symbols)} symbols")

        # Use synchronized loading from StreamingDataLoader for efficiency
        # This returns Dict[str, pl.DataFrame]
        try:
             pl_dfs = self.data_loader.load_symbols_synchronized(symbols, start_date, end_date)
        except Exception as e:
             logger.error(f"Failed to load synchronized data: {e}")
             raise

        # Convert to Pandas for internal processing
        # We need a unified price DataFrame and individual symbol DataFrames
        prices_dict = {}
        all_entries = {}
        all_exits = {}
        
        # We need to construct the 'price_data' DF used by MultiAssetPortfolio
        # It expects a multi-index (timestamp, symbol) usually, or we pass dict?
        # MultiAssetPortfolio signature: price_data: pd.DataFrame (multi-index usually)
        
        # Let's reconstruct the needed data structures
        
        for symbol, pl_df in pl_dfs.items():
            pdf = pl_df.to_pandas().set_index('timestamp')
            prices_dict[symbol] = pdf[price_type]

            # Tell strategy which symbol we're processing (for sentiment loading, etc.)
            strategy._current_symbol = symbol

            # Generate signals - handle both standard (2 signals) and LongShort (4 signals) strategies
            try:
                if isinstance(strategy, LongShortStrategy):
                    long_entries, long_exits, short_entries, short_exits = strategy.generate_signals(pdf)
                    # Combine: entry = long entry or short cover, exit = long exit or short entry
                    entries = (long_entries | short_exits).fillna(False).astype(bool)
                    exits = (long_exits | short_entries).fillna(False).astype(bool)
                else:
                    entries, exits = strategy.generate_signals(pdf)
                    entries = entries.fillna(False).astype(bool)
                    exits = exits.fillna(False).astype(bool)
                all_entries[symbol] = entries
                all_exits[symbol] = exits
            except Exception as e:
                logger.warning(f"Error generating signals for {symbol}: {e}")
                
        prices_df = pd.DataFrame(prices_dict)
        entries_df = pd.DataFrame(all_entries)
        exits_df = pd.DataFrame(all_exits)
        
        # Reconstruct multi-index 'data' for the portfolio if needed
        # Or simply pass what we have if MultiAssetPortfolio supports it.
        # Looking at original code: price_data=data (which was multi-index).
        
        # Let's rebuild the multi-index DF lazily or just skip if not strictly used or build it:
        dfs_list = []
        for sym, pl_df in pl_dfs.items():
            p = pl_df.to_pandas().set_index('timestamp')
            p['symbol'] = sym
            dfs_list.append(p)
        
        if dfs_list:
             data = pd.concat(dfs_list).reset_index().set_index(['symbol', 'timestamp']).sort_index()
        else:
             data = pd.DataFrame()


        # Create multi-asset portfolio
        # Pass max_positions from risk_config for concentrated portfolios
        max_pos = self.risk_config.max_positions if self.risk_config else 10

        # Extract max_hold_bars from strategy if available (for time-based exits)
        max_hold_bars = getattr(strategy, 'max_hold_bars', 0)

        portfolio = MultiAssetPortfolio(
            symbols=symbols,
            prices=prices_df,
            entries=entries_df,
            exits=exits_df,
            init_cash=self.initial_capital,
            fees=self.fees,
            slippage=self.slippage,
            freq=self.freq,
            market_hours_only=self.market_hours_only,
            risk_config=self.risk_config,
            max_positions=max_pos,
            price_data=data,
            max_hold_bars=max_hold_bars
        )

        return portfolio  # type: ignore[return-value]

    def _detect_strategy_type(self, strategy: BaseStrategy) -> bool:
        """
        Detect if strategy is a multi-symbol strategy.

        Args:
            strategy: Strategy instance to check

        Returns:
            True if strategy is a MultiSymbolStrategy, False otherwise
        """
        return isinstance(strategy, MultiSymbolStrategy)

    def _synchronize_symbol_data(
        self,
        data: pd.DataFrame,
        symbols: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Synchronize data across multiple symbols (align timestamps).

        For multi-symbol strategies, all symbols must have aligned timestamps.
        This method extracts per-symbol data and ensures they share a common index.

        Args:
            data: Multi-index DataFrame with (timestamp, symbol) index
            symbols: List of symbols to synchronize

        Returns:
            Dictionary mapping symbol to synchronized DataFrame

        Example:
            >>> data_dict = engine._synchronize_symbol_data(data, ['AAPL', 'MSFT'])
            >>> # All DataFrames in data_dict now have the same timestamp index
        """
        # Extract data for each symbol
        symbol_data = {}
        for symbol in symbols:
            try:
                symbol_df: pd.DataFrame = data.xs(symbol, level='symbol')  # type: ignore
                symbol_data[symbol] = symbol_df
            except KeyError:
                raise ValueError(f"Symbol {symbol} not found in data")

        # Find common timestamp index (intersection of all symbol timestamps)
        common_index = symbol_data[symbols[0]].index
        for symbol in symbols[1:]:
            common_index = common_index.intersection(symbol_data[symbol].index)

        if len(common_index) == 0:
            raise ValueError(
                f"No overlapping timestamps found between symbols {symbols}. "
                f"Cannot synchronize data for multi-symbol strategy."
            )

        # Reindex all symbols to common timestamps
        synchronized_data = {}
        for symbol, df in symbol_data.items():
            synchronized_data[symbol] = df.loc[common_index]

        logger.info(
            f"Synchronized {len(symbols)} symbols: {len(common_index)} common timestamps"
        )

        return synchronized_data

    def _run_multi_symbol_strategy(
        self,
        strategy: MultiSymbolStrategy,
        symbols: List[str],
        start_date: str,
        end_date: str,
        price_type: str
    ) -> Portfolio:
        """
        Run backtest for a multi-symbol strategy (e.g., PairsTrading).

        Multi-symbol strategies receive all symbol data at once and generate
        synchronized signals across all symbols.

        Args:
            strategy: MultiSymbolStrategy instance
            data: Multi-index DataFrame with (timestamp, symbol) index
            symbols: List of symbols to trade
            price_type: Price column to use for execution

        Returns:
            Portfolio object with backtest results
        """
        logger.info(f"Running multi-symbol strategy: {strategy.__class__.__name__}")
        logger.info(f"Symbols: {', '.join(symbols)}")

        # Synchronize data using streaming loader
        pl_dfs = self.data_loader.load_symbols_synchronized(symbols, start_date, end_date)
        
        # Convert to pandas and build dict
        data_dict = {}
        dfs_list = []
        
        for sym, pl_df in pl_dfs.items():
             pdf = pl_df.to_pandas().set_index('timestamp')
             data_dict[sym] = pdf
             
             # For PairsPortfolio 'price_data1/2' args
             # And for consistency
             

        # Generate multi-symbol signals
        signals_dict = strategy.generate_multi_signals(data_dict)

        # Check if this is a pairs strategy
        if isinstance(strategy, PairsStrategy) and len(symbols) == 2:
            # Use PairsPortfolio for proper pairs execution
            from src.backtesting.engine.pairs_portfolio import PairsPortfolio

            logger.info("Using PairsPortfolio for synchronized pair execution")

            symbol1, symbol2 = symbols
            signals1 = signals_dict[symbol1]
            signals2 = signals_dict[symbol2]

            # Expect 4-tuple for pairs
            if len(signals1) != 4 or len(signals2) != 4:
                raise ValueError(
                    f"PairsStrategy must return 4-tuple (long_e, long_x, short_e, short_x), "
                    f"got {len(signals1)}-tuple for {symbol1} and {len(signals2)}-tuple for {symbol2}"
                )

            # Unpack signals for symbol1
            long_entries1, long_exits1, short_entries1, short_exits1 = signals1

            # For pairs: we use symbol1's signals to determine spread direction
            # Long spread: short sym1, long sym2 (entries = long_entries1 inverted)
            # Short spread: long sym1, short sym2 (entries = short_entries1)

            # The strategy should return signals where:
            # - symbol1 long_entries means "long symbol1" (short spread)
            # - symbol1 short_entries means "short symbol1" (long spread)

            # So we use short_entries1 as long_spread_entries
            # And long_entries1 as short_spread_entries

            portfolio = PairsPortfolio(
                symbols=(symbol1, symbol2),
                prices1=data_dict[symbol1][price_type],
                prices2=data_dict[symbol2][price_type],
                entries=short_entries1,  # Long spread entries
                exits=short_exits1,      # Long spread exits
                short_entries=long_entries1,  # Short spread entries
                short_exits=long_exits1,      # Short spread exits
                init_cash=self.initial_capital,
                fees=self.fees,
                slippage=self.slippage,
                freq=self.freq,
                market_hours_only=self.market_hours_only,
                risk_config=self.risk_config,
                price_data1=data_dict[symbol1],
                price_data2=data_dict[symbol2]
            )

            return portfolio  # type: ignore

        else:
            # Fallback to single-symbol execution for other multi-symbol strategies
            logger.warning(
                f"Non-pairs multi-symbol strategy: executing first symbol only. "
                f"Full multi-asset support pending."
            )

            first_symbol = symbols[0]
            signal_tuple = signals_dict[first_symbol]

            # Check if 2-tuple (long-only) or 4-tuple (long-short)
            if len(signal_tuple) == 2:
                # Long-only strategy
                entries, exits = signal_tuple
                short_entries = pd.Series(False, index=entries.index)
                short_exits = pd.Series(False, index=entries.index)
            elif len(signal_tuple) == 4:
                # Long-short strategy
                entries, exits, short_entries, short_exits = signal_tuple
            else:
                raise ValueError(
                    f"Invalid signal tuple length: {len(signal_tuple)}. "
                    f"Expected 2 (long-only) or 4 (long-short)."
                )

            # Get price data
            price = data_dict[first_symbol][price_type]

            # Ensure boolean signals
            entries = entries.fillna(False).astype(bool)
            exits = exits.fillna(False).astype(bool)
            short_entries = short_entries.fillna(False).astype(bool)
            short_exits = short_exits.fillna(False).astype(bool)

            # Create portfolio using existing from_signals()
            portfolio = from_signals(
                close=price,
                entries=entries,
                exits=exits,
                short_entries=short_entries,
                short_exits=short_exits,
                init_cash=self.initial_capital,
                fees=self.fees,
                slippage=self.slippage,
                freq=self.freq,
                market_hours_only=self.market_hours_only,
                risk_config=self.risk_config,
                price_data=data_dict[first_symbol],
                allow_shorts=self.allow_shorts
            )

            return portfolio

    def run_with_data(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        price_type: str = 'close'
    ) -> Portfolio:
        """
        Run backtest with pre-loaded data.

        Args:
            strategy: Strategy instance to backtest
            data: DataFrame with OHLCV columns and timestamp index
            price_type: Price to use for signals

        Returns:
            Portfolio object with backtest results
        """
        logger.blank()
        logger.separator()
        logger.header(f"Running backtest: {strategy}")
        logger.info(f"Data shape: {data.shape}")
        logger.metric(f"Initial capital: ${self.initial_capital:,.2f}")
        if self.market_hours_only:
            logger.info(f"Market hours only: 9:35 AM - 3:55 PM EST")
        else:
            logger.info(f"Market hours filtering: Disabled")
        logger.separator()
        logger.blank()

        # Validate data: check for duplicate timestamps
        if data.index.duplicated().any():
            n_duplicates = data.index.duplicated().sum()
            logger.warning(f"Found {n_duplicates} duplicate timestamps - removing duplicates (keeping first occurrence)")
            # Remove duplicates, keeping first occurrence
            data = data[~data.index.duplicated(keep='first')]
            logger.info(f"Data shape after removing duplicates: {data.shape}")

        entries, exits = strategy.generate_signals(data)
        price = data[price_type]

        entries = entries.fillna(False).astype(bool)
        exits = exits.fillna(False).astype(bool)
        if isinstance(price, pd.DataFrame):
            price = price.ffill().bfill()

        portfolio = from_signals(
            close=price,
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            fees=self.fees,
            slippage=self.slippage,
            freq=self.freq,
            market_hours_only=self.market_hours_only,
            risk_config=self.risk_config,
            price_data=data  # Pass OHLC data for ATR-based position sizing
        )

        self._print_summary(portfolio)

        return portfolio

    def optimize(
        self,
        strategy_class: type,
        param_grid: Dict[str, List[Any]],
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters over a grid.

        Delegates to GridSearchOptimizer for the actual optimization.

        Args:
            strategy_class: Strategy class to optimize
            param_grid: Dictionary mapping parameter names to lists of values
            symbols: Symbol or list of symbols
            start_date: Start date
            end_date: End date
            metric: Metric to optimize ('sharpe_ratio', 'total_return', 'max_drawdown')

        Returns:
            Dictionary with best parameters and results
        """
        from src.backtesting.optimization.grid_search import GridSearchOptimizer

        optimizer = GridSearchOptimizer(self)
        return optimizer.optimize(
            strategy_class=strategy_class,
            param_grid=param_grid,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            metric=metric
        )

    def _print_summary(self, portfolio: Portfolio) -> None:
        """
        Print backtest summary statistics.
        """
        try:
            stats = portfolio.stats()

            if stats is None:
                logger.warning("Could not generate statistics for portfolio")
                return

            logger.blank()
            logger.separator()
            logger.header("BACKTEST RESULTS")
            logger.separator()

            # Color-code returns based on positive/negative
            total_return = stats.get('Total Return [%]', 0)
            annual_return = stats.get('Annual Return [%]', 0)
            if total_return >= 0:
                logger.profit(f"Total Return:       {total_return:.2f}%")
            else:
                logger.loss(f"Total Return:       {total_return:.2f}%")

            if annual_return >= 0:
                logger.profit(f"Annual Return:      {annual_return:.2f}%")
            else:
                logger.loss(f"Annual Return:      {annual_return:.2f}%")

            # Sharpe ratio (higher is better)
            sharpe = stats.get('Sharpe Ratio', 0)
            if sharpe >= 1.0:
                logger.profit(f"Sharpe Ratio:       {sharpe:.2f}")
            elif sharpe >= 0:
                logger.metric(f"Sharpe Ratio:       {sharpe:.2f}")
            else:
                logger.loss(f"Sharpe Ratio:       {sharpe:.2f}")

            # Max drawdown (negative value, so use loss color)
            max_dd = stats.get('Max Drawdown [%]', 0)
            if max_dd >= -5:
                logger.metric(f"Max Drawdown:       {max_dd:.2f}%")
            elif max_dd >= -15:
                logger.warning(f"Max Drawdown:       {max_dd:.2f}%")
            else:
                logger.loss(f"Max Drawdown:       {max_dd:.2f}%")

            # Win rate
            win_rate = stats.get('Win Rate [%]', 0)
            if win_rate >= 50:
                logger.profit(f"Win Rate:           {win_rate:.2f}%")
            else:
                logger.metric(f"Win Rate:           {win_rate:.2f}%")

            logger.metric(f"Total Trades:       {stats.get('Total Trades', 0)}")
            logger.metric(f"Final Value:        ${stats.get('End Value', 0):.2f}")
            logger.separator()
            logger.blank()
        except Exception as e:
            logger.warning(f"Could not print summary statistics: {e}")

    def _print_regime_analysis(self, portfolio: Portfolio) -> Optional['RegimeAnalysisResults']:
        """
        Print regime-based performance analysis and store results (Level 4).

        Analyzes strategy performance across different market regimes:
        - Trend regimes (Bull/Bear/Sideways)
        - Volatility regimes (High/Low)
        - Drawdown regimes (Drawdown/Recovery/Calm)

        Returns:
            RegimeAnalysisResults object if analysis successful, None otherwise
        """
        try:
            from src.backtesting.regimes.analyzer import RegimeAnalyzer

            logger.blank()
            logger.separator()
            logger.header("REGIME-BASED ANALYSIS")
            logger.separator()
            logger.info("Analyzing performance across market regimes...")
            logger.blank()

            # Get portfolio returns
            returns = portfolio.returns()

            if len(returns) == 0:
                logger.warning("No returns available for regime analysis")
                return None

            # Get market data (need to resample to daily for regime detection)
            if self._last_market_data is None or self._last_symbols is None:
                logger.warning("Market data not available for regime analysis")
                return None

            # Use the first symbol for regime detection
            primary_symbol = self._last_symbols[0]
            market_prices = self._last_market_data.xs(primary_symbol, level='symbol')['close']

            # Resample to daily to avoid too many regime changes
            logger.info("Resampling market data to daily frequency for regime detection...")
            daily_prices = market_prices.resample('D').last().dropna()

            # Create regime analyzer
            analyzer = RegimeAnalyzer(
                trend_lookback=60,
                vol_lookback=20,
                drawdown_threshold=10.0
            )

            # Analyze by regime
            regime_results = analyzer.analyze(
                portfolio_returns=returns,
                market_prices=daily_prices,
                trades=None
            )

            # Store results in portfolio object (Level 4: for GUI/export access)
            portfolio.regime_analysis = regime_results

            # Results are printed automatically by analyzer.analyze()

            return regime_results

        except ImportError:
            logger.warning("Regime analysis module not available")
            return None
        except Exception as e:
            logger.warning(f"Could not perform regime analysis: {e}")
            return None

    def _generate_windows(
        self,
        start_date: str,
        end_date: str,
        window_days: int,
        step_days: int
    ) -> List[tuple]:
        """
        Generate rolling window date ranges based on TRADING DAYS.

        Uses MarketCalendar to ensure windows respect actual trading days,
        not calendar days. This is critical for accurate backtesting.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            window_days: Number of TRADING days per window (e.g., 60 = ~3 months)
            step_days: Number of TRADING days to step forward

        Returns:
            List of (window_start, window_end) tuples as YYYY-MM-DD strings
        """
        calendar = MarketCalendar()

        # Get all trading days in range
        trading_days = calendar.get_trading_days(start_date, end_date)

        if len(trading_days) < window_days:
            logger.warning(
                f"Only {len(trading_days)} trading days available, "
                f"but window_days={window_days}. Returning single window."
            )
            if len(trading_days) > 0:
                return [(
                    trading_days[0].strftime('%Y-%m-%d'),
                    trading_days[-1].strftime('%Y-%m-%d')
                )]
            return []

        windows = []
        i = 0
        while i + window_days <= len(trading_days):
            window_start = trading_days[i].strftime('%Y-%m-%d')
            window_end = trading_days[i + window_days - 1].strftime('%Y-%m-%d')
            windows.append((window_start, window_end))
            i += step_days

        logger.info(f"Generated {len(windows)} rolling windows of {window_days} trading days each")

        return windows

    def _run_rolling(
        self,
        strategy: BaseStrategy,
        symbols: List[str],
        start_date: str,
        end_date: str,
        window_days: int,
        step_days: int,
        price_type: str
    ) -> RollingWindowResults:
        """
        Execute backtest with rolling windows.

        Memory efficient: Only holds current window in memory (~1-2 GB per window).
        LRU cache in StreamingDataLoader accelerates overlapping window loads.

        Args:
            strategy: Strategy instance to backtest
            symbols: List of symbols to trade
            start_date: Overall start date (YYYY-MM-DD)
            end_date: Overall end date (YYYY-MM-DD)
            window_days: Number of trading days per window
            step_days: Number of trading days to step forward
            price_type: Price column to use for signals

        Returns:
            RollingWindowResults with stitched equity curve and per-window stats
        """
        # 1. Generate window schedule (trading days)
        windows = self._generate_windows(start_date, end_date, window_days, step_days)

        if not windows:
            raise ValueError(f"No valid windows generated for {start_date} to {end_date}")

        results = RollingWindowResults()

        logger.info(f"Running {len(windows)} rolling windows...")

        for window_idx, (window_start, window_end) in enumerate(windows):
            logger.info(f"Window {window_idx + 1}/{len(windows)}: {window_start} to {window_end}")

            # 2. Load ONLY this window's data (LRU cache helps with overlap)
            window_data = {}
            for symbol, pl_df in self.data_loader.iter_symbols(
                symbols, window_start, window_end
            ):
                if not pl_df.is_empty():
                    window_data[symbol] = pl_df.to_pandas().set_index('timestamp')

            if not window_data:
                logger.warning(f"No data for window {window_idx + 1}, skipping")
                continue

            # 3. Generate signals for this window
            signals = {}
            for symbol, data in window_data.items():
                try:
                    if isinstance(strategy, LongShortStrategy):
                        signal_tuple = strategy.generate_signals(data)
                    else:
                        entries, exits = strategy.generate_signals(data)
                        signal_tuple = (entries, exits)
                    signals[symbol] = signal_tuple
                except Exception as e:
                    logger.warning(f"Signal generation failed for {symbol}: {e}")

            if not signals:
                logger.warning(f"No signals generated for window {window_idx + 1}, skipping")
                continue

            # 4. Simulate portfolio for this window
            try:
                window_portfolio = self._simulate_window(
                    window_data=window_data,
                    signals=signals,
                    price_type=price_type
                )
                results.add_window(window_start, window_end, window_portfolio)
            except Exception as e:
                logger.error(f"Window {window_idx + 1} simulation failed: {e}")

            # Window data garbage collected here (~1-2 GB freed)

        # 5. Stitch results
        if results.windows:
            results.stitch_equity_curves()
            results.compute_combined_stats()

            logger.success(f"Rolling backtest complete: {len(results.windows)} windows")
            if results.combined_stats:
                sharpe = results.combined_stats.get('sharpe', 0)
                total_ret = results.combined_stats.get('total_return', 0)
                logger.metric(f"Combined Sharpe: {sharpe:.2f}")
                logger.metric(f"Combined Return: {total_ret:.1%}")
        else:
            logger.warning("No windows completed successfully")

        return results

    def _simulate_window(
        self,
        window_data: Dict[str, pd.DataFrame],
        signals: Dict[str, tuple],
        price_type: str
    ) -> Portfolio:
        """
        Simulate a single window's portfolio.

        Args:
            window_data: Dict mapping symbol -> DataFrame with OHLCV
            signals: Dict mapping symbol -> (entries, exits) or 4-tuple for long-short
            price_type: Price column to use

        Returns:
            Portfolio object for this window
        """
        # For single symbol, use simple path
        if len(window_data) == 1:
            symbol = list(window_data.keys())[0]
            data = window_data[symbol]
            signal_tuple = signals[symbol]

            if len(signal_tuple) == 4:
                entries, exits, short_entries, short_exits = signal_tuple
            else:
                entries, exits = signal_tuple
                short_entries = pd.Series(False, index=entries.index)
                short_exits = pd.Series(False, index=entries.index)

            entries = entries.fillna(False).astype(bool)
            exits = exits.fillna(False).astype(bool)
            short_entries = short_entries.fillna(False).astype(bool)
            short_exits = short_exits.fillna(False).astype(bool)

            return from_signals(
                close=data[price_type],
                entries=entries,
                exits=exits,
                short_entries=short_entries,
                short_exits=short_exits,
                init_cash=self.initial_capital,
                fees=self.fees,
                slippage=self.slippage,
                freq=self.freq,
                market_hours_only=self.market_hours_only,
                risk_config=self.risk_config,
                price_data=data,
                allow_shorts=self.allow_shorts
            )

        # Multi-symbol: use MultiAssetPortfolio
        from src.backtesting.engine.multi_asset_portfolio import MultiAssetPortfolio

        symbols = list(window_data.keys())
        prices_dict = {}
        all_entries = {}
        all_exits = {}

        for symbol, data in window_data.items():
            prices_dict[symbol] = data[price_type]

            if symbol in signals:
                signal_tuple = signals[symbol]
                if len(signal_tuple) >= 2:
                    entries, exits = signal_tuple[0], signal_tuple[1]
                    all_entries[symbol] = entries.fillna(False).astype(bool)
                    all_exits[symbol] = exits.fillna(False).astype(bool)

        prices_df = pd.DataFrame(prices_dict)
        entries_df = pd.DataFrame(all_entries)
        exits_df = pd.DataFrame(all_exits)

        # Build multi-index data
        dfs_list = []
        for symbol, data in window_data.items():
            df_copy = data.copy()
            df_copy['symbol'] = symbol
            dfs_list.append(df_copy)

        combined = pd.concat(dfs_list)
        combined = combined.reset_index().set_index(['symbol', 'timestamp']).sort_index()

        portfolio = MultiAssetPortfolio(
            symbols=symbols,
            prices=prices_df,
            entries=entries_df,
            exits=exits_df,
            init_cash=self.initial_capital,
            fees=self.fees,
            slippage=self.slippage,
            freq=self.freq,
            market_hours_only=self.market_hours_only,
            risk_config=self.risk_config,
            price_data=combined
        )

        return portfolio
