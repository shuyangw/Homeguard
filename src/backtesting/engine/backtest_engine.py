"""
Core backtesting engine with custom portfolio simulator.
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, Any, Optional, List, TYPE_CHECKING
from pathlib import Path

from backtesting.base.strategy import BaseStrategy
from backtesting.engine.data_loader import DataLoader
from backtesting.engine.portfolio_simulator import Portfolio, from_signals
from backtesting.utils.risk_config import RiskConfig
from visualization.reports.quantstats_reporter import QuantStatsReporter
from utils import logger

if TYPE_CHECKING:
    from backtesting.engine.multi_asset_portfolio import MultiAssetPortfolio

# Type alias for portfolio return types
PortfolioType = Union[Portfolio, 'MultiAssetPortfolio']


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
        enable_regime_analysis: bool = False
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
        """
        self.initial_capital = initial_capital
        self.fees = fees
        self.slippage = slippage
        self.freq = freq
        self.market_hours_only = market_hours_only
        self.risk_config = risk_config or RiskConfig.moderate()
        self.enable_regime_analysis = enable_regime_analysis
        self.data_loader = DataLoader()
        self.reporter = QuantStatsReporter(benchmark=benchmark)

        # Cache for regime analysis
        self._last_market_data = None
        self._last_symbols = None
        self._last_start_date = None
        self._last_end_date = None

    def run(
        self,
        strategy: BaseStrategy,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        price_type: str = 'close',
        portfolio_mode: str = 'single'
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

        Returns:
            Portfolio object with backtest results
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

        logger.separator()
        logger.blank()

        data = self.data_loader.load_symbols(symbols, start_date, end_date)

        # Cache for regime analysis
        if self.enable_regime_analysis:
            self._last_market_data = data
            self._last_symbols = symbols
            self._last_start_date = start_date
            self._last_end_date = end_date

        # Route based on portfolio mode
        if portfolio_mode == 'single':
            # SINGLE-SYMBOL MODE (backward compatible)
            if len(symbols) == 1:
                portfolio = self._run_single_symbol(strategy, data, symbols[0], price_type)
            else:
                # Sweep mode: run each symbol separately
                portfolio = self._run_multiple_symbols(strategy, data, symbols, price_type)

        elif portfolio_mode == 'multi':
            # MULTI-SYMBOL PORTFOLIO MODE (new)
            if len(symbols) == 1:
                # Single symbol in multi mode = same as single mode
                portfolio = self._run_single_symbol(strategy, data, symbols[0], price_type)
            else:
                # True multi-asset portfolio
                portfolio = self._run_multi_asset_portfolio(strategy, data, symbols, price_type)

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
        include_pdf: bool = False
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
            >>> portfolio = engine.run_and_report(
            ...     strategy=strategy,
            ...     symbols=['AAPL'],
            ...     start_date='2024-01-01',
            ...     end_date='2024-12-31',
            ...     output_dir=Path('logs/backtest_001')
            ... )
        """
        # Run the backtest
        portfolio = self.run(
            strategy=strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            price_type=price_type
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
            'fees': f"{self.fees:.4f}" if self.fees else "0.0000"
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
        data: pd.DataFrame,
        symbol: str,
        price_type: str
    ) -> Portfolio:
        """
        Run backtest for a single symbol.
        """
        symbol_data: pd.DataFrame = data.xs(symbol, level='symbol')  # type: ignore[assignment]

        entries, exits = strategy.generate_signals(symbol_data)
        price = symbol_data[price_type]

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
            price_data=symbol_data  # Pass OHLC data for ATR-based position sizing
        )

        return portfolio

    def _run_multiple_symbols(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        symbols: List[str],
        price_type: str
    ) -> Portfolio:
        """
        Run backtest for multiple symbols in sweep mode (each symbol separately).

        Note: This is the old behavior - each symbol tested independently.
        For true portfolio mode, use _run_multi_asset_portfolio.
        """
        logger.warning(f"Multi-symbol backtesting simplified to first symbol only.")
        return self._run_single_symbol(strategy, data, symbols[0], price_type)

    def _run_multi_asset_portfolio(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        symbols: List[str],
        price_type: str
    ) -> Portfolio:
        """
        Run backtest for multiple symbols in portfolio mode (hold all simultaneously).

        This is the new multi-asset portfolio implementation.
        """
        from backtesting.engine.multi_asset_portfolio import MultiAssetPortfolio

        logger.info(f"Running multi-asset portfolio with {len(symbols)} symbols")

        # Generate signals for each symbol
        all_entries = {}
        all_exits = {}

        for symbol in symbols:
            try:
                symbol_data: pd.DataFrame = data.xs(symbol, level='symbol')  # type: ignore[assignment]
                entries, exits = strategy.generate_signals(symbol_data)

                # Ensure boolean type
                entries = entries.fillna(False).astype(bool)
                exits = exits.fillna(False).astype(bool)

                all_entries[symbol] = entries
                all_exits[symbol] = exits

            except Exception as e:
                logger.warning(f"Could not generate signals for {symbol}: {e}")
                continue

        # Create DataFrames with symbols as columns
        entries_df = pd.DataFrame(all_entries)
        exits_df = pd.DataFrame(all_exits)

        # Extract prices for all symbols
        prices_dict = {}
        for symbol in symbols:
            try:
                symbol_data: pd.DataFrame = data.xs(symbol, level='symbol')  # type: ignore[assignment]
                prices_dict[symbol] = symbol_data[price_type]
            except Exception as e:
                logger.warning(f"Could not extract prices for {symbol}: {e}")
                continue

        prices_df = pd.DataFrame(prices_dict)

        # Create multi-asset portfolio
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
            price_data=data
        )

        return portfolio  # type: ignore[return-value]

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
        from backtesting.optimization.grid_search import GridSearchOptimizer

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
            from backtesting.regimes.analyzer import RegimeAnalyzer

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
