"""
Integration helpers for connecting visualization with backtesting engine.
"""

import pandas as pd
import vectorbt as vbt
from typing import List, Optional, Any
from pathlib import Path

from src.visualization.config import VisualizationConfig, LogLevel
from src.visualization.logger import TradeLogger, TradeEvent
from src.visualization.charts.candlestick import CandlestickChart
from src.visualization.reports.report_generator import ReportGenerator
from src.visualization.utils.output_manager import OutputManager
from src.backtesting.engine.data_loader import DataLoader


class BacktestVisualizer:
    """
    Integrates visualization with backtest results.
    """

    def __init__(self, config: VisualizationConfig):
        """
        Initialize visualizer.

        Args:
            config: Visualization configuration
        """
        self.config = config
        self.logger = TradeLogger(log_level=config.log_level)
        self.output_mgr = OutputManager(config.log_dir)

    def visualize_backtest(
        self,
        portfolio: Any,  # vbt.Portfolio - using Any due to incomplete type stubs
        strategy_name: str,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float,
        fees: float,
        price_data: Optional[pd.DataFrame] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Create comprehensive visualization for backtest results.

        Args:
            portfolio: VectorBT Portfolio object
            strategy_name: Name of the strategy
            symbols: List of symbols traded
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Initial capital
            fees: Transaction fees
            price_data: Optional price data for charts (if not provided, will load from storage)
        """
        if not self.config.is_enabled():
            return

        # Use provided output_dir or create new one
        if output_dir is not None:
            run_dir = Path(output_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            self.output_mgr.run_dir = run_dir
        else:
            run_dir = self.output_mgr.create_run_directory(
                strategy_name=strategy_name,
                symbols=symbols
            )

        self.logger.log_message(
            f"\nGenerating visualization for {strategy_name} on {', '.join(symbols)}...",
            LogLevel.MINIMAL
        )

        self._extract_trades_from_portfolio(portfolio, symbols)

        if self.config.save_logs:
            self._save_logs()

        if self.config.save_charts:
            self._create_charts(portfolio, symbols, start_date, end_date, price_data)

        stats = portfolio.stats()
        if stats is not None:
            performance_stats = dict(stats)
        else:
            performance_stats = {}

        trade_summary = self.logger.get_trade_summary()

        ReportGenerator.generate_summary_report(
            strategy_name=strategy_name,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            fees=fees,
            performance_stats=performance_stats,
            trade_summary=trade_summary,
            output_path=self.output_mgr.get_report_path('summary_report.txt')
        )

        ReportGenerator.generate_json_report(
            strategy_name=strategy_name,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            fees=fees,
            performance_stats=performance_stats,
            trade_summary=trade_summary,
            output_path=self.output_mgr.get_report_path('backtest_results.json')
        )

        trades_df = self.logger.get_trades_dataframe()
        if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
            ReportGenerator.generate_trade_log_summary(
                trades_df=trades_df,
                output_path=self.output_mgr.get_report_path('trade_summary.txt')
            )

        self.logger.log_message(
            f"\n✓ Visualization complete! Output saved to: {run_dir}",
            LogLevel.MINIMAL
        )

    def _extract_trades_from_portfolio(self, portfolio: Any, symbols: List[str]):
        """Extract trade events from custom Portfolio object."""
        try:
            # Get trades from portfolio (custom Portfolio uses a list of dicts)
            if not hasattr(portfolio, 'trades'):
                self.logger.log_message("No trades attribute found on portfolio.", LogLevel.NORMAL)
                return

            trades_list = portfolio.trades  # type: ignore[attr-defined]

            # Handle custom Portfolio format (list of trade dicts)
            if isinstance(trades_list, list):
                if len(trades_list) == 0:
                    self.logger.log_message("No trades executed.", LogLevel.NORMAL)
                    return

                # Get equity curve for portfolio values
                equity_curve = portfolio.equity_curve if hasattr(portfolio, 'equity_curve') else None

                # Process trades from custom format
                for trade in trades_list:
                    trade_type = trade.get('type')
                    timestamp = trade.get('timestamp')
                    price = trade.get('price', 0.0)
                    shares = trade.get('shares', 0.0)

                    # Get portfolio value at this timestamp
                    if equity_curve is not None and timestamp in equity_curve.index:
                        portfolio_value = float(equity_curve.loc[timestamp])
                    else:
                        portfolio_value = 0.0

                    # Determine action and create event
                    if trade_type == 'entry':
                        action = 'BUY'
                        cost = trade.get('cost', price * shares)
                        cash = portfolio_value - (price * shares)  # Approximate
                        position_value = price * shares

                        event = TradeEvent(
                            timestamp=timestamp,
                            symbol=symbols[0] if symbols else 'UNKNOWN',
                            action=action,
                            price=price,
                            size=shares,
                            portfolio_value=portfolio_value,
                            cash=cash,
                            position_value=position_value
                        )
                        self.logger.log_trade(event)

                    elif trade_type == 'exit':
                        action = 'SELL'
                        proceeds = trade.get('proceeds', price * shares)
                        cash = portfolio_value  # After exit, position is closed
                        position_value = 0.0

                        event = TradeEvent(
                            timestamp=timestamp,
                            symbol=symbols[0] if symbols else 'UNKNOWN',
                            action=action,
                            price=price,
                            size=shares,
                            portfolio_value=portfolio_value,
                            cash=cash,
                            position_value=position_value
                        )
                        self.logger.log_trade(event)

                return

            # If not a list, log warning
            self.logger.log_message(
                f"Warning: Unexpected trades format. Type: {type(trades_list)}",
                LogLevel.MINIMAL
            )

        except Exception as e:
            self.logger.log_message(f"Warning: Could not extract trades: {e}", LogLevel.MINIMAL)

    def _save_logs(self):
        """Save log files."""
        self.logger.save_log(self.output_mgr.get_log_path('trade_log.txt'))
        self.logger.save_trades_csv(self.output_mgr.get_log_path('trades.csv'))

    def _create_charts(
        self,
        portfolio: Any,
        symbols: List[str],
        start_date: str,
        end_date: str,
        price_data: Optional[pd.DataFrame] = None
    ):
        """Create visualization charts."""
        trades_df = self.logger.get_trades_dataframe()

        # Ensure trades_df is a DataFrame (defensive check)
        if not isinstance(trades_df, pd.DataFrame):
            self.logger.log_message(
                f"Warning: get_trades_dataframe returned {type(trades_df)}, expected DataFrame",
                LogLevel.MINIMAL
            )
            trades_df = pd.DataFrame()

        for symbol in symbols:
            try:
                if price_data is None:
                    loader = DataLoader()
                    symbol_data = loader.load_single_symbol(symbol, start_date, end_date)
                else:
                    if len(symbols) == 1:
                        symbol_data = price_data
                    else:
                        symbol_data = price_data.xs(symbol, level='symbol')  # type: ignore[assignment]

                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in symbol_data.columns for col in required_cols):
                    self.logger.log_message(
                        f"Warning: Missing required columns for {symbol}",
                        LogLevel.MINIMAL
                    )
                    continue

                # Get trades for this symbol (with defensive checks)
                symbol_trades = pd.DataFrame()
                if isinstance(trades_df, pd.DataFrame) and not trades_df.empty and 'symbol' in trades_df.columns:
                    try:
                        filtered = trades_df[trades_df['symbol'] == symbol]
                        if isinstance(filtered, pd.DataFrame):
                            symbol_trades = filtered
                        else:
                            self.logger.log_message(
                                f"Warning: Filtered trades is {type(filtered)}, expected DataFrame",
                                LogLevel.MINIMAL
                            )
                            symbol_trades = pd.DataFrame()
                    except Exception as e:
                        self.logger.log_message(
                            f"Warning: Error filtering trades for {symbol}: {e}",
                            LogLevel.MINIMAL
                        )
                        symbol_trades = pd.DataFrame()

                # Final defensive check before passing to create_chart
                if not isinstance(symbol_trades, pd.DataFrame):
                    self.logger.log_message(
                        f"Warning: symbol_trades is {type(symbol_trades)}, setting to empty DataFrame",
                        LogLevel.MINIMAL
                    )
                    symbol_trades = pd.DataFrame()

                # Create lightweight-charts chart
                # Use proper defensive check that won't fail if symbol_trades is somehow not a DataFrame
                try:
                    if isinstance(symbol_trades, pd.DataFrame) and not symbol_trades.empty:
                        trades_to_pass = symbol_trades
                    else:
                        trades_to_pass = None
                except Exception as e:
                    self.logger.log_message(
                        f"Warning: Error checking symbol_trades: {e}. Type: {type(symbol_trades)}",
                        LogLevel.MINIMAL
                    )
                    trades_to_pass = None

                # Get charts directory path (without filename)
                charts_dir = self.output_mgr.get_chart_path("")
                if charts_dir.name == "":
                    # Remove empty path component
                    charts_dir = charts_dir.parent

                # Save charts in all formats (HTML, PNG, SVG)
                chart_results = CandlestickChart.save_all_formats(
                    data=symbol_data,
                    output_dir=charts_dir,
                    symbol=symbol,
                    trades_df=trades_to_pass,
                    title=f"{symbol} Backtest Results",
                    show_volume=True,
                    max_data_points_html=10000,  # More points for interactive HTML
                    max_data_points_static=5000,  # Fewer points for static charts
                    style='yahoo',
                    figsize=(14, 8),
                    dpi=150  # Higher DPI for better quality PNG
                )

                # Check if at least one format was saved successfully
                successful_formats = [fmt for fmt, path in chart_results.items() if path is not None]
                if not successful_formats:
                    self.logger.log_message(
                        f"Warning: Failed to save any chart format for {symbol}",
                        LogLevel.MINIMAL
                    )
                else:
                    self.logger.log_message(
                        f"Saved {symbol} charts in formats: {', '.join(successful_formats)}",
                        LogLevel.NORMAL
                    )

            except Exception as e:
                import traceback
                self.logger.log_message(
                    f"Warning: Could not create chart for {symbol}: {e}",
                    LogLevel.MINIMAL
                )
                # Log full traceback for debugging
                self.logger.log_message(
                    f"Full error: {traceback.format_exc()}",
                    LogLevel.NORMAL
                )
