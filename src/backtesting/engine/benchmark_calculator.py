"""
Benchmark calculations for backtesting performance comparison.

Provides buy-and-hold equity curves and outperformance metrics
to compare strategy performance against passive investing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from src.utils import logger


class BenchmarkCalculator:
    """
    Calculate benchmark performance metrics for strategy comparison.

    Supports:
    - Buy-and-hold equity curves for individual symbols
    - S&P 500 (SPY) benchmark comparison
    - Outperformance calculations
    """

    @staticmethod
    def calculate_buy_and_hold_equity(
        symbol: str,
        start_date: str,
        end_date: str,
        initial_capital: float,
        data_loader
    ) -> Optional[pd.Series]:
        """
        Calculate buy-and-hold equity curve for a symbol.

        Simulates buying the symbol at open on start_date and holding
        until end_date, tracking portfolio value over time.

        Args:
            symbol: Symbol to calculate buy-and-hold for
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Initial capital to invest
            data_loader: DataLoader instance to fetch data

        Returns:
            Series with DatetimeIndex and equity values, or None if data unavailable
        """
        try:
            # Load symbol data
            df = data_loader.load_single_symbol(symbol, start_date, end_date)

            if df.empty:
                logger.warning(f"No data available for {symbol} buy-and-hold calculation")
                return None

            # Get first price (buy price)
            first_price = df['open'].iloc[0]

            if first_price <= 0 or pd.isna(first_price):
                logger.warning(f"Invalid first price for {symbol}: {first_price}")
                return None

            # Calculate shares purchased
            shares = initial_capital / first_price

            # Calculate equity over time using close prices
            equity = df['close'] * shares

            # Rename series
            equity.name = f'{symbol}_buy_hold'

            return equity

        except Exception as e:
            logger.warning(f"Could not calculate buy-and-hold for {symbol}: {e}")
            return None

    @staticmethod
    def calculate_spy_benchmark(
        start_date: str,
        end_date: str,
        initial_capital: float,
        data_loader
    ) -> Optional[pd.Series]:
        """
        Calculate S&P 500 (SPY) buy-and-hold equity curve.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Initial capital to invest
            data_loader: DataLoader instance to fetch data

        Returns:
            Series with DatetimeIndex and equity values, or None if SPY not available
        """
        return BenchmarkCalculator.calculate_buy_and_hold_equity(
            'SPY', start_date, end_date, initial_capital, data_loader
        )

    @staticmethod
    def calculate_outperformance(
        strategy_equity: pd.Series,
        benchmark_equity: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate outperformance metrics comparing strategy vs benchmark.

        Args:
            strategy_equity: Strategy equity curve
            benchmark_equity: Benchmark equity curve

        Returns:
            Dict with outperformance metrics:
                - 'strategy_return_pct': Total return of strategy
                - 'benchmark_return_pct': Total return of benchmark
                - 'outperformance_pct': Difference in returns
                - 'alpha': Annualized excess return
        """
        if strategy_equity.empty or benchmark_equity.empty:
            return {
                'strategy_return_pct': 0.0,
                'benchmark_return_pct': 0.0,
                'outperformance_pct': 0.0,
                'alpha': 0.0
            }

        # Align series on common timestamps
        aligned = pd.DataFrame({
            'strategy': strategy_equity,
            'benchmark': benchmark_equity
        }).dropna()

        if aligned.empty:
            logger.warning("No overlapping data between strategy and benchmark")
            return {
                'strategy_return_pct': 0.0,
                'benchmark_return_pct': 0.0,
                'outperformance_pct': 0.0,
                'alpha': 0.0
            }

        # Calculate returns
        strategy_start = aligned['strategy'].iloc[0]
        strategy_end = aligned['strategy'].iloc[-1]
        benchmark_start = aligned['benchmark'].iloc[0]
        benchmark_end = aligned['benchmark'].iloc[-1]

        strategy_return = ((strategy_end - strategy_start) / strategy_start) * 100
        benchmark_return = ((benchmark_end - benchmark_start) / benchmark_start) * 100
        outperformance = strategy_return - benchmark_return

        # Calculate annualized alpha (excess return)
        num_days = len(aligned)
        if num_days > 0:
            years = num_days / 252  # Approximate trading days per year
            if years > 0:
                strategy_annual = ((strategy_end / strategy_start) ** (1 / years) - 1) * 100
                benchmark_annual = ((benchmark_end / benchmark_start) ** (1 / years) - 1) * 100
                alpha = strategy_annual - benchmark_annual
            else:
                alpha = outperformance
        else:
            alpha = 0.0

        return {
            'strategy_return_pct': float(strategy_return),
            'benchmark_return_pct': float(benchmark_return),
            'outperformance_pct': float(outperformance),
            'alpha': float(alpha)
        }

    @staticmethod
    def calculate_all_benchmarks(
        portfolios: Dict,
        symbols: list,
        start_date: str,
        end_date: str,
        initial_capital: float,
        data_loader,
        include_spy: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate benchmarks for all symbols and aggregate portfolio.

        Args:
            portfolios: Dict of {symbol: Portfolio}
            symbols: List of symbols tested
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Initial capital per symbol
            data_loader: DataLoader instance
            include_spy: Whether to include SPY benchmark (default True)

        Returns:
            Dict with structure:
            {
                'per_symbol': {
                    'AAPL': {
                        'buy_hold_equity': Series,
                        'strategy_equity': Series,
                        'metrics': {...}
                    },
                    ...
                },
                'spy': {
                    'equity': Series or None,
                    'metrics': {...} or None
                },
                'outperformers': ['AAPL', 'GOOGL'],  # Symbols that beat buy-and-hold
                'underperformers': ['MSFT']
            }
        """
        result = {
            'per_symbol': {},
            'spy': {'equity': None, 'metrics': None},
            'outperformers': [],
            'underperformers': []
        }

        # Calculate per-symbol benchmarks
        for symbol in symbols:
            portfolio = portfolios.get(symbol)
            if portfolio is None:
                continue

            # Get strategy equity
            try:
                if hasattr(portfolio, 'equity_curve'):
                    strategy_equity = portfolio.equity_curve
                elif hasattr(portfolio, 'value'):
                    strategy_equity = getattr(portfolio, 'value')
                    if callable(strategy_equity):
                        strategy_equity = strategy_equity()
                else:
                    logger.warning(f"Cannot extract equity for {symbol}")
                    continue

                if strategy_equity is None or strategy_equity.empty:
                    continue

            except Exception as e:
                logger.warning(f"Error extracting equity for {symbol}: {e}")
                continue

            # Calculate buy-and-hold
            buy_hold = BenchmarkCalculator.calculate_buy_and_hold_equity(
                symbol, start_date, end_date, initial_capital, data_loader
            )

            if buy_hold is None:
                continue

            # Calculate outperformance
            metrics = BenchmarkCalculator.calculate_outperformance(
                strategy_equity, buy_hold
            )

            result['per_symbol'][symbol] = {
                'buy_hold_equity': buy_hold,
                'strategy_equity': strategy_equity,
                'metrics': metrics
            }

            # Categorize as outperformer or underperformer
            if metrics['outperformance_pct'] > 0:
                result['outperformers'].append(symbol)
            else:
                result['underperformers'].append(symbol)

        # Calculate SPY benchmark if requested
        if include_spy:
            spy_equity = BenchmarkCalculator.calculate_spy_benchmark(
                start_date, end_date,
                initial_capital * len(symbols),  # Total capital across all symbols
                data_loader
            )

            if spy_equity is not None:
                result['spy']['equity'] = spy_equity
                logger.success(f"SPY benchmark data loaded for comparison")
            else:
                logger.warning("SPY data not available - skipping S&P 500 comparison")

        return result
