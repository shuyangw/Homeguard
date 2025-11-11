"""
Regime-Aware Optimization Framework

Integrates regime detection with parameter optimization to find regime-specific
optimal parameters. This dramatically improves strategy performance by using
different parameters in different market conditions.

Key Insight:
- Moving Average Crossover works in TRENDING markets (Bull/Bear)
- Mean Reversion works in SIDEWAYS/RANGING markets
- High Volatility requires different parameters than Low Volatility

Instead of finding one set of "optimal" parameters for all conditions,
regime-aware optimization finds the best parameters for each regime separately.

Example:
    optimizer = RegimeAwareOptimizer(engine, base_optimizer)
    results = optimizer.optimize(
        strategy_class=MovingAverageCrossover,
        param_grid={'fast': [10, 20], 'slow': [50, 100]},
        symbols='AAPL',
        start_date='2020-01-01',
        end_date='2024-01-01',
        regime_type='trend'  # 'trend', 'volatility', or 'drawdown'
    )

This returns optimal parameters for each regime:
    {
        'BULL': {'fast': 20, 'slow': 100, 'sharpe': 1.8},
        'BEAR': {'fast': 30, 'slow': 150, 'sharpe': 1.2},
        'SIDEWAYS': {'fast': 10, 'slow': 50, 'sharpe': -0.5}  # Don't trade!
    }

In live trading: Detect current regime → Use regime-specific parameters
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.optimization.base_optimizer import BaseOptimizer
from backtesting.regimes.detector import TrendDetector, VolatilityDetector, DrawdownDetector
from backtesting.regimes.analyzer import RegimeAnalyzer
from utils import logger


class RegimeAwareOptimizer:
    """
    Optimizes strategy parameters separately for each market regime.

    Uses existing regime detection infrastructure to split historical data
    into regime periods, then optimizes parameters for each regime.
    """

    def __init__(self, engine: BacktestEngine, base_optimizer: BaseOptimizer):
        """
        Initialize regime-aware optimizer.

        Args:
            engine: BacktestEngine instance
            base_optimizer: Underlying optimizer (Grid, Random, Bayesian, etc.)
        """
        self.engine = engine
        self.base_optimizer = base_optimizer

    def optimize(
        self,
        strategy_class: type,
        param_space: Dict[str, Any],
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        regime_type: str = 'trend',  # 'trend', 'volatility', 'drawdown'
        metric: str = 'sharpe_ratio',
        min_regime_days: int = 60,  # Minimum days per regime for valid optimization
        export_results: bool = True,
        **optimizer_kwargs
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters for each market regime.

        Args:
            strategy_class: Strategy to optimize
            param_space: Parameter grid/ranges for optimizer
            symbols: Symbol(s) to optimize on
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            regime_type: Type of regime detection
                'trend': BULL, BEAR, SIDEWAYS
                'volatility': HIGH, LOW
                'drawdown': DRAWDOWN, RECOVERY, CALM
            metric: Optimization metric
            min_regime_days: Minimum days in regime for optimization
            export_results: Export results to CSV
            **optimizer_kwargs: Additional args for base optimizer

        Returns:
            Dictionary with regime-specific results:
                - regime_params: Best parameters per regime
                - regime_performance: Performance metrics per regime
                - regime_stats: Regime statistics (duration, frequency)
                - recommendations: Which regimes to trade, which to avoid
        """
        logger.blank()
        logger.separator()
        logger.header("REGIME-AWARE OPTIMIZATION")
        logger.separator()
        logger.blank()

        logger.info(f"Strategy: {strategy_class.__name__}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Regime Type: {regime_type.upper()}")
        logger.blank()

        # Detect regimes in historical data
        logger.info("Step 1: Detecting market regimes...")
        logger.blank()
        regimes = self._detect_regimes(symbols, start_date, end_date, regime_type)

        # Analyze regime distribution
        regime_stats = self._analyze_regime_distribution(
            regimes, min_regime_days
        )

        logger.info("Regime Distribution:")
        for regime_name, stats in regime_stats.items():
            logger.metric(f"  {regime_name}:")
            logger.info(f"    Total days: {stats['total_days']}")
            logger.info(f"    Percentage: {stats['percentage']:.1f}%")
            logger.info(f"    Periods: {stats['n_periods']}")

            if stats['total_days'] < min_regime_days:
                logger.warning(f"    ⚠️  Insufficient data (< {min_regime_days} days)")

        logger.blank()

        # Optimize for each regime
        logger.info("Step 2: Optimizing parameters per regime...")
        logger.blank()

        regime_results = {}

        for regime_name, stats in regime_stats.items():
            if stats['total_days'] < min_regime_days:
                logger.warning(f"Skipping {regime_name}: Insufficient data")
                regime_results[regime_name] = {
                    'skipped': True,
                    'reason': f'Insufficient data ({stats["total_days"]} days < {min_regime_days})'
                }
                continue

            logger.separator()
            logger.header(f"OPTIMIZING FOR {regime_name} REGIME")
            logger.separator()
            logger.blank()

            try:
                # Get date ranges for this regime
                regime_periods = self._get_regime_periods(regimes, regime_name)

                logger.info(f"Optimizing on {len(regime_periods)} {regime_name} periods...")
                logger.info(f"Total days: {stats['total_days']}")
                logger.blank()

                # Optimize on regime periods
                result = self._optimize_on_regime(
                    strategy_class=strategy_class,
                    param_space=param_space,
                    symbols=symbols,
                    regime_periods=regime_periods,
                    metric=metric,
                    **optimizer_kwargs
                )

                regime_results[regime_name] = {
                    'best_params': result['best_params'],
                    'best_value': result['best_value'],
                    'metric': metric,
                    'total_days': stats['total_days'],
                    'n_periods': stats['n_periods']
                }

                logger.success(f"{regime_name} Optimization Complete!")
                logger.profit(f"  Best {metric}: {result['best_value']:.4f}")
                logger.metric(f"  Best parameters: {result['best_params']}")

            except Exception as e:
                logger.error(f"{regime_name} optimization failed: {e}")
                regime_results[regime_name] = {
                    'error': str(e)
                }

            logger.blank()

        # Generate recommendations
        recommendations = self._generate_recommendations(
            regime_results, strategy_class.__name__
        )

        # Compile final analysis
        analysis = {
            'strategy': strategy_class.__name__,
            'symbols': symbols,
            'regime_type': regime_type,
            'regime_stats': regime_stats,
            'regime_params': regime_results,
            'recommendations': recommendations,
            'metric': metric
        }

        # Export if requested
        if export_results:
            self._export_results(analysis, strategy_class, symbols)

        # Print summary
        self._print_summary(analysis)

        return analysis

    def _detect_regimes(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        regime_type: str
    ) -> pd.DataFrame:
        """Detect market regimes using appropriate detector."""
        # Get price data
        if isinstance(symbols, list):
            symbol = symbols[0]  # Use first symbol for regime detection
        else:
            symbol = symbols

        # Load data
        from backtesting.engine.data_loader import DataLoader
        loader = DataLoader()
        data = loader.load_data([symbol], start_date, end_date)

        if data.empty:
            raise ValueError(f"No data available for {symbol}")

        # Extract close prices
        prices = data['close'].unstack(level='symbol')[symbol]

        # Detect regimes based on type
        if regime_type == 'trend':
            detector = TrendDetector()
            regimes = detector.detect(prices)
        elif regime_type == 'volatility':
            detector = VolatilityDetector()
            regimes = detector.detect(prices)
        elif regime_type == 'drawdown':
            detector = DrawdownDetector()
            regimes = detector.detect(prices)
        else:
            raise ValueError(f"Unknown regime type: {regime_type}")

        return regimes

    def _analyze_regime_distribution(
        self,
        regimes: pd.Series,
        min_days: int
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze distribution of regimes in data."""
        total_days = len(regimes)
        regime_names = regimes.unique()

        stats = {}
        for regime in regime_names:
            regime_mask = regimes == regime
            regime_days = regime_mask.sum()

            # Count number of continuous periods
            regime_changes = (regime_mask != regime_mask.shift()).cumsum()
            n_periods = regime_changes[regime_mask].nunique()

            stats[regime] = {
                'total_days': regime_days,
                'percentage': (regime_days / total_days) * 100,
                'n_periods': n_periods,
                'sufficient_data': regime_days >= min_days
            }

        return stats

    def _get_regime_periods(
        self,
        regimes: pd.Series,
        regime_name: str
    ) -> List[tuple]:
        """Get list of (start_date, end_date) tuples for a regime."""
        regime_mask = regimes == regime_name

        # Find continuous periods
        regime_groups = (regime_mask != regime_mask.shift()).cumsum()

        periods = []
        for group_id in regime_groups[regime_mask].unique():
            group_dates = regimes.index[regime_groups == group_id]
            if len(group_dates) > 0:
                start = group_dates[0].strftime('%Y-%m-%d')
                end = group_dates[-1].strftime('%Y-%m-%d')
                periods.append((start, end))

        return periods

    def _optimize_on_regime(
        self,
        strategy_class: type,
        param_space: Dict[str, Any],
        symbols: Union[str, List[str]],
        regime_periods: List[tuple],
        metric: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize on specific regime periods.

        Combines multiple regime periods into single optimization.
        """
        # For now, optimize on the longest regime period
        # TODO: Could optimize on concatenated periods
        longest_period = max(regime_periods, key=lambda p: (
            pd.to_datetime(p[1]) - pd.to_datetime(p[0])
        ).days)

        start_date, end_date = longest_period

        logger.info(f"Using longest period: {start_date} to {end_date}")

        # Run optimization
        return self.base_optimizer.optimize(
            strategy_class=strategy_class,
            param_grid=param_space if hasattr(self.base_optimizer, 'optimize_parallel') else None,
            param_ranges=param_space if not hasattr(self.base_optimizer, 'optimize_parallel') else None,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            metric=metric,
            export_results=False,
            **kwargs
        )

    def _generate_recommendations(
        self,
        regime_results: Dict[str, Dict],
        strategy_name: str
    ) -> Dict[str, str]:
        """Generate trading recommendations based on regime performance."""
        recommendations = {}

        for regime, result in regime_results.items():
            if 'skipped' in result:
                recommendations[regime] = f"SKIP: {result['reason']}"
            elif 'error' in result:
                recommendations[regime] = f"ERROR: {result['error']}"
            else:
                sharpe = result['best_value']

                if sharpe > 1.0:
                    recommendations[regime] = f"✅ TRADE: Excellent performance (Sharpe {sharpe:.2f})"
                elif sharpe > 0.5:
                    recommendations[regime] = f"✅ TRADE: Good performance (Sharpe {sharpe:.2f})"
                elif sharpe > 0.0:
                    recommendations[regime] = f"⚠️  MARGINAL: Low profitability (Sharpe {sharpe:.2f})"
                else:
                    recommendations[regime] = f"❌ AVOID: Negative performance (Sharpe {sharpe:.2f})"

        return recommendations

    def _export_results(
        self,
        analysis: Dict[str, Any],
        strategy_class: type,
        symbols: Union[str, List[str]]
    ):
        """Export regime-aware optimization results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        symbol_str = symbols if isinstance(symbols, str) else '_'.join(symbols)
        output_dir = Path(f"C:/Users/qwqw1/Dropbox/cs/stonk/logs/{timestamp}_RegimeAware_{strategy_class.__name__}_{symbol_str}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export regime parameters
        params_data = []
        for regime, result in analysis['regime_params'].items():
            if 'best_params' in result:
                row = {
                    'regime': regime,
                    'sharpe': result['best_value'],
                    'total_days': result.get('total_days', 0),
                    'n_periods': result.get('n_periods', 0)
                }
                row.update({f'param_{k}': v for k, v in result['best_params'].items()})
                params_data.append(row)

        if params_data:
            df = pd.DataFrame(params_data)
            df.to_csv(output_dir / 'regime_parameters.csv', index=False)

        # Export summary
        with open(output_dir / 'regime_optimization_summary.txt', 'w') as f:
            f.write("REGIME-AWARE OPTIMIZATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Strategy: {analysis['strategy']}\n")
            f.write(f"Symbols: {analysis['symbols']}\n")
            f.write(f"Regime Type: {analysis['regime_type']}\n\n")
            f.write("REGIME PARAMETERS:\n\n")

            for regime, result in analysis['regime_params'].items():
                f.write(f"{regime}:\n")
                if 'best_params' in result:
                    f.write(f"  Sharpe: {result['best_value']:.4f}\n")
                    f.write(f"  Parameters: {result['best_params']}\n")
                else:
                    f.write(f"  {result.get('reason', result.get('error', 'Unknown'))}\n")
                f.write("\n")

            f.write("\nRECOMMENDATIONS:\n\n")
            for regime, rec in analysis['recommendations'].items():
                f.write(f"{regime}: {rec}\n")

        logger.info(f"Results exported to: {output_dir}")

    def _print_summary(self, analysis: Dict[str, Any]):
        """Print regime-aware optimization summary."""
        logger.blank()
        logger.separator()
        logger.header("REGIME-AWARE OPTIMIZATION SUMMARY")
        logger.separator()
        logger.blank()

        logger.header("REGIME-SPECIFIC PARAMETERS")
        logger.blank()

        for regime, result in analysis['regime_params'].items():
            logger.metric(f"{regime}:")
            if 'best_params' in result:
                logger.profit(f"  Sharpe: {result['best_value']:.4f}")
                logger.info(f"  Parameters: {result['best_params']}")
                logger.info(f"  Data: {result['total_days']} days across {result['n_periods']} periods")
            elif 'skipped' in result:
                logger.warning(f"  {result['reason']}")
            else:
                logger.error(f"  ERROR: {result.get('error', 'Unknown error')}")
            logger.blank()

        logger.separator()
        logger.header("TRADING RECOMMENDATIONS")
        logger.separator()
        logger.blank()

        for regime, rec in analysis['recommendations'].items():
            if '✅' in rec:
                logger.profit(f"{regime}: {rec}")
            elif '⚠️' in rec:
                logger.warning(f"{regime}: {rec}")
            elif '❌' in rec:
                logger.error(f"{regime}: {rec}")
            else:
                logger.info(f"{regime}: {rec}")

        logger.blank()
        logger.separator()

        logger.header("HOW TO USE IN LIVE TRADING")
        logger.separator()
        logger.blank()

        logger.info("1. Detect current market regime (use RegimeDetector)")
        logger.info("2. Look up regime-specific parameters from table above")
        logger.info("3. Use those parameters for strategy")
        logger.info("4. Monitor regime changes and update parameters accordingly")
        logger.info("5. Avoid trading in regimes marked with ❌ (negative Sharpe)")
        logger.blank()

        logger.info("Example code:")
        logger.blank()
        logger.code("from backtesting.regimes.detector import TrendDetector")
        logger.code("detector = TrendDetector()")
        logger.code("current_regime = detector.detect_current(recent_prices)")
        logger.code("params = regime_params[current_regime]")
        logger.code("strategy = StrategyClass(**params)")
        logger.blank()

        logger.separator()
