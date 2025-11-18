"""
Validation Framework - Standardized backtest validation pipeline.

This framework eliminates ~2,000 lines of duplicated validation code across
17+ validation scripts by providing a common pipeline for:
- Strategy configuration and initialization
- Data loading and preparation
- Backtest execution
- Performance analysis and reporting
- Result export (CSV, JSON, Markdown)

Usage:
    from frameworks.validation_framework import ValidationFramework
    from strategies.advanced.pairs_trading import PairsTrading

    # Initialize framework with config
    framework = ValidationFramework(config)

    # Run validation
    results = framework.run_validation(
        strategy_class=PairsTrading,
        symbols=["SPY", "IWM"],
        strategy_params={
            "entry_zscore": 2.0,
            "exit_zscore": 0.5
        }
    )

    # Export results
    framework.export_results(format='csv')
    framework.generate_report()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Tuple
from datetime import datetime
import json

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()

from src.backtesting.engine.backtest_engine import BacktestEngine
from src.utils.logger import logger
from utils.config_loader import get_nested
from utils.cli_args import get_output_filename


class ValidationFramework:
    """
    Standardized framework for strategy validation.

    This framework provides a common pipeline that eliminates code duplication
    across validation scripts.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validation framework.

        Args:
            config: Configuration dictionary (from config_loader)
        """
        self.config = config
        self.results = []
        self.summary_stats = {}
        self.backtest_start_time = None
        self.backtest_end_time = None

        # Extract common settings
        self.initial_cash = get_nested(config, 'backtest.initial_cash', 100000)
        self.commission = get_nested(config, 'costs.commission', 0.001)
        self.slippage = get_nested(config, 'costs.slippage', 0.0005)
        self.start_date = get_nested(config, 'backtest.start_date', '2020-01-01')
        self.end_date = get_nested(config, 'backtest.end_date', '2024-12-31')

        # Output settings
        self.save_trades = get_nested(config, 'output.save_trades', True)
        self.save_reports = get_nested(config, 'output.save_reports', True)
        self.output_dir = Path(get_nested(config, 'output.output_dir', 'output/validation'))
        self.reports_dir = Path(get_nested(config, 'output.reports_dir', 'reports/validation'))

        # Logging
        self.verbose = get_nested(config, 'logging.level', 'INFO') == 'DEBUG'

        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run_validation(
        self,
        strategy_class: Type,
        symbols: List[str],
        strategy_params: Optional[Dict[str, Any]] = None,
        description: str = "Strategy Validation"
    ) -> Dict[str, Any]:
        """
        Run backtest validation on a strategy.

        Args:
            strategy_class: Strategy class to instantiate
            symbols: List of symbols to trade
            strategy_params: Optional strategy parameters (overrides config)
            description: Description of this validation run

        Returns:
            Dictionary with validation results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"VALIDATION: {description}")
        logger.info(f"{'='*80}")
        logger.info(f"Strategy: {strategy_class.__name__}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Period: {self.start_date} to {self.end_date}")
        logger.info(f"Initial Cash: ${self.initial_cash:,.2f}")
        logger.info(f"Commission: {self.commission*100:.2f}%")
        logger.info(f"Slippage: {self.slippage*100:.2f}%")

        # Start timer
        self.backtest_start_time = datetime.now()

        try:
            # Merge strategy params from config and overrides
            final_params = get_nested(self.config, 'strategy.parameters', {}).copy()
            if strategy_params:
                final_params.update(strategy_params)

            # Initialize strategy
            strategy = strategy_class(**final_params) if final_params else strategy_class()

            # Create backtest engine
            engine = BacktestEngine(
                initial_capital=self.initial_cash,
                fees=self.commission,
                slippage=self.slippage
            )

            # Run backtest
            logger.info("\nRunning backtest...")
            portfolio = engine.run(
                strategy=strategy,
                symbols=symbols,
                start_date=self.start_date,
                end_date=self.end_date
            )

            # End timer
            self.backtest_end_time = datetime.now()
            duration = (self.backtest_end_time - self.backtest_start_time).total_seconds()

            # Get statistics
            stats = portfolio.stats()

            if stats is None:
                logger.error(f"Failed to generate stats for {symbols}")
                return self._create_error_result(symbols, description, "No stats generated")

            # Extract key metrics
            result = self._extract_metrics(stats, symbols, description, duration)

            # Print results
            self._print_results(result)

            # Store result
            self.results.append(result)

            return result

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            self.backtest_end_time = datetime.now()
            return self._create_error_result(symbols, description, str(e))

    def run_multiple_validations(
        self,
        strategy_class: Type,
        symbol_groups: List[Tuple[List[str], str]],
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run validation on multiple symbol groups.

        Args:
            strategy_class: Strategy class to instantiate
            symbol_groups: List of (symbols, description) tuples
            strategy_params: Optional strategy parameters

        Returns:
            List of result dictionaries

        Example:
            >>> symbol_groups = [
            ...     (["SPY", "IWM"], "Large vs Small Cap"),
            ...     (["QQQ", "DIA"], "Tech vs Industrials"),
            ... ]
            >>> results = framework.run_multiple_validations(PairsTrading, symbol_groups)
        """
        results = []
        total = len(symbol_groups)

        for i, (symbols, description) in enumerate(symbol_groups, 1):
            logger.info(f"\n\nValidation {i}/{total}")
            result = self.run_validation(
                strategy_class=strategy_class,
                symbols=symbols,
                strategy_params=strategy_params,
                description=description
            )
            results.append(result)

        # Calculate summary statistics
        self._calculate_summary_stats()

        return results

    def _extract_metrics(
        self,
        stats: Dict[str, Any],
        symbols: List[str],
        description: str,
        duration: float
    ) -> Dict[str, Any]:
        """Extract standardized metrics from backtest stats."""
        return {
            'symbols': symbols,
            'description': description,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_cash': self.initial_cash,
            'final_equity': float(stats.get('End Value', 0)),
            'total_return_pct': float(stats.get('Total Return [%]', 0)),
            'sharpe_ratio': float(stats.get('Sharpe Ratio', 0)),
            'sortino_ratio': float(stats.get('Sortino Ratio', 0)),
            'max_drawdown_pct': float(stats.get('Max Drawdown [%]', 0)),
            'win_rate_pct': float(stats.get('Win Rate [%]', 0)),
            'profit_factor': float(stats.get('Profit Factor', 0)),
            'total_trades': int(stats.get('Total Trades', 0)),
            'avg_trade_pct': float(stats.get('Avg Trade [%]', 0)),
            'duration_seconds': duration,
            'success': True,
            'error': None,
            'timestamp': datetime.now().isoformat()
        }

    def _create_error_result(
        self,
        symbols: List[str],
        description: str,
        error_msg: str
    ) -> Dict[str, Any]:
        """Create error result dictionary."""
        return {
            'symbols': symbols,
            'description': description,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_cash': self.initial_cash,
            'final_equity': 0,
            'total_return_pct': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown_pct': 0,
            'win_rate_pct': 0,
            'profit_factor': 0,
            'total_trades': 0,
            'avg_trade_pct': 0,
            'duration_seconds': 0,
            'success': False,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }

    def _print_results(self, result: Dict[str, Any]) -> None:
        """Print validation results in standardized format."""
        logger.info(f"\n{'='*80}")
        logger.info("BACKTEST RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Total Return:       {result['total_return_pct']:.2f}%")
        logger.info(f"Sharpe Ratio:       {result['sharpe_ratio']:.3f}")
        logger.info(f"Sortino Ratio:      {result['sortino_ratio']:.3f}")
        logger.info(f"Max Drawdown:       {result['max_drawdown_pct']:.2f}%")
        logger.info(f"Win Rate:           {result['win_rate_pct']:.2f}%")
        logger.info(f"Profit Factor:      {result['profit_factor']:.2f}")
        logger.info(f"Total Trades:       {result['total_trades']}")
        logger.info(f"Final Equity:       ${result['final_equity']:,.2f}")
        logger.info(f"Duration:           {result['duration_seconds']:.1f}s")
        logger.info(f"{'='*80}")

        # Assessment
        assessment = self._assess_performance(result)
        logger.info(f"\nASSESSMENT: {assessment}")
        logger.info(f"{'='*80}\n")

    def _assess_performance(self, result: Dict[str, Any]) -> str:
        """
        Assess strategy performance.

        Returns:
            Assessment string
        """
        if not result['success']:
            return "❌ FAILED"

        sharpe = result['sharpe_ratio']
        total_return = result['total_return_pct']
        win_rate = result['win_rate_pct']

        if sharpe >= 1.5 and total_return > 20 and win_rate > 55:
            return "🟢 EXCELLENT - Production Ready"
        elif sharpe >= 1.0 and total_return > 10:
            return "🟡 GOOD - Consider Production"
        elif sharpe >= 0.5 and total_return > 0:
            return "🟠 MODERATE - Needs Optimization"
        elif total_return > 0:
            return "🟡 MARGINAL - Barely Profitable"
        else:
            return "🔴 POOR - Not Viable"

    def _calculate_summary_stats(self) -> None:
        """Calculate summary statistics across all validations."""
        if not self.results:
            return

        successful_results = [r for r in self.results if r['success']]

        if not successful_results:
            self.summary_stats = {
                'total_validations': len(self.results),
                'successful_validations': 0,
                'failed_validations': len(self.results)
            }
            return

        returns = [r['total_return_pct'] for r in successful_results]
        sharpes = [r['sharpe_ratio'] for r in successful_results]
        drawdowns = [r['max_drawdown_pct'] for r in successful_results]

        self.summary_stats = {
            'total_validations': len(self.results),
            'successful_validations': len(successful_results),
            'failed_validations': len(self.results) - len(successful_results),
            'avg_return_pct': np.mean(returns),
            'median_return_pct': np.median(returns),
            'avg_sharpe': np.mean(sharpes),
            'median_sharpe': np.median(sharpes),
            'avg_drawdown_pct': np.mean(drawdowns),
            'best_return_pct': max(returns),
            'worst_return_pct': min(returns),
            'best_sharpe': max(sharpes),
            'worst_sharpe': min(sharpes)
        }

    def export_results(self, format: str = 'csv', filename: Optional[str] = None) -> Path:
        """
        Export validation results.

        Args:
            format: Export format ('csv', 'json', or 'both')
            filename: Optional custom filename (without extension)

        Returns:
            Path to exported file
        """
        if not self.results:
            logger.warning("No results to export")
            return None

        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_validation_results"

        if format in ['csv', 'both']:
            csv_path = self.output_dir / f"{filename}.csv"
            df = pd.DataFrame(self.results)
            df.to_csv(csv_path, index=False)
            logger.info(f"Results exported to: {csv_path}")

        if format in ['json', 'both']:
            json_path = self.output_dir / f"{filename}.json"
            with open(json_path, 'w') as f:
                json.dump({
                    'results': self.results,
                    'summary': self.summary_stats,
                    'config': self.config
                }, f, indent=2)
            logger.info(f"Results exported to: {json_path}")

        return csv_path if format == 'csv' else json_path

    def generate_report(self, filename: Optional[str] = None) -> Path:
        """
        Generate markdown validation report.

        Args:
            filename: Optional custom filename (without .md extension)

        Returns:
            Path to generated report
        """
        if not self.results:
            logger.warning("No results to report")
            return None

        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_validation_report"

        report_path = self.reports_dir / f"{filename}.md"

        # Calculate summary if not done
        if not self.summary_stats:
            self._calculate_summary_stats()

        # Generate report content
        report = self._generate_report_content()

        # Write report
        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Report generated: {report_path}")
        return report_path

    def _generate_report_content(self) -> str:
        """Generate markdown report content."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        report = f"""# Validation Report

**Generated**: {timestamp}
**Period**: {self.start_date} to {self.end_date}
**Initial Cash**: ${self.initial_cash:,.2f}
**Commission**: {self.commission*100:.2f}%
**Slippage**: {self.slippage*100:.2f}%

---

## Summary Statistics

"""
        if self.summary_stats:
            report += f"""
- **Total Validations**: {self.summary_stats['total_validations']}
- **Successful**: {self.summary_stats['successful_validations']}
- **Failed**: {self.summary_stats['failed_validations']}

### Performance Metrics

- **Average Return**: {self.summary_stats.get('avg_return_pct', 0):.2f}%
- **Median Return**: {self.summary_stats.get('median_return_pct', 0):.2f}%
- **Average Sharpe**: {self.summary_stats.get('avg_sharpe', 0):.3f}
- **Median Sharpe**: {self.summary_stats.get('median_sharpe', 0):.3f}
- **Average Drawdown**: {self.summary_stats.get('avg_drawdown_pct', 0):.2f}%

### Best/Worst

- **Best Return**: {self.summary_stats.get('best_return_pct', 0):.2f}%
- **Worst Return**: {self.summary_stats.get('worst_return_pct', 0):.2f}%
- **Best Sharpe**: {self.summary_stats.get('best_sharpe', 0):.3f}
- **Worst Sharpe**: {self.summary_stats.get('worst_sharpe', 0):.3f}

---
"""

        report += "\n## Individual Results\n\n"

        for i, result in enumerate(self.results, 1):
            assessment = self._assess_performance(result)
            symbols_str = ", ".join(result['symbols'])

            report += f"""
### {i}. {result['description']}

**Symbols**: {symbols_str}
**Assessment**: {assessment}

| Metric | Value |
|--------|-------|
| Total Return | {result['total_return_pct']:.2f}% |
| Sharpe Ratio | {result['sharpe_ratio']:.3f} |
| Sortino Ratio | {result['sortino_ratio']:.3f} |
| Max Drawdown | {result['max_drawdown_pct']:.2f}% |
| Win Rate | {result['win_rate_pct']:.2f}% |
| Profit Factor | {result['profit_factor']:.2f} |
| Total Trades | {result['total_trades']} |
| Final Equity | ${result['final_equity']:,.2f} |

"""
            if not result['success']:
                report += f"**Error**: {result['error']}\n"

            report += "---\n"

        return report
